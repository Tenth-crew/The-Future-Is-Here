import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    为输入序列添加标准的正弦/余弦位置编码。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 形状: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HistoricalEncoder(nn.Module):
    """
    将输入的历史序列编码为 Key 和 Value，为 Cross-Attention 做准备。
    """
    def __init__(self, enc_in, d_model, dropout=0.1):
        super(HistoricalEncoder, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(enc_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, enc_in]
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.layer_norm(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        return key, value

class CrossAttentionLayer(nn.Module):
    """
    封装了多头注意力、残差连接和层归一化。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # PyTorch > 1.9.0 已经支持 batch_first=True
        # query: [batch_size, pred_len, d_model]
        # key, value: [batch_size, seq_len, d_model]
        residual = query
        attn_output, _ = self.attention(query, key, value)
        # output = self.dropout(attn_output) + residual
        output = self.dropout(attn_output)# 不要残差的方案
        output = self.layer_norm(output)
        return output

class GatingNetwork(nn.Module):
    """
    门控网络，根据融合后的特征为每个基模型生成动态权重。
    (从 GatingNetWork.py 复制一份到这里，确保模块独立)
    """
    def __init__(self, input_dim, num_base_models, hidden_dim, softmax_temperature, dropout_rate=0.1):
        super(GatingNetwork, self).__init__()
        self.softmax_temperature = softmax_temperature
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_base_models),
            # nn.Softmax(dim=-1) # 在 forward 里加上 Softmax，forward会控制温度参数
        )
    def forward(self, x):
        logits = self.network(x)
        # 在 forward 的最后应用带温度的 Softmax
        return nn.Softmax(dim=-1)(logits / self.softmax_temperature)


class Model(nn.Module):
    """
    这是我们将暴露给 Exp 函数的主模型类，命名为 Model 以符合工厂模式。
    它整合了历史编码、Cross-Attention 和门控网络。
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_base_models = len(args.base_model_names)
        self.c_out = 1 if args.features in ['S', 'MS'] else args.c_out
        self.d_model = args.d_model

        # 基模型原始预测拼接后的维度
        base_model_pred_dim = self.num_base_models * self.c_out
        
        # 模块 1: 历史序列编码器
        self.historical_encoder = HistoricalEncoder(enc_in=args.enc_in, d_model=args.d_model)

        # 模块 2: 将基模型的原始预测投影到 d_model，作为 Query
        self.query_projection = nn.Linear(base_model_pred_dim, args.d_model)

        # 模块 3: Cross-Attention 层
        self.cross_attention = CrossAttentionLayer(d_model=args.d_model, n_heads=args.n_heads)
        
        # 模块 4: 最终的门控网络
        # 它的输入来自三部分拼接：
        # 1. Cross-Attention 的输出 (d_model)
        # 2. 历史信息的一个全局上下文表示 (d_model)
        # 3. 基模型的原始预测 (base_model_pred_dim)
        # gating_input_dim = args.d_model + args.d_model + base_model_pred_dim
        gating_input_dim = args.d_model + base_model_pred_dim #不要历史信息的全局上下文表示
        self.gating_network = GatingNetwork(gating_input_dim, self.num_base_models, args.meta_hidden_dim, args.softmax_temperature, args.meta_dropout_rate)

    def forward(self, stacked_features, context_input):
        # stacked_features (基模型预测): [B, pred_len, base_model_pred_dim]
        # context_input (历史序列 batch_x): [B, seq_len, enc_in]
        
        # 1. 生成 Query
        query = self.query_projection(stacked_features) # -> [B, pred_len, d_model]

        # 2. 编码历史信息，生成 Key 和 Value
        key, value = self.historical_encoder(context_input) # -> [B, seq_len, d_model]

        # 3. 执行 Cross-Attention
        attn_output = self.cross_attention(query, key, value) # -> [B, pred_len, d_model]

        # 4. 准备门控网络的输入
        # history_context_vector = value.mean(dim=1).unsqueeze(1).repeat(1, stacked_features.shape[1], 1)

        # 5. 拼接所有信息并生成动态权重
        # fusion_input = torch.cat([attn_output, history_context_vector, stacked_features], dim=-1)
        fusion_input = torch.cat([attn_output, stacked_features], dim=-1)
        dynamic_weights = self.gating_network(fusion_input)

        # 6. 使用权重进行加权求和
        bs, pred_len, _ = stacked_features.shape
        reshaped_features = stacked_features.view(bs, pred_len, self.num_base_models, self.c_out)
        weights_view = dynamic_weights.unsqueeze(-1)
        final_outputs = (reshaped_features * weights_view).sum(dim=2)

        return final_outputs, dynamic_weights