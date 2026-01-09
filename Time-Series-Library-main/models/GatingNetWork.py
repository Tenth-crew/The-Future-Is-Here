import torch
import torch.nn as nn
import warnings

# ==================== 新增：上下文编码器 ====================
class ContextEncoder(nn.Module):
    """
    使用1D-CNN从输入序列(seq_len)中提取一个固定大小的上下文向量。
    """
    def __init__(self, input_channels, output_dim=32):
        super(ContextEncoder, self).__init__()
        # 1D-CNN期望的输入形状为 (batch, channels, seq_len)
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # 使用自适应平均池化来处理可变seq_len并输出一个固定大小的向量
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x 形状: [batch_size, seq_len, input_channels]
        # 调整维度以适应Conv1d
        x = x.permute(0, 2, 1) # -> [batch_size, input_channels, seq_len]
        context_vector = self.network(x) # -> [batch_size, output_dim]
        return context_vector

# =========================================================

class GatingNetwork(nn.Module):
    # ... (这个类保持不变)
    def __init__(self, input_dim, num_base_models, hidden_dim, dropout_rate=0.1):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_base_models),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

# ==================== 修改：GatedFusionModel ====================
class GatedFusionModel(nn.Module):
    def __init__(self, num_base_models, c_out, gating_input_dim, gating_hidden_dim, gating_dropout_rate, 
                 use_context_aware=False, seq_len=96, enc_in=5, context_output_dim=32):
        super(GatedFusionModel, self).__init__()
        self.num_base_models = num_base_models
        self.c_out = c_out
        self.use_context_aware = use_context_aware

        final_gating_input_dim = gating_input_dim
        self.context_encoder = None

        if self.use_context_aware:
            # 初始化上下文编码器
            self.context_encoder = ContextEncoder(input_channels=enc_in, output_dim=context_output_dim)
            final_gating_input_dim += context_output_dim
        
        # 使用最终计算出的维度来初始化门控网络
        self.gating_network = GatingNetwork(final_gating_input_dim, num_base_models, gating_hidden_dim, gating_dropout_rate)

    def forward(self, stacked_features, context_input=None):
        # stacked_features 形状: [batch_size, pred_len, num_base_models * c_out]
        # context_input (即 batch_x) 形状: [batch_size, seq_len, enc_in]
        
        gating_input = stacked_features
        bs, pred_len, _ = stacked_features.shape

        if self.use_context_aware:
            if context_input is None:
                warnings.warn("GatedFusionModel is set to use context, but none was provided.")
            else:
                # 1. 使用编码器提取上下文向量
                context_vector = self.context_encoder(context_input) # -> [batch_size, context_output_dim]
                
                # 2. 扩展/重复上下文向量以匹配 pred_len
                context_features = context_vector.unsqueeze(1).repeat(1, pred_len, 1) # -> [batch_size, pred_len, context_output_dim]
                
                # 3. 拼接特征
                gating_input = torch.cat([stacked_features, context_features], dim=-1)

        # 使用最终的gating_input计算动态权重
        dynamic_weights = self.gating_network(gating_input)
        
        # ... (后续的加权求和逻辑保持不变) ...
        reshaped_features = stacked_features.view(bs, pred_len, self.num_base_models, self.c_out)
        weights_view = dynamic_weights.unsqueeze(-1)
        final_outputs = (reshaped_features * weights_view).sum(dim=2)

        return final_outputs, dynamic_weights
    
class ContextOnlyPredictor(nn.Module):
    """
    一个仅依赖历史上下文进行预测的模型，用于验证 ContextEncoder 的独立预测能力。
    它接收 context_input，忽略 stacked_features。
    """
    def __init__(self, args):
        super(ContextOnlyPredictor, self).__init__()
        # 复用 ContextEncoder 来提取历史特征
        self.context_encoder = ContextEncoder(input_channels=args.enc_in, output_dim=args.context_output_dim)
        
        # 添加一个“预测头”，将上下文向量映射到最终的预测形状
        self.prediction_head = nn.Sequential(
            nn.Linear(args.context_output_dim, args.meta_hidden_dim), # 增加一个隐藏层
            nn.ReLU(),
            nn.Linear(args.meta_hidden_dim, args.pred_len * args.c_out)
        )
        
        self.pred_len = args.pred_len
        self.c_out = args.c_out

    def forward(self, stacked_features, context_input=None):
        # 这个模型故意忽略 stacked_features
        if context_input is None:
            raise ValueError("ContextOnlyPredictor 需要 context_input，但没有提供。")

        # 1. 使用编码器提取上下文向量
        context_vector = self.context_encoder(context_input) # -> [batch_size, context_output_dim]

        # 2. 使用预测头进行预测
        prediction_flat = self.prediction_head(context_vector) # -> [batch_size, pred_len * c_out]

        # 3. 将展平的预测重塑为正确的输出形状
        final_outputs = prediction_flat.view(-1, self.pred_len, self.c_out)

        # 为了与现有框架兼容，返回两个值，第二个可以是 None
        return final_outputs, None