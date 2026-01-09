import torch
from torch import nn
# 假设原始的layers.* 模块在PYTHONPATH中，或者在同一目录下
# 如果找不到，下方提供了最简化的模拟版本
try:
    from layers.Transformer_EncDec import Encoder, EncoderLayer
    from layers.SelfAttention_Family import FullAttention, AttentionLayer
    from layers.Embed import PatchEmbedding as OriginalPatchEmbedding # 重命名原始的
except ImportError:
    print("警告: layers.* 模块未找到，将使用精简版模拟模块。")
    # --- Minimal Mock for OriginalPatchEmbedding (renamed) ---
    class OriginalPatchEmbedding(nn.Module):
        def __init__(self, d_model, patch_len, stride, padding, dropout):
            super().__init__()
            self.d_model = d_model
            self.patch_len = patch_len
            self.stride = stride
            self.value_embedding = nn.Linear(patch_len, d_model)
            print(f"模拟 OriginalPatchEmbedding: patch_len={patch_len}, stride={stride}, d_model={d_model}")

        def forward(self, x): # x 应该已经是 [bs * nvars, patch_len, num_patches] permute后
                              # 或者 [bs * nvars, num_patches, patch_len]
            # 原始PatchTST的PatchEmbedding输入x的形状是 [B, N, L] (permute后)
            # 内部会处理成 [B*N, P, PL] -> [B*N, P, D]
            # 此处为了模拟，假设输入x已经是 [B*N, num_patches, patch_len]
            # 简化模拟：
            bs_nvars, num_patches, patch_len = x.shape
            # 这是非常简化的模拟，仅用于保证维度匹配
            # 真实实现中，x首先是 (bs, n_vars, seq_len)
            # 然后PatchEmbedding会将其转换为 (bs * n_vars, num_patches, d_model)
            # 和 n_vars
            # 我们需要根据PatchTST.py中PatchEmbedding的真实输入输出来模拟
            # x_enc.permute(0, 2, 1) -> [bs, nvars, seq_len]
            # 假设这是输入
            n_vars = x.shape[1]
            seq_len = x.shape[2]
            
            # 简化计算num_patches，实际更复杂并涉及padding
            num_patches_calculated = (seq_len - self.patch_len) // self.stride + 1
            
            # 模拟输出形状 [bs * nvars, num_patches, d_model]
            # 这是一个非常粗略的模拟，不执行实际的patching和embedding
            mock_output = torch.randn(x.shape[0] * n_vars, num_patches_calculated, self.d_model, device=x.device)
            return mock_output, n_vars

    # --- Minimal Mock for Transformer Encoder ---
    class Encoder(nn.Module): # (和之前一样)
        def __init__(self, attn_layers, norm_layer=None):
            super().__init__()
            self.layers = nn.ModuleList(attn_layers)
            self.norm = norm_layer
        def forward(self, x, attn_mask=None):
            attns = []
            for layer in self.layers:
                x, attn = layer(x, attn_mask=attn_mask) # 模拟层返回 (x, attn)
                if attn is not None: attns.append(attn) # 确保attn存在
            if self.norm is not None:
                x = self.norm(x)
            return x, attns if attns else None # 返回None如果attns为空

    class EncoderLayer(nn.Module): # (和之前一样)
        def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attention = attention
            self.conv1 = nn.Conv1d(d_model, d_ff, 1)
            self.conv2 = nn.Conv1d(d_ff, d_model, 1)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        def forward(self, x, attn_mask=None):
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
            x = x + self.dropout(new_x)
            y = x = self.norm1(x)
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            x = self.norm2(x + y)
            return x, attn

    class AttentionLayer(nn.Module): # (和之前一样)
        def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
            super().__init__()
            self.inner_attention = attention
            self.query_projection = nn.Linear(d_model, d_model)
            self.key_projection = nn.Linear(d_model, d_model)
            self.value_projection = nn.Linear(d_model, d_model)
            self.out_projection = nn.Linear(d_model, d_model)
        def forward(self, queries, keys, values, attn_mask=None):
            Q = self.query_projection(queries)
            K = self.key_projection(keys)
            V = self.value_projection(values)
            out, attn = self.inner_attention(Q, K, V, attn_mask)
            return self.out_projection(out), attn

    class FullAttention(nn.Module): # (和之前一样)
        def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
            super().__init__()
            self.scale = scale
            self.mask_flag = mask_flag
            self.output_attention = output_attention
            self.dropout = nn.Dropout(attention_dropout)
        def forward(self, queries, keys, values, attn_mask): # B, L, H, E or B, N, L, E
            # 简化模拟，仅保证维度和返回类型
            # 假设输入是 (B, N_patches, D_model) for Q,K,V from EncoderLayer perspective
            # FullAttention 内部可能期望多头 (B, H, N_patches, D_head)
            # 为简单起见，直接返回 values 和一个虚拟 attn
            # 实际的FullAttention非常复杂
            if self.output_attention:
                return values, torch.randn(*values.shape[:-1], values.shape[-2], device=values.device) # dummy attention
            else:
                return values, None


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# --- 新增模块 ---
class EnhancedPatchEmbedding(nn.Module):
    """
    增强的PatchEmbedding模块:
    1. 计算原始序列和其一阶差分序列的Patch Embedding。
    2. 融合这两个Embedding。
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout,
                 input_c=1, # 通常PatchEmbedding处理单变量或已经分离的多变量
                 use_diff=True):
        super().__init__()
        self.use_diff = use_diff
        self.patch_embed_orig = OriginalPatchEmbedding(d_model, patch_len, stride, padding, dropout)
        if self.use_diff:
            self.patch_embed_diff = OriginalPatchEmbedding(d_model, patch_len, stride, padding, dropout)
            # 融合层，例如一个简单的线性层或直接相加
            # 如果d_model很大，直接相加可能更好；如果想学习融合权重，用线性层
            self.fusion_layer = nn.Linear(d_model * 2, d_model)
            # 或者 self.fusion_alpha = nn.Parameter(torch.tensor(0.5)) # 用于加权和

    def forward(self, x_orig_permuted): # x_orig_permuted: [bs, nvars, seq_len]
        # 原始序列的Patch Embedding
        # OriginalPatchEmbedding期望的输入是 [B*N, num_patches, patch_len] or similar after internal processing
        # 它内部会处理 x_orig_permuted.reshape(bs*nvars, 1, seq_len) 或类似操作
        # 我们需要直接传递 [bs, nvars, seq_len] 给它，让它自行处理
        # 但为了适配模拟的OriginalPatchEmbedding，这里我们先不改变它的调用方式
        # 假设OriginalPatchEmbedding的forward现在接受 [bs, nvars, seq_len]
        
        # 为了正确调用原版PatchEmbedding(如果能加载的话)
        # 它内部处理 permute 和 reshape
        # x: [bs, seq_len, n_vars] in PatchTST Model constructor before permute
        # after permute x_enc = x_enc.permute(0, 2, 1) -> [bs, nvars, seq_len]
        # This x_orig_permuted is the input here.
        
        # 模拟的OriginalPatchEmbedding forward(self, x) 期望 x 是 [bs, nvars, seq_len]
        # 所以直接调用
        
        enc_out_orig, n_vars = self.patch_embed_orig(x_orig_permuted)
        # enc_out_orig: [bs * nvars, patch_num, d_model]

        if not self.use_diff:
            return enc_out_orig, n_vars

        # 计算一阶差分序列
        # x_orig_permuted: [bs, nvars, seq_len]
        x_diff = torch.diff(x_orig_permuted, dim=-1) # 差分后 seq_len 会减1
        
        # 对差分后的序列补齐长度以便Patching (简单用0填充在末尾)
        # 或者调整patch_len/stride，但简单填充更直接
        # 如果stride很大，可能差分后的序列太短不足以形成patch
        # 需要 careful padding or ensure seq_len is large enough
        padding_diff = x_orig_permuted.shape[-1] - x_diff.shape[-1]
        if padding_diff > 0:
             x_diff_padded = nn.functional.pad(x_diff, (0, padding_diff), mode='replicate') # 在序列末尾填充
        else:
             x_diff_padded = x_diff

        enc_out_diff, _ = self.patch_embed_diff(x_diff_padded)
        # enc_out_diff: [bs * nvars, patch_num_diff, d_model]

        # 确保 patch_num 一致，如果不一致则需要截断或填充（取决于patching策略）
        # 简单假设它们因为相同的patch_len, stride, padding 而有相同的patch_num
        # 如果差分后序列长度变化导致patch_num不同，这是个需要仔细处理的问题
        # 简单起见，这里假设它们patch_num相同
        min_patch_num = min(enc_out_orig.shape[1], enc_out_diff.shape[1])
        enc_out_orig = enc_out_orig[:, :min_patch_num, :]
        enc_out_diff = enc_out_diff[:, :min_patch_num, :]

        # 融合
        fused_emb = torch.cat((enc_out_orig, enc_out_diff), dim=-1) # [bs * nvars, patch_num, d_model * 2]
        enc_out_fused = self.fusion_layer(fused_emb) # [bs * nvars, patch_num, d_model]
        # 或者: enc_out_fused = enc_out_orig + self.fusion_alpha * enc_out_diff

        return enc_out_fused, n_vars


class IntraPatchContextEncoder(nn.Module):
    """
    Patch内部上下文编码器
    使用一个轻量级的Transformer层来增强每个Patch的表示。
    输入: [bs * nvars, patch_num, d_model] (PatchTST的Encoder的输入格式)
    输出: [bs * nvars, patch_num, d_model] (同样格式，但内容被提炼)
    """
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu", num_layers=1):
        super().__init__()
        # 注意：这里复用了主Encoder的EncoderLayer和AttentionLayer结构
        # 但这里的上下文是 "patch_num" 维度，即不同的patch之间
        # 如果要对patch内部（长度为patch_len的序列）进行编码，结构会更复杂
        # PatchTST的设计是将patch视为一个token，所以这里的上下文编码器作用于这些token序列
        # 这更像是一个 "Pre-Encoder" 或者 "Refinement Layer"
        
        # 如果目标是对 patch 内部进行编码，那么PatchEmbedding后，每个patch是d_model维
        # 就不能直接用TransformerEncoder了，除非把patch再序列化
        # 原始论文的 Patching and Embedding:
        # x: [bs, seq_len, n_vars] -> permute(0,2,1) -> [bs, n_vars, seq_len]
        # PatchTST's PatchEmbedding:
        #   Input: x_patched: [bs*n_vars, patch_len, num_patches] (after unfold and permute)
        #   value_embedding: nn.Linear(patch_len, d_model) applied on dim=-1
        #   Output: [bs*n_vars, num_patches, d_model] (after permute)

        # 所以，"IntraPatchContextEncoder"这个名字如果指代对patch内部编码，那么应该作用于
        # patch_len 这个维度。但PatchTST已经把patch_len压缩到d_model了。
        # 因此，这个模块更适合叫 "PatchSequenceRefiner" 或类似。
        # 它的作用是在主Encoder之前，对patch序列进行一次初步的上下文编码。
        
        self.refinement_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, attention_dropout=dropout, output_attention=False),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x): # x: [bs * nvars, patch_num, d_model]
        attns = [] # 如果需要收集注意力图
        for layer in self.refinement_layers:
            x, attn = layer(x) # EncoderLayer的forward是 (x, attn_mask=None)
            if attn is not None: attns.append(attn)
        x = self.norm(x)
        return x # 如果需要，也可以返回attns

class Model(nn.Module):
    """
    CaPTD-PatchTST: Context-Aware Patch with Temporal Difference Enhancement
    """
    def __init__(self, configs, patch_len=16, stride=8, use_diff_embed=True, use_intra_patch_encoder=True):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model # 新增，确保configs里有

        padding = stride # 和原始PatchTST一致

        # 1. 增强的Patching和Embedding (包含差分特征)
        self.use_diff_embed = use_diff_embed
        self.patch_embedding = EnhancedPatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout,
            use_diff=self.use_diff_embed
        )

        # 2. (可选的) Patch内部上下文编码器 / Patch序列提炼器
        self.use_intra_patch_encoder = use_intra_patch_encoder
        if self.use_intra_patch_encoder:
            self.intra_patch_encoder = IntraPatchContextEncoder(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, configs.activation, num_layers=configs.refine_layers # 新增configs.refine_layers
            )

        # 3. 主Encoder (与PatchTST相同)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), 
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # 4. 预测头 (与PatchTST相同)
        # head_nf的计算依赖于patch_num, patch_len, stride。
        # EnhancedPatchEmbedding输出的patch_num应该和OriginalPatchEmbedding一致（如果输入seq_len等相同）
        self.num_patches = int((configs.seq_len - patch_len) / stride + 2) # 和PatchTST计算方式保持一致
        self.head_nf = configs.d_model * self.num_patches
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        # ... (其他任务的头也类似初始化，这里省略以保持简洁)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)
        else: # 默认一个，避免报错
             self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)


    def _normalize_input(self, x_enc, task_type="forecast", mask=None):
        if task_type == "imputation":
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc_norm = x_enc - means
            x_enc_norm = x_enc_norm.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc_norm * x_enc_norm, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc_norm /= stdev
        else: # forecast, anomaly_detection, classification
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc_norm = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_norm /= stdev
        return x_enc_norm, means, stdev

    def _embed_and_encode(self, x_enc_processed):
        # x_enc_processed: [bs, seq_len, n_vars] (已经归一化)
        
        # 1. Permute and Patch Embedding (with temporal difference)
        x_enc_permuted = x_enc_processed.permute(0, 2, 1) # -> [bs, nvars, seq_len]
        # enc_out: [bs * nvars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc_permuted)

        # 2. (Optional) Intra-Patch Context Encoding / Patch Sequence Refinement
        if self.use_intra_patch_encoder:
            enc_out = self.intra_patch_encoder(enc_out)

        # 3. Main Encoder
        # enc_out: [bs * nvars, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out) # attns可能是None

        # Reshape and Permute for head
        # enc_out: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # enc_out: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        return enc_out, n_vars, attns


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_norm, means, stdev = self._normalize_input(x_enc, "forecast")
        
        enc_out, _, _ = self._embed_and_encode(x_enc_norm) # n_vars, attns not used here

        # Decoder
        dec_out = self.head(enc_out)  # dec_out: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1) # dec_out: [bs x target_window x nvars]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc_norm, means, stdev = self._normalize_input(x_enc, "imputation", mask=mask)
        enc_out, _, _ = self._embed_and_encode(x_enc_norm)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc): # x_mark_enc not used by default in original
        x_enc_norm, means, stdev = self._normalize_input(x_enc, "anomaly_detection")
        enc_out, _, _ = self._embed_and_encode(x_enc_norm)
        dec_out = self.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc_norm, _, _ = self._normalize_input(x_enc, "classification") # means/stdev not for de-norm
        
        # enc_out_processed: [bs x nvars x d_model x patch_num]
        enc_out_processed, n_vars, attns = self._embed_and_encode(x_enc_norm)
        
        # Decoder for classification
        output = self.flatten(enc_out_processed) # output: [bs x nvars x (d_model * patch_num)]
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1) # Flatten to [bs, nvars * d_model * patch_num]
        output = self.projection(output) # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: [Batch, SeqLen, Variates]
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc) # Pass x_mark_enc if anomaly_detection uses it
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None