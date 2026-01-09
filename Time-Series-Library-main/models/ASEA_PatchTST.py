import torch
from torch import nn
# 假设原始的layers.* 模块在PYTHONPATH中，或者在同一目录下
# 如果找不到，下方提供了最简化的模拟版本 (与您之前提供的类似)
try:
    from scipy.stats import skew
    _SCIPY_STATS_SKEW_AVAILABLE = True
except ImportError:
    _SCIPY_STATS_SKEW_AVAILABLE = False
    # 如果希望在 scipy 不可用时有一个基于 torch 的备用方案，可以在这里定义或导入
    # def skew_pytorch(data_tensor, axis=0, bias=True): ...
    pass
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import OriginalPatchEmbedding
import numpy as np

class SkewnessHandler(nn.Module):
    """
    处理数据偏态的模块。
    检测输入数据的偏态，如果超过阈值，则应用log1p变换。
    记录哪些变量被变换，以便进行逆变换。
    """
    def __init__(self, skew_threshold=0.5, warning_threshold=1.0, verbose=False):
        super().__init__()
        self.skew_threshold = skew_threshold
        self.warning_threshold = warning_threshold # 较高的阈值，用于发出更强的警告
        self.verbose = verbose
        if not _SCIPY_STATS_SKEW_AVAILABLE:
            print("警告: scipy.stats.skew 未找到或无法导入。SkewnessHandler 将无法计算偏态。")
            print("请确保已安装 scipy 库 (pip install scipy)，或者提供基于 PyTorch 的偏态计算函数。")
            self.enabled = False
            self._calculate_skewness_func = None # 没有可用的偏态计算函数
        else:
            self.enabled = True
            self._calculate_skewness_func = skew # 将 scipy.stats.skew 赋值给一个成员变量
            if self.verbose:
                print("Scipy.stats.skew 加载成功，SkewnessHandler 已启用。")

    def _calculate_skewness(self, x_np_var):
        """计算单个变量序列的偏态"""
        if not self.enabled or self._calculate_skewness_func is None:
            # 如果未启用或没有偏态计算函数，则返回0或抛出错误
            if self.verbose:
                print("Skewness calculation skipped as handler is disabled or skew function is unavailable.")
            return 0.0 # 或者可以 raise RuntimeError("Skewness function not available.")

        # scipy.stats.skew 默认使用有偏估计 (bias=False 对应三阶矩的无偏估计)
        # 对于时间序列，通常沿着时间轴计算
        s = self._calculate_skewness_func(x_np_var, axis=0, bias=True) # axis=0 因为期望的输入是 [seq_len] or [bs, seq_len]
        if np.isscalar(s):
            return s
        return np.mean(s) # 如果输入是 [bs, seq_len]，则取所有批次该变量偏态的均值 (或可调整)

    def transform(self, x):
        """
        检测偏态并对超过阈值的变量应用log1p变换。
        x: 输入张量，形状 [bs, seq_len, n_vars]
        返回:
            x_transformed: 变换后的张量
            transformed_flags:布尔张量，形状 [bs, n_vars]，标记哪些变量被变换了
        """
        if not self.enabled:
            bs, _, n_vars = x.shape
            return x, torch.zeros((bs, n_vars), dtype=torch.bool, device=x.device)

        bs, seq_len, n_vars = x.shape
        x_transformed = x.clone()
        # transformed_flags 标记每个batch的每个variable是否被转换了
        transformed_flags = torch.zeros((bs, n_vars), dtype=torch.bool, device=x.device)

        for i in range(n_vars):
            # 考虑每个batch内的序列分别计算偏态，或者整个batch一起计算
            # 为简单起见，我们对每个 (batch_sample, var) 计算偏态
            # 或者更进一步，对一个变量在所有batch样本中的数据计算整体偏态
            # 这里选择对每个 (batch_idx, var_idx) 的序列独立判断和变换
            for batch_idx in range(bs):
                var_data_np = x[batch_idx, :, i].detach().cpu().numpy()
                
                # 检查是否有足够的非零值来计算有意义的偏态，且数据是否太平坦
                if np.all(var_data_np == var_data_np[0]) or np.count_nonzero(var_data_np) < 5: # 避免对常量序列或稀疏序列计算
                    current_skewness = 0 # 假设常量序列无偏态
                else:
                    current_skewness = self._calculate_skewness(var_data_np)

                if abs(current_skewness) > self.skew_threshold:
                    if self.verbose:
                        print(f"Batch {batch_idx}, Variable {i}: Skewness {current_skewness:.2f} > threshold {self.skew_threshold}. Applying log1p.")
                    if abs(current_skewness) > self.warning_threshold and self.verbose:
                         print(f"警告: Batch {batch_idx}, Variable {i} 偏态 ({current_skewness:.2f}) 非常严重。")
                    
                    # log1p 要求输入 >= 0
                    x_clamped = torch.clamp(x[batch_idx, :, i], min=0.0)
                    x_transformed[batch_idx, :, i] = torch.log1p(x_clamped)
                    transformed_flags[batch_idx, i] = True
        return x_transformed, transformed_flags

    def inverse_transform(self, x_pred, transformed_flags):
        """
        对预测结果中被log1p变换过的变量应用expm1逆变换。
        x_pred: 预测张量，形状 [bs, pred_len, n_vars]
        transformed_flags: 布尔张量，形状 [bs, n_vars]
        返回: 逆变换后的张量
        """
        if not self.enabled or torch.sum(transformed_flags) == 0:
            return x_pred

        x_inv = x_pred.clone()
        bs, pred_len, n_vars = x_pred.shape
        for i in range(n_vars):
            for batch_idx in range(bs):
                if transformed_flags[batch_idx, i]:
                    x_inv[batch_idx, :, i] = torch.expm1(x_pred[batch_idx, :, i])
        return x_inv


class EventFeatureExtractor(nn.Module):
    """
    从原始时间序列数据中提取事件特征。
    """
    def __init__(self, activity_threshold=1e-6):
        super().__init__()
        self.activity_threshold = activity_threshold

    def forward(self, x_raw):
        """
        x_raw: 原始输入张量，形状 [bs, seq_len, n_vars]
        返回:
            event_features: 事件特征张量，形状 [bs, seq_len, n_vars, num_event_features] (这里是2)
        """
        # 1. 活动状态 (s_t): 1 if x > threshold else 0
        activity_status = (x_raw > self.activity_threshold).float() # [bs, seq_len, n_vars]

        # 2. 活动开始标记 (c_t): 1 if s_t=1 and s_{t-1}=0 else 0
        # 填充s_{t-1}。在时间序列开始处补0 (dim=1是seq_len维度)
        # (left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad)
        # 我们只需要在seq_len维度左边填充1个0
        padded_activity_status = nn.functional.pad(activity_status, (0, 0, 1, 0), value=0) # pads dim 1 (seq_len)
        
        s_t_minus_1 = padded_activity_status[:, :-1, :] # [bs, seq_len, n_vars]
        
        activity_started = (activity_status == 1) & (s_t_minus_1 == 0)
        activity_started = activity_started.float() # [bs, seq_len, n_vars]

        # 3. 拼接特征
        # unsqueeze(-1) 在最后增加一个维度用于拼接特征
        event_features = torch.stack(
            [activity_status, activity_started],
            dim=-1 # 新的最后一个维度是特征维度
        ) # event_features: [bs, seq_len, n_vars, 2]
        return event_features

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): super().__init__(); self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars # n_vars 在这里可能不再直接使用，因为PatchTST通常独立处理变量或在头部合并
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    def forward(self, x): # x: [bs, nvars, d_model, num_patches]
        # PatchTST的典型做法是将nvars视为batch的一部分，或者在head处独立处理再合并
        # 如果 x 是 [bs * nvars, d_model, num_patches]
        # 或者在输入时已经是 [bs, nvars * d_model * num_patches] (较少见)
        # 假设这里的输入是encoder的直接输出reshape后的 [bs, nvars, d_model, num_patches]
        # 并且我们希望为每个nvar独立预测，或者合并它们
        # FlattenHead 的原始实现可能假设 n_vars 维度已经被整合或将在之后处理
        # 这里的 x 传入时是 [bs, n_vars, d_model, num_patches] (来自Model._embed_and_encode的输出)
        # 我们需要将其展平为 [bs * n_vars, d_model * num_patches] 给Linear层
        
        # bs, nvars, d_model, num_patches = x.shape
        # x = x.reshape(bs * nvars, d_model * num_patches) # 兼容原始FlattenHead的Linear层输入
        #                                                  # 或者修改Linear层以接受不同的输入形状
        # 如果FlattenHead的 nf 参数已经是 d_model * num_patches
        # x = x.permute(0, 1, 3, 2) # -> [bs, nvars, num_patches, d_model]
        # x = self.flatten(x) # -> [bs, nvars, num_patches * d_model]
        
        # 基于PatchTST原版FlattenHead，它处理 [B*N, P, D], 然后展平 P*D
        # 我们的 enc_out 在送入 head 之前是 [bs, nvars, d_model, patch_num]
        # 如果要适配PatchTST的FlattenHead，输入应是 [bs * nvars, patch_num, d_model]
        # Model._embed_and_encode 的输出 enc_out 是 [bs, nvars, d_model, patch_num]
        # 我们先在 Model.forecast 中调整维度再送入 head
        
        # 假设输入 x 已经是 [bs * n_vars_count, features_per_var]
        # features_per_var = d_model * num_patches
        x = self.flatten(x) # 如果输入已经是 [bs * n_vars, num_patches, d_model], flatten后是 [bs*n_vars, num_patches*d_model]
        x = self.linear(x)
        x = self.dropout(x)
        return x


class AugmentedPatchEmbedding(nn.Module):
    """
    增强的PatchEmbedding模块:
    1. 接收 值数据流 (可能经过对数变换和归一化) 和 事件特征数据流 (来自原始数据)。
    2. 将它们在特征维度上拼接。
    3. 使用OriginalPatchEmbedding对这个增强后的序列进行Patch化和Embedding。
    """
    def __init__(self, d_model, patch_len, stride, padding_for_original_embed, dropout, # padding_for_original_embed 是给 OriginalPatchEmbedding 构造函数用的
                 num_event_features=2):
        super().__init__()
        self.input_dims_per_step = 1 + num_event_features # 1 (值流) + num_event_features

        # 实例化您提供的 OriginalPatchEmbedding
        # 注意：您 OriginalPatchEmbedding 的构造函数中的 padding 参数是用于 nn.ReplicationPad1d((0, padding))
        # 它指定在序列的右边填充多少。
        self.actual_patch_embedder = OriginalPatchEmbedding(
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            padding=padding_for_original_embed, # 这是传递给 OriginalPatchEmbedding 内部的 nn.ReplicationPad1d 的 padding 值
            dropout=dropout,
            input_dims_per_step=self.input_dims_per_step
        )
        print(f"AugmentedPatchEmbedding: input_dims_per_step={self.input_dims_per_step} (1 value + {num_event_features} events)")
        print(f"  Passing padding={padding_for_original_embed} to OriginalPatchEmbedding.")


    def forward(self, x_values_permuted, event_features_permuted):
        """
        x_values_permuted: 值数据流, 形状 [bs, nvars, seq_len]。
                           (已归一化, 可能已log变换)
        event_features_permuted: 事件特征流, 形状 [bs, nvars, seq_len, num_event_features]。
                                (来自原始数据, 未归一化)
        """
        # 1. 拼接特征
        # x_values_permuted: [bs, nvars, seq_len] -> [bs, nvars, seq_len, 1]
        # event_features_permuted: [bs, nvars, seq_len, num_event_features]
        # 目标: [bs, nvars, seq_len, 1 + num_event_features]

        augmented_input_permuted = torch.cat(
            [x_values_permuted.unsqueeze(-1), event_features_permuted],
            dim=-1
        )
        # augmented_input_permuted 形状: [bs, nvars, seq_len, self.input_dims_per_step]
        # 这正是您 OriginalPatchEmbedding 的 forward 方法所期望的输入形状。

        # 2. 直接调用 self.actual_patch_embedder 的 forward 方法
        # 它期望输入 [bs, n_vars, seq_len, input_dims_per_step]
        # 并返回 (dropout(x_final_embedding), n_vars_count)
        # x_final_embedding 形状是 (bs*n_vars, num_patches, d_model)
        patched_representation, n_vars_out = self.actual_patch_embedder(augmented_input_permuted)

        return patched_representation, n_vars_out


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8, skew_threshold=0.5, verbose_skew=False):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_vars = configs.enc_in # Number of variables

        self.event_feature_extractor = EventFeatureExtractor()
        self.num_event_features = 2 # activity_status, activity_started
        
        self.skew_handler = SkewnessHandler(skew_threshold=skew_threshold, verbose=verbose_skew)

        # Padding for PatchTST's OriginalPatchEmbedding is often 'replicate' or specific
        # and handled internally or via a padding layer before unfold.
        # The 'padding' argument here might be for the nn.Conv1d inside OriginalPatchEmbedding if it uses one for patching.
        # Or it could refer to the padding amount for nn.Unfold.
        # PatchTST's OriginalPatchEmbedding might use nn.ReplicationPad1d((0, stride))
        # For now, we assume OriginalPatchEmbedding handles its padding.
        # The 'padding' argument for an nn.Conv1d based patcher would be (patch_len - stride)//2
        # Let's assume `padding` here refers to the padding argument for `OriginalPatchEmbedding` constructor if needed.
        # If OriginalPatchEmbedding uses `unfold`, it might not need a padding argument in its constructor
        # but rather apply padding to the input tensor directly.
        # We use the `stride` as a placeholder for what might be the padding amount in some contexts.
        padding_for_embedder = stride # This is a common setup in some PatchTST versions if padding is related to stride

        self.patch_embedding = AugmentedPatchEmbedding(
            configs.d_model, patch_len, stride, padding_for_embedder, configs.dropout,
            num_event_features=self.num_event_features
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), # output_attention is False for forecast
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model) # Typical Transformer encoder norm
            # Original PatchTST used: norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
            # Using LayerNorm as it's more standard for Transformers if applied per patch embedding.
            # If BatchNorm is desired, it should be applied correctly to the [B*N, num_patches, d_model] tensor.
            # For example, on the feature dimension (d_model). So permute, BatchNorm1d, permute back.
        )

        # Number of patches: PatchTST's formula given its padding strategy
        self.num_patches = (configs.seq_len - patch_len) // stride + 1 # Standard formula with no extra padding beyond seq_len
                                                                     # Or, if specific padding in PatchTST's OriginalPatchEmbedding:
                                                                     # (configs.seq_len + stride - patch_len) // stride + 1  (if pad right by stride)
                                                                     # (configs.seq_len + 2*stride - patch_len) // stride + 1 (if pad both sides by stride)
        # Let's use the formula consistent with many PatchTST public implementations (pad right by stride, then unfold)
        # This often results in: num_patches = (seq_len + stride - patch_len) // stride + 1
        # Given the user code had `+2` in `int((configs.seq_len - patch_len) / stride + 2)`,
        # this implies a formula like `num_patches = (seq_len - patch_len + 2*stride) // stride`.
        # Or, if `padding_patch_layer = nn.ReplicationPad1d((0, stride))`, then L_new = L + stride.
        # Patches = floor((L_new - P)/S + 1) = floor((L+S-P)/S + 1).
        # If S (stride in unfold) == stride (padding amount), then ( (L-P)/S + 2 ).
        self.num_patches = (configs.seq_len - patch_len) // stride + 2 # Matching the original code's implied calculation.


        self.head_nf = configs.d_model * self.num_patches
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # The FlattenHead's n_vars argument is not used in its forward if x is shaped as [B*N, ...].
            # enc_in (configs.enc_in) is n_vars.
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        # ... (other task heads remain similar, ensure FlattenHead input is [B*N, num_patches, d_model])

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2) # Flattens [num_patches, d_model]
            self.dropout = nn.Dropout(configs.dropout)
            # Input to projection: bs * n_vars * num_patches * d_model
            self.projection = nn.Linear(
                configs.enc_in * self.head_nf, configs.num_class) # This assumes n_vars are concatenated
        else: # Default
             self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)


    def _normalize_input(self, x_enc_values, task_type="forecast", mask=None):
        # x_enc_values: [bs, seq_len, n_vars], (potentially log-transformed) values
        # Normalization is applied per-variable (channel-independence in PatchTST)
        # RevIN: Subtract mean, divide by std dev, applied per instance per variable.
        
        # Permute to [bs, n_vars, seq_len] for per-variable statistics
        x_enc_p = x_enc_values.permute(0, 2, 1) # [bs, n_vars, seq_len]
        
        if task_type == "imputation":
            # Mask should be [bs, seq_len, n_vars], permute it too
            mask_p = mask.permute(0, 2, 1) if mask is not None else torch.ones_like(x_enc_p)
            
            active_elements = mask_p.float()
            num_active = torch.sum(active_elements, dim=-1, keepdim=True) # [bs, n_vars, 1]
            num_active[num_active == 0] = 1 # Avoid division by zero

            sum_val = torch.sum(x_enc_p * active_elements, dim=-1, keepdim=True) # [bs, n_vars, 1]
            means = sum_val / num_active
            means = means.detach() # [bs, n_vars, 1]

            x_enc_norm_p = x_enc_p - means
            x_enc_norm_p = x_enc_norm_p.masked_fill(active_elements == 0, 0)
            
            sum_sq_err = torch.sum(x_enc_norm_p * x_enc_norm_p * active_elements, dim=-1, keepdim=True)
            stdev = torch.sqrt(sum_sq_err / num_active + 1e-5) # [bs, n_vars, 1]
            stdev = stdev.detach()
            x_enc_norm_p /= stdev
        else: # forecast, anomaly_detection, classification
            means = torch.mean(x_enc_p, dim=-1, keepdim=True).detach() # [bs, n_vars, 1]
            x_enc_norm_p = x_enc_p - means
            stdev = torch.sqrt(
                torch.var(x_enc_norm_p, dim=-1, keepdim=True, unbiased=False) + 1e-5 # unbiased=False for consistency if comparing
            ).detach() # [bs, n_vars, 1]
            x_enc_norm_p /= stdev
            
        # Permute back to [bs, seq_len, n_vars]
        x_enc_norm_values = x_enc_norm_p.permute(0, 2, 1) # [bs, seq_len, n_vars]
        
        # Means and stdev should be [bs, 1, n_vars] for easy broadcasting during denorm
        means_out = means.permute(0, 2, 1) # [bs, 1, n_vars]
        stdev_out = stdev.permute(0, 2, 1) # [bs, 1, n_vars]
        
        return x_enc_norm_values, means_out, stdev_out


    def _embed_and_encode(self, x_norm_values, event_features):
        # x_norm_values: [bs, seq_len, n_vars] (normalized, possibly log-transformed values)
        # event_features: [bs, seq_len, n_vars, num_event_features] (from raw data)
        
        # Permute for AugmentedPatchEmbedding's expected input format
        x_norm_values_permuted = x_norm_values.permute(0, 2, 1) # [bs, n_vars, seq_len]
        event_features_permuted = event_features.permute(0, 3, 1, 2) # [bs, num_event_features, seq_len, n_vars]
                                                                     # This needs to be [bs, n_vars, seq_len, num_event_features]
        event_features_permuted = event_features.permute(0, 2, 1, 3) # [bs, n_vars, seq_len, num_event_features]

        # Augmented Patch Embedding
        # enc_out: [bs * n_vars, patch_num, d_model]
        enc_out, n_vars_count = self.patch_embedding(x_norm_values_permuted, event_features_permuted)
        
        # Main Encoder
        enc_out, attns = self.encoder(enc_out) # Input: [B*N, P, D], Output: [B*N, P, D]

        # Reshape for head: [bs, n_vars, num_patches, d_model] then permute for FlattenHead
        # FlattenHead might expect [bs * n_vars, num_patches, d_model] or [bs, n_vars, num_patches*d_model]
        # Standard PatchTST head input: [bs * n_vars, num_patches, d_model]
        # enc_out is already [bs * n_vars_count, self.num_patches, self.d_model]
        
        # For classification, we might need to bring n_vars back explicitly
        if self.task_name == 'classification':
             enc_out = enc_out.reshape(-1, n_vars_count, self.num_patches, self.d_model)
             # enc_out is now [bs, n_vars, num_patches, d_model]

        return enc_out, n_vars_count, attns


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Extract event features from raw data
        # x_enc: [bs, seq_len, n_vars]
        event_features = self.event_feature_extractor(x_enc) # [bs, seq_len, n_vars, num_event_f]

        # 2. Skewness check and conditional log transform on raw values
        x_values, transformed_vars_flags = self.skew_handler.transform(x_enc) # x_values: [bs, seq_len, n_vars], flags: [bs, n_vars]

        # 3. Normalize the (potentially log-transformed) value stream
        x_norm_values, means, stdev = self._normalize_input(x_values, "forecast") # x_norm_values: [bs, seq_len, n_vars]
                                                                               # means, stdev: [bs, 1, n_vars]
        
        # 4. Embedding and Encoding
        # enc_out for forecast will be [bs * n_vars, num_patches, d_model]
        enc_out, n_vars_count, _ = self._embed_and_encode(x_norm_values, event_features)
        
        # 5. Prediction Head
        # Input to FlattenHead: [bs * n_vars, num_patches, d_model]
        # Output from FlattenHead: [bs * n_vars, pred_len]
        dec_out_flat = self.head(enc_out) # Pass [bs * n_vars, num_patches, d_model]
        
        # Reshape dec_out_flat to [bs, n_vars, pred_len] then permute to [bs, pred_len, n_vars]
        dec_out = dec_out_flat.reshape(-1, n_vars_count, self.pred_len).permute(0, 2, 1)
                                                                        # [bs, pred_len, n_vars]

        # 6. Denormalize
        dec_out = dec_out * stdev # stdev is [bs, 1, n_vars], broadcasts correctly
        dec_out = dec_out + means # means is [bs, 1, n_vars], broadcasts correctly
        
        # 7. Inverse transform for skewness (if applied)
        dec_out = self.skew_handler.inverse_transform(dec_out, transformed_vars_flags)
        
        return dec_out

    # --- Other task methods (imputation, anomaly_detection, classification) would follow a similar pattern ---
    # 1. Event feature extraction from raw x_enc
    # 2. Skewness handling on raw x_enc
    # 3. Normalization of (skew-transformed) values
    # 4. Embedding and Encoding
    # 5. Head processing
    # 6. Denormalization and inverse skew transform (for regression tasks)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        event_features = self.event_feature_extractor(x_enc)
        x_values, transformed_vars_flags = self.skew_handler.transform(x_enc)
        x_norm_values, means, stdev = self._normalize_input(x_values, "imputation", mask=mask)
        
        enc_out, n_vars_count, _ = self._embed_and_encode(x_norm_values, event_features)
        
        dec_out_flat = self.head(enc_out)
        dec_out = dec_out_flat.reshape(-1, n_vars_count, self.seq_len).permute(0, 2, 1)
        
        dec_out = dec_out * stdev + means
        dec_out = self.skew_handler.inverse_transform(dec_out, transformed_vars_flags)
        return dec_out

    def anomaly_detection(self, x_enc): # Simplified, adjust as needed
        event_features = self.event_feature_extractor(x_enc)
        x_values, transformed_vars_flags = self.skew_handler.transform(x_enc)
        x_norm_values, means, stdev = self._normalize_input(x_values, "anomaly_detection")
        
        enc_out, n_vars_count, _ = self._embed_and_encode(x_norm_values, event_features)

        dec_out_flat = self.head(enc_out)
        dec_out = dec_out_flat.reshape(-1, n_vars_count, self.seq_len).permute(0, 2, 1)

        dec_out = dec_out * stdev + means
        dec_out = self.skew_handler.inverse_transform(dec_out, transformed_vars_flags)
        return dec_out # Typically, anomaly score is derived from reconstruction error

    def classification(self, x_enc, x_mark_enc):
        event_features = self.event_feature_extractor(x_enc)
        x_values, transformed_vars_flags = self.skew_handler.transform(x_enc) # For consistency, though inverse won't be applied to logits
        x_norm_values, _, _ = self._normalize_input(x_values, "classification") # No means/stdev needed for denorm
        
        # enc_out for classification is [bs, n_vars, num_patches, d_model]
        enc_out_classified, _, _ = self._embed_and_encode(x_norm_values, event_features)
        
        # enc_out_classified: [bs, n_vars, num_patches, d_model]
        output = self.flatten(enc_out_classified) # -> [bs, n_vars, num_patches * d_model]
                                                  # head_nf = num_patches * d_model
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1) # -> [bs, n_vars * num_patches * d_model]
        output = self.projection(output) # Linear(n_vars * head_nf, num_class)
        return output


    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc) # Pass other args if needed
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc) # Pass other args if needed
            return dec_out
        return None
