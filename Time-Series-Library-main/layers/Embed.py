import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class OriginalPatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout, input_dims_per_step=1):
        super(OriginalPatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.input_dims_per_step = input_dims_per_step # 新增：每个时间点的特征维度

        # 调整padding层以适应多维输入
        # nn.ReplicationPad1d 期望输入 (N, C, L_in) or (C, L_in)
        # 我们的输入在forward中会是 (bs * n_vars, seq_len, input_dims_per_step)
        # 需要先 permute 成 (bs * n_vars, input_dims_per_step, seq_len) 再padding
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding)) # padding应用在最后一个维度(seq_len)

        # Backbone, Input encoding
        # 输入维度现在是 patch_len * input_dims_per_step
        self.value_embedding = nn.Linear(patch_len * input_dims_per_step, d_model, bias=False)

        # Positional embedding (与原始代码一致，作用于d_model维度)
        # 假设PositionalEmbedding的max_len足够覆盖patch数量
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_augmented_permuted):
        # x_augmented_permuted 的期望输入形状: [bs, n_vars, seq_len, input_dims_per_step]
        # 这是从 AugmentedPatchEmbedding 传过来的
        
        bs, n_vars_count, seq_len_val, aug_dims = x_augmented_permuted.shape
        
        if aug_dims != self.input_dims_per_step:
            raise ValueError(f"Input feature dimension per step ({aug_dims}) "
                             f"does not match instantiated input_dims_per_step ({self.input_dims_per_step})")

        # 为了进行Patching，我们将bs和n_vars合并，并将特征维度视为"通道"
        # 目标形状给padding_patch_layer: (N, C, L) -> (bs*n_vars, input_dims_per_step, seq_len)
        x_reshaped = x_augmented_permuted.reshape(bs * n_vars_count, seq_len_val, aug_dims)
        x_for_padding_permuted = x_reshaped.permute(0, 2, 1) # -> (bs*n_vars, input_dims_per_step, seq_len)

        # 1. Padding
        x_padded = self.padding_patch_layer(x_for_padding_permuted) # Output: (bs*n_vars, input_dims_per_step, seq_len_padded)

        # 2. Patching (unfold)
        # unfold作用于最后一个维度 (seq_len_padded)
        # 输出: (bs*n_vars, input_dims_per_step, num_patches, patch_len)
        x_unfolded = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 调整维度以匹配value_embedding的期望输入
        # 目标: (bs*n_vars, num_patches, patch_len, input_dims_per_step)
        x_patched = x_unfolded.permute(0, 2, 3, 1)
        
        # 3. Flatten patch_len 和 input_dims_per_step 维度
        #  (bs*n_vars, num_patches, patch_len * input_dims_per_step)
        num_patches = x_patched.shape[1] # 保存num_patches给PositionalEmbedding
        x_flattened_patches = torch.reshape(x_patched,
                                            (x_patched.shape[0], num_patches, self.patch_len * self.input_dims_per_step))
        
        # 4. Input encoding (value_embedding)
        x_embedded = self.value_embedding(x_flattened_patches) # Output: (bs*n_vars, num_patches, d_model)
        
        # 5. Add Positional embedding
        x_final_embedding = x_embedded + self.position_embedding(x_embedded)
        
        return self.dropout(x_final_embedding), n_vars_count