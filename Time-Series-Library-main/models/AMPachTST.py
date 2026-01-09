import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

# Assuming layers.Transformer_EncDec and layers.SelfAttention_Family
# and layers.Embed are in the python path or same directory.
# If not, you might need to mock them up for the code to be runnable standalone.
# For demonstration, let's create minimal mock-ups if they are not available.

try:
    from layers.Transformer_EncDec import Encoder, EncoderLayer
    from layers.SelfAttention_Family import FullAttention, AttentionLayer
    from layers.Embed import PatchEmbedding
except ImportError:
    print("Warning: layers.* modules not found. Using minimal mocks.")
    # Minimal Mock for PatchEmbedding
    class PatchEmbedding(nn.Module):
        def __init__(self, d_model, patch_len, stride, padding, dropout):
            super().__init__()
            self.d_model = d_model
            self.patch_len = patch_len
            self.stride = stride
            # Simplified: just a linear layer to project patch to d_model
            self.value_embedding = nn.Linear(patch_len, d_model)
            print(f"Mock PatchEmbedding: patch_len={patch_len}, stride={stride}, d_model={d_model}")

        def forward(self, x):
            # x: [bs, seq_len, n_vars] -> permuted to [bs, n_vars, seq_len] in model
            # Expected input to PatchEmbedding in original code: [bs * n_vars, seq_len_for_patching, 1]
            # Here, for simplicity, assume x is already [bs * n_vars, num_patches, patch_len]
            # Or rather, let's simulate the patching process roughly
            n_vars = x.shape[1] # x is [bs, n_vars, seq_len] after permute in main model
            bs = x.shape[0]
            seq_len = x.shape[2]

            num_patches = (seq_len - self.patch_len) // self.stride + 1
            # This is a gross simplification for mock, real one is more complex
            # Patches are [bs * n_vars, num_patches, patch_len]
            # We'll just take first patch_len elements and project
            # This is NOT how actual patching works, just for dimensionality
            # Proper patching would involve unfolding or strided slicing
            
            # The original PatchEmbedding reshapes x to be [batch_size * num_variates, patch_len, 1] 
            # and then applies a linear layer.
            # Let's simulate its output shape: [bs * n_vars, num_patches, d_model]
            
            # Create dummy patches
            # This is a placeholder for actual patching logic
            # In reality, you'd use x.unfold(...) or similar
            # For now, let's assume x is [bs * nvars, seq_len_for_patching, 1]
            # and we create patches from this
            
            # Correcting the input assumption for mock based on original PatchTST.py:
            # x_enc is permuted to [bs, n_vars, seq_len]
            # then reshaped for PatchEmbedding to something like [bs * n_vars, seq_len, 1]
            # then patched.
            # Let's mimic the output shape:
            # Output: [bs * n_vars, num_patches, d_model]
            
            # Calculate num_patches based on how PatchEmbedding calculates it
            # (seq_len - patch_len) / stride + 1 (integer division)
            # The +2 in original head_nf implies padding adds effective patches.
            # For simplicity, let's use the formula from head_nf for num_patches
            # effective_seq_len = seq_len + padding (from patch_embedding if padding is not 0)
            # num_patches = int((seq_len - self.patch_len)/self.stride + 2) # from original head_nf calc
            # This calculation for num_patches is specific to how head_nf was built.
            # A more robust way for PatchEmbedding:
            padding_val = self.stride # as per original: padding = stride
            x_padded = torch.nn.functional.pad(x, (0,0, padding_val, padding_val)) # Pad sequence dim
            seq_len_padded = x_padded.shape[2]
            
            num_patches = (seq_len_padded - self.patch_len) // self.stride + 1


            # This is a very rough mock
            # The patch_embedding permutes input x from (bs, n_vars, seq_len)
            # to (bs*n_vars, patch_len, num_patches) and then (bs*n_vars, num_patches, patch_len)
            # then applies linear layer
            
            # Let's just create output of the right shape for now for the mock
            # The input to PatchEmbedding in the code is x_enc.permute(0,2,1) which is [bs, n_vars, seq_len]
            # PatchEmbedding internally unnSqueezes and permutes to make it (bs*n_vars, num_patches, patch_len)
            # then projects to d_model.
            # Output should be (bs*n_vars, num_patches, d_model)
            
            # A simple way to get the shapes roughly right for the mock
            n_vars_internal = x.shape[1]
            x_for_patching = x.permute(0, 2, 1) # [bs, seq_len, n_vars]
            x_for_patching = x_for_patching.reshape(bs * n_vars_internal, seq_len, 1) # Matches expectation somewhat

            # Simplified patching:
            patches = []
            for i in range(0, seq_len - self.patch_len + 1, self.stride):
                patches.append(x_for_patching[:, i:i+self.patch_len, :].reshape(bs*n_vars_internal, self.patch_len))
            if not patches: # Handle case where seq_len < patch_len
                 # Fallback: create one zero patch of the expected dimension if no patches can be formed
                 # This is a tricky edge case. The original paper might handle this with padding.
                 # For now, let's assume seq_len >= patch_len.
                 # The original head_nf calculation suggests num_patches is at least 2 due to padding.
                 # Let's assume num_patches as calculated above.
                 patched_x = torch.randn(bs * n_vars_internal, num_patches, self.patch_len, device=x.device)
            else:
                patched_x = torch.stack(patches, dim=1) # [bs*n_vars, num_patches_calc, patch_len]
                # Ensure num_patches matches calculation (due to stride and padding effects)
                if patched_x.shape[1] != num_patches:
                     # This mock's patching is too simple. For now, force shape for testing downstream.
                     # A more accurate mock for PatchEmbedding's patching logic is needed for perfect shape matching.
                     # Let's create a dummy tensor of the target shape.
                     patched_x = torch.randn(bs * n_vars_internal, num_patches, self.patch_len, device=x.device)


            output = self.value_embedding(patched_x) # [bs*n_vars, num_patches, d_model]
            return output, n_vars_internal


    # Minimal Mock for Transformer Encoder
    class Encoder(nn.Module):
        def __init__(self, attn_layers, norm_layer=None):
            super().__init__()
            self.layers = nn.ModuleList(attn_layers)
            self.norm = norm_layer
        def forward(self, x, attn_mask=None):
            attns = []
            for layer in self.layers:
                x, attn = layer(x, attn_mask=attn_mask)
                attns.append(attn)
            if self.norm is not None:
                x = self.norm(x)
            return x, attns

    class EncoderLayer(nn.Module):
        def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attention = attention
            self.conv1 = nn.Conv1d(d_model, d_ff, 1) # Using 1D conv for FFN
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

    class AttentionLayer(nn.Module):
        def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
            super().__init__()
            self.inner_attention = attention # This is FullAttention instance
            self.query_projection = nn.Linear(d_model, d_model) # Simplified
            self.key_projection = nn.Linear(d_model, d_model)   # Simplified
            self.value_projection = nn.Linear(d_model, d_model) # Simplified
            self.out_projection = nn.Linear(d_model, d_model)   # Simplified
            self.n_heads = n_heads # Not directly used in this simplified mock forward

        def forward(self, queries, keys, values, attn_mask=None):
            # In self-attention, Q, K, V are usually the same
            Q = self.query_projection(queries)
            K = self.key_projection(keys)
            V = self.value_projection(values)
            # Pass to FullAttention mock or actual
            # FullAttention's forward is: (self, queries, keys, values, attn_mask)
            out, attn = self.inner_attention(Q, K, V, attn_mask)
            return self.out_projection(out), attn


    class FullAttention(nn.Module):
        def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
            super().__init__()
            self.scale = scale
            self.mask_flag = mask_flag
            self.output_attention = output_attention
            self.dropout = nn.Dropout(attention_dropout)
        def forward(self, queries, keys, values, attn_mask):
            # Simplified: just pass values through, return dummy attention
            # Actual implementation involves QK^T, softmax, etc.
            B, L, H, E = queries.shape # Assuming queries might be multi-headed
            _, S, _, D = values.shape  # Using S for key/value sequence length
            
            # Simplistic attention logic
            scale = self.scale or 1. / (E ** 0.5)
            scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale
            
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, -1e9)
                
            attn_weights = self.dropout(torch.softmax(scores, dim=-1))
            output = torch.einsum("bhls,bshe->blhe", attn_weights, values)
            
            if self.output_attention:
                return output.contiguous(), attn_weights
            else:
                return output.contiguous(), None # None for attn if not output_attention


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
        x = self.flatten(x) # x: [bs x nvars x (d_model * patch_num)]
        x = self.linear(x)  # x: [bs x nvars x target_window]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    AM-PatchTST: Attentive Multi-Scale Patch Time Series Transformer
    Paper link: https://arxiv.org/pdf/2211.14730.pdf (Original PatchTST)
    """

    def __init__(self, configs, patch_len_scales=[16, 32], stride_scales=[8, 16]): # MODIFIED: Added multi-scale params
        """
        configs: configuration object
        patch_len_scales: list of int, patch lengths for each scale
        stride_scales: list of int, strides for each scale
        """
        super().__init__()

        if len(patch_len_scales) != len(stride_scales):
            raise ValueError("patch_len_scales and stride_scales must have the same length.")

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # MODIFIED: Multi-scale patching and embedding
        self.patch_embeddings = nn.ModuleList()
        self.num_patch_scales = len(patch_len_scales)
        
        total_num_patches = 0
        for i in range(self.num_patch_scales):
            patch_len = patch_len_scales[i]
            stride = stride_scales[i]
            padding = stride # As per original PatchTST setup

            self.patch_embeddings.append(PatchEmbedding(
                configs.d_model, patch_len, stride, padding, configs.dropout))
            
            # Calculate number of patches for this scale.
            # This formula is based on the original PatchTST's head_nf calculation:
            # (seq_len - patch_len) / stride + 2.
            # The +2 comes from (seq_len - patch_len + 2*padding) / stride + 1, where padding = stride.
            # So, (seq_len - patch_len + 2*stride) / stride + 1 = (seq_len - patch_len)/stride + 2 -1 +1 = (seq_len - patch_len)/stride + 2
            # A more direct way from PatchEmbedding logic itself assuming padding adds to effective length:
            # effective_seq_len = configs.seq_len + 2 * padding
            # num_patches_scale = (configs.seq_len + 2*padding - patch_len) // stride + 1
            # Let's stick to the original paper's implicit patch count derivation:
            num_patches_scale = int((configs.seq_len - patch_len) / stride + 2)
            total_num_patches += num_patches_scale
            
        print(f"Total number of patches combined from all scales: {total_num_patches}")

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2)) # Original: nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        # MODIFIED: self.head_nf now depends on the total number of patches from all scales
        self.head_nf = configs.d_model * total_num_patches
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'stacking': # Stacking might need a different head structure if features are desired.
                                            # For now, keep similar to forecast.
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, 
                                    head_dropout=configs.dropout) # Or output features directly
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, # Target window is seq_len
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class) # head_nf * n_vars

    def _process_input(self, x_enc):
        # Normalization from Non-stationary Transformer (common for many tasks)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm /= stdev
        return x_enc_norm, means, stdev

    def _process_input_imputation(self, x_enc, mask):
        # Normalization for imputation
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc_norm = x_enc - means
        x_enc_norm = x_enc_norm.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc_norm * x_enc_norm, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc_norm /= stdev
        return x_enc_norm, means, stdev

    def _embed_and_encode(self, x_enc_processed):
        # MODIFIED: Apply multi-scale patch embedding and concatenate
        x_enc_processed = x_enc_processed.permute(0, 2, 1) # x: [bs x nvars x seq_len]

        all_scale_enc_outs = []
        n_vars = None
        for patch_embedding_layer in self.patch_embeddings:
            # Input to patch_embedding_layer: [bs, n_vars, seq_len]
            # Output: enc_out_scale [bs * nvars x patch_num_scale x d_model], n_vars_scale
            enc_out_scale, current_n_vars = patch_embedding_layer(x_enc_processed)
            all_scale_enc_outs.append(enc_out_scale)
            if n_vars is None:
                n_vars = current_n_vars
            # else: ensure n_vars are consistent if necessary, though they should be
            #   assert n_vars == current_n_vars, "n_vars mismatch between scales"

        # Concatenate along the patch_num dimension
        # Each enc_out_scale is [bs * nvars x patch_num_scale x d_model]
        enc_out_combined = torch.cat(all_scale_enc_outs, dim=1)
        # enc_out_combined is now [bs * nvars x total_patch_num x d_model]

        # Encoder
        # z: [bs * nvars x total_patch_num x d_model]
        enc_out_combined, attns = self.encoder(enc_out_combined)
        
        # Reshape back: [bs x nvars x total_patch_num x d_model]
        enc_out_reshaped = torch.reshape(
            enc_out_combined, (-1, n_vars, enc_out_combined.shape[-2], enc_out_combined.shape[-1]))
        # Permute for head: [bs x nvars x d_model x total_patch_num]
        enc_out_final = enc_out_reshaped.permute(0, 1, 3, 2)
        
        return enc_out_final, n_vars # attns could also be returned if needed

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_norm, means, stdev = self._process_input(x_enc)
        
        enc_out, _ = self._embed_and_encode(x_enc_norm) # _ is n_vars

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1) # [bs x target_window x nvars]

        # De-Normalization
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def stacking(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # For stacking, we want to output the features from the encoder typically
        x_enc_norm, means, stdev = self._process_input(x_enc)
        enc_out, n_vars = self._embed_and_encode(x_enc_norm)
        # enc_out is [bs x nvars x d_model x total_patch_num]
        # Depending on what stacking expects, you might flatten this or use it directly.
        # Original PatchTST forecast method also calls self.head.
        # If stacking means feature extraction before the final linear layer:
        # Option 1: Return encoder output (potentially flattened or pooled)
        # output_features = self.head.flatten(enc_out) # [bs x nvars x (d_model * total_patch_num)]
        # return output_features.permute(0,2,1) # [bs x (d_model * total_patch_num) x nvars]
        
        # Option 2: Apply the head like in forecast if stacking means generating predictions to be used by another model
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0,2,1)
        # De-normalize if these are actual value predictions
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc_norm, means, stdev = self._process_input_imputation(x_enc, mask)
        
        enc_out, _ = self._embed_and_encode(x_enc_norm)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window (seq_len for imputation)]
        dec_out = dec_out.permute(0, 2, 1) # [bs x seq_len x nvars]

        # De-Normalization
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc_norm, means, stdev = self._process_input(x_enc)
        
        enc_out, _ = self._embed_and_encode(x_enc_norm)

        # Decoder (reconstructs the input)
        dec_out = self.head(enc_out) # z: [bs x nvars x target_window (seq_len for AD)]
        dec_out = dec_out.permute(0, 2, 1) # [bs x seq_len x nvars]

        # De-Normalization
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out # Anomaly score often derived from reconstruction error: (x_enc - dec_out)^2

    def classification(self, x_enc, x_mark_enc):
        x_enc_norm, _, _ = self._process_input(x_enc) # means and stdev not used for final output scaling in classification

        enc_out, n_vars = self._embed_and_encode(x_enc_norm) # enc_out: [bs x nvars x d_model x total_patch_num]
        
        # Decoder for classification
        # enc_out is [bs x nvars x d_model x total_patch_num]
        output = self.flatten(enc_out) # [bs x nvars x (d_model * total_patch_num)]
        output = self.dropout(output)
        
        # Reshape to [bs, nvars * d_model * total_patch_num] to be fed into the linear layer
        output = output.reshape(output.shape[0], -1) # Flatten across nvars and features
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'stacking':
            # For stacking, the interpretation can vary.
            # If it means using PatchTST as a feature extractor for another model:
            # One option is to return the output of the encoder, possibly flattened.
            # x_enc_norm, _, _ = self._process_input(x_enc)
            # enc_out, n_vars = self._embed_and_encode(x_enc_norm) # enc_out: [bs x nvars x d_model x total_patch_num]
            # return enc_out # Or some processed version of it.
            # For consistency with other tasks that produce predictions, let's assume stacking implies
            # generating initial predictions that can be stacked.
            dec_out = self.stacking(x_enc, x_mark_enc, x_dec, x_mark_dec) # Uses the head like forecast
            return dec_out[:, -self.pred_len:, :] # [B, L, D]

        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None