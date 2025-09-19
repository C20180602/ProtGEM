import math
import torch
from torch import nn
import torch.nn.functional as F
import model.transformer as tf

class EnhancedSeqEncoder(nn.Module):
    """Enhanced sequence encoder with improved transformer processing and multi-scale attention"""
    def __init__(self, in_dim=1280, n_heads=8, layer_cnt=3, ffn_hid_dim=2048,
                 global_dims=512, part_dims=256, dropout=0.1, use_positional_encoding=True):
        super().__init__()
        self.glob_dims = global_dims
        self.in_dim = in_dim
        self.part_dims = part_dims
        self.use_pe = use_positional_encoding

        # Input preprocessing
        self.input_norm = nn.LayerNorm(in_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Positional encoding
        if self.use_pe:
            self.pe = tf.PositionalEncoding(in_dim, dropout)

        # Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            tf.TransformerEncoderLayer(in_dim, n_heads, ffn_hid_dim, dropout)
            for _ in range(layer_cnt)
        ])

        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(in_dim, in_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]
        ])

        # Attention-based pooling for global features
        self.global_attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, 1)
        )

        # Feature projection layers
        self.global_proj = nn.Sequential(
            nn.Linear(in_dim, global_dims),
            nn.LayerNorm(global_dims),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Enhanced part feature projection with random orthogonal projection
        self.part_proj = nn.Sequential(
            nn.Linear(in_dim, part_dims),
            nn.LayerNorm(part_dims),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable pooling weights for different positions
        self.position_weights = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, in_dim]
        # mask: [batch_size, seq_len]
        batch_size, seq_len, _ = x.shape

        # Input preprocessing
        x = self.input_norm(x)

        # Apply mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            x = x * mask_expanded

        # Add positional encoding
        if self.use_pe:
            x = self.pe(x)

        # Enhanced transformer processing
        for layer in self.transformer_layers:
            residual = x
            x = layer(x, mask)
            # Add residual connection with gating
            gate = torch.sigmoid(self.position_weights)
            x = gate * x + (1 - gate) * residual

        # Multi-scale convolution features
        x_conv = x.transpose(1, 2)  # [batch_size, in_dim, seq_len]
        multi_scale_features = []

        for conv in self.multi_scale_conv:
            features = F.relu(conv(x_conv))
            # Global average pooling for each scale
            pooled = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
            multi_scale_features.append(pooled)

        multi_scale_concat = torch.cat(multi_scale_features, dim=1)  # [batch_size, in_dim]

        # Attention-based global pooling
        attention_weights = self.global_attention(x)  # [batch_size, seq_len, 1]

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        attention_weights = F.softmax(attention_weights, dim=1)
        global_features = torch.sum(x * attention_weights, dim=1)  # [batch_size, in_dim]

        # Combine different global features
        combined_global = global_features + multi_scale_concat

        # Project to final dimensions
        glob_emb = self.global_proj(combined_global)
        part_emb = self.part_proj(x)

        return glob_emb, part_emb

class ImprovedSeqEncoder(nn.Module):
    """Drop-in replacement for existing SeqEncoder with the same interface"""
    def __init__(self, in_dim=1280, n_heads=8, layer_cnt=2, ffn_hid_dim=1280,
                 global_dims=512, part_dims=64, dropout=0.4):
        super().__init__()
        self.enhanced_encoder = EnhancedSeqEncoder(
            in_dim=in_dim,
            n_heads=n_heads,
            layer_cnt=layer_cnt,
            ffn_hid_dim=ffn_hid_dim,
            global_dims=global_dims,
            part_dims=part_dims,
            dropout=dropout
        )
        self.glob_dims = global_dims
        self.part_dims = part_dims

    def forward(self, x, mask=None):
        return self.enhanced_encoder(x, mask)