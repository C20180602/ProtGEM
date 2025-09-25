import torch
from torch import nn
import torch.nn.functional as F
import math
from model.protein_model import ProteinEncoder, ProteinEncoderv2
from model.go_model import GoEncoder

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, pred, labels, t=0.05):
        pred = torch.exp(pred/t)
        pos = torch.sum(pred * labels, dim=-1)
        neg = torch.sum(pred * (1-labels), dim=-1)
        frac = pos / (pos + neg)
        return torch.mean(-torch.log(frac))

class CrossAttention(nn.Module):
    """Cross-attention mechanism for protein-GO interaction"""
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context), attn

class AdaptiveGOEncoder(nn.Module):
    """Enhanced GO encoder with contextual adaptation"""
    def __init__(self, vocab_dim=42093, emb_dim=512, context_dim=512, n_layers=3):
        super().__init__()
        self.base_encoder = GoEncoder(vocab_dim, emb_dim)

        # Context adaptation layers
        self.context_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim + context_dim, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(emb_dim)

    def forward(self, go_indices, protein_context=None):
        # Base GO embeddings
        go_emb = self.base_encoder(go_indices)
        if protein_context is not None:
            # Expand protein context to match GO dimensions
            go_emb = go_emb.unsqueeze(0).tile(protein_context.shape[0], 1, 1)
            go_count = go_emb.shape[1]
            protein_context = protein_context.unsqueeze(1).tile(1, go_count, 1)

            # Apply context adaptation
            for layer in self.context_layers:
                combined = torch.cat([go_emb, protein_context], dim=-1)
                go_emb = go_emb + layer(combined)
        # go_emb: [batch_size, go_count, emb_dim] or [go_count, emb_dim]
        return self.final_norm(go_emb)

class HierarchicalAttentionPooling(nn.Module):
    """Multi-scale attention pooling for protein features"""
    def __init__(self, in_dim, out_dim, scales=[1, 2, 4]):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scales = scales
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, in_dim // 4),
                nn.Tanh(),
                nn.Linear(in_dim // 4, 1)
            ) for _ in scales
        ])
        self.output_proj = nn.Linear(in_dim * len(scales), out_dim)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, in_dim = x.shape
        assert in_dim == self.in_dim
        
        outputs = []

        for scale, attn_layer in zip(self.scales, self.attention_layers):
            # Multi-scale pooling
            if scale > 1:
                pad_len = (scale - seq_len % scale) % scale
                if pad_len > 0:
                    padding = torch.zeros(batch_size, pad_len, in_dim, device=x.device)
                    x_scaled = torch.cat([x, padding], dim=1)
                else:
                    x_scaled = x

                x_scaled = x_scaled.view(batch_size, -1, scale, in_dim).mean(dim=2)
            else:
                x_scaled = x
            # x_scaled: [batch_size, pad_seq_len//scale, in_dim]

            # Attention weights
            attn_weights = attn_layer(x_scaled)
            # x: [batch_size, pad_seq_len//scale, 1]
            if mask is not None and scale == 1:
                attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

            attn_weights = F.softmax(attn_weights, dim=1)

            # Weighted pooling
            pooled = torch.sum(x_scaled * attn_weights, dim=1)
            # pooled: [batch_size, in_dim]
            outputs.append(pooled)

        # Combine multi-scale features
        combined = torch.cat(outputs, dim=-1)
        # combined: [batch_size, in_dim * len(scales)]
        return self.output_proj(combined)
        # return: [batch_size, out_dim]

class EnhancedProteinGoModel(nn.Module):
    """Enhanced protein-GO prediction model with advanced architectures"""
    def __init__(self, pm_param, gm_param, enhanced_config=None):
        super().__init__()

        # Default enhanced configuration
        if enhanced_config is None:
            enhanced_config = {
                'use_cross_attention': True,
                'use_adaptive_go': True,
                'use_hierarchical_pooling': True,
                'interaction_method': 'cross_attention',  # 'element_wise', 'cross_attention', 'bilinear'
                'n_attention_heads': 8,
                'n_interaction_layers': 2,
                'dropout_rate': 0.1
            }

        self.config = enhanced_config

        # Core encoders
        self.pm = ProteinEncoder(**pm_param)

        # Enhanced GO encoder
        if enhanced_config['use_adaptive_go']:
            self.gm = AdaptiveGOEncoder(
                vocab_dim=gm_param['vocab_dim'],
                emb_dim=gm_param['emb_dim'],
                context_dim=pm_param['final_out_dim']
            )
        else:
            self.gm = GoEncoder(**gm_param)

        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.d_model = pm_param['final_out_dim']

        # Enhanced pooling
        if enhanced_config['use_hierarchical_pooling']:
            self.protein_pooling = HierarchicalAttentionPooling(pm_param['model_args']['SeqEncoder']['in_dim'], self.d_model)

        # Interaction mechanisms
        if enhanced_config['interaction_method'] == 'cross_attention':
            self.cross_attention = CrossAttention(
                self.d_model,
                enhanced_config['n_attention_heads'],
                enhanced_config['dropout_rate']
            )

        elif enhanced_config['interaction_method'] == 'bilinear':
            self.bilinear = nn.Bilinear(self.d_model, self.d_model, self.d_model)

        # Multi-layer interaction processing
        self.interaction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Dropout(enhanced_config['dropout_rate'])
            ) for _ in range(enhanced_config['n_interaction_layers'])
        ])

        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(enhanced_config['dropout_rate']),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat=[]):
        # Encode protein features
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat)
        # protein_emb: [batch_size, d_model]

        # Enhanced hierarchical pooling if enabled
        if self.config['use_hierarchical_pooling'] and len(seq_embs.shape) == 3:
            protein_context = self.protein_pooling(seq_embs, mask)
        else:
            protein_context = protein_emb

        # Encode GO terms with protein context
        if self.config['use_adaptive_go']:
            go_emb = self.gm(go_idx, protein_context)
        else:
            go_emb = self.gm(go_idx)
        # go_emb: [batch_size, go_count, d_model] or [go_count, d_model]

        if len(go_emb.shape) == 2:
            go_emb = go_emb.unsqueeze(0).expand(protein_emb.size(0), -1, -1)

        batch_size, go_count, _ = go_emb.shape
        protein_emb = protein_emb.unsqueeze(1).expand(-1, go_count, -1)

        # Enhanced interaction computation
        if self.config['interaction_method'] == 'cross_attention':
            interaction_features, attention_weights = self.cross_attention(protein_emb, go_emb, go_emb)

        elif self.config['interaction_method'] == 'bilinear':
            interaction_features = self.bilinear(protein_emb, go_emb)

        else:  # element_wise
            interaction_features = protein_emb * go_emb

        # Multi-layer interaction processing
        for layer in self.interaction_layers:
            residual = interaction_features
            interaction_features = layer(interaction_features) + residual

        # Final prediction
        predictions = self.prediction_head(interaction_features)
        predictions = predictions.squeeze(-1)  # [batch_size, go_count]

        return predictions

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class EnhancedProteinGoModelv2(nn.Module):
    """Enhanced v2 model with ProteinEncoderv2"""
    def __init__(self, pm_param, gm_param, enhanced_config=None):
        super().__init__()

        if enhanced_config is None:
            enhanced_config = {
                'use_cross_attention': True,
                'use_adaptive_go': True,
                'interaction_method': 'cross_attention',
                'n_attention_heads': 8,
                'n_interaction_layers': 2,
                'dropout_rate': 0.1
            }

        self.config = enhanced_config
        self.pm = ProteinEncoderv2(**pm_param)

        if enhanced_config['use_adaptive_go']:
            self.gm = AdaptiveGOEncoder(
                vocab_dim=gm_param['vocab_dim'],
                emb_dim=gm_param['emb_dim'],
                context_dim=pm_param['final_out_dim'] // 2
            )
        else:
            self.gm = GoEncoder(**gm_param)

        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.d_model = pm_param['final_out_dim']

        if enhanced_config['interaction_method'] == 'cross_attention':
            self.cross_attention = CrossAttention(
                self.d_model,
                enhanced_config['n_attention_heads'],
                enhanced_config['dropout_rate']
            )

        self.interaction_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
                nn.ReLU(),
                nn.Dropout(enhanced_config['dropout_rate'])
            ) for _ in range(enhanced_config['n_interaction_layers'])
        ])

        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(enhanced_config['dropout_rate']),
            nn.Linear(self.d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat=[]):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, ablation_feat)

        if self.config['use_adaptive_go']:
            go_emb = self.gm(go_idx, protein_emb)
        else:
            go_emb = self.gm(go_idx)

        if len(go_emb.shape) == 2:
            go_emb = go_emb.unsqueeze(0).expand(protein_emb.size(0), -1, -1)

        batch_size, go_count, _ = go_emb.shape
        protein_emb = protein_emb.unsqueeze(1).expand(-1, go_count, -1)

        if self.config['interaction_method'] == 'cross_attention':
            interaction_features, _ = self.cross_attention(protein_emb, go_emb, go_emb)
            # interaction_features = interaction_features.mean(dim=1).unsqueeze(1).expand(-1, go_count, -1)
        else:
            interaction_features = protein_emb * go_emb

        for layer in self.interaction_layers:
            residual = interaction_features
            interaction_features = layer(interaction_features) + residual

        predictions = self.prediction_head(interaction_features)
        return predictions.squeeze(-1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))