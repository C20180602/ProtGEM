import math

import torch
from torch import nn
import torch.nn.functional as F

# TODO: 换为nn中的Transformer
class MultiHeadAttention(nn.Module):
    # !!! hid_dim must be divisible by n_heads
    def __init__(self, q_dim, k_dim, v_dim, hid_dim, n_heads, dropout, bias=False):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.each_head_dim = hid_dim // n_heads
        
        self.w_q = nn.Linear(q_dim, hid_dim, bias=bias)
        self.w_k = nn.Linear(k_dim, hid_dim, bias=bias)
        self.w_v = nn.Linear(v_dim, hid_dim, bias=bias)
        self.out = nn.Linear(hid_dim, hid_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.each_head_dim]))

    def forward(self, query, key, value, mask=None):
        # query [batch_size, seq_length_query, q_dim]
        # key [batch_size, seq_length_key, k_dim]
        # value [batch_size, seq_length_value, v_dim]
        # (seq_length_key = seq_length_value) 键值对数目
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 实现多头注意力
        Q = Q.view(batch_size, -1, self.n_heads, self.each_head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.each_head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.each_head_dim).permute(0, 2, 1, 3)
        self.scale = self.scale.to(query.device)
        mul = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # mul [batch_size, n_heads, seq_length_query, seq_length_key]
        
        # mask=0时，将该位置的数换为-inf
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            mul = mul.masked_fill(mask == 0, -1e10)
        attention = self.drop(F.softmax(mul, dim=-1))
        # attention [batch_size, n_heads, seq_length_query, seq_length_key]
        
        x = torch.matmul(attention, V)
        # x [batch_size, n_heads, seq_length_query, each_head_dim]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, -1, self.hid_dim)
        x = self.out(x)
        # x [batch_size, seq_length_query, hid_dim]
        return x

class PositionWiseFFN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, hid_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x [batch_size, seq_len, in_dim] 
        return self.dense2(self.relu(self.dense1(x)))
        # x [batch_size, seq_len, out_dim] 

class TransformerEncoderLayer(nn.Module):
    def __init__(self, in_dim, n_heads, ffn_hid_dim, dropout):
        super().__init__()
        self.MHA = MultiHeadAttention(in_dim, in_dim, in_dim, in_dim, n_heads, dropout)
        self.LN1 = nn.LayerNorm(in_dim)
        self.FFN = PositionWiseFFN(in_dim, ffn_hid_dim, in_dim)
        self.LN2 = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x 形状输入输出保持不变 [batch_size, seq_len, in_dim] 
        x = self.LN1(x + self.drop(self.MHA(x, x, x, mask)))
        x = self.LN2(x + self.drop(self.FFN(x)))
        return x

# 去除第二个残差网络的Encoderlayer，压缩特征维数
class TransformerEncoderLayer_NoRes(nn.Module):
    def __init__(self, in_dim, n_heads, ffn_hid_dim, out_dim, dropout):
        super().__init__()
        self.MHA = MultiHeadAttention(in_dim, in_dim, in_dim, in_dim, n_heads, dropout)
        self.LN1 = nn.LayerNorm(in_dim)
        self.FFN = PositionWiseFFN(in_dim, ffn_hid_dim, out_dim)
        self.LN2 = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x 形状输入输出保持不变 [batch_size, seq_len, in_dim] 
        x = self.LN1(x + self.drop(self.MHA(x, x, x, mask)))
        x = self.LN2(self.drop(self.FFN(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=5000): # 默认蛋白质序列长度在5000以内
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(5000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # i为序列坐标，j为embedding坐标 (i/5000^(2j/hid_dim))
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)