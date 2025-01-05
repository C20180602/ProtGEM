import math

import torch
from torch import nn
import torch.nn.functional as F
import model.transformer as tf

import pickle
    
class SeqEncoder(nn.Module):
    '''
    将预训练的ESM2模型输出的token级特征,通过多层transformer后取第一个token特征,作为该序列的整体特征
    in_dim: 从ESM2模型中输出的特征维数默认为 1280
    layer_num: 经过的Transformer层数
    '''
    def __init__(self, in_dim=1280, n_heads=2, layer_cnt=2, ffn_hid_dim=1280, global_dims=512, part_dims=64, dropout=0.4):
        super(SeqEncoder, self).__init__()
        self.scale =math.sqrt(in_dim) # 放缩系数sqrt(in_dim)
        self.LN = nn.LayerNorm(in_dim)
        self.PE = tf.PositionalEncoding(in_dim,dropout)
        self.layers = nn.Sequential()
        for i in range(layer_cnt):
            self.layers.add_module("Layer"+str(i),tf.TransformerEncoderLayer(in_dim,n_heads,ffn_hid_dim,dropout))
        
        self.part_readout = tf.TransformerEncoderLayer_NoRes(in_dim, n_heads, ffn_hid_dim, part_dims, dropout)
        self.glob_readout = nn.Linear(in_dim, global_dims)
    
    def forward(self, x, mask=None):
        # x [batch_size, seq_len, in_dim]
        x = self.PE(self.LN(x)*self.scale)
        for layer in self.layers:
            x = layer(x, mask)
        # x [batch_size, seq_len, in_dim]
        
        part_emb_out = self.part_readout(x, mask)
        # part_emb_out [batch_size, seq_len, part_dims]
        
        x = x[:,0,:]
        # x [batch_size, in_dim]
        glob_emb_out = F.relu(self.glob_readout(x))
        # glob_emb_out [batch_size, global_dims]
        
        return glob_emb_out, part_emb_out

# DEBUG
if __name__ == "__main__":
    import os
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    model = SeqEncoder()
    with open(os.path.join("dataset","ECOLI","P77302",'seq_emb.pkl'),"rb") as f:
        se = torch.tensor(pickle.load(f))
        se = se.unsqueeze(0)
        print(se.shape)
        glob,part = model.forward(se)
        print(glob.shape, part.shape)