import math

import torch
from torch import nn
import torch.nn.functional as F
import transformer as tf

import pickle
    
class SeqEncoder(nn.Module):
    '''
    将预训练的ESM2模型输出的token级特征,通过多层transformer后取第一个token特征,作为该序列的整体特征
    in_dim: 从ESM2模型中输出的特征维数默认为 1280
    layer_num: 经过的Transformer层数
    '''
    def __init__(self, in_dim=1280, n_heads=4, layer_cnt=3, ffn_hid_dim=1280, out_dim=512, dropout=0.1):
        super(SeqEncoder, self).__init__()
        self.scale =math.sqrt(in_dim) # 放缩系数sqrt(in_dim)
        self.LN = nn.LayerNorm(in_dim)
        self.PE = tf.PositionalEncoding(in_dim,dropout)
        self.layers = nn.Sequential()
        for i in range(layer_cnt):
            self.layers.add_module("Layer"+str(i),tf.TransformerEncoderLayer(in_dim,n_heads,ffn_hid_dim,dropout))
        self.dense = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, mask=None):
        # x [batch_size, seq_len, in_dim]
        x = self.PE(self.LN(x)*self.scale)
        for layer in self.layers:
            x = layer(x, mask)
        # x [batch_size, seq_len, in_dim]
        x = x[:,0,:]
        # x [batch_size, in_dim]
        return F.relu(self.dense(x))
        # x [batch_size, out_dim]

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
        x = model.forward(se)
        print(x.shape)