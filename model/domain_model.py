import math

import torch
from torch import nn
import torch.nn.functional as F
import transformer as tf

import pickle
    
class DomainEncoder(nn.Module):
    '''
    输入蛋白质的interpro信息
    输出全局特征与局部特征: (batch,global_dims), list((ni,part_dims))
    in_dims为interpro标签的总数
    '''
    def __init__(self, in_dims, global_dims=512, part_dims=64):
        super(DomainEncoder, self).__init__()
        self.glob_emb_out = nn.EmbeddingBag(in_dims, global_dims, mode='sum', include_last_offset=True)
        
    def forward(self, x):
        pass

# DEBUG
if __name__ == "__main__":
    model = DomainEncoder()