import math
import os
import torch
import dgl
from torch import nn
import torch.nn.functional as F
import transformer as tf

import pickle
    
class GoEncoder(nn.Module):
    '''
    对GO标签进行编码
    每个标签都对应一个1024维的向量
    '''
    def __init__(self, in_dim, layer_num = 3, use_biobert = True):
        super(GoEncoder, self).__init__()
        
        self.emb = nn.Embedding(in_dim, 1024)
        if use_biobert :
            with open(os.path.join("dataset","go_embedding_biobert.pkl"),"rb") as f:
                go_embedding_biobert = pickle.load(f)
            self.emb.weight.data.copy_(go_embedding_biobert)

        
    def forward(self, graph):
        pass

# DEBUG
if __name__ == "__main__":
    model = GoEncoder()