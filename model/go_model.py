import math
import os
import torch
import dgl
from dgl.nn.pytorch.conv import GATv2Conv
from torch import nn
import torch.nn.functional as F
import transformer as tf

import pickle
    
class GoEncoder(nn.Module):
    '''
    对GO标签进行编码
    每个标签都对应一个1024维的向量
    '''
    def __init__(self, in_dim=42093, hid_dim=1024, layer_num = 3, n_heads = 1, manual_init = False, init_tensor = None):
        super(GoEncoder, self).__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.layer_num = layer_num
        self.n_heads = n_heads
        
        # 读取手动设置的初始embedding参数
        if manual_init :
            assert init_tensor.shape[0] == in_dim
            assert init_tensor.shape[1] == hid_dim
            self.emb = nn.Parameter(init_tensor,requires_grad = True)
        # 否则使用随机参数
        else:
            self.emb = nn.Parameter(torch.randn(in_dim, hid_dim),requires_grad = True)
        
        self.layers = nn.Sequential()
        for i in range(layer_num):
            self.layers.append(GATv2Conv(hid_dim,hid_dim,n_heads,bias=False))
    
    def forward(self, graph):
        graph = dgl.add_self_loop(graph)
        x = self.emb.data
        # x: [in_dim(node_cnt), hid_dim]
        for i in range(self.layer_num):
            x = self.layers[i](graph,x)
        # x: [in_dim, 1, hid_dim]
        return x.squeeze()
        # x: [in_dim, hid_dim]

# DEBUG
if __name__ == "__main__":
    from torchinfo import summary
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
        go_graph = pickle.load(f)
    with open(os.path.join("dataset","go_embedding_biobert.pkl"),"rb") as f:
        go_embedding_biobert = pickle.load(f)
    model = GoEncoder(manual_init=True,init_tensor=go_embedding_biobert)
    summary(model)
    x = model(go_graph)
    print(x.shape)