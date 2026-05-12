import os
import torch
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATv2Conv
from torch import nn

import pickle
         
class GoEncoder(nn.Module):
    '''
    对GO标签进行编码
    每个标签都对应一个1024维的向量
    '''
    def __init__(self, vocab_dim=42093, emb_dim=1024, manual_init = False, init_tensor = None):
        super(GoEncoder, self).__init__()
        
        self.vocab_dim = vocab_dim
        
        # 读取手动设置的初始embedding参数
        if manual_init :
            assert init_tensor.shape[0] == vocab_dim
            assert init_tensor.shape[1] == emb_dim
            self.emb = nn.Parameter(init_tensor,requires_grad = True)
        # 否则使用随机参数
        else:
            self.emb = nn.Parameter(torch.randn(vocab_dim, emb_dim),requires_grad = True)
    
    def forward(self, indices):
        x = self.emb[indices]
        # x = x / x.norm(p=2, dim=-1, keepdim=True)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

# DEBUG
if __name__ == "__main__":
    from torchinfo import summary
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    
    device = torch.device("cuda:0")
    with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
        go_graph = pickle.load(f)
    with open(os.path.join("dataset","go_embedding_biobert.pkl"),"rb") as f:
        go_embedding_biobert = pickle.load(f)
    model = GoEncoder(manual_init=True,init_tensor=go_embedding_biobert)
    model.to(device)
    summary(model)
    go_graph = go_graph.to(device)
    x = model(go_graph)
    print(x.shape)