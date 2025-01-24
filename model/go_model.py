import os
import torch
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GATv2Conv
from torch import nn

import pickle

class PretrainLoss(nn.Module):
    '''
    预训练任务的损失函数
    '''
    def __init__(self, pos_k, neg_k):
        super(PretrainLoss, self).__init__()
        self.pos_k = pos_k
        self.neg_k = neg_k
    
    def forward(self, x):
        # x: [batch_size, ( 1 + pos_k + neg_k ), hid_dim]
        idx = x[:,:1,:]
        # idx: [batch_size, 1, hid_dim]
        idx = torch.transpose(idx, 1, 2)
        # idx: [batch_size, hid_dim, 1]
        obj = x[:,1:,:]
        # obj: [batch_size, pos_k + neg_k, hid_dim]
        sim = torch.bmm(obj , idx)
        # sim: [batch_size, pos_k + neg_k, 1]
        sim = sim.squeeze(-1)
        # sim: [batch_size, pos_k + neg_k]
        pos = sim[:,:self.pos_k]
        # pos: [batch_size, pos_k]
        neg = sim[:,self.pos_k:]
        # neg: [batch_size, neg_k]
        # print("--------------------")
        # print(torch.mean(pos.max(dim=-1)[0]).item(),torch.mean(neg.max(dim=-1)[0]).item())
        # print(torch.mean(pos.min(dim=-1)[0]).item(),torch.mean(neg.min(dim=-1)[0]).item())
        # print(torch.mean(torch.mean(pos,dim=-1)).item(),torch.mean(torch.mean(neg,dim=-1)).item())
        pos = torch.exp(pos/0.07)
        neg = torch.exp(neg/0.07)
        
        # return torch.mean(torch.sum(neg, dim=-1)) / torch.mean(torch.sum(pos, dim=-1))
        frac = torch.sum(pos, dim=-1) / (torch.sum(pos, dim=-1) + torch.sum(neg, dim=-1))
        # frac: [batch_size]
        return torch.mean(-torch.log(frac))
         
class GoEncoder(nn.Module):
    '''
    对GO标签进行编码
    每个标签都对应一个1024维的向量
    '''
    def __init__(self, vocab_dim=42093, emb_dim=1024, hid_dim=512, out_dim=1024, layer_num = 3, n_heads = 1, manual_init = False, init_tensor = None):
        super(GoEncoder, self).__init__()
        
        self.vocab_dim = vocab_dim
        self.hid_dim = hid_dim
        self.layer_num = layer_num
        self.n_heads = n_heads
        
        # 读取手动设置的初始embedding参数
        if manual_init :
            assert init_tensor.shape[0] == vocab_dim
            assert init_tensor.shape[1] == emb_dim
            self.emb = nn.Parameter(init_tensor,requires_grad = True)
        # 否则使用随机参数
        else:
            self.emb = nn.Parameter(torch.randn(vocab_dim, emb_dim),requires_grad = True)
        
        self.lin1 = nn.Linear(emb_dim,hid_dim)
        self.layers = nn.Sequential()
        self.liners = nn.Sequential()
        for i in range(layer_num):
            # self.layers.append(GATv2Conv(hid_dim,hid_dim,n_heads))
            self.liners.append(nn.Linear(n_heads*hid_dim,hid_dim))
        self.LN = nn.LayerNorm(hid_dim)
        self.lin2 = nn.Linear(hid_dim,out_dim)
    
    def forward(self, indices, graph):
        # graph = dgl.add_self_loop(graph)
        x = self.emb[indices]
        x = self.lin1(x)
        x = F.relu(x)
        # x: [node_cnt, hid_dim]
        for i in range(self.layer_num):
            # x = self.layers[i](graph,x)
            # # x: [node_cnt, n_head, hid_dim]
            # x = x.view(self.vocab_dim,-1)
            # # x: [node_cnt, n_head*hid_dim]
            x = self.liners[i](x)
            x = F.relu(x)
        # x = x[indices]
        # x: [indices_cnt, hid_dim]
        x = self.lin2(x)
        x = x / x.norm(p=2, dim=-1, keepdim=True)
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