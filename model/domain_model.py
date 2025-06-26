import math

import torch
from torch import nn
import torch.nn.functional as F
import model.transformer as tf

import pickle
    
class DomainEncoder(nn.Module):
    '''
    输入蛋白质的interpro信息
    输出全局特征与局部特征: (batch,global_dims), list((ni,part_dims))
    in_dims为interpro标签的总数
    '''
    def __init__(self, in_dims=46061, global_dims=1024, part_dims=128):
        super(DomainEncoder, self).__init__()
        self.in_dims = in_dims
        self.glob_dims = global_dims
        self.part_dims = part_dims
        
        self.glob_emb_out = nn.EmbeddingBag(in_dims, global_dims, mode='sum')
        self.part_emb = nn.Embedding(in_dims, part_dims)
        
        self.glob_LN = nn.LayerNorm(global_dims)
        self.part_LN = nn.LayerNorm(part_dims)
    
    def get_device(self):
        return next(self.parameters()).device
    
    # TODO 多进程优化？向量化？
    def forward(self, glob_dm_id, glob_dm_offset, plens, domain_list):
        glob_emb_out = self.glob_emb_out(glob_dm_id, glob_dm_offset)
        # glob_emb_out :[batch_size, global_dims]
        
        part_emb_out = []
        device = self.get_device()
        for plen,dm in zip(plens,domain_list):
            part_emb = torch.zeros([plen, self.part_dims], device=device)
            for domain in dm:
                domain[1] = min(domain[1],plen)
                domain[2] = min(domain[2],plen)
                pos_flag = torch.stack([torch.tensor([0.])] * (domain[1]-1)
                                       + [torch.tensor([1.])] * (domain[2]-domain[1]+1)
                                       + [torch.tensor([0.])] * (plen-domain[2]))
                pos_flag = pos_flag.to(device)
                domain_emb = self.part_emb(torch.tensor([domain[0]], device=device))
                part_emb += pos_flag @ domain_emb
            part_emb = part_emb.to(device)
            part_emb = self.part_LN(part_emb)
            part_emb_out.append(part_emb)
        # part_emb_out :list(len=batch_size)[seq_len, part_dims]
        
        glob_emb_out = glob_emb_out.to(device)
        
        glob_emb_out = self.glob_LN(glob_emb_out)
        return glob_emb_out, part_emb_out

# DEBUG
if __name__ == "__main__":
    import os
    import json
    from dgl.dataloading import GraphDataLoader
    from dataset import MyDataset, my_collate_fn
    
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    
    device = torch.device("cuda:0")
    model = DomainEncoder()
    model.to(device)
    ds = MyDataset("HUMAN")
    dl = GraphDataLoader(ds,batch_size=2,shuffle=False, collate_fn=my_collate_fn)
    for pid, _, _,domain_info,_  in dl:
        domain_info[0] = domain_info[0].to(device)
        domain_info[1] = domain_info[1].to(device)
        
        glob,part = model(*domain_info)
        print(glob.shape, len(part), part[0].shape, part[1].shape)
        print(glob.device)
        print(part[0].device)
        break