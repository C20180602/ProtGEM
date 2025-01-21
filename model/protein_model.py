import torch
from torch import nn
import torch.nn.functional as F
from model.seq_model import SeqEncoder
from model.struct_model import StructEncoder
from model.domain_model import DomainEncoder
import model.transformer as tf

class ProteinEncoder(nn.Module):
    '''
    蛋白质进行编码
    每个蛋白质最终对应一个1024维的向量
    '''
    def __init__(self, final_out_dim=1024, fusion_layers=2, fusion_att_heads=2, fusion_hid_dim=256, fusion_dropout=0.5, model_args = None):
        super(ProteinEncoder, self).__init__()
        # 没有给定参数文件则使用默认参数
        if model_args is None:
            self.seq_enc = SeqEncoder()
            self.struct_enc = StructEncoder()
            self.domain_enc = DomainEncoder()
        else:
            self.seq_args = model_args['SeqEncoder']
            self.struct_args = model_args['StructEncoder']
            self.domain_args = model_args['DomainEncoder']
            
            # 检验参数是否符合融合条件
            assert self.seq_args['global_dims'] + self.struct_args['global_dims'] == self.domain_args['global_dims']
            assert self.seq_args['part_dims'] + self.struct_args['part_dims'] == self.domain_args['part_dims']
            
            self.seq_enc = SeqEncoder(**self.seq_args)
            self.struct_enc = StructEncoder(**self.struct_args)
            self.domain_enc = DomainEncoder(**self.domain_args)
        
        # 根据现在的特征融合方式，需要暂时和domain的输出保持一致
        self.glob_dims = self.domain_enc.glob_dims
        self.part_dims = self.domain_enc.part_dims
        
        self.fusion_layers = nn.Sequential()
        for i in range(fusion_layers):
            self.fusion_layers.add_module("Layer"+str(i),tf.TransformerEncoderLayer(self.part_dims, fusion_att_heads, fusion_hid_dim, fusion_dropout))
        
        # TODO 这里直接拼接+线性层有点草率，想想更好的方法
        self.final_out = nn.Linear(self.glob_dims + self.part_dims, final_out_dim)
        
    # TODO mask目前只对seq有效
    def forward(self, seq_embs, batched_graph, domain_info, mask=None):
        seq_glob, seq_part = self.seq_enc(seq_embs,mask)
        struct_glob, struct_part = self.struct_enc(batched_graph)
        domain_glob, domain_part = self.domain_enc(*domain_info)
        # xxx_glob: [batch_size, global_dims]
        # seq_part: [batch_size, max_len, part_dims]
        # (struct/domain)_part: list(len=batch_size)[len_i, part_dims]
        
        # 对struct_part和domain_part进行padding
        max_len = seq_part.shape[1]
        
        padded_struct_part = []
        for struct_emb in struct_part:
            padding = torch.zeros(max_len - struct_emb.shape[0], struct_emb.shape[1], device=struct_emb.device)
            padded_struct_part.append(torch.cat([struct_emb, padding],dim=0))
        padded_struct_part = torch.stack(padded_struct_part)
        
        padded_domain_part = []
        for domain_emb in domain_part:
            padding = torch.zeros(max_len - domain_emb.shape[0], domain_emb.shape[1], device=domain_emb.device)
            padded_domain_part.append(torch.cat([domain_emb, padding],dim=0))
        padded_domain_part = torch.stack(padded_domain_part)
        
        # padded_(domain/struct)_part: [batch_size, max_len, (domain/struct)_part_dims]
        
        # 融合全局特征
        glob_emb = torch.cat([seq_glob,struct_glob],dim=1) + domain_glob
        # glob_emb: [batch_size, domain_glob_dims]
        # 融合局部特征：
        part_emb = torch.cat([seq_part,padded_struct_part],dim=2) + padded_domain_part
        for layer in self.fusion_layers:
            part_emb = layer(part_emb, mask)
        # part_emb: [batch_size, max_len, domain_part_dims]
        part_emb = part_emb[:,0,:]
        # part_emb: [batch_size, domain_part_dims]
        
        # 最终融合，得到最终的embedding
        emb = torch.cat([glob_emb,part_emb],dim=1)
        emb = self.final_out(emb)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        return emb

# DEBUG
if __name__ == "__main__":
    import os
    import json
    from dgl.dataloading import GraphDataLoader
    from dataset import MyDataset, my_collate_fn
    from torchinfo import summary
    
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    
    device = torch.device("cuda:0")
    model = ProteinEncoder()
    model.to(device)
    summary(model)
    ds = MyDataset("HUMAN")
    dl = GraphDataLoader(ds,batch_size=6,shuffle=False, collate_fn=my_collate_fn)
    for pid, seq_embs, batched_graph, domain_info, _  in dl:
        seq_embs = seq_embs.to(device)
        batched_graph = batched_graph.to(device)
        domain_info[0] = domain_info[0].to(device)
        domain_info[1] = domain_info[1].to(device)
        emb = model(seq_embs, batched_graph, domain_info)
        print(emb)
        print(emb.shape)
        break