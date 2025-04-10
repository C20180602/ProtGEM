import torch
from torch import nn
import torch.nn.functional as F
from model.seq_model import SeqEncoder
from model.struct_model import StructEncoder, StructEncoderv2
from model.domain_model import DomainEncoder
import model.transformer as tf

class ProteinEncoder(nn.Module):
    '''
    蛋白质进行编码
    每个蛋白质最终对应一个1024维的向量
    '''
    def __init__(self, final_out_dim=1024, final_hid_dim = 768, fusion_layers=2, fusion_att_heads=2, fusion_hid_dim=256, fusion_dropout=0.5, model_args = None):
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
            
            self.seq_enc = SeqEncoder(**self.seq_args)
            self.struct_enc = StructEncoder(**self.struct_args)
            self.domain_enc = DomainEncoder(**self.domain_args)
        
        self.glob_dims = self.seq_enc.glob_dims + self.struct_enc.glob_dims + self.domain_enc.glob_dims
        self.part_dims = self.seq_enc.part_dims + self.struct_enc.part_dims + self.domain_enc.part_dims
        self.final_hid_dim = final_hid_dim
        self.fusion_layers = nn.Sequential()
        for i in range(fusion_layers):
            self.fusion_layers.add_module("Layer"+str(i),tf.TransformerEncoderLayer(self.part_dims, fusion_att_heads, fusion_hid_dim, fusion_dropout))
        
        self.dropout = nn.Dropout(fusion_dropout)
        self.glob_scale = nn.Linear(self.glob_dims, self.final_hid_dim)
        self.part_scale = nn.Linear(self.part_dims, self.final_hid_dim)
        self.final_out = nn.Linear(2 * self.final_hid_dim, final_out_dim)
        
    def forward(self, seq_embs, batched_graph, domain_info, mask=None, ablation_feat = []):
        # seq_embs: [batch_size, seq_len, seq_dim]
        # mask [batch_size, seq_len]
        batch_size = seq_embs.shape[0]
        max_len = seq_embs.shape[1]
        
        # Ablation Study 消融实验
        if "seq" not in ablation_feat:
            seq_glob, seq_part = self.seq_enc(seq_embs,mask)
        else:
            seq_glob = torch.zeros(batch_size, self.seq_enc.glob_dims, device=seq_embs.device)
            seq_part = torch.zeros(batch_size, max_len, self.seq_enc.part_dims, device=seq_embs.device)
        
        if "struct" not in ablation_feat:
            struct_glob, struct_part = self.struct_enc(batched_graph)
        else:
            struct_glob = torch.zeros(batch_size, self.struct_enc.glob_dims, device=batched_graph.device)
            struct_part = [torch.zeros(max_len, self.struct_enc.part_dims, device=batched_graph.device) for _ in range(batch_size)]
        
        if "domain" not in ablation_feat:
            domain_glob, domain_part = self.domain_enc(*domain_info)
        else:
            domain_glob = torch.zeros(batch_size, self.domain_enc.glob_dims, device=domain_info[0].device)
            domain_part = [torch.zeros(max_len, self.domain_enc.part_dims, device=domain_info[0].device) for _ in range(batch_size)]
        # xxx_glob: [batch_size, global_dims]
        # seq_part: [batch_size, max_len, part_dims]
        # (struct/domain)_part: list(len=batch_size)[len_i, part_dims]
        
        # 对struct_part和domain_part进行padding
        
        padded_struct_part = []
        for struct_emb in struct_part:
            padding = torch.zeros(max_len - struct_emb.shape[0], struct_emb.shape[1], device=struct_emb.device)
            padded_struct_part.append(torch.cat([struct_emb, padding],dim=0))
        padded_struct_part = torch.stack(padded_struct_part)
        if mask is not None:
            padded_struct_part = padded_struct_part * mask.unsqueeze(-1)
        
        padded_domain_part = []
        for domain_emb in domain_part:
            padding = torch.zeros(max_len - domain_emb.shape[0], domain_emb.shape[1], device=domain_emb.device)
            padded_domain_part.append(torch.cat([domain_emb, padding],dim=0))
        padded_domain_part = torch.stack(padded_domain_part)
        if mask is not None:
            padded_domain_part = padded_domain_part * mask.unsqueeze(-1)
        
        # padded_(domain/struct)_part: [batch_size, max_len, (domain/struct)_part_dims]
        
        # 融合全局特征
        glob_emb = torch.cat([seq_glob,struct_glob,domain_glob],dim=1)
        glob_emb = self.glob_scale(glob_emb)
        glob_emb = F.relu(glob_emb)
        # glob_emb: [batch_size, final_hid_dim]
        
        # 融合局部特征：
        part_emb = torch.cat([seq_part,padded_struct_part,padded_domain_part],dim=2)
        for layer in self.fusion_layers:
            part_emb = layer(part_emb, mask)
        # part_emb: [batch_size, max_len, part_dims]
        part_emb = part_emb[:,0,:]
        # part_emb: [batch_size, part_dims]
        part_emb = self.part_scale(part_emb)
        part_emb = F.relu(part_emb)
        # part_emb: [batch_size, final_hid_dim]
        
        # Ablation Study 消融实验
        if "glob" in ablation_feat:
            glob_emb = torch.zeros(batch_size, self.final_hid_dim, device=glob_emb.device)
        if "part" in ablation_feat:
            part_emb = torch.zeros(batch_size, self.final_hid_dim, device=part_emb.device)
        
        # 最终融合，得到最终的embedding
        emb = torch.cat([glob_emb,part_emb],dim=1)
        emb = self.dropout(emb)
        emb = self.final_out(emb)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        return emb
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class ProteinEncoderv2(nn.Module):
    '''
    蛋白质进行编码
    每个蛋白质最终对应一个1024维的向量
    '''
    def __init__(self, final_out_dim=1024, fusion_dropout=0.5, esm_dim = 1280, model_args = None):
        super(ProteinEncoderv2, self).__init__()
        
        if model_args == None:
            self.struct_enc = StructEncoderv2()
            self.domain_enc = DomainEncoder()
        else:
            self.struct_args = model_args['StructEncoder']
            self.domain_args = model_args['DomainEncoder']
            
            self.struct_enc = StructEncoderv2(**self.struct_args)
            self.domain_enc = DomainEncoder(**self.domain_args)
        
        self.total_dims = self.struct_enc.out_dims + self.domain_enc.glob_dims
        
        self.dropout = nn.Dropout(fusion_dropout)
        self.glob_scale = nn.Linear(self.total_dims, final_out_dim)
        self.final_out = nn.Linear(final_out_dim, final_out_dim)
        
    def forward(self, seq_embs, batched_graph, domain_info, mask=None, ablation_feat = []):
        batch_size = batched_graph.batch_num_nodes().shape[0]
        
        # Ablation Study 消融实验
        if "seq" in ablation_feat:
            batched_graph.ndata["esm"] = torch.zeros((batched_graph.ndata["esm"].shape), device=seq_embs.device)
        
        if "struct" in ablation_feat:
            struct_emb = torch.zeros(batch_size, self.struct_enc.glob_dims, device=batched_graph.device)
        else:
            struct_emb = self.struct_enc(batched_graph)
        
        if "domain" in ablation_feat:
            domain_emb = torch.zeros(batch_size, self.domain_enc.glob_dims, device=domain_info[0].device)
        else:
            domain_emb, _ = self.domain_enc(*domain_info)
        
        emb = torch.cat([struct_emb, domain_emb],dim=1)
        emb = self.glob_scale(emb)
        emb = F.relu(emb)
        # emb: [batch_size, final_hid_dim]
        
        # 最终融合，得到最终的embedding
        emb = self.dropout(emb)
        emb = self.final_out(emb)
        emb = emb / emb.norm(p=2, dim=1, keepdim=True)
        return emb
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

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