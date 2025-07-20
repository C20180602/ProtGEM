# only sequence feature
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.transformer as tf
from model.domain_model import DomainEncoder
from model.go_model import GoEncoder

class ProteinEncoder_domain_abl(nn.Module):
    '''
    蛋白质进行编码
    每个蛋白质最终对应一个1024维的向量
    '''
    def __init__(self, final_out_dim=1024, final_hid_dim = 768, fusion_layers=2, fusion_att_heads=2, fusion_hid_dim=256, fusion_dropout=0.5, model_args = None):
        super(ProteinEncoder_domain_abl, self).__init__()
        # 没有给定参数文件则使用默认参数
        if model_args is None:
            self.domain_enc = DomainEncoder()
        else:
            self.domain_args = model_args['DomainEncoder']
            
            self.domain_enc = DomainEncoder(**self.domain_args)
        
        self.glob_dims = self.domain_enc.glob_dims
        self.part_dims = self.domain_enc.part_dims
        self.final_hid_dim = final_hid_dim
        
        self.fusion_layers = nn.Sequential()
        for i in range(fusion_layers):
            self.fusion_layers.add_module("Layer"+str(i),tf.TransformerEncoderLayer(self.part_dims, fusion_att_heads, fusion_hid_dim, fusion_dropout))
        
        self.dropout = nn.Dropout(fusion_dropout)
        self.glob_scale = nn.Linear(self.glob_dims, self.final_hid_dim)
        self.part_scale = nn.Linear(self.part_dims, self.final_hid_dim)
        self.final_out = nn.Linear(2 * self.final_hid_dim, final_out_dim)
        
    def forward(self, seq_embs, batched_graph, domain_info, mask=None, graph_mask=None):
        # seq_embs: [batch_size, seq_len, seq_dim]
        # mask [batch_size, seq_len]
        batch_size = seq_embs.shape[0]
        max_len = seq_embs.shape[1]
        
        domain_glob, domain_part = self.domain_enc(*domain_info)
        # xxx_glob: [batch_size, global_dims]
        # seq_part: [batch_size, max_len, part_dims]
        # (struct/domain)_part: list(len=batch_size)[len_i, part_dims]
        
        # 对domain_part进行padding
        padded_domain_part = []
        for domain_emb in domain_part:
            padding = torch.zeros(max_len - domain_emb.shape[0], domain_emb.shape[1], device=domain_emb.device)
            padded_domain_part.append(torch.cat([domain_emb, padding],dim=0))
        padded_domain_part = torch.stack(padded_domain_part)
        if mask is not None:
            padded_domain_part = padded_domain_part * mask.unsqueeze(-1)
        
        # padded_(domain/struct)_part: [batch_size, max_len, (domain/struct)_part_dims]
        
        # 融合全局特征
        glob_emb = domain_glob
        glob_emb = self.glob_scale(glob_emb)
        glob_emb = F.relu(glob_emb)
        # glob_emb: [batch_size, final_hid_dim]
        
        # 融合局部特征：
        part_emb =padded_domain_part
        for layer in self.fusion_layers:
            part_emb = layer(part_emb, mask)
        # part_emb: [batch_size, max_len, part_dims]
        part_emb = part_emb[:,0,:]
        # part_emb: [batch_size, part_dims]
        part_emb = self.part_scale(part_emb)
        part_emb = F.relu(part_emb)
        # part_emb: [batch_size, final_hid_dim]
        
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
        
class ProteinGoModel_domain_abl(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param, layer_cnt=1, hid_dim=256):
        super(ProteinGoModel_domain_abl, self).__init__()
        self.pm = ProteinEncoder_domain_abl(**pm_param)
        self.gm = GoEncoder(**gm_param)
        
        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential()
        for i in range(layer_cnt):
            if i==0:
                self.MLP.append(nn.Linear(self.in_dim, self.hid_dim))
            else:
                self.MLP.append(nn.Linear(self.hid_dim, self.hid_dim))
            self.MLP.append(nn.LayerNorm(self.hid_dim))
            self.MLP.append(nn.ELU())
        self.MLP.append(nn.Linear(self.hid_dim, 1))
        self.MLP.append(nn.Sigmoid())
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask)
        go_emb = self.gm(go_idx)
        # protein_emb: [batch_size, hid_dim]
        # go_emb: [go_nums, hid_dim]
        protein_emb = protein_emb.unsqueeze(1)
        go_emb = go_emb.unsqueeze(0)
        predict_matrix = go_emb * protein_emb
        # predict_matrix: [batch_size, go_nums, hid_dim]
        predict_matrix = self.MLP(predict_matrix)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))