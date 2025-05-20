import torch
from torch import nn
import torch.nn.functional as F
from model.protein_model import ProteinEncoder, ProteinEncoderv2
from model.go_model import GoEncoder

class ContrastLoss(nn.Module):
    '''
    对比学习的损失函数
    '''
    def __init__(self):
        super(ContrastLoss, self).__init__()
    
    def forward(self, pred, labels, t = 0.05):
        # pred: [batch_size, go_nums]
        # labels: [batch_size, go_nums]
        pred = torch.exp(pred/t)
        pos = torch.sum(pred * labels, dim=-1)
        neg = torch.sum(pred * (1-labels), dim=-1)
        frac = pos / (pos + neg)
        return torch.mean(-torch.log(frac))

class ProteinGoModel(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param, layer_cnt=1, hid_dim=256):
        super(ProteinGoModel, self).__init__()
        self.pm = ProteinEncoder(**pm_param)
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
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat = []):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat)
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
        
class ProteinGoModelv2(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param):
        super(ProteinGoModelv2, self).__init__()
        self.pm = ProteinEncoderv2(**pm_param)
        self.gm = GoEncoder(**gm_param)
        
        assert pm_param['final_out_dim'] == gm_param['emb_dim']
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = 256
        self.MLP = nn.Sequential(
            nn.Linear(self.in_dim, self.hid_dim),
            nn.ELU(),
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat = []):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat)
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