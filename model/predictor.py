import torch
from torch import nn
import torch.nn.functional as F
from model.protein_model import ProteinEncoder, ProteinEncoderv2, ProteinEncoderv3
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

class SequenceLoss(nn.Module):
    '''
    序列重构损失函数
    '''
    def __init__(self, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, seq_pred, seq_target, mask=None):
        """
        seq_pred: [batch_size, seq_len, 21] 预测的logits或概率
        seq_target: [batch_size, seq_len] 真实的氨基酸类别(0-19)
        mask: [batch_size, seq_len] 需要计算损失的位置(True表示需要计算)
        """
        batch_size, seq_len, _ = seq_pred.shape
        
        # 如果提供了mask，只计算mask位置的损失
        if mask is not None:
            # 展平
            mask = mask.long()
            seq_pred_flat = seq_pred[mask].view(-1, 21)
            seq_target_flat = seq_target[mask].view(-1, 21)
        else:
            seq_pred_flat = seq_pred.view(-1, 21)
            seq_target_flat = seq_target.view(-1, 21)
        
        # 使用交叉熵损失，可加入标签平滑
        if self.label_smoothing > 0:
            loss = F.cross_entropy(
                seq_pred_flat, 
                seq_target_flat,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing
            )
        else:
            loss = F.cross_entropy(
                seq_pred_flat, 
                seq_target_flat,
                reduction=self.reduction
            )
        
        return loss

class GeometricStructureLoss(nn.Module):
    def __init__(self, 
                 distance_loss_weight=1.0,
                 angle_loss_weight=0.5,
                 dihedral_loss_weight=0.2):
        super().__init__()
        self.distance_loss_weight = distance_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.dihedral_loss_weight = dihedral_loss_weight
        
    def compute_pairwise_distances(self, coords):
        """计算所有原子对之间的距离矩阵"""
        # coords: [seq_len, 3]
        
        # 扩展维度用于广播计算
        coords_expanded1 = coords.unsqueeze(1)  # [L, 1, 3]
        coords_expanded2 = coords.unsqueeze(0)  # [1, L, 3]
        
        # 计算欧氏距离
        distances = torch.sqrt(
            torch.sum((coords_expanded1 - coords_expanded2) ** 2, dim=-1) + 1e-8
        )
        return distances  # [L, L]
    
    def compute_bond_angles(self, coords):
        """计算连续三个原子之间的键角"""
        # 连续原子向量
        vec1 = coords[1:-1] - coords[:-2]  # v_i-1,i
        vec2 = coords[2:] - coords[1:-1]   # v_i,i+1
        
        # 归一化
        vec1_norm = F.normalize(vec1, dim=-1)
        vec2_norm = F.normalize(vec2, dim=-1)
        
        # 计算角度 (弧度)
        cos_angles = torch.sum(vec1_norm * vec2_norm, dim=-1)
        angles = torch.acos(torch.clamp(cos_angles, -0.9999, 0.9999))
        
        return angles  # [B, L-2]
    
    def compute_dihedral_angles(self, coords):
        """计算二面角 (phi, psi, omega)"""
        
        # 计算四个连续原子形成的平面
        p0, p1, p2, p3 = coords[:-3], coords[1:-2], coords[2:-1], coords[3:]
        
        # 计算向量
        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2
        
        # 计算法向量
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        
        # 归一化
        n1_norm = F.normalize(n1, dim=-1)
        n2_norm = F.normalize(n2, dim=-1)
        b2_norm = F.normalize(b2, dim=-1)
        
        # 计算二面角
        cos_phi = torch.sum(n1_norm * n2_norm, dim=-1)
        sin_phi = torch.sum(torch.cross(n1_norm, n2_norm, dim=-1) * b2_norm, dim=-1)
        dihedrals = torch.atan2(sin_phi, cos_phi)
        
        return dihedrals  # [B, L-3]
    
    def forward(self, coord_pred, coord_target):
        # 1、距离矩阵损失（保持全局结构）
        dist_pred = self.compute_pairwise_distances(coord_pred)
        dist_target = self.compute_pairwise_distances(coord_target)
        
        # 只考虑局部距离
        local_mask = (dist_target < 10.0).float()
        distance_loss = F.smooth_l1_loss(
            dist_pred * local_mask,
            dist_target * local_mask,
            reduction='none'
        )
        distance_loss = distance_loss.sum(dim=(-1, -2)) / (local_mask.sum(dim=(-1, -2)) + 1e-8)
        
        # 2、键角损失
        angle_pred = self.compute_bond_angles(coord_pred)
        angle_target = self.compute_bond_angles(coord_target)
        angle_loss = F.smooth_l1_loss(angle_pred, angle_target, reduction='mean')
        
        # 3、二面角损失
        dihedral_pred = self.compute_dihedral_angles(coord_pred)
        dihedral_target = self.compute_dihedral_angles(coord_target)
        
        # 处理周期性：二面角在-pi到pi之间是连续的
        dihedral_diff = dihedral_pred - dihedral_target
        dihedral_diff = torch.atan2(torch.sin(dihedral_diff), torch.cos(dihedral_diff))
        dihedral_loss = torch.mean(torch.abs(dihedral_diff))
        
        # 组合所有损失
        total_loss = (
            self.distance_loss_weight * distance_loss.mean() +
            self.angle_loss_weight * angle_loss +
            self.dihedral_loss_weight * dihedral_loss
        )
        
        return total_loss

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
        
    def forward(self, seq_embs, batched_graph, domain_info, go_idx, mask=None, graph_mask=None, ablation_feat = [], pert_mode=False, get_protein_feat=False):
        if get_protein_feat:
            protein_emb, protein_emb_token = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat, pert_mode, True)
        else:
            protein_emb = self.pm(seq_embs, batched_graph, domain_info, mask, graph_mask, ablation_feat, pert_mode)
        go_emb = self.gm(go_idx)
        # protein_emb: [batch_size, hid_dim]
        # go_emb: [go_nums, hid_dim]
        protein_emb = protein_emb.unsqueeze(1)
        go_emb = go_emb.unsqueeze(0)
        predict_matrix = go_emb * protein_emb
        # predict_matrix: [batch_size, go_nums, hid_dim]
        predict_matrix = self.MLP(predict_matrix)
        predict_matrix = predict_matrix.squeeze(-1)
        if get_protein_feat:
            return predict_matrix, protein_emb, protein_emb_token
        else:
            return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class ReconstructDecoder(nn.Module):
    '''
    重构器，输入为[batch_size, hidden_dim(final_out_dim)] 和 [batch_size, seq_len, token_hid_dim]
    输出重构序列和重构坐标[batch_size, seq_len, seq_vocal_size] 和 [batch_size, seq_len, 3]
    '''
    def __init__(self, hidden_dim, token_hid_dim, num_tokens=21, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        max_seq_len = kwargs.get('max_seq_len', 2100)  # 最大2048
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)

        # 把位置和glob拼一起，融合
        self.query_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, token_hid_dim)
        )
        
        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_hid_dim,
            nhead=kwargs.get('num_heads', 2),
            dim_feedforward=token_hid_dim,
            dropout=kwargs.get('dropout', 0.1),
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=kwargs.get('num_layers', 1)  # 较少的层数
        )
        
        # 预测头
        self.token_head = nn.Linear(token_hid_dim, num_tokens)
        self.coord_head = nn.Sequential(
            nn.Linear(token_hid_dim, token_hid_dim // 2),
            nn.GELU(),
            nn.Linear(token_hid_dim // 2, 3),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, global_feature, encoder_hidden, lens, src_mask=None):
        batch_size, seq_len, _ = encoder_hidden.shape
        device = encoder_hidden.device
        
        # 1、投影全局特征
        global_proj = self.global_projection(global_feature)  # [batch, hidden]
        
        # 2、生成位置编码
        positions = torch.arange(seq_len, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.position_embeddings(positions)  # [batch, seq, hidden]
        
        # 3、生成查询（融合全局特征和位置信息）
        global_expanded = global_proj.expand(-1, seq_len, -1)
        query_input = torch.cat([global_expanded, pos_embeds], dim=-1)
        tgt_queries = self.query_generator(query_input)
        tgt_queries = self.dropout(tgt_queries)
        
        # 4、Transformer解码器
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        decoder_output = self.transformer_decoder(
            tgt=tgt_queries,
            memory=encoder_hidden,
            tgt_mask=causal_mask,
            memory_key_padding_mask=src_mask
        )
        
        # 5、输出
        token_logits = self.token_head(decoder_output)
        coordinates = self.coord_head(decoder_output)
        coords_logits = []
        for i, len in enumerate(lens):
            coords_logits.append(coordinates[i,:len,:])
        
        return token_logits, coords_logits

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
        
class ProteinGoModelv3(nn.Module):
    '''
    蛋白质和GO标签的联合模型
    '''
    def __init__(self, pm_param, gm_param, layer_cnt=1, hid_dim=256):
        super(ProteinGoModelv3, self).__init__()
        self.pm = ProteinEncoderv3(**pm_param)
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

class PPIModel(nn.Module):
    '''
    PPI预测模型
    '''
    def __init__(self, pm_param, layer_cnt=1, hid_dim=256):
        super(PPIModel, self).__init__()
        self.pm = ProteinEncoder(**pm_param)
        
        self.in_dim = pm_param['final_out_dim']
        self.hid_dim = hid_dim
        self.MLP = nn.Sequential()
        for i in range(layer_cnt):
            if i==0:
                self.MLP.append(nn.Linear(self.in_dim*2, self.hid_dim))
                # self.MLP.append(nn.Linear(self.in_dim, self.hid_dim))
            else:
                self.MLP.append(nn.Linear(self.hid_dim, self.hid_dim))
            self.MLP.append(nn.LayerNorm(self.hid_dim))
            self.MLP.append(nn.ELU())
        self.MLP.append(nn.Linear(self.hid_dim, 1))
        self.MLP.append(nn.Sigmoid())
        
    def forward(self, seq_embs1, batched_graph1, domain_info1,
                      seq_embs2, batched_graph2, domain_info2,
                      mask1=None, graph_mask1=None, mask2=None, graph_mask2=None, ablation_feat = []):
        
        protein_emb1 = self.pm(seq_embs1, batched_graph1, domain_info1, mask1, graph_mask1, ablation_feat)
        protein_emb2 = self.pm(seq_embs2, batched_graph2, domain_info2, mask2, graph_mask2, ablation_feat)
        # protein_emb: [batch_size, hid_dim]
        protein_emb = torch.cat([protein_emb1 , protein_emb2],dim=1)
        # protein_emb = protein_emb1 * protein_emb2
        predict_matrix = self.MLP(protein_emb)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
        # predict_matrix: [batch_size]
    
    def upstream_infer(self, seq_embs, batched_graph, domain_info, ablation_feat = []):
        protein_emb = self.pm(seq_embs, batched_graph, domain_info, ablation_feat=ablation_feat)
        return protein_emb
    
    def downstream_infer(self, emb1, emb2):
        protein_emb = torch.cat([emb1 , emb2],dim=1)
        # protein_emb = emb1 * emb2
        predict_matrix = self.MLP(protein_emb)
        predict_matrix = predict_matrix.squeeze(-1)
        return predict_matrix
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))