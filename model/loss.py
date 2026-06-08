import torch
from torch import nn
import torch.nn.functional as F

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

class GoContrastLoss(nn.Module):
    '''
    GO嵌入对比学习的损失函数
    '''
    def __init__(self, pos_k, neg_k):
        super(GoContrastLoss, self).__init__()
        self.pos_k = pos_k
        self.neg_k = neg_k
    
    def forward(self, x, t = 0.07):
        # x: [batch_size, ( 1 + pos_k + neg_k ), hid_dim]
        idx = x[:,:1,:]
        # idx: [batch_size, 1, hid_dim]
        obj = x[:,1:,:]
        # obj: [batch_size, pos_k + neg_k, hid_dim]
        sim = torch.norm(obj-idx,dim=-1)
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
        # pos = torch.exp(pos/t)
        # neg = torch.exp(neg/t)
        
        # return torch.mean(torch.sum(neg, dim=-1)) / torch.mean(torch.sum(pos, dim=-1))
        frac = torch.sum(pos, dim=-1) / (torch.sum(pos, dim=-1) + torch.sum(neg, dim=-1))
        # frac: [batch_size]
        return torch.mean(-torch.log(frac))

class DPOLoss(nn.Module):
    '''
    后训练DPO直接偏好优化损失
    '''
    def __init__(self, kl_coef=0.1):
        super(DPOLoss, self).__init__()
        self.kl_coef = kl_coef
    def forward(self, pos_new_mat, pos_ref_mat, neg_new_mat, neg_ref_mat):
        # 使用抽取的偏好对预测值，计算DPO损失
        # pos_new_mat, neg_new_mat: [batch_size, go_nums]
        # pos_ref_mat, neg_ref_mat: [batch_size, go_nums]
        # 计算新旧预测值的差异
        pos_diff = torch.log(pos_new_mat / (pos_ref_mat+1e-8) +1e-8)
        neg_diff = torch.log(neg_new_mat / (neg_ref_mat+1e-8) +1e-8)
        # 计算DPO损失
        dpo_loss = -torch.mean(torch.log(torch.sigmoid(pos_diff - neg_diff)))
        # 计算KL散度作为正则化项
        kl_loss = F.kl_div(torch.log(pos_new_mat+1e-8), pos_ref_mat, reduction='batchmean') + F.kl_div(torch.log(neg_new_mat+1e-8), neg_ref_mat, reduction='batchmean')
        # 综合损失
        total_loss = dpo_loss + self.kl_coef * kl_loss
        return total_loss

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