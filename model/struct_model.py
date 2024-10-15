import torch
from torch import nn
import torch.nn.functional as F
import dgl

def L2(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    # 计算向量2-范数
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

class GVP(nn.Module):
    '''
    几何向量感知机
    in_dims, out_dims: (scalar_dim, vector_dim) (标量维度和向量维度)
    (s,v) -> (relu(wm@(s,L2(wh@v))), sigmoid(L2(wu@wh@v))*(wu@wh@v)) (自身GLU)
    或
    (s,v) -> (relu(wm@(s,L2(wh@v))), sigmoid(wm@(s,L2(wh@v)))*(wu@wh@v)) (交叉GLU)
    @ 为矩阵乘法， * 为哈达玛乘积
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), cross_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations
        self.cross_gate = cross_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.wm = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wu = nn.Linear(self.h_dim, self.vo, bias=False)
                # 用标量特征对向量特征进行门控限制
                if self.cross_gate:
                    self.wmv = nn.Linear(self.so, self.vo)
        else:
            self.wm = nn.Linear(self.si, self.so)
        
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        if self.vi:
            s, v = x
            # s=[batch, si], v=[batch, vi, 3]
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            # vh=[batch, 3, h]  
            s = self.wm(torch.cat([s, L2(vh, axis=-2)], -1))
            # s=[batch, so]
            if self.vo: 
                v = torch.transpose(self.wu(vh), -1, -2)
                # 交叉GLU
                if self.cross_gate:
                    v = v * torch.sigmoid(self.wmv(s)).unsqueeze(-1)
                # 普通GLU
                elif self.vector_act:
                    v = v * self.vector_act(L2(v, axis=-1, keepdims=True))
                else:
                    v = v * torch.sigmoid(L2(v, axis=-1, keepdims=True))
                # v=[batch, vo, 3]
        else:
            s = self.wm(x)
            # s=[batch, so]
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
            # v=[batch, vo, 3]
        if self.scalar_act:
            s = self.scalar_act(s)
        return (s, v) if self.vo else s
    
