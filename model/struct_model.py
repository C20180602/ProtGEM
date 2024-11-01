import functools
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

def L2(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    # x: [...,vector_num, vector_dim]
    # 计算向量2-范数
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    # out: [...,vector_num] || [vector_num, 1]
    return torch.sqrt(out) if sqrt else out

class VectorDropout(nn.Module):
    '''
    对vector整体进行Dropout
    '''
    def __init__(self, p):
        super(VectorDropout, self).__init__()
        self.p = p

    def forward(self, x):
        # x: [..., vector_num, vector_dim]
        if not self.training:
            return x
        
        mask = ((torch.rand(x.shape[-2])) > self.p).float().unsqueeze(-1)
        x = mask*x/(1-self.p)
        return x
    
class VectorLayerNorm(nn.Module):
    '''
    对单个节点的所有vector整体进行LayerNorm
    限制所有vector的模长
    '''
    def __init__(self, vec_num):
        super(VectorLayerNorm, self).__init__()
        self.vec_num = vec_num

    def forward(self, v):
        # v: [..., node_num, vector_num, vector_dim]
        vn = L2(v, axis=-1, keepdims=True, sqrt=False)
        # vn: [..., node_num, vector_num, 1]
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        # vn: [..., node_num, 1, 1]
        return v / vn

class GVP(nn.Module):
    '''
    几何向量感知机
    
    :in_dims, out_dims: (scalar_dim, vector_dim) (标量维度和向量特征的个数)
    
    :
    
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
    
class GVPConvLayer(nn.Module):
    '''
    使用DGL实现的GVP卷积层, 一层图卷积+一层节点前馈网络
    
    在消息函数处使用GVP, 聚合函数使用均值聚合mean
    
    由于使用了残差网络, in_dims和out_dims需要一致
    
    :num_layers: GVP层数
    
    :in_dims, out_dims: (scalar_dim, vector_dim) (标量维度和向量特征的个数)
    
    :edge_dims: (scalar_dim, vector_dim) (边的标量维度和向量特征的个数)
    
    :dropout: 标量特征和向量特征使用同样的Dropout率, 向量特征进行整体Dropout
    
    '''
    def __init__(self, in_dims, out_dims, edge_dims, conv_gvp_layers=3, ffn_gvp_layers=2, dropout=0.5, h_dim=None, 
                 activations=(F.relu, torch.sigmoid), cross_gate=False):
        super(GVPConvLayer, self).__init__()
        
        self.ins, self.inv = in_dims
        self.outs,self.outv= out_dims
        self.edge_s,self.edge_v= edge_dims
        # TODO: edge没有向量特征，可以考虑后期添加
        
        self.num_layers = conv_gvp_layers
        tmp_gvp = functools.partial(GVP, activations=activations, cross_gate=cross_gate)
        self.gvp_layers1 = nn.Sequential(tmp_gvp(( 2*self.ins + self.edge_s, self.inv + self.edge_v ), out_dims, h_dim))
        for _ in range(conv_gvp_layers-1):
            self.gvp_layers1.append(tmp_gvp(out_dims,out_dims,h_dim))
        
        self.gvp_layers2 = nn.Sequential()
        for _ in range(ffn_gvp_layers):
            self.gvp_layers2.append(tmp_gvp(out_dims,out_dims,h_dim))
            
        self.dropout_s = nn.Dropout(dropout)
        self.dropout_v = VectorDropout(dropout)
        self.layernorm_s = nn.LayerNorm(self.outs)
        self.layernorm_v = VectorLayerNorm(self.outv)
        
        
    def edge_udf(self,edges):
        # edges.src/dst: 'res': [batch_size(dgl), 23], 'pos': [batch_size(dgl), 1, 3]
        
        s = torch.cat((edges.src["res"],edges.dst["res"],edges.data["feat"]),dim=1)
        # s: [batch_size, 23 * 2 + 3]
        
        v = edges.dst["pos"]-edges.src["pos"]
        # v: [batch_size, 1, 3]
        
        s,v = self.gvp_layers1((s,v))
        v = v.view(v.shape[0],-1)
        # v: [batch_size, 1*3]
        # 这里把s和v拼接到一起，方便后续使用DGL内置函数进行处理
        return {"gvp_out" : torch.cat((s,v),dim=1)}
    
    # 完成残差拼接和
    def update_udf(self,g,s,v):
        g.ndata["res"] += s
        g.ndata["pos"] += v
        g.ndata["res"] = self.layernorm_s(g.ndata["res"])
        g.ndata["pos"] = self.layernorm_v(g.ndata["pos"])
    
    def forward(self, g: dgl.DGLGraph):
        g.update_all(self.edge_udf,fn.mean("gvp_out","merge"))
        merge_sv = g.ndata["merge"].split(split_size = [self.outs, self.outv*3],dim=1)
        s = self.dropout_s(merge_sv[0])
        v = merge_sv[1]
        # v: [num_nodes, 3]
        v = v.view(v.shape[0],self.outv,-1)
        # v: [num_nodes, num_vec(outv), 3]
        v = self.dropout_v(v)
        self.update_udf(g,s,v)
        # feed forward network
        s,v = self.gvp_layers2((g.ndata["res"],g.ndata["pos"]))
        # s:[num_nodes, s_dim] v: [num_nodes, v_dim, 3]
        self.update_udf(g,s,v)
        return g

class StructEncoder(nn.Moduel):
    def __init__(self, in_dims, out_dims, edge_dims, conv_gvp_layers=3, ffn_gvp_layers=2, dropout=0.5, h_dim=None, 
                 activations=(F.relu, torch.sigmoid), cross_gate=False): 
        super(GVPConvLayer, self).__init__()
    
    def forward(self, g):
        pass

# DEBUG
import os
# 设置工作目录为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")
model = GVPConvLayer((23,1),(23,1),(3,0),cross_gate=True)
g,_ = dgl.load_graphs(os.path.join("dataset","ECOLI","P77302",'struct.dgl'))
g=g[0]
model.forward(g)