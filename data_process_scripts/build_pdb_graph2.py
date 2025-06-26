'''
获取蛋白质结构图DGLGraph
使用了GVP论文中的建图模式
建议先运行pretrain/seq_esm.py后再使用此脚本

图特征：
node: "s": 6, "v": 3
edge: "s": 32, "v": 1
'''
import math
import torch
import numpy as np
import torch.nn.functional as F
import torch_cluster
import Bio.PDB.PDBParser
import dgl

import os
import warnings
import json
from tqdm import tqdm

class PDB_Builder:
    '''
    Modified From https://github.com/SIAT-code/MASSA
    '''
    def __init__(self, num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cuda:0"):
        
        super(PDB_Builder, self).__init__()
        
        self.parser = Bio.PDB.PDBParser()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
    
    def _normalize(self, tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


    def _rbf(self, D, D_min=0., D_max=20., D_count=16, device='cuda:0'):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        
        Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
        That is, if `D` has shape [...dims], then the returned tensor will have
        shape [...dims, D_count].
        '''
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = self._normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = self._normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features


    def _positional_embeddings(self, edge_index):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
        
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        forward = self._normalize(X[1:] - X[:-1])
        backward = self._normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = self._normalize(c - origin), self._normalize(n - origin)
        bisector = self._normalize(c + n)
        perp = self._normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 

    def featurize_as_graph(self, coords):
        with torch.no_grad():
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1]
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index)
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
            rbf = self._rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
            
            dihedrals = self._dihedrals(coords)                     
            orientations = self._orientations(X_ca)
            sidechains = self._sidechains(coords)
            
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
            edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
            edge_v = self._normalize(E_vectors).unsqueeze(-2)
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                    (node_s, node_v, edge_s, edge_v))
            # Create a DGL graph
            g = dgl.graph((edge_index[0],edge_index[1]))
            g.ndata["s"] = node_s
            g.ndata["v"] = node_v
            g.edata["s"] = edge_s
            g.edata["v"] = edge_v

        return g

    def get_graph_info(self, pdbfilepath):
        struct = self.parser.get_structure(0, pdbfilepath)
        # 只取第一个模型
        model = struct.get_models().__next__()
        graph = dgl.DGLGraph()
        residue_list = []
        for chain in model.get_chains():
            residues = chain.get_residues()
            for residue in residues:
                pos_list = []
                pos_list.append(torch.from_numpy(residue['N'].get_coord()))
                pos_list.append(torch.from_numpy(residue['CA'].get_coord()))
                pos_list.append(torch.from_numpy(residue['C'].get_coord()))
                pos_list.append(torch.from_numpy(residue['O'].get_coord()))
                pos_tensor = torch.stack(pos_list,dim=0)
                residue_list.append(pos_tensor)
        protein_tensor = torch.stack(residue_list,dim=0)
        protein_tensor = protein_tensor.to(self.device)
        graph = self.featurize_as_graph(protein_tensor)
        return graph

            
if __name__ == "__main__":
    # 忽略使用DGLGraph建立空图的警告
    warnings.filterwarnings("ignore")
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    builder = PDB_Builder()
    
    with open("config/global.conf","r") as f:
        global_var = json.load(f)
    species_name = global_var["species_name"]

    with open(os.path.join("processed_data",species_name+"_protein_info.json"),"r") as f:
        protein_info = json.load(f)
    pdb_data_dir = os.path.join('raw_data',species_name,'pdb')
    save_dir = os.path.join('dataset',species_name)
    save_dir_list = os.listdir(save_dir)
    model_fnum_log = {}
    for pid in tqdm(protein_info):
        filename="AF-{}-F1-model_v4.pdb".format(pid)
        
        graph = builder.get_graph_info(os.path.join(pdb_data_dir, filename))
        print(filename)
        print(graph)
        dgl.save_graphs(os.path.join(save_dir,pid,'struct_gvp.dgl'),graph)
        # break

    print(len(model_fnum_log))