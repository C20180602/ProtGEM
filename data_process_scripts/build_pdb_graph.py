'''
获取蛋白质结构图DGLGraph
建议先运行pretrain/seq_esm.py后再使用此脚本

图特征：
node: "pos": 6, "res": 46
edge: "feat": 3
'''
import Bio.PDB.PDBParser
import os
import numpy as np
import torch
import dgl
import warnings
import json
from tqdm import tqdm

parser = Bio.PDB.PDBParser()
# 连接Cα的肽键的边特征向量
Peptide_bond_feature = torch.tensor([1.0,0.0,0.0])
# 连接Cα的二硫键的边特征向量
Disulfide_bond_feature = torch.tensor([0.0,1.0,0.0])
# 因为相邻距离小于 10 AA （1AA = 0.1纳米） 而连接的Cα的边特征向量
Close_bond_feature = torch.tensor([0.0,0.0,1.0])

# 二硫键长度大约在2.0~2.2AA之间，在CYS之间产生
thresh_ss = 2.5
thresh_aa = 10.0

three2one= {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}
# 氨基酸的特征 23维
AA_feature_onehot = {
    'A' : torch.tensor([ 1.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'R' : torch.tensor([ 0.,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'N' : torch.tensor([ 0.,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'D' : torch.tensor([ 0.,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'C' : torch.tensor([ 0.,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'Q' : torch.tensor([ 0.,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'E' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'G' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'H' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'I' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'L' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'K' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'M' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'F' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0]),
    'P' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0]),
    'S' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0]),
    'T' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0]),
    'W' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]),
    'Y' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0]),
    'V' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0]),
    'B' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0]),
    'Z' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0]),
    'X' : torch.tensor([ 0.,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1]),
}
AA_feature_blosum80 = {
    'A' : torch.tensor([ 7., -3, -3, -3, -1, -2, -2,  0, -3, -3, -3, -1, -2, -4, -1,  2,  0, -5, -4, -1, -3, -2, -1]),
    'R' : torch.tensor([-3.,  9, -1, -3, -6,  1, -1, -4,  0, -5, -4,  3, -3, -5, -3, -2, -2, -5, -4, -4, -2,  0, -2]),
    'N' : torch.tensor([-3., -1,  9,  2, -5,  0, -1, -1,  1, -6, -6,  0, -4, -6, -4,  1,  0, -7, -4, -5,  5, -1, -2]),
    'D' : torch.tensor([-3., -3,  2, 10, -7, -1,  2, -3, -2, -7, -7, -2, -6, -6, -3, -1, -2, -8, -6, -6,  6,  1, -3]),
    'C' : torch.tensor([-1., -6, -5, -7, 13, -5, -7, -6, -7, -2, -3, -6, -3, -4, -6, -2, -2, -5, -5, -2, -6, -7, -4]),
    'Q' : torch.tensor([-2.,  1,  0, -1, -5,  9,  3, -4,  1, -5, -4,  2, -1, -5, -3, -1, -1, -4, -3, -4, -1,  5, -2]),
    'E' : torch.tensor([-2., -1, -1,  2, -7,  3,  8, -4,  0, -6, -6,  1, -4, -6, -2, -1, -2, -6, -5, -4,  1,  6, -2]),
    'G' : torch.tensor([ 0., -4, -1, -3, -6, -4, -4,  9, -4, -7, -7, -3, -5, -6, -5, -1, -3, -6, -6, -6, -2, -4, -3]),
    'H' : torch.tensor([-3.,  0,  1, -2, -7,  1,  0, -4, 12, -6, -5, -1, -4, -2, -4, -2, -3, -4,  3, -5, -1,  0, -2]),
    'I' : torch.tensor([-3., -5, -6, -7, -2, -5, -6, -7, -6,  7,  2, -5,  2, -1, -5, -4, -2, -5, -3,  4, -6, -6, -2]),
    'L' : torch.tensor([-3., -4, -6, -7, -3, -4, -6, -7, -5,  2,  6, -4,  3,  0, -5, -4, -3, -4, -2,  1, -7, -5, -2]),
    'K' : torch.tensor([-1.,  3,  0, -2, -6,  2,  1, -3, -1, -5, -4,  8, -3, -5, -2, -1, -1, -6, -4, -4, -1,  1, -2]),
    'M' : torch.tensor([-2., -3, -4, -6, -3, -1, -4, -5, -4,  2,  3, -3,  9,  0, -4, -3, -1, -3, -3,  1, -5, -3, -2]),
    'F' : torch.tensor([-4., -5, -6, -6, -4, -5, -6, -6, -2, -1,  0, -5,  0, 10, -6, -4, -4,  0,  4, -2, -6, -6, -3]),
    'P' : torch.tensor([-1., -3, -4, -3, -6, -3, -2, -5, -4, -5, -5, -2, -4, -6, 12, -2, -3, -7, -6, -4, -4, -2, -3]),
    'S' : torch.tensor([ 2., -2,  1, -1, -2, -1, -1, -1, -2, -4, -4, -1, -3, -4, -2,  7,  2, -6, -3, -3,  0, -1, -1]),
    'T' : torch.tensor([ 0., -2,  0, -2, -2, -1, -2, -3, -3, -2, -3, -1, -1, -4, -3,  2,  8, -5, -3,  0, -1, -2, -1]),
    'W' : torch.tensor([-5., -5, -7, -8, -5, -4, -6, -6, -4, -5, -4, -6, -3,  0, -7, -6, -5, 16,  3, -5, -8, -5, -5]),
    'Y' : torch.tensor([-4., -4, -4, -6, -5, -3, -5, -6,  3, -3, -2, -4, -3,  4, -6, -3, -3,  3, 11, -3, -5, -4, -3]),
    'V' : torch.tensor([-1., -4, -5, -6, -2, -4, -4, -6, -5,  4,  1, -4,  1, -2, -4, -3,  0, -5, -3,  7, -6, -4, -2]),
    'B' : torch.tensor([-3., -2,  5,  6, -6, -1,  1, -2, -1, -6, -7, -1, -5, -6, -4,  0, -1, -8, -5, -6,  6,  0, -3]),
    'Z' : torch.tensor([-2.,  0, -1,  1, -7,  5,  6, -4,  0, -6, -5,  1, -3, -6, -2, -1, -2, -5, -4, -4,  0,  6, -1]),
    'X' : torch.tensor([-1., -2, -2, -3, -4, -2, -2, -3, -2, -2, -2, -2, -2, -3, -3, -1, -1, -5, -3, -2, -3, -1, -2]),
}

# 将蛋白质中的每个氨基酸的Cα原子作为
def get_graph_info(pdbfilepath):
    struct = parser.get_structure(0,pdbfilepath)
    # 只取第一个模型
    model = struct.get_models().__next__()
    graph = dgl.DGLGraph()
    
    peptide_cnt = 0
    ss_cnt = 0
    close_cnt = 0
    res_list = []
    ca_pos_list = []
    sg_pos_list = []
    # i,j表示当前氨基酸的序号
    i = 0
    for chain in model.get_chains():
        residues = chain.get_residues()
        # print(chain)
        chain_start_flag = 1
        las_C = 0
        las_O = 0
        for residue in residues:
            res_name = three2one[residue.get_resname()]
            res_list.append(res_name)
            # pos: [node_nums,6,3]
            pos_feature = []
            pos_feature.append(torch.from_numpy(residue['N'].get_coord()))
            pos_feature.append(torch.from_numpy(residue['CA'].get_coord()))
            pos_feature.append(torch.from_numpy(residue['C'].get_coord()))
            pos_feature.append(torch.from_numpy(residue['O'].get_coord()))
            
            ca_pos_list.append(torch.from_numpy(residue['CA'].get_coord()))
            if res_name == "C":
                sg_pos_list.append(torch.from_numpy(residue['SG'].get_coord()))
            else:
                sg_pos_list.append(torch.tensor([1e9,1e9,1e9]))
            N = residue['N'].get_coord()
            CA = residue['CA'].get_coord()
            C = residue['C'].get_coord()
            O = residue['O'].get_coord()
            pos_feature.append(torch.from_numpy(np.cross(N-CA,C-CA)))
            pos_feature.append(torch.from_numpy(np.cross(CA-C,O-C)))
            pos_feature = torch.stack(pos_feature,dim=0)
            # res: [node_nums,23+23]
            residue_feature = torch.cat([AA_feature_blosum80[res_name],AA_feature_onehot[res_name]],dim=0)
            graph.add_nodes(1, {'v': pos_feature.unsqueeze(0), 's': residue_feature.unsqueeze(0)})
            if chain_start_flag == 0 :
                peptide_plane = torch.from_numpy(np.cross(las_O-las_C,N-las_C))
                bond_feature = torch.cat([Peptide_bond_feature,peptide_plane],dim=0)
                graph.add_edges(torch.tensor([i-1]), torch.tensor([i]), {'s': bond_feature.unsqueeze(0)})
                graph.add_edges(torch.tensor([i]), torch.tensor([i-1]), {'s': bond_feature.unsqueeze(0)})
                peptide_cnt += 1
            i += 1
            chain_start_flag = 0
            las_C = C
            las_O = O
    
    n = len(res_list)
    for i in range(n):
        res1 = res_list[i]
        Cpos1 = ca_pos_list[i]
        ca_pos_tensor = torch.stack(ca_pos_list)
        dis_tensor = torch.norm(ca_pos_tensor-Cpos1,dim=1)
        j_tensor = torch.where(dis_tensor < thresh_aa)[0]
        cadidate_num = j_tensor.shape[0]
        # print(cadidate_num)
        bond_feature = torch.cat([Close_bond_feature.unsqueeze(0).expand(cadidate_num,-1), ca_pos_tensor[j_tensor] - Cpos1], dim=1)
        # print(bond_feature.shape)
        graph.add_edges(torch.tensor([i]*cadidate_num), j_tensor, {'feat': bond_feature})
        close_cnt += cadidate_num
        
        if res1 == "C":
            Spos1 = sg_pos_list[i]
            sg_pos_tensor = torch.stack(sg_pos_list)

            dis_tensor = torch.norm(sg_pos_tensor-Spos1,dim=1)
            j_tensor = torch.where(dis_tensor < thresh_ss)[0]
            for j in j_tensor:
                j = j.item()
                if i == j:
                    continue
                res2 = res_list[j]
                # 当枚举的两个氨基酸均为半胱氨酸时，判断是否可能存在二硫键
                if res2=='C':
                    Spos2 = sg_pos_list[j]
                    bond_feature = torch.cat([Disulfide_bond_feature,Spos2-Spos1],dim=0)
                    graph.add_edges(torch.tensor([i]), torch.tensor([j]), {'s': bond_feature.unsqueeze(0)})
                    ss_cnt += 1
    print("edge cnt(pep, ss, close):",peptide_cnt, ss_cnt, close_cnt," => ",peptide_cnt*2+ss_cnt+close_cnt)
    print("edge_data:",graph.edata["s"].shape)
    return graph

if __name__ == "__main__":
    # 忽略使用DGLGraph建立空图的警告
    warnings.filterwarnings("ignore")
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")

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
        
        graph = get_graph_info(os.path.join(pdb_data_dir, filename))
        # print(filename)
        # print(graph)
        dgl.save_graphs(os.path.join(save_dir,pid,'struct.dgl'),graph)
        # break

    print(len(model_fnum_log))