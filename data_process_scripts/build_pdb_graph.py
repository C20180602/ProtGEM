'''
获取蛋白质结构图DGLGraph
建议先运行pretrain/seq_esm.py后再使用此脚本

图特征：
node: "pos": 3, "res": 23
edge: "feat": 3

TODO: 尝试使用GVP中的建图模式
'''
import Bio.PDB.PDBParser
import os
import numpy
import torch
import dgl
import warnings
from tqdm import tqdm

# 忽略使用DGLGraph建立空图的警告
warnings.filterwarnings("ignore")
# 设置工作目录为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

parser = Bio.PDB.PDBParser()
# 连接Cα的肽键的边特征向量
Peptide_bond_feature = torch.tensor([[1.0,0.0,0.0]])
# 连接Cα的二硫键的边特征向量
Disulfide_bond_feature = torch.tensor([[0.0,1.0,0.0]])
# 因为相邻距离小于 10 AA （1AA = 0.1纳米） 而连接的Cα的边特征向量
Close_bond_feature = torch.tensor([[0.0,0.0,1.0]])

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
    
    # i,j表示当前氨基酸的序号
    i = 0
    for chain in model.get_chains():
        residues = chain.get_residues()
        # print(chain)
        chain_start_flag = 1
        for residue in residues:
            res_name = three2one[residue.get_resname()]
            # pos: [node_nums,1,3]
            pos_feature = torch.from_numpy(residue['CA'].get_coord()).unsqueeze(0)
            # res: [node_nums,23]
            residue_feature = AA_feature_blosum80[res_name]
            graph.add_nodes(1, {'pos': pos_feature.unsqueeze(0), 'res': residue_feature.unsqueeze(0)})
            if chain_start_flag == 0 :
                graph.add_edges(torch.tensor([i-1]), torch.tensor([i]), {'feat': Peptide_bond_feature})
                graph.add_edges(torch.tensor([i]), torch.tensor([i-1]), {'feat': Peptide_bond_feature})
            i += 1
            chain_start_flag = 0
    i = 0
    for residue1 in model.get_residues():
        j = 0
        for residue2 in model.get_residues():
            if j <= i:
                continue
            res_name1 = three2one[residue1.get_resname()]
            res_name2 = three2one[residue2.get_resname()]

            Cpos1 = torch.from_numpy(residue1['CA'].get_coord()).unsqueeze(0)
            Cpos2 = torch.from_numpy(residue2['CA'].get_coord()).unsqueeze(0)
            distance = float(torch.norm(Cpos1-Cpos2))
            if distance <= thresh_aa:
                graph.add_edges(torch.tensor([i]), torch.tensor([j]), {'feat': Close_bond_feature})
                graph.add_edges(torch.tensor([j]), torch.tensor([i]), {'feat': Close_bond_feature})
            
            # 当枚举的两个氨基酸均为半胱氨酸时，判断是否可能存在二硫键
            if res_name1=='C' and res_name2=='C':
                Spos1 = torch.from_numpy(residue1['SG'].get_coord()).unsqueeze(0)
                Spos2 = torch.from_numpy(residue2['SG'].get_coord()).unsqueeze(0)
                if float(torch.norm(Spos1-Spos2)) <= thresh_ss:
                    graph.add_edges(torch.tensor([i]), torch.tensor([j]), {'feat': Disulfide_bond_feature})
                    graph.add_edges(torch.tensor([j]), torch.tensor([i]), {'feat': Disulfide_bond_feature})
                
            j += 1
        i += 1
    return graph

species_name = 'HUMAN'
pdb_data_dir = os.path.join('raw_data',species_name,'pdb')
save_dir = os.path.join('dataset',species_name)
save_dir_list = os.listdir(save_dir)
cnt=0
for filename in tqdm(os.listdir(pdb_data_dir)):
    if filename.endswith('.pdb'):
        # 只使用AlphaFold预测的F1模型
        if not (filename.split('-')[2] == 'F1'):
            continue
        
        protein_name=filename.split('-')[1]
        cnt+=1
        # print(cnt,protein_name)
        
        # 为保证数据集完整性,暂时不考虑没有Uniprot序列记录的数据
        if protein_name not in save_dir_list:
            print("Warning: ",protein_name," dosen't exist directory to save file")
            # os.mkdir(os.path.join(save_dir,protein_name))
            continue
        
        graph = get_graph_info(os.path.join(pdb_data_dir, filename))
        print(filename)
        print(graph)
        dgl.save_graphs(os.path.join(save_dir,protein_name,'struct.dgl'),graph)