import Bio.PDB.PDBParser
import os
import numpy
import torch
import dgl
 
parser = Bio.PDB.PDBParser()
# 连接Cα的肽键的边特征向量
Peptide_bond_feature = torch.tensor([1.0,0.0])
# 连接Cα的二硫键的边特征向量
S_S_bond_feature = torch.tensor([0.0,1.0])
 
three2one= {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}
# 氨基酸的特征
AA_feature_blosum80 = {
    'A' : torch.tensor([ 7, -3, -3, -3, -1, -2, -2,  0, -3, -3, -3, -1, -2, -4, -1,  2,  0, -5, -4, -1, -3, -2, -1]),
    'R' : torch.tensor([-3,  9, -1, -3, -6,  1, -1, -4,  0, -5, -4,  3, -3, -5, -3, -2, -2, -5, -4, -4, -2,  0, -2]),
    'N' : torch.tensor([-3, -1,  9,  2, -5,  0, -1, -1,  1, -6, -6,  0, -4, -6, -4,  1,  0, -7, -4, -5,  5, -1, -2]),
    'D' : torch.tensor([-3, -3,  2, 10, -7, -1,  2, -3, -2, -7, -7, -2, -6, -6, -3, -1, -2, -8, -6, -6,  6,  1, -3]),
    'C' : torch.tensor([-1, -6, -5, -7, 13, -5, -7, -6, -7, -2, -3, -6, -3, -4, -6, -2, -2, -5, -5, -2, -6, -7, -4]),
    'Q' : torch.tensor([-2,  1,  0, -1, -5,  9,  3, -4,  1, -5, -4,  2, -1, -5, -3, -1, -1, -4, -3, -4, -1,  5, -2]),
    'E' : torch.tensor([-2, -1, -1,  2, -7,  3,  8, -4,  0, -6, -6,  1, -4, -6, -2, -1, -2, -6, -5, -4,  1,  6, -2]),
    'G' : torch.tensor([ 0, -4, -1, -3, -6, -4, -4,  9, -4, -7, -7, -3, -5, -6, -5, -1, -3, -6, -6, -6, -2, -4, -3]),
    'H' : torch.tensor([-3,  0,  1, -2, -7,  1,  0, -4, 12, -6, -5, -1, -4, -2, -4, -2, -3, -4,  3, -5, -1,  0, -2]),
    'I' : torch.tensor([-3, -5, -6, -7, -2, -5, -6, -7, -6,  7,  2, -5,  2, -1, -5, -4, -2, -5, -3,  4, -6, -6, -2]),
    'L' : torch.tensor([-3, -4, -6, -7, -3, -4, -6, -7, -5,  2,  6, -4,  3,  0, -5, -4, -3, -4, -2,  1, -7, -5, -2]),
    'K' : torch.tensor([-1,  3,  0, -2, -6,  2,  1, -3, -1, -5, -4,  8, -3, -5, -2, -1, -1, -6, -4, -4, -1,  1, -2]),
    'M' : torch.tensor([-2, -3, -4, -6, -3, -1, -4, -5, -4,  2,  3, -3,  9,  0, -4, -3, -1, -3, -3,  1, -5, -3, -2]),
    'F' : torch.tensor([-4, -5, -6, -6, -4, -5, -6, -6, -2, -1,  0, -5,  0, 10, -6, -4, -4,  0,  4, -2, -6, -6, -3]),
    'P' : torch.tensor([-1, -3, -4, -3, -6, -3, -2, -5, -4, -5, -5, -2, -4, -6, 12, -2, -3, -7, -6, -4, -4, -2, -3]),
    'S' : torch.tensor([ 2, -2,  1, -1, -2, -1, -1, -1, -2, -4, -4, -1, -3, -4, -2,  7,  2, -6, -3, -3,  0, -1, -1]),
    'T' : torch.tensor([ 0, -2,  0, -2, -2, -1, -2, -3, -3, -2, -3, -1, -1, -4, -3,  2,  8, -5, -3,  0, -1, -2, -1]),
    'W' : torch.tensor([-5, -5, -7, -8, -5, -4, -6, -6, -4, -5, -4, -6, -3,  0, -7, -6, -5, 16,  3, -5, -8, -5, -5]),
    'Y' : torch.tensor([-4, -4, -4, -6, -5, -3, -5, -6,  3, -3, -2, -4, -3,  4, -6, -3, -3,  3, 11, -3, -5, -4, -3]),
    'V' : torch.tensor([-1, -4, -5, -6, -2, -4, -4, -6, -5,  4,  1, -4,  1, -2, -4, -3,  0, -5, -3,  7, -6, -4, -2]),
    'B' : torch.tensor([-3, -2,  5,  6, -6, -1,  1, -2, -1, -6, -7, -1, -5, -6, -4,  0, -1, -8, -5, -6,  6,  0, -3]),
    'Z' : torch.tensor([-2,  0, -1,  1, -7,  5,  6, -4,  0, -6, -5,  1, -3, -6, -2, -1, -2, -5, -4, -4,  0,  6, -1]),
    'X' : torch.tensor([-1, -2, -2, -3, -4, -2, -2, -3, -2, -2, -2, -2, -2, -3, -3, -1, -1, -5, -3, -2, -3, -1, -2]),
}
 
def get_seq_graph_info(pdbfilepath):
    struct = parser.get_structure(0,pdbfilepath)
    model = struct.get_models().__next__()
    seq = []
    graph = dgl.DGLGraph()
 
    residue_num = 0
    for chain in model.get_chains():
        residues = chain.get_residues()
        # print(chain)
        chain_start_flag = 1
        for residue in residues:
            res_name = three2one[residue.get_resname()]
            seq.append(res_name)
            pos_feature = torch.from_numpy(residue['CA'].get_coord()).unsqueeze(0)
            residue_feature = AA_feature_blosum80[res_name].unsqueeze(0)
            graph.add_nodes(1, {'pos': pos_feature, 'res': residue_feature})
            if chain_start_flag == 0 :
                graph.add_edges(torch.tensor([residue_num-1]), torch.tensor([residue_num]), {'feat': Peptide_bond_feature})
                graph.add_edges(torch.tensor([residue_num]), torch.tensor([residue_num-1]), {'feat': Peptide_bond_feature})
            residue_num += 1
            chain_start_flag = 0
    return seq,graph

# 这里物种以大肠杆菌ecoli为例
pdb_data_dir = os.path.join('.','data','ecoli')
file_list = [os.path.join(pdb_data_dir, file) for file in os.listdir(pdb_data_dir)]
for file in file_list:
    if file.endswith('.pdb'):
        print("processing "+file)
        seq,graph = get_seq_graph_info(file)
        print(seq)
        print(graph)
        print("finished!")