'''
用于生成序列中每个氨基酸的特征向量
'''
import json
import os
import pickle
import click

import torch
import esm
import Bio.PDB.PDBParser
from pathlib import Path
from tqdm import tqdm

three2one= {
    'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
    'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
    'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
    'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
}

def get_seq_info(pdbfilepath):
    parser = Bio.PDB.PDBParser()
    struct = parser.get_structure(0,pdbfilepath)
    # 只取第一个模型
    model = struct.get_models().__next__()
    # i表示当前氨基酸的序号
    seq = []
    for res in model.get_residues():
        acid = three2one[res.get_resname()]
        seq.append(acid)
    return "".join(seq)

# 设置CUDA设备编号
device = "cuda:0"

# ESM预训练模型初始化,取消梯度
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
converter = alphabet.get_batch_converter()
model.to(device)
for p in model.parameters():
    p.requires_grad = False
model.eval()

@click.command()
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
def main(species_name):
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    pdb_dir_path = Path("raw_data",species_name,"pdb")
    for pdb_file in tqdm(os.listdir(pdb_dir_path)):
        if not pdb_file.endswith(".pdb"):
            continue
        if "AF-" in pdb_file:
            pid = pdb_file.split("-")[1]
        else:
            pid = pdb_file.split(".")[0]
        seq = get_seq_info(Path(pdb_dir_path,pdb_file))
        if len(seq) > 2048:
            print(f"Warning: {pid} length greater than 2048, skipped!")
            continue
        labels, strs, tokens = converter([(pid,seq)])
        tokens = tokens.to(device)
        seq_emb = model(tokens, repr_layers=[33])['representations'][33]
        seq_emb = seq_emb.squeeze(0)
        seq_emb = seq_emb.cpu().numpy()
        
        dir_path = Path("dataset",species_name,pid)
        os.makedirs(dir_path, exist_ok=True)
        with open(Path(dir_path,"seq_emb.pkl"), 'wb') as f:
            pickle.dump(seq_emb, f)
        del tokens,seq_emb
        torch.cuda.empty_cache()

    print("序列特征向量生成完毕")
    
if __name__ == "__main__":
    main()