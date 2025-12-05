'''
蛋白质数据处理第一步
将某物种的uniprot_id:{id:,seq:,domain:[(interpro_id,start,end)],GO:set(go_id)}处理为dict保存为.json文件
并统计信息
'''
import Bio.PDB.PDBParser
import os
import json
import click
import numpy as np
import pickle as pkl
from tqdm import tqdm

@click.command()
@click.option('-s', '--species-name', type=click.STRING)
@click.option('-ml', '--max-length', type=click.INT, default=2048, help="过滤掉序列长度大于该值的蛋白质")
def main(species_name, max_length):
    # 设置工作目录为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")

    protein_cnt = 0
    protein_info = {}
    protein_coords = {}

    os.chdir(os.path.join("raw_data",species_name))

    ''' sequence '''
    parser = Bio.PDB.PDBParser()
    three2one= {
        'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
        'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 
        'TYR':'Y', 'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 
        'MET':'M', 'ALA':'A', 'GLY':'G', 'PRO':'P', 'CYS':'C'
    }
    one2num = {
        'A' : 0,'R' : 1,'N' : 2,'D' : 3,'C' : 4,
        'Q' : 5,'E' : 6,'G' : 7,'H' : 8,'I' : 9,
        'L' : 10,'K' : 11,'M' : 12,'F' : 13,'P' : 14,
        'S' : 15,'T' : 16,'W' : 17,'Y' : 18,'V' : 19,
    }
    def get_seq_coords_info(pdbfilepath):
        struct = parser.get_structure(0,pdbfilepath)
        # 只取第一个模型
        model = struct.get_models().__next__()
        # i表示当前氨基酸的序号
        seq = ""
        coords = []
        for res in model.get_residues():
            acid = three2one[res.get_resname()]
            seq = seq + acid
            coords.append(res['CA'].get_coord())
        return seq, np.array(coords)

    seq_data_cnt = 0
    pdb_data_dir = os.path.join('pdb')
    for filename in tqdm(os.listdir(pdb_data_dir), desc="Read sequences from PDB files"):
        pdbfilepath = os.path.join(pdb_data_dir, filename)
        seq, coords = get_seq_coords_info(pdbfilepath)
        # 过滤过长的蛋白质
        if len(seq) > max_length:
            print("Warning: Length Limit Exceeded")
            continue
        # 读取完序列数据之后 初始化该蛋白质对应的dict
        if "AF-" in filename:
            pid = filename.split('-')[1]
        else:
            pid = filename.split('.')[0]
        id = protein_cnt
        protein_cnt += 1
        protein_info[pid] = {}
        protein_info[pid]["id"] = id
        protein_info[pid]["seq"] = seq
        protein_info[pid]["go"] = []
        protein_info[pid]["domain"] = []
        protein_coords[pid] = coords
        seq_data_cnt += 1
    print("protein_cnt: ",protein_cnt)
    print("seq_data_cnt: ",seq_data_cnt)


    ''' GO '''
    # 统计所有GO标注数量，重复标注只算一个
    go_data_cnt = 0
    # 统计没有任何GO标签的蛋白质数量
    zero_go_protein_cnt = 0
    # 优先使用已经处理好的GO json文件，格式为:{"Uniprot ID":["GO:xxxxxxx", "GO:xxxxxxx", ...], ...}
    if os.path.exists(os.path.join("go","pgo_dict.json")):
        with open(os.path.join("go","pgo_dict.json"), "r") as f:
            pgo_dict = json.load(f)
        for pid, go_ids in pgo_dict.items():
            for go_id in go_ids:
                if pid in protein_info:
                    protein_info[pid]["go"].append(go_id)
    else:
        with open(os.path.join("go","go.tsv"),'r') as f:
            line = f.readline()
            line = f.readline()
            while line:
                pid = line.split('\t')[0]
                if pid not in protein_info:
                    line = f.readline()
                    continue
                for go in line.split('\t')[2].split("; "):
                    protein_info[pid]["go"].append(go.strip())
                line = f.readline()

    for p in protein_info.values():
        if len(p["go"]) == 0:
            zero_go_protein_cnt += 1
        else:
            go_data_cnt += len(p["go"])
    print("go_data_cnt: ",go_data_cnt)
    print("zero_go_protein_cnt: ",zero_go_protein_cnt)

    ''' domain '''
    # 统计所有interpro标注数量
    interpro_data_cnt = {}
    interpro_data_sum = 0
    # 统计没有任何interpro标签的蛋白质数量
    zero_interpro_protein_cnt = 0
    for file in os.listdir("domain"):
        pid = file.split('.')[0]
        # 跳过没有序列数据的蛋白质
        if pid not in protein_info.keys():
            continue
        with open(os.path.join("domain", file), 'r') as f:
            line = f.readline()
            while line:
                interpro_id = line.split("\t")[0]
                range_list = line.split("\t")[-2].split(",")
                for range in range_list:
                    start = int(range.split("..")[0])
                    end = int(range.split("..")[-1])
                    protein_info[pid]["domain"].append((interpro_id,start,end))
                
                # 统计该物种中各个类型的interpro标签的数据量
                data_type = line.split("\t")[3]
                if data_type not in interpro_data_cnt.keys():
                    interpro_data_cnt[data_type] = 0
                interpro_data_cnt[data_type] += 1
                
                line = f.readline()
    for p in protein_info.values():
        if len(p["domain"]) == 0:
            zero_interpro_protein_cnt += 1
    for cnt in interpro_data_cnt.values():
        interpro_data_sum += cnt
    print("interpro_data_sum: ", interpro_data_sum)
    print(interpro_data_cnt) 
    print("zero_interpro_protein_cnt: ",zero_interpro_protein_cnt)

    # 回到工程文件目录
    os.chdir(os.path.join("..",".."))
    with open(os.path.join("processed_data",species_name+"_protein_info.json"), 'w') as f:
        json.dump(protein_info, f, indent=4)
    with open(os.path.join("processed_data",species_name+"_protein_coords.pkl"), 'wb') as f:
        pkl.dump(protein_coords, f)

if __name__ == "__main__":
    main()