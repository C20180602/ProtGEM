'''
将-s的物种PPI迁移到-d物种
输出迁移融合后的PPI json文件
'''
import os
import json
import click
import subprocess
from tqdm import tqdm
from pathlib import Path
from data_export_scripts.export_fasta import export_fasta

def quick_blast_analysis(db_fasta: str, query_fasta: str, evalue: float = 0.001):
    """
    快速BLAST分析
    """
    # 创建输出目录
    output_dir = Path("blast_output")
    output_dir.mkdir(exist_ok=True)
    
    db_name = str(output_dir / "protein_db")
    
    # 创建数据库
    print("Building BLAST database...")
    subprocess.run([
        "makeblastdb", "-in", db_fasta, "-dbtype", "prot",
        "-out", db_name, "-parse_seqids"
    ], check=True)
    
    # 运行BLAST
    print("Running BLAST...")
    blast_output = output_dir / "results.tab"
    subprocess.run([
        "blastp", "-query", query_fasta, "-db", db_name,
        "-out", str(blast_output), "-evalue", str(evalue),
        "-outfmt", "6",
        "-num_threads", "4",
        "-max_target_seqs", "5"
    ], check=True)
    
    print(f"Finish! Result saved to: {blast_output}")
    
    # 读取结果
    DB2Q_dict = {}
    with open(blast_output, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            prot_Q = fields[0]
            prot_DB = fields[1]
            identity = float(fields[2])
            if prot_DB not in DB2Q_dict:
                DB2Q_dict[prot_DB] = []
            DB2Q_dict[prot_DB].append((prot_Q, identity))
            # print(f"Query: {fields[0]}, Target: {fields[1]}, "
            #       f"Identity: {fields[2]}%, E-value: {fields[10]}")
    # 清理数据库文件
    for ext in [".phr", ".pin", ".psq"]:
        db_file = output_dir / f"protein_db{ext}"
        if db_file.exists():
            os.remove(db_file)
    # 将结果dict返回
    return DB2Q_dict

@click.command()
@click.option('-s', '--species-name', type=click.STRING, required=True)
@click.option('-d', '--dest-species', type=click.STRING, required=True)
def main(species_name, dest_species):
    # 已有数据统计
    D_set = set()
    with open(Path("processed_data",dest_species+"_protein_info.json"),'r') as f:
        dest_prot_info = json.load(f)
    for pid in dest_prot_info.keys():
        D_set.add(pid)
    print(len(D_set))
    with open(Path("raw_data",species_name,"ppi",f"ppi_list.json"),'r') as f:
        S_ppi_list = json.load(f)
    with open(Path("raw_data",dest_species,"ppi",f"ppi_list.json"),'r') as f:
        D_ppi_list = json.load(f)
    D_ppi = {}
    for prot1, prot2 in D_ppi_list:
        if prot1 in D_set and prot2 in D_set:
            if prot1 not in D_ppi:
                D_ppi[prot1] = set()
            if prot2 not in D_ppi:
                D_ppi[prot2] = set()
            D_ppi[prot1].add(prot2)
            D_ppi[prot2].add(prot1)
    ppi_count = len(D_ppi_list)
    covered_prot_count = len(D_ppi)

    # blast比对
    export_fasta(species_name)
    export_fasta(dest_species)
    S2D_dict = quick_blast_analysis(Path("processed_data",f"{species_name}.fasta"), Path("processed_data",f"{dest_species}.fasta"))

    # 迁移数据
    for s1, s2 in tqdm(S_ppi_list, desc="Transfering PPI"):
        if s1 in S2D_dict and s2 in S2D_dict:
            for d1, id1 in S2D_dict[s1]:
                if d1 not in D_set:
                    continue
                for d2, id2 in S2D_dict[s2]:
                    if d2 not in D_set:
                        continue
                    # 在s物种已有的ppi上，与两端蛋白质相似度均高于50%才进行迁移
                    if id1 >= 50.0 and id2 >= 50.0:
                        if d1 not in D_ppi:
                            D_ppi[d1] = set()
                        if d2 not in D_ppi:
                            D_ppi[d2] = set()
                        D_ppi[d1].add(d2)
                        D_ppi[d2].add(d1)
    print("Before PPI Transfer:")
    print(f"ppi_count: {ppi_count}")
    print(f"covered_prot_count: {covered_prot_count}")
    # 迁移后数据统计
    D_ppi_list = []
    for d1, neighbors in D_ppi.items():
        for d2 in neighbors:
            D_ppi_list.append([d1, d2])
    ppi_count = len(D_ppi_list)
    covered_prot_count = len(D_ppi)
    print("After PPI Transfer:")
    print(f"ppi_count: {ppi_count}")
    print(f"covered_prot_count: {covered_prot_count}")

    with open(Path("raw_data",dest_species,"ppi",f"transferred_ppi_list.json"),'w') as f:
        json.dump(D_ppi_list, f, indent=4)

if __name__ == '__main__':
    main()