'''
在更新数据集时使用(也可以不运行)
清扫过往的下载数据和建图数据, 避免空间占用
'''
import os
import json
import click
from pathlib import Path

# 清除掉不在pid_list.json里面的蛋白质结构文件
def clean_pdb(species_name):
    pid_list_file = Path("raw_data", species_name, "pid_list.json")
    with open(pid_list_file, 'r') as f:
        pid_list = set(json.load(f))

    pdb_dir = Path("raw_data", species_name, "pdb")
    for pdb_file in pdb_dir.glob("*.pdb"):
        pdb_id = pdb_file.stem
        if pdb_id not in pid_list:
            os.remove(pdb_file)
            print(f"Deleted: {pdb_file}")
    print("PDB file cleanup completed.")

# 清除掉不在pid_list.json里面的蛋白质结构域.tsv文件
def clean_domain(species_name):
    pid_list_file = Path("raw_data", species_name, "pid_list.json")
    with open(pid_list_file, 'r') as f:
        pid_list = set(json.load(f))

    domain_dir = Path("raw_data", species_name, "domain")
    for domain_file in domain_dir.glob("*.tsv"):
        domain_id = domain_file.stem
        if domain_id not in pid_list:
            os.remove(domain_file)
            print(f"Deleted: {domain_file}")
    print("Domain file cleanup completed.")

# 清除掉不在pid_list.json里面的dataset中的蛋白质文件夹
def clean_dataset(species_name):
    pid_list_file = Path("raw_data", species_name, "pid_list.json")
    with open(pid_list_file, 'r') as f:
        pid_list = set(json.load(f))

    dataset_dir = Path("dataset", species_name)
    for prot_dir in dataset_dir.iterdir():
        if prot_dir.is_dir():
            prot_id = prot_dir.name
            if prot_id not in pid_list:
                os.rmdir(prot_dir)
                print(f"Deleted dataset folder: {prot_dir}")
    print("Dataset file cleanup completed.")

@click.command()
@click.option('-s', '--species-name', type=click.STRING, required=True)
def main(species_name):
    clean_pdb(species_name)
    clean_domain(species_name)
    clean_dataset(species_name)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')
    main()