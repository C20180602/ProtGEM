'''
将processed_data/{species_name}_protein_info.json导出为fasta格式
'''
import json
import os
import click
from pathlib import Path

@click.command()
@click.option('-s', '--species-name', type=click.STRING, required=True, help='物种名称')
def export_fasta(species_name):
    # 定义文件路径
    input_file = Path(f"processed_data/{species_name}_protein_info.json")
    output_file = Path(f"processed_data/{species_name}.fasta")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        protein_info = json.load(f)
    
    # 写入FASTA文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for pid,info in protein_info.items():
            protein_id = pid
            sequence = info["seq"]
            f.write(f">{protein_id}\n")
            f.write(sequence+'\n')

    print(f"FASTA file exported to {output_file}")

if __name__ == '__main__':
    export_fasta()