import subprocess
import os
import click
from pathlib import Path

def quick_blast_analysis(db_fasta: str, query_fasta: str, evalue: float = 0.001):
    """
    快速BLAST分析
    """
    # 创建输出目录
    output_dir = Path("blast_output")
    output_dir.mkdir(exist_ok=True)
    
    db_name = str(output_dir / "protein_db")
    
    # 1. 创建数据库
    print("创建BLAST数据库...")
    subprocess.run([
        "makeblastdb", "-in", db_fasta, "-dbtype", "prot",
        "-out", db_name, "-parse_seqids"
    ], check=True)
    
    # 2. 运行BLAST
    print("运行BLAST比对...")
    blast_output = output_dir / "results.tab"
    subprocess.run([
        "blastp", "-query", query_fasta, "-db", db_name,
        "-out", str(blast_output), "-evalue", str(evalue),
        "-outfmt", "6",  # 表格格式
        "-num_threads", "4",
        "-max_target_seqs", "5"
    ], check=True)
    
    print(f"完成! 结果保存到: {blast_output}")
    
    # 3. 读取结果
    with open(blast_output, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            print(f"Query: {fields[0]}, Target: {fields[1]}, "
                  f"Identity: {fields[2]}%, E-value: {fields[10]}")

# 使用示例
quick_blast_analysis("database.fasta", "query.fasta")


@click.command()
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_ppi_v1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2']), default='v1.1')
@click.option('-thresh', '--thresh', type=click.FLOAT, default=None)
@click.option('-ds', '--dest-species', type=click.STRING)
def main(species_name, model_name, model_type, thresh, dest_species):
