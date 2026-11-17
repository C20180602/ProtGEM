'''
假设先从一个TSV文件中读取数据，然后将其中的第一列数据保存为JSON格式的文件：
'''
import os
import pandas as pd
import json
import click

@click.command()
@click.option('-f', '--file-path', type=click.STRING)
@click.option('-s', '--species-name', type=click.STRING)# 输出到目标物种的raw_data文件夹
def main(file_path, species_name):
    # 设置工作路径为项目文件夹
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    df = pd.read_csv(file_path, sep="\t", header=None)
    first_column = df.iloc[:, 0].tolist()
    os.makedirs(os.path.join("raw_data", species_name), exist_ok=True)
    with open(os.path.join("raw_data", species_name, "pid_list.json"), "w") as f:
        json.dump(first_column, f, indent=4)

if __name__ == "__main__":
    main()