'''
用于解压当文件夹下的所有压缩pdb文件 (xxx.pdb.gz)
需要操作系统支持gzip命令
'''
import os
import json

# 设置工作路径为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

# 切换路径，进入需要处理的文件夹
with open("config/global.conf","r") as f:
    global_var = json.load(f)
species_name = global_var["species_name"]
data_path = os.path.join('raw_data',species_name,'pdb')
os.chdir(data_path)
for file_path in os.listdir('.'):
    if file_path.endswith(".pdb.gz"):
        # gzip文件解压后，压缩包会消失
        os.system(f"gzip -d {file_path}")