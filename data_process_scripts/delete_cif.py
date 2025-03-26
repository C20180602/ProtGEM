'''
用于删除AlphaFold数据库下载的物种蛋白质数据中的CIF文件
'''
import os
import json

# 设置工作路径为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

with open("config/global.conf","r") as f:
    global_var = json.load(f)
species_name = global_var["species_name"]
data_path = os.path.join('raw_data',species_name,'pdb')
for file_path in os.listdir(data_path):
    if file_path.endswith(".cif.gz") or file_path.endswith(".cif"):
        os.remove(os.path.abspath(os.path.join(data_path,file_path)))