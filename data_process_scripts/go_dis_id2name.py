import os
import json
import torch
import dgl
import pickle
import warnings
import networkx as nx
from tqdm import tqdm

# 设置工作目录为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

with open("processed_data/go_id.json","r") as f:
    go_id = json.load(f)
with open("dataset/go_term_dis.json","r") as f:
    go_dis = json.load(f)

go_name_dis = {}
for i, name in enumerate(go_id):
    go_name_dis[name] = go_dis[str(go_id[name])]

with open("dataset/go_name_dis.json","w") as f:
    json.dump(go_name_dis,f,indent=4)