'''
GO数据处理第二步
根据上一步处理的结果进行建图
'''
import os
import json
import torch
import dgl

# 设置工作目录为项目文件夹
os.chdir("..")

unique_keys = {'id', 'name', 'namespace', 'def'}
# 在go-basic集合中只有5中关系类型：
edge_feature = {
    'is_a':                torch.tensor([1, 0, 0, 0]),
    'part_of':             torch.tensor([0, 1, 0, 0]),
    'regulates':           torch.tensor([0, 0, 1, 0]),
    'positively_regulates':torch.tensor([0, 0, 1, 0.5]),
    'negatively_regulates':torch.tensor([0, 0, 1,-0.5]),
}
terms = []
go_map = {}
graph = dgl.DGLGraph()

go_map_path = os.path.join('.','processed_data','go_map.txt')
go_terms_path = os.path.join('.','processed_data','go_terms.txt')

# 读取go_map信息，获取每个GO标签对应的节点编号
with open(go_map_path,"r") as file:
    go_map = json.load(file)

# 读取go_terms，建图
'''
relationship    edge
A is_a B        B->A
A part_of B     B->A
A regulates B   A->B
'''
with open(go_terms_path,"r") as file:
    terms = json.load(file)
    for term in terms:
        for key in term.keys():
            if key not in unique_keys:
                