'''
GO数据处理第二步
根据上一步处理的结果进行建图
'''
import os
import json
import torch
import dgl
import warnings
import networkx as nx

# 设置工作目录为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

unique_keys = {'id', 'name', 'namespace', 'def'}
# 在go-basic集合中只有5中关系类型：
edge_feature = {
    'is_a':                torch.tensor([1., 0., 0., 0.]),
    'part_of':             torch.tensor([0., 1., 0., 0.]),
    'regulates':           torch.tensor([0., 0., 1., 0.]),
    'positively_regulates':torch.tensor([0., 0., 1., 0.5]),
    'negatively_regulates':torch.tensor([0., 0., 1.,-0.5]),
}
terms = []
go_id = {}
graph = dgl.DGLGraph()

go_id_path = os.path.join('.','processed_data','go_id.json')
go_terms_path = os.path.join('.','processed_data','go_terms.json')

# 读取go_map信息，获取每个GO标签对应的节点编号
with open(go_id_path,"r") as f:
    go_id = json.load(f)

# 读取go_terms，建图
'''
relationship    edge
u is_a v        v->u
u part_of v     v->u
u regulates v   u->v

在networkx中单独建图主要为了方便后续传播标签的过程
'''
# 忽略使用DGLGraph建立空图的警告
warnings.filterwarnings("ignore")
go_graph = dgl.DGLGraph()
go_graph_nx = nx.DiGraph()
with open(go_terms_path,"r") as file:
    terms = json.load(file)
    for term in terms:
        term_id = term["id"]
        go_graph.add_nodes
        # 为节点添加BioBERT特征（可选？还是锁定？）
    for term in terms:
        term_id1 = term["id"]
        for key in term.keys():
            if key in edge_feature:
                for term_id2 in term[key]:
                    u = go_id[term_id1]
                    v = go_id[term_id2]
                    if key == "is_a" or key == "part_of":
                        go_graph.add_edges(torch.tensor([v]), torch.tensor([u]), {'feat': edge_feature[key]})
                        go_graph_nx.add_edge(term_id1, term_id2)
                    else:
                        go_graph.add_edges(torch.tensor([u]), torch.tensor([v]), {'feat': edge_feature[key]})
    print(go_graph)
    