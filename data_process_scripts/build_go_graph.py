'''
GO数据处理第三步
根据第一步处理的结果与第二步获得的BioBert嵌入作为GO节点特征进行建图
'''
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

unique_keys = {'id', 'name', 'namespace', 'def'}
# 在go-basic集合中只有5中关系类型：
edge_feature = {
    'is_a':                torch.tensor([[1., 0., 0., 0.]]),
    'part_of':             torch.tensor([[0., 1., 0., 0.]]),
    'regulates':           torch.tensor([[0., 0., 1., 0.]]),
    'positively_regulates':torch.tensor([[0., 0., 1., 0.5]]),
    'negatively_regulates':torch.tensor([[0., 0., 1.,-0.5]]),
}
terms = []
go_id = {}
graph = dgl.DGLGraph()

go_id_path = os.path.join('processed_data','go_id.json')
go_terms_path = os.path.join('processed_data','go_terms.json')
go_biobert_emb_path = os.path.join("dataset","GO")

go_graph_out_path = os.path.join("dataset","go_graph_dgl.pkl")
go_graph_nx_out_path = os.path.join("dataset","go_graph_nx.pkl")

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
    
go_graph.add_nodes(len(terms))
# 这里的1024来自于biobert-large的输出向量维数
go_graph.ndata['feat'] = torch.randn(len(terms),1024)
print("reading biobert embedding...")
for term in tqdm(terms):
    term_id = term["id"]
    u = go_id[term_id]
    # 为节点添加BioBERT特征（是否锁定？）（如果不使用biobert可以注释掉以下两行）
    with open(os.path.join(go_biobert_emb_path,term_id+'.pkl'),'rb') as f_go:
        go_graph.ndata["feat"][u] = pickle.load(f_go)

print("building edges...")
for term in tqdm(terms):
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
print(go_graph_nx)
with open(go_graph_out_path,"wb") as f:
    pickle.dump(go_graph,f)
with open(go_graph_nx_out_path,"wb") as f:
    pickle.dump(go_graph_nx,f)
    