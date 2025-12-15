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

go_id_path = os.path.join('processed_data','go_id.json')
go_terms_path = os.path.join('processed_data','go_terms.json')

go_graph_out_path = os.path.join("dataset","go_graph_dgl.pkl")
go_graph_nx_out_path = os.path.join("dataset","go_graph_nx.pkl")
go_term_dis_path = os.path.join("dataset","go_term_dis.json")

# 读取go_map信息，获取每个GO标签对应的节点编号
with open(go_id_path,"r") as f:
    go_id = json.load(f)

# 读取go_terms，建图
'''
go_graph building method
relationship    edge
u is_a v        v->u
u part_of v     v->u
u regulates v   u->v

在networkx中单独建图主要为了方便后续传播标签的过程
go_graph_nx building method
u is_a v        u->v
u part_of v     u->v
'''
# 忽略使用DGLGraph建立空图的警告
warnings.filterwarnings("ignore")
go_graph = dgl.DGLGraph()
go_graph_nx = nx.DiGraph()
with open(go_terms_path,"r") as file:
    terms = json.load(file)
go_graph.add_nodes(len(terms))
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
                    go_graph_nx.add_edge(u, v)
                else:
                    go_graph.add_edges(torch.tensor([u]), torch.tensor([v]), {'feat': edge_feature[key]})

print(go_graph)
print(go_graph_nx)
with open(go_graph_out_path,"wb") as f:
    pickle.dump(go_graph,f)
with open(go_graph_nx_out_path,"wb") as f:
    pickle.dump(go_graph_nx,f)

# DP求最长路作为GO标签的距离值，越大表示标签描述的功能越具体
# 这里会比较confusing，因为go_graph_nx的边是从具体标签指向抽象标签的，所以DP采用的是反向DP，三个根节点bp、cc、mf的dis值为1
def dfs(u,go_graph,dis):
    dis[u]=1
    for edge in go_graph.out_edges(u):
        v = edge[1]
        if v not in dis:
            dfs(v,go_graph,dis)
        dis[u]=max(dis[u],dis[v]+1)

def DAG_DP(go_graph:nx.DiGraph):
    dis={}
    for i in go_graph.nodes():
        if i not in dis:
            dfs(i,go_graph,dis)
    return dis

dis = DAG_DP(go_graph_nx)
with open(go_term_dis_path,"w") as f:
    json.dump(dis,f,indent=4)