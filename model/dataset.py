import os
import json
import queue
import pickle
import networkx as nx
import torch
import dgl
from torch.utils.data import Dataset
from tqdm import tqdm

# use BFS to propagate goterms O(n)
def propagate(go_list, go_graph:nx.DiGraph):
    vis={}
    q = queue.Queue()
    all_nodes = go_graph.nodes()
    for go in go_list:
        if go in all_nodes:
            q.put(go)
            vis[go]=1
    while not q.empty():
        u = q.get()
        for edge in go_graph.out_edges(u):
            v = edge[1]
            if v not in vis:
                vis[v]=1
                q.put(v)
    return vis.keys()

class MyDataset(Dataset):
    def __init__(self, species_name, use_cache = True):
        # GO数据与interpro数据, 较小直接读入
        # seq_emb, struct较大，采用懒加载
        
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        self.species_name = species_name
        
        # 是否使用缓存
        if self.has_cache() and use_cache:
           print("Using Cached Dataset!")
           self.load_dataset()
           return
           
        # 读取各类文件
        print("loading all kinds of files...")
        with open(os.path.join("processed_data",species_name+"_protein_info.json"), 'r') as f:
            self.protein_info = json.load(f)
        with open(os.path.join("processed_data","go_id.json"), 'r') as f:
            self.vocab_go = json.load(f)
        with open(os.path.join("processed_data","interpro_id.json"), 'r') as f:
            interpro_list = json.load(f)
            self.vocab_interpro = {interpro:i for i,interpro in enumerate(interpro_list)}
            
        with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
            go_graph_nx = pickle.load(f)
        
        print("processing dataset...")
        species_path = os.path.join("dataset",species_name)
        protein_list = os.listdir(species_path)
        self.data_info = []
        for pname in tqdm(protein_list):
            seq_path = os.path.join(species_path,pname,"seq_emb.pkl")
            struct_path = os.path.join(species_path,pname,"struct.dgl")
            
            # 当一个数据在序列, 结构, GO标签都存在时, 将其加入数据集
            if os.path.exists(seq_path) and os.path.exists(struct_path) and self.protein_info[pname]["go"] != []:
                plen = len(self.protein_info[pname]["seq"])
                # 保存数据为: (蛋白质名, 序列嵌入路径, 结构图路径, interpro标签, GO标签)
                go_id_list = []
                for go in self.protein_info[pname]["go"]:
                    if go in self.vocab_go:
                        go_id_list.append(self.vocab_go[go])
                propagated_go_list = list(propagate(go_id_list,go_graph_nx))
                
                # TODO?: domain mapping的过程可以提前到process_protein中，不过耗时很短，似乎并不必要
                domain_list = []
                for domain in self.protein_info[pname]["domain"]:
                    if domain[0] in self.vocab_interpro:
                        domain_list.append([self.vocab_interpro[domain[0]],domain[1],domain[2]])
                self.data_info.append((pname,plen,
                                       os.path.abspath(seq_path), os.path.abspath(struct_path),
                                       domain_list, propagated_go_list))
        
        self.save_dataset()
        
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        info = self.data_info[idx]
        pname = info[0]
        plen = info[1]
        # 数据集在访问时才读取大型文件
        with open(info[2],"rb") as f:
            seq_emb = torch.tensor(pickle.load(f))
        struct_graph,_ = dgl.load_graphs(info[3])
        struct_graph = struct_graph[0]
        
        domain = info[4]
        label = info[5]
        return pname,plen,seq_emb,struct_graph,domain, label

    def save_dataset(self):
        save_path = os.path.join("dataset",self.species_name+"_cache.pkl")
        with open(save_path,"wb") as f:
            pickle.dump(self.data_info,f)
    
    def load_dataset(self):
        load_path = os.path.join("dataset",self.species_name+"_cache.pkl")
        with open(load_path,"rb") as f:
            self.data_info = pickle.load(f)
    
    def has_cache(self):
        cache_path = os.path.join("dataset",self.species_name+"_cache.pkl")
        if os.path.exists(cache_path):
            return True
        return False
    
def my_collate_fn(batch):
    pnames, plens, seq_embs, struct_graphs, domain_list, labels= zip(*batch)
    
    # 处理seq_emb，使用padding
    max_length = max([seq_emb.shape[0] for seq_emb in seq_embs])
    padded_seq_embs = []
    for seq_emb in seq_embs:
        padding = torch.zeros(max_length - seq_emb.shape[0], seq_emb.shape[1])
        padded_seq_embs.append(torch.cat([seq_emb, padding],dim=0))
    
    padded_seq_embs = torch.stack(padded_seq_embs)
    
    # 结构图处理为batch_graph
    batched_graph = dgl.batch(struct_graphs)
    
    # domain处理
    glob_dm_id = []
    glob_dm_offset = []
    for dm in domain_list:
        glob_dm_offset.append(len(glob_dm_id))
        for domain in dm:
            glob_dm_id.append(domain[0])
    glob_dm_id = torch.tensor(glob_dm_id)
    glob_dm_offset = torch.tensor(glob_dm_offset)

    return pnames, padded_seq_embs, batched_graph, [glob_dm_id,glob_dm_offset,plens,domain_list], labels