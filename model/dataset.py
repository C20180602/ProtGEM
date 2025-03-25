import os
import json
import queue
import pickle
import networkx as nx
import torch
import dgl
import numpy as np
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

def go_list_to_tensor(go_list, balanced_go_map_inv):
    multihot = torch.zeros([len(balanced_go_map_inv)])
    for go in go_list:
        if go in balanced_go_map_inv:
            multihot[balanced_go_map_inv[go]] = 1.0
    return multihot

def random_rotation_matrix():
    # 随机生成四元数
    u1 = np.random.rand()
    u2 = np.random.rand()
    u3 = np.random.rand()

    # 根据四元数公式计算旋转矩阵
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    # 旋转矩阵
    rotation_matrix = np.array([
        [q1*q1 + q4*q4 - q2*q2 - q3*q3, 2*(q2*q1 + q3*q4), 2*(q3*q1 - q2*q4)],
        [2*(q2*q1 - q3*q4), q2*q2 + q4*q4 - q1*q1 - q3*q3, 2*(q3*q2 + q1*q4)],
        [2*(q3*q1 + q2*q4), 2*(q3*q2 - q1*q4), q3*q3 + q4*q4 - q1*q1 - q2*q2]
    ])

    return rotation_matrix

def probablity_propagate(prob_vector, go_graph:nx.DiGraph, topo_order, balanced_go_map_inv):
    # DP
    for u in topo_order:
        if u not in balanced_go_map_inv:
            continue
        b_u = balanced_go_map_inv[u]
        for v in go_graph.predecessors(u):
            if v not in balanced_go_map_inv:
                continue
            b_v = balanced_go_map_inv[v]
            prob_vector[b_u] = torch.max(prob_vector[b_u],prob_vector[b_v])
    return prob_vector

class MyDataset(Dataset):
    def __init__(self, species_name, target_device, go_max_percent=1.0, go_min_count=0, use_cache = False):
        # GO数据与interpro数据, 较小直接读入
        # seq_emb, struct较大，采用懒加载
        
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        self.species_name = species_name
        self.device = target_device
        self.train_flag = True
        
        # 读取各类文件
        print("loading all kinds of files...")
        with open(os.path.join("processed_data",species_name+"_protein_info.json"), 'r') as f:
            self.protein_info = json.load(f)
        with open(os.path.join("processed_data","go_id.json"), 'r') as f:
            self.vocab_go = json.load(f)
        self.go_count_map = {}
        with open(os.path.join("processed_data","interpro_id.json"), 'r') as f:
            interpro_list = json.load(f)
            self.vocab_interpro = {interpro:i for i,interpro in enumerate(interpro_list)}
            
        with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
            self.go_graph_nx = pickle.load(f)
        
        # 初始化随机旋转矩阵
        print("initializing random rotation matrix...")
        self.mat_R_list = []
        for i in range(1000):
            self.mat_R_list.append(torch.tensor(random_rotation_matrix(), dtype=torch.float, device=self.device))
        
        # 是否使用缓存
        if self.has_cache() and use_cache:
            print("Using Cached Dataset!")
            self.load_dataset()
            self.balanced_go_map_inv = {go:b_go for b_go,go in enumerate(self.balanced_go_list)}
            return
        
        print("processing dataset...")
        species_path = os.path.join("dataset",species_name)
        protein_list = os.listdir(species_path)
        self.data_info = []
        for pname in tqdm(protein_list):
            seq_path = os.path.join(species_path,pname,"seq_emb.pkl")
            struct_path = os.path.join(species_path,pname,"struct.dgl")
            
            # 当一个数据在序列, 结构, GO标签都存在时, 将其加入数据集
            if os.path.exists(seq_path) and os.path.exists(struct_path) and pname in self.protein_info and self.protein_info[pname]["go"] != []:
                struct_graph,_ = dgl.load_graphs(struct_path)
                struct_graph = struct_graph[0]
                plen = len(self.protein_info[pname]["seq"])
                # 只选择pdb文件与序列长度相等的进入数据集
                if len(struct_graph.nodes()) == plen:
                    # go数据统计
                    go_id_list = []
                    for go in self.protein_info[pname]["go"]:
                        if go in self.vocab_go:
                            go_id_list.append(self.vocab_go[go])
                    propagated_go_list = list(propagate(go_id_list,self.go_graph_nx))
                    for go in propagated_go_list:
                        if go not in self.go_count_map:
                            self.go_count_map[go] = 0
                        self.go_count_map[go] += 1
                    
                    domain_list = []
                    for domain in self.protein_info[pname]["domain"]:
                        if domain[0] in self.vocab_interpro:
                            domain_list.append([self.vocab_interpro[domain[0]],domain[1],domain[2]])
                    
                    # 保存数据为: (蛋白质名, 序列长度, 序列嵌入路径, 结构图路径, interpro标签, 传递后的GO标签)
                    if propagated_go_list != []:
                        self.data_info.append((pname,plen,
                                            os.path.abspath(seq_path), os.path.abspath(struct_path),
                                            domain_list, propagated_go_list))
        # balance of posivite/negative samples
        self.balanced_go_list = []
        for go in self.go_count_map:
            if self.go_count_map[go] >= go_min_count and self.go_count_map[go] <= len(self.data_info)*go_max_percent:
                self.balanced_go_list.append(go)
        self.balanced_go_list = sorted(self.balanced_go_list)
        self.balanced_go_map_inv = {go:i for i,go in enumerate(self.balanced_go_list)}
        
        # 重新筛选数据集
        new_data_info = []
        balanced_go_set = set(self.balanced_go_list)
        for info in self.data_info:
            go_list = info[5]
            for go in go_list:
                if go in balanced_go_set:
                    new_data_info.append(info)
                    break
        self.data_info = new_data_info
        self.save_dataset()
        
    def __len__(self):
        return len(self.data_info)
    
    def structure_data_enhance(self, g: dgl.DGLGraph):
        # spin
        mat_R = self.mat_R_list[np.random.randint(0, 1000)]
        g.ndata['pos'] = g.ndata['pos'] @ mat_R
        return g

    def train_mode(self):
        self.train_flag = True
    
    def valid_mode(self):
        self.train_flag = False
    
    def __getitem__(self, idx):
        info = self.data_info[idx]
        pname = info[0]
        plen = info[1]
        
        # 数据集在访问时才读取大型文件
        # sequence
        with open(info[2],"rb") as f:
            seq_emb = torch.tensor(pickle.load(f))
        seq_emb = seq_emb[1:-1]
        seq_emb = seq_emb.to(self.device)
        
        # structure
        struct_graph,_ = dgl.load_graphs(info[3])
        struct_graph = struct_graph[0]
        assert len(struct_graph.nodes()) == plen
        struct_graph = struct_graph.to(self.device)
        if self.train_flag:
            struct_graph = self.structure_data_enhance(struct_graph)
        
        # domain
        domain_list = info[4]
        domain_id_list = torch.tensor([domain[0] for domain in domain_list], dtype=torch.long, device=self.device)
        
        # GO terms
        label = go_list_to_tensor(info[5],self.balanced_go_map_inv)
        label = label.to(self.device)
        
        return pname,plen, seq_emb,struct_graph, domain_list, domain_id_list, label

    def save_dataset(self):
        save_path = os.path.join("dataset",self.species_name+"_cache.pkl")
        balanced_go_list_path = os.path.join("dataset",self.species_name+"_balanced_go_list_cache.json")
        with open(save_path,"wb") as f:
            pickle.dump(self.data_info,f)
        with open(balanced_go_list_path,"w") as f:
            json.dump(self.balanced_go_list,f,indent=4)
    
    def load_dataset(self):
        load_path = os.path.join("dataset",self.species_name+"_cache.pkl")
        balanced_go_list_path = os.path.join("dataset",self.species_name+"_balanced_go_list_cache.json")
        with open(load_path,"rb") as f:
            self.data_info = pickle.load(f)
        with open(balanced_go_list_path,"r") as f:
            self.balanced_go_list = json.load(f)
    
    def has_cache(self):
        cache_path = os.path.join("dataset",self.species_name+"_cache.pkl")
        balanced_go_list_path = os.path.join("dataset",self.species_name+"_balanced_go_list_cache.json")
        if os.path.exists(cache_path) and os.path.exists(balanced_go_list_path):
            return True
        return False
    
    def show_dataset_info(self, logger):
        len_max = 0
        len_min = 100000
        len_sum = 0
        domain_max = 0
        domain_min = 100000
        domain_sum = 0
        go_max = 0
        go_min = 100000
        go_sum = 0
        for info in self.data_info:
            len_max = max(len_max, info[1])
            len_min = min(len_min, info[1])
            len_sum += info[1]
            domain_max = max(domain_max, len(info[4]))
            domain_min = min(domain_min, len(info[4]))
            domain_sum += len(info[4])
            go_max = max(go_max, len(info[5]))
            go_min = min(go_min, len(info[5]))
            go_sum += len(info[5])
        logger.info("Dataset Info:")
        logger.info("Protein Num: %d"%(len(self.data_info)))
        logger.info("Protein Length: Max: %d, Min: %d, Avg: %.2f"%(len_max, len_min, len_sum/len(self.data_info)))
        logger.info("Domain Num: Max: %d, Min: %d, Avg: %.2f"%(domain_max, domain_min, domain_sum/len(self.data_info)))
        logger.info("GO Terms: Max: %d, Min: %d, Avg: %.2f"%(go_max, go_min, go_sum/len(self.data_info)))
    
def my_collate_fn(batch):
    pnames, plens, seq_embs, struct_graphs, domain_lists, domain_id_lists, labels = zip(*batch)
    
    # 处理seq_emb，使用padding
    max_length = max([seq_emb.shape[0] for seq_emb in seq_embs])
    padded_seq_embs = []
    for seq_emb in seq_embs:
        padding = torch.zeros(max_length - seq_emb.shape[0], seq_emb.shape[1], device=seq_emb.device)
        padded_seq_embs.append(torch.cat([seq_emb, padding],dim=0))
    
    padded_seq_embs = torch.stack(padded_seq_embs)
    
    # 结构图处理为batch_graph
    batched_graph = dgl.batch(struct_graphs)
    
    # domain处理
    glob_dm_id = torch.cat(domain_id_lists)
    glob_dm_offset = []
    offset = 0
    for domain_id_list in domain_id_lists:
        glob_dm_offset.append(offset)
        offset += domain_id_list.shape[0]
    glob_dm_offset = torch.tensor(glob_dm_offset, dtype=torch.long, device=glob_dm_id.device)
    
    # label处理
    labels = torch.stack(labels)

    return pnames, padded_seq_embs, batched_graph, [glob_dm_id, glob_dm_offset, plens, domain_lists], labels