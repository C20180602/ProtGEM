import os
import json
import queue
import pickle
from pathlib import Path
from tqdm import tqdm

import networkx as nx
import torch
import dgl
import esm
import numpy as np
from torch.utils.data import Dataset

one2num = {
    'A' : 0,'R' : 1,'N' : 2,'D' : 3,'C' : 4,
    'Q' : 5,'E' : 6,'G' : 7,'H' : 8,'I' : 9,
    'L' : 10,'K' : 11,'M' : 12,'F' : 13,'P' : 14,
    'S' : 15,'T' : 16,'W' : 17,'Y' : 18,'V' : 19,
}

ns_map = {"biological_process":"bp","cellular_component":"cc","molecular_function":"mf"}

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

def bfs_get_dis_go(go_list, go_graph:nx.DiGraph):
    vis = set()
    dis = {}
    q = queue.Queue()
    all_nodes = go_graph.nodes()
    for go in all_nodes:
        dis[go] = -1
    for go in go_list:
        if go in all_nodes:
            q.put(go)
            dis[go]=0
            vis.add(go)
    while not q.empty():
        u = q.get()
        for edge in go_graph.in_edges(u):
            v = edge[1]
            if v not in vis:
                vis.add(v)
                dis[v] = dis[u] + 1
                q.put(v)
    return dis

def bfs_get_dis(start_pid, ppi_dict):
    vis = set()
    dis = {}
    q = queue.Queue()
    for pid in ppi_dict:
        dis[pid] = -1
    dis[start_pid] = 0
    vis.add(start_pid)
    q.put(start_pid)
    while not q.empty():
        u = q.get()
        for v in ppi_dict[u]:
            if v not in vis:
                vis.add(v)
                dis[v] = dis[u] + 1
                q.put(v)
    return dis

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
    prob_vector = prob_vector.transpose(1,0)
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
    return prob_vector.transpose(1,0)

def SimGO(go_set1, go_set2, go_terms, go_term_dis, go_diff_weight = {'bp':0.4, 'cc':0.4, 'mf':0.2}):
    # 防止除以0
    go_sim = 1e-8
    ns_max = {'bp':0, 'cc':0, 'mf':0}
    ns_max_cnt = {'bp':0, 'cc':0, 'mf':0}
    # 求两个GO集合交集
    # 将三大类的GO标签中，分别统计出GO交集中所在的最大标签深度，以及处于该深度的标签个数
    intersection = go_set1 & go_set2
    for go in intersection:
        ns = ns_map[go_terms[go]["namespace"]]
        go = str(go)
        if ns_max[ns] < go_term_dis[go]:
            ns_max[ns] = go_term_dis[go]
            ns_max_cnt[ns] = 1
        elif ns_max[ns] == go_term_dis[go]:
            ns_max_cnt[ns] += 1
    for ns in ns_max:
        go_sim += go_diff_weight[ns]*ns_max[ns]*np.sqrt(1.0*ns_max_cnt[ns])
    return go_sim

class UpstreamDataset(Dataset):
    def __init__(self, species_name, cache_dir=Path("dataset"), target_device=torch.device("cuda:0"),
                 inference_flag = False, exclude_pid_list = [], use_cache = False):
        # GO数据与interpro数据, 较小直接读入
        # seq_emb, struct较大, 采用懒加载
        
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        self.species_name = species_name
        self.cache_dir = cache_dir
        self.device = target_device
        self.train_flag = not inference_flag
        
        # 读取各类文件
        print("loading all kinds of files...")
        with open(Path("processed_data",species_name+"_protein_info.json"), 'r') as f:
            self.protein_info = json.load(f)
        with open(Path("processed_data","interpro_id.json"), 'r') as f:
            interpro_list = json.load(f)
            self.vocab_interpro = {interpro:i for i,interpro in enumerate(interpro_list)}
        
        # 初始化随机旋转矩阵
        print("initializing random rotation matrix...")
        self.mat_R_list = []
        for i in range(1000):
            self.mat_R_list.append(torch.tensor(random_rotation_matrix(), dtype=torch.float, device=self.device))
        
        # 是否使用缓存
        if self.has_cache() and use_cache:
            print("Using Cached Dataset!")
            self.load_dataset()
            return
        
        print("processing dataset...")
        species_path = Path("dataset",species_name)
        protein_list = list(self.protein_info.keys())
        self.data_info = []
        ex_set = set(exclude_pid_list)
        for pname in tqdm(protein_list):
            if pname in ex_set:
                continue
            seq_path = Path(species_path,pname,"seq_emb.pkl")
            # 因为某些愚蠢的原因，有些蛋白质的序列嵌入文件是损坏的
            with open(seq_path,"rb") as f:
                seq_emb = pickle.load(f)
            if len(seq_emb.shape) != 2:
                continue
            struct_path = os.path.join(species_path,pname,"struct_gvp.dgl")
            
            # 当一个数据在序列, 结构, GO标签都存在时, 将其加入数据集
            if Path.exists(seq_path) and os.path.exists(struct_path) and pname in self.protein_info:
                struct_graph,_ = dgl.load_graphs(struct_path)
                struct_graph = struct_graph[0]
                plen = len(self.protein_info[pname]["seq"])
                # pdb文件与序列长度相等
                if len(struct_graph.nodes()) == plen:
                    domain_list = []
                    for domain in self.protein_info[pname]["domain"]:
                        if domain[0] in self.vocab_interpro:
                            domain_list.append([self.vocab_interpro[domain[0]],domain[1],domain[2]])
                    
                    # 保存数据为: (蛋白质名, 序列长度, 序列嵌入路径, 结构图路径, interpro标签)
                    self.data_info.append((pname,plen, seq_path.resolve(), os.path.abspath(struct_path),domain_list))
        self.save_dataset()
        
    def __len__(self):
        return len(self.data_info)
    
    def structure_data_enhance(self, g: dgl.DGLGraph):
        # spin
        mat_R = self.mat_R_list[np.random.randint(0, 1000)]
        g.ndata['v'] = g.ndata['v'] @ mat_R
        return g

    def train_mode(self):
        self.train_flag = True
    
    def valid_mode(self):
        self.train_flag = False
        
    def get_pid_list(self):
        pid_list = [pinfo[0] for pinfo in self.data_info]
        return pid_list
    
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
        # if self.train_flag:
        #     struct_graph = self.structure_data_enhance(struct_graph)
        
        # domain
        domain_list = info[4]
        domain_id_list = torch.tensor([domain[0] for domain in domain_list], dtype=torch.long, device=self.device)
        
        return idx, pname,plen, seq_emb,struct_graph, domain_list, domain_id_list

    def save_dataset(self):
        dataset_path = Path(self.cache_dir,f"upstream_dataset.pkl")
        with open(dataset_path,"wb") as f:
            pickle.dump(self.data_info,f)
            
    def load_dataset(self):
        dataset_path = Path(self.cache_dir,f"upstream_dataset.pkl")
        with open(dataset_path,"rb") as f:
            self.data_info = pickle.load(f)
    
    def has_cache(self):
        dataset_path = Path(self.cache_dir,f"upstream_dataset.pkl")
        if Path.exists(dataset_path):
            return True
        return False
    
    def show_dataset_info(self, logger):
        len_max = 0
        len_min = 100000
        len_sum = 0
        domain_max = 0
        domain_min = 100000
        domain_sum = 0
        for info in self.data_info:
            len_max = max(len_max, info[1])
            len_min = min(len_min, info[1])
            len_sum += info[1]
            domain_max = max(domain_max, len(info[4]))
            domain_min = min(domain_min, len(info[4]))
            domain_sum += len(info[4])
        logger.info("Dataset Info:")
        logger.info("Species:{}".format(self.species_name))
        logger.info("Protein Num: %d"%(len(self.data_info)))
        logger.info("Protein Length: Max: %d, Min: %d, Avg: %.2f"%(len_max, len_min, len_sum/len(self.data_info)))
        logger.info("Domain Num: Max: %d, Min: %d, Avg: %.2f"%(domain_max, domain_min, domain_sum/len(self.data_info)))
        logger.info("----------------------------")

class MyDataset(Dataset):
    def __init__(self, species_name, ont, cache_dir=Path("dataset"), target_device=torch.device("cuda:0"),
                 go_max_percent=1.0, go_min_count=0, inference_flag = False, force_balance_go_list = None, exclude_pid_list = [], 
                 use_cache = False, light_mode = False):
        # GO数据与interpro数据, 较小直接读入
        # seq_emb, struct较大, 采用懒加载
        
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")

        # 读取输入参数
        self.species_name = species_name
        self.ont = ont
        self.cache_dir = cache_dir
        self.device = target_device
        self.train_flag = not inference_flag
        self.light_mode = light_mode

        # 硬编码缓存相对路径
        self.dataset_path = Path(self.cache_dir,f"{self.ont}_dataset.pkl")
        self.balanced_go_list_path = Path(self.cache_dir,f"{self.ont}_balanced_go_list.json")
        self.propagated_go_path = Path(self.cache_dir,f"{self.ont}_propagated_go.json")
        self.re_seq_path = Path(self.cache_dir,"re_seq.pkl")
        
        # 磁盘存储节省模式，需要在线ESM推理
        if light_mode:
            print("loading ESM model...")
            self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.converter = alphabet.get_batch_converter()
            self.esm_model.to(self.device)
            for p in self.esm_model.parameters():
                p.requires_grad = False
            self.esm_model.eval()
        
        # 读取各类文件
        print("loading all kinds of files...")
        with open(Path("processed_data",species_name+"_protein_info.json"), 'r') as f:
            self.protein_info = json.load(f)
        with open(os.path.join("processed_data",species_name+"_protein_coords.pkl"), 'rb') as f:
            self.protein_coords = pickle.load(f)
        with open(Path("processed_data","go_id.json"), 'r') as f:
            self.vocab_go = json.load(f)
        with open(Path("processed_data","go_terms.json"), 'r') as f:
            self.go_terms = json.load(f)
        self.go_count_map = {}
        with open(Path("processed_data","interpro_id.json"), 'r') as f:
            interpro_list = json.load(f)
            self.vocab_interpro = {interpro:i for i,interpro in enumerate(interpro_list)}
            
        with open(Path("dataset","go_graph_nx.pkl"),"rb") as f:
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
        
        print("Reading Reconstruct targets...")
        self.re_seq = {}
        for pid in self.protein_info:
            info = self.protein_info[pid]
            seq_tensor_list = []
            for aa in info['seq']:
                tmp = [0]*21
                tmp[one2num[aa]] = 1.0
                seq_tensor_list.append(tmp)
            self.re_seq[pid] = np.array(seq_tensor_list)

        print("processing dataset...")
        species_path = Path("dataset",species_name)
        protein_list = list(self.protein_info.keys())
        self.data_info = []
        self.propagated_go = {}
        ex_set = set(exclude_pid_list)
        for pname in tqdm(protein_list):
            if pname in ex_set:
                continue
            if not light_mode:
                seq_path = Path(species_path,pname,"seq_emb.pkl")
                # 因为某些愚蠢的原因，有些蛋白质的序列嵌入文件是损坏的
                with open(seq_path,"rb") as f:
                    seq_emb = pickle.load(f)
                if len(seq_emb.shape) != 2:
                    continue
            struct_path = os.path.join(species_path,pname,"struct_gvp.dgl")
            
            # 当一个数据在序列, 结构, GO标签都存在时, 将其加入数据集
            if  (light_mode or Path.exists(seq_path)) \
                and os.path.exists(struct_path) and pname in self.protein_info \
                and (self.protein_info[pname]["go"] != [] or inference_flag):
                
                seq = self.protein_info[pname]["seq"]
                struct_graph,_ = dgl.load_graphs(struct_path)
                struct_graph = struct_graph[0]
                plen = len(seq)
                # pdb文件与序列长度相等
                if len(struct_graph.nodes()) == plen:
                    # go数据统计
                    go_id_list = []
                    for go in self.protein_info[pname]["go"]:
                        if go in self.vocab_go:
                            go_id_list.append(self.vocab_go[go])
                    propagated_go_list = list(propagate(go_id_list,self.go_graph_nx))
                    self.propagated_go[pname] = []
                    for go in propagated_go_list:
                        if (self.ont == "all") or (ns_map[self.go_terms[go]["namespace"]] == self.ont):
                            self.propagated_go[pname].append(self.go_terms[go]['id'])
                        if go not in self.go_count_map:
                            self.go_count_map[go] = 0
                        self.go_count_map[go] += 1
                    
                    domain_list = []
                    for domain in self.protein_info[pname]["domain"]:
                        if domain[0] in self.vocab_interpro:
                            domain_list.append([self.vocab_interpro[domain[0]],domain[1],domain[2]])
                    
                    # 保存数据为: (蛋白质名, 序列长度, 序列嵌入路径, 结构图路径, interpro标签, 传递后的GO标签)
                    if propagated_go_list != [] or inference_flag:
                        if light_mode:
                            self.data_info.append((pname,plen,
                                            seq, os.path.abspath(struct_path),
                                            domain_list, propagated_go_list))
                        else:
                            self.data_info.append((pname,plen,
                                            seq_path.resolve(), os.path.abspath(struct_path),
                                            domain_list, propagated_go_list))
        # balance of posivite/negative samples
        if force_balance_go_list is None:
            self.balanced_go_list = []
            for go in self.go_count_map:
                if (self.ont == "all") or (ns_map[self.go_terms[go]["namespace"]] == self.ont):
                    if self.go_count_map[go] >= go_min_count and self.go_count_map[go] <= len(self.data_info)*go_max_percent:
                        self.balanced_go_list.append(go)
        else:
            self.balanced_go_list = force_balance_go_list
        self.balanced_go_list = sorted(self.balanced_go_list)
        self.balanced_go_map_inv = {go:i for i,go in enumerate(self.balanced_go_list)}
        
        # 重新筛选数据集
        new_data_info = []
        balanced_go_set = set(self.balanced_go_list)
        for info in self.data_info:
            go_list = info[5]
            if inference_flag:
                new_data_info.append(info)
                continue
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
        g.ndata['v'] = g.ndata['v'] @ mat_R
        return g

    def train_mode(self):
        self.train_flag = True
    
    def valid_mode(self):
        self.train_flag = False
        
    def get_pid_list(self):
        pid_list = [pinfo[0] for pinfo in self.data_info]
        return pid_list
    
    def __getitem__(self, idx):
        info = self.data_info[idx]
        pname = info[0]
        plen = info[1]
        
        # 数据集在访问时才读取大型文件
        # sequence
        if self.light_mode:
            labels, strs, tokens = self.converter([(pname,info[2])])
            tokens = tokens.to(self.device)
            seq_emb = self.esm_model(tokens, repr_layers=[33])['representations'][33]
            seq_emb = seq_emb.squeeze(0)
            seq_emb = seq_emb[1:-1]
        else:
            with open(info[2],"rb") as f:
                seq_emb = torch.tensor(pickle.load(f))
            seq_emb = seq_emb[1:-1]
            seq_emb = seq_emb.to(self.device)
        
        # structure
        struct_graph,_ = dgl.load_graphs(info[3])
        struct_graph = struct_graph[0]
        assert len(struct_graph.nodes()) == plen
        struct_graph = struct_graph.to(self.device)
        # if self.train_flag:
        #     struct_graph = self.structure_data_enhance(struct_graph)
        
        # domain
        domain_list = info[4]
        domain_id_list = torch.tensor([domain[0] for domain in domain_list], dtype=torch.long, device=self.device)
        
        # GO terms
        label = go_list_to_tensor(info[5],self.balanced_go_map_inv)
        label = label.to(self.device)
        
        return idx, pname,plen, seq_emb,struct_graph, domain_list, domain_id_list, label

    def generate_re_seq(self, pids, padded_len, device):
        re_seq_tgt = []
        for pid in pids:
            if padded_len - len(self.re_seq[pid]) > 0:
                pad_vec = [0.0]*20 + [1.0]
                pad_mat = np.array([pad_vec] * (padded_len - len(self.re_seq[pid])))
                padded_seq = np.vstack([self.re_seq[pid], pad_mat])
                re_seq_tgt.append(padded_seq)
            else:
                re_seq_tgt.append(self.re_seq[pid])
        return torch.tensor(np.array(re_seq_tgt), device=device)

    def generate_re_struct(self, pids, padded_len, device):
        re_struct_tgt = []
        for pid in pids:
            tmp_tensor = torch.tensor(self.protein_coords[pid], device=device)
            re_struct_tgt.append(tmp_tensor)
        return re_struct_tgt

    def save_dataset(self):
        with open(self.dataset_path,"wb") as f:
            pickle.dump(self.data_info,f)
        with open(self.balanced_go_list_path,"w") as f:
            json.dump(self.balanced_go_list,f,indent=4)
        with open(self.propagated_go_path,"w") as f:
            json.dump(self.propagated_go,f,indent=4)
        with open(self.re_seq_path,"wb") as f:
            pickle.dump(self.re_seq,f)
    
    def load_dataset(self):
        with open(self.dataset_path,"rb") as f:
            self.data_info = pickle.load(f)
        with open(self.balanced_go_list_path,"r") as f:
            self.balanced_go_list = json.load(f)
        with open(self.re_seq_path,"rb") as f:
            self.re_seq = pickle.load(f)
    
    def has_cache(self):
        if Path.exists(self.dataset_path) and Path.exists(self.balanced_go_list_path) and Path.exists(self.re_seq_path):
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
        logger.info("Species:{} ontology:{}".format(self.species_name,self.ont))
        logger.info("Protein Num: %d"%(len(self.data_info)))
        logger.info("Protein Length: Max: %d, Min: %d, Avg: %.2f"%(len_max, len_min, len_sum/len(self.data_info)))
        logger.info("Domain Num: Max: %d, Min: %d, Sum: %d, Avg: %.2f"%(domain_max, domain_min, domain_sum, domain_sum/len(self.data_info)))
        logger.info("GO Terms: Max: %d, Min: %d, Sum: %d, Avg: %.2f"%(go_max, go_min, go_sum, go_sum/len(self.data_info)))
        logger.info("Balanced GO List length: %d"%(len(self.balanced_go_list)))
        logger.info("----------------------------")
        
class PPIDataset(Dataset):
    def __init__(self, species_name, cache_dir=Path("dataset"), specified_ppi_file_name="ppi_list.json", target_device=torch.device("cuda:0"),
                 go_diff_weight = {'bp':0.4, 'cc':0.4, 'mf':0.2}, ppi_weight = 0.7, go_weight = 0.3, pick_neg_rate = 4.0, random_neg_rate = 1.0,
                 inference_flag = False, exclude_pid_list = [], ignore_go = False, use_cache = False):
        # GO数据与interpro数据, 较小直接读入
        # seq_emb, struct较大, 采用懒加载
        
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        self.species_name = species_name
        self.cache_dir = cache_dir
        self.device = target_device
        self.train_flag = not inference_flag
        
        # 读取各类文件
        print("loading all kinds of files...")
        with open(Path("processed_data",species_name+"_protein_info.json"), 'r') as f:
            self.protein_info = json.load(f)
        with open(Path("raw_data",species_name,"ppi",specified_ppi_file_name), 'r') as f:
            self.ppi_list = json.load(f)
        with open(Path("processed_data","interpro_id.json"), 'r') as f:
            interpro_list = json.load(f)
            self.vocab_interpro = {interpro:i for i,interpro in enumerate(interpro_list)}

        if not ignore_go:
            with open(Path("processed_data","go_id.json"), 'r') as f:
                self.vocab_go = json.load(f)
            with open(Path("processed_data","go_terms.json"), 'r') as f:
                self.go_terms = json.load(f)
            with open(Path("dataset","go_graph_nx.pkl"),"rb") as f:
                self.go_graph_nx = pickle.load(f)
            with open(Path("dataset","go_term_dis.json"), 'r') as f:
                self.go_term_dis = json.load(f)
            
        # 初始化随机旋转矩阵
        print("initializing random rotation matrix...")
        self.mat_R_list = []
        for i in range(1000):
            self.mat_R_list.append(torch.tensor(random_rotation_matrix(), dtype=torch.float, device=self.device))
        
        # 是否使用缓存
        if self.has_cache() and use_cache:
            print("Using Cached Dataset!")
            self.load_dataset()
            return
        
        print("processing dataset...")
        species_path = Path("dataset",species_name)
        protein_list = list(self.protein_info.keys())
        self.data_info = []
        ex_set = set(exclude_pid_list)
        for pname in tqdm(protein_list):
            if pname in ex_set:
                continue
            seq_path = Path(species_path,pname,"seq_emb.pkl")
            # 因为某些愚蠢的原因，有些蛋白质的序列嵌入文件是损坏的
            with open(seq_path,"rb") as f:
                seq_emb = pickle.load(f)
            if len(seq_emb.shape) != 2:
                continue
            struct_path = os.path.join(species_path,pname,"struct_gvp.dgl")
            
            # 当一个数据在序列, 结构都存在时, 将其加入数据集
            if Path.exists(seq_path) and os.path.exists(struct_path) and pname in self.protein_info:
                struct_graph,_ = dgl.load_graphs(struct_path)
                struct_graph = struct_graph[0]
                plen = len(self.protein_info[pname]["seq"])
                # pdb文件与序列长度相等
                if len(struct_graph.nodes()) == plen:
                    # go数据统计
                    if not ignore_go:
                        go_id_list = []
                        for go in self.protein_info[pname]["go"]:
                            if go in self.vocab_go:
                                go_id_list.append(self.vocab_go[go])
                        propagated_go_list = list(propagate(go_id_list,self.go_graph_nx))
                    else:
                        propagated_go_list = []
                    
                    domain_list = []
                    for domain in self.protein_info[pname]["domain"]:
                        if domain[0] in self.vocab_interpro:
                            domain_list.append([self.vocab_interpro[domain[0]],domain[1],domain[2]])
                    
                    # 保存数据为: (蛋白质名, 序列长度, 序列嵌入路径, 结构图路径, interpro标签, 传递后的GO标签)
                    self.data_info.append((pname,plen,
                                            seq_path.resolve(), os.path.abspath(struct_path),
                                            domain_list, propagated_go_list))
        
        pid_list = [d[0] for d in self.data_info]
        # 读取正样本
        pos_ppi_dict = {}
        for pid in pid_list:
            pos_ppi_dict[pid] = set()
        for cp in self.ppi_list:
            x,y = cp[0],cp[1]
            if x not in pos_ppi_dict:
                continue
            if y not in pos_ppi_dict:
                continue
            pos_ppi_dict[x].add(y)
            pos_ppi_dict[y].add(x)
        
        # 采样负样本
        neg_ppi_dict = {}
        pid2go_est = {}
        for pid in pid_list:
            neg_ppi_dict[pid] = set()
        for data in self.data_info:
            pid = data[0]
            go_list = data[5]
            pid2go_est[pid] = set(go_list)
        for u in tqdm(neg_ppi_dict, desc="negative sampling"):
            # 不存在正样本的蛋白质不进行负采样
            if u not in pos_ppi_dict or len(pos_ppi_dict[u])==0:
                continue
            ppi_dis_dict = bfs_get_dis(u, pos_ppi_dict)
            max_ppi_dis = max(max(ppi_dis_dict.values()),0)
            final_dis_list = []
            for v in pid_list:
                # 剔除已经在正样本集的蛋白质
                if v in pos_ppi_dict[u]:
                    continue
                # 不选择没有任何PPI信息的蛋白质作为负样本
                if v not in pos_ppi_dict or len(pos_ppi_dict[v])==0:
                    continue
                # 防止除以0
                go_sim = 1e-8
                if not ignore_go:
                    go_sim = SimGO(pid2go_est[u], pid2go_est[v], self.go_terms, self.go_term_dis, go_diff_weight)
                ppi_dis = ppi_dis_dict[v]
                if ppi_dis == -1:
                    ppi_dis = max_ppi_dis + 1
                if not ignore_go:
                    final_dis_list.append((-ppi_weight*(ppi_dis/(max_ppi_dis+1))-go_weight/go_sim, v))
                else:
                    final_dis_list.append((-ppi_dis, v))
                
            # 采取筛选的高可能性负样本:
            final_dis_list.sort()
            pick_neg_num = min(int(pick_neg_rate*len(pos_ppi_dict[u])),len(final_dis_list))
            for i in range(pick_neg_num):
                v = final_dis_list[i][1]
                if v not in pos_ppi_dict[u]:
                    neg_ppi_dict[u].add(v)
            # 采取随机负样本:
            shuffle_pid_list = pid_list.copy()
            np.random.shuffle(shuffle_pid_list)
            random_neg_num = min(int((pick_neg_rate+random_neg_rate) * len(pos_ppi_dict[u])) - pick_neg_num,
                                 len(pid_list)-1 - len(neg_ppi_dict[u]) - len(pos_ppi_dict[u]))
            if random_neg_num > 0:
                for v in shuffle_pid_list:
                    if v not in pos_ppi_dict or len(pos_ppi_dict[v])==0:
                        continue
                    if v!=u and (v not in neg_ppi_dict[u]) and (v not in pos_ppi_dict[u]):
                        if random_neg_num > 0:
                            random_neg_num -= 1
                            neg_ppi_dict[u].add(v)
                        else:
                            break
            
        # 转为序列中编号
        self.pid_id_map = {d[0]:i for i,d in enumerate(self.data_info)}
        self.pair_info = []
        for u in pid_list:
            for v in pos_ppi_dict[u]:
                self.pair_info.append((self.pid_id_map[u],self.pid_id_map[v],1))
            for v in neg_ppi_dict[u]:
                self.pair_info.append((self.pid_id_map[u],self.pid_id_map[v],0))
        
        self.save_dataset()
        
    def __len__(self):
        return len(self.pair_info)
    
    def structure_data_enhance(self, g: dgl.DGLGraph):
        # spin
        mat_R = self.mat_R_list[np.random.randint(0, 1000)]
        g.ndata['v'] = g.ndata['v'] @ mat_R
        return g

    def train_mode(self):
        self.train_flag = True
    
    def valid_mode(self):
        self.train_flag = False
        
    def get_pid_list(self):
        pid_list = [pinfo[0] for pinfo in self.data_info]
        return pid_list
    
    def get_protein_info(self, idx):
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
        # if self.train_flag:
        #     struct_graph = self.structure_data_enhance(struct_graph)
        
        # domain
        domain_list = info[4]
        domain_id_list = torch.tensor([domain[0] for domain in domain_list], dtype=torch.long, device=self.device)
        
        return (pname, plen, seq_emb, struct_graph, domain_list, domain_id_list)
    
    def __getitem__(self, idx):
        info = self.pair_info[idx]
        p1 = self.get_protein_info(info[0])
        p2 = self.get_protein_info(info[1])
        label = torch.tensor(info[2], dtype=torch.float,device=self.device)
        
        return idx, p1, p2, label

    def export_all_pairs(self, save_path=None):
        pair_list = []
        for info in self.pair_info:
            p1 = self.data_info[info[0]][0]
            p2 = self.data_info[info[1]][0]
            pair_list.append((p1,p2))
        return pair_list


    def export_all_posivite_pairs(self, save_path=None):
        pair_set_dict = {}
        for info in self.pair_info:
            if info[2] < 1.0:
                continue
            p1 = self.data_info[info[0]][0]
            p2 = self.data_info[info[1]][0]
            if p1 not in pair_set_dict:
                pair_set_dict[p1] = set()
            pair_set_dict[p1].add(p2)
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(pair_set_dict,f,indent=4)
        return pair_set_dict
    
    def save_dataset(self):
        data_info_path = Path(self.cache_dir,f"ppi_dataset_data.pkl")
        pair_info_path = Path(self.cache_dir,f"ppi_dataset_pair.pkl")
        with open(data_info_path,"wb") as f:
            pickle.dump(self.data_info,f)
        with open(pair_info_path,"wb") as f:
            pickle.dump(self.pair_info,f)
    
    def load_dataset(self):
        data_info_path = Path(self.cache_dir,f"ppi_dataset_data.pkl")
        pair_info_path = Path(self.cache_dir,f"ppi_dataset_pair.pkl")
        with open(data_info_path,"rb") as f:
            self.data_info = pickle.load(f)
        with open(pair_info_path,"rb") as f:
            self.pair_info = pickle.load(f)
    
    def has_cache(self):
        data_info_path = Path(self.cache_dir,f"ppi_dataset_data.pkl")
        pair_info_path = Path(self.cache_dir,f"ppi_dataset_pair.pkl")
        if Path.exists(data_info_path) and Path.exists(pair_info_path):
            return True
        return False
    
    def show_dataset_info(self, logger):
        len_max = 0
        len_min = 100000
        len_sum = 0
        domain_max = 0
        domain_min = 100000
        domain_sum = 0
        for info in self.data_info:
            len_max = max(len_max, info[1])
            len_min = min(len_min, info[1])
            len_sum += info[1]
            domain_max = max(domain_max, len(info[4]))
            domain_min = min(domain_min, len(info[4]))
            domain_sum += len(info[4])
        pos_num = 0
        neg_num = 0
        cover_pid_set = set()
        for info in self.pair_info:
            cover_pid_set.add(info[0])
            cover_pid_set.add(info[1])
            if info[2] >= 0.5:
                pos_num += 1
            else:
                neg_num += 1
        logger.info("----------Dataset Info----------")
        logger.info("Species:{}".format(self.species_name))
        logger.info("Protein Num: %d"%(len(self.data_info)))
        logger.info("Covered Protein Num: %d"%(len(cover_pid_set)))
        logger.info("Protein Length: Max: %d, Min: %d, Avg: %.2f"%(len_max, len_min, len_sum/len(self.data_info)))
        logger.info("Domain Num: Max: %d, Min: %d, Avg: %.2f"%(domain_max, domain_min, domain_sum/len(self.data_info)))
        logger.info("pos_pair: %d, neg_pair: %d"%(pos_num, neg_num))
        logger.info("----------Dataset Info End----------")

def upstream_collate_fn(batch):
    idxs, pnames, plens, seq_embs, struct_graphs, domain_lists, domain_id_lists = zip(*batch)
    
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

    return idxs, pnames, padded_seq_embs, batched_graph, [glob_dm_id, glob_dm_offset, plens, domain_lists]

def my_collate_fn(batch):
    idxs, pnames, plens, seq_embs, struct_graphs, domain_lists, domain_id_lists, labels = zip(*batch)
    
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

    return idxs, pnames, padded_seq_embs, batched_graph, [glob_dm_id, glob_dm_offset, plens, domain_lists], labels

def my_collate_fn_v2(batch):
    idxs, pnames, plens, seq_embs, struct_graphs, domain_lists, domain_id_lists, labels = zip(*batch)
    
    # 处理seq_emb，使用padding
    max_length = max([seq_emb.shape[0] for seq_emb in seq_embs])
    padded_seq_embs = []
    for seq_emb in seq_embs:
        padding = torch.zeros(max_length - seq_emb.shape[0], seq_emb.shape[1], device=seq_emb.device)
        padded_seq_embs.append(torch.cat([seq_emb, padding],dim=0))
    padded_seq_embs = torch.stack(padded_seq_embs)
    
    # 处理seq_emb，并入graph中
    for seq_emb, struct_graph in zip(seq_embs,struct_graphs):
        struct_graph.ndata["esm"] = seq_emb
        struct_graph = dgl.add_self_loop(struct_graph)
    
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

    return idxs, pnames, padded_seq_embs, batched_graph, [glob_dm_id, glob_dm_offset, plens, domain_lists], labels

def collate_each_pack(ps):
    pnames, plens, seq_embs, struct_graphs, domain_lists, domain_id_lists = zip(*ps)
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
    
    return pnames, padded_seq_embs, batched_graph, [glob_dm_id, glob_dm_offset, plens, domain_lists]

def ppi_collate_fn(batch):
    idxs, p1s, p2s, labels = zip(*batch)
    labels = torch.stack(labels)
    return idxs, collate_each_pack(p1s), collate_each_pack(p2s), labels