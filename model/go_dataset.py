import os
import json
import queue
import pickle
import numpy as np
import networkx as nx
import torch
import dgl
from torch.utils.data import Dataset
from tqdm import tqdm
from model.dataset import propagate

class GoDataset(Dataset):
    def __init__(self, pos_k, pos_max, neg_k, neg_max, pos_rate, use_cache = True):
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        assert pos_k <= pos_max
        assert neg_k <= neg_max
        
        self.pos_k = pos_k
        self.pos_max = pos_max
        self.neg_k = neg_k
        self.neg_max = neg_max
        
        with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
            self.go_graph_nx = pickle.load(f)
        with open(os.path.join("processed_data","go_id.json"), 'r') as f:
            self.vocab_go = json.load(f)
        n = self.go_graph_nx.number_of_nodes()
        
        if self.has_cache() and use_cache:
            print("Using Cached GO Dataset!")
            self.load_dataset()
            return
        
        # propagate swissprot GO terms
        if self.has_sp_cache():
            print("Using Cached Swissprot propagate data cache!")
            self.load_sp_data()
        else:
            print("propagate swissprot GO terms...")
            self.sp_gos = {}
            with open(os.path.join("raw_data","swissprot_go.tsv"), 'r') as f:
                line = f.readline()
                while line:
                    line = f.readline()
                    if line == "":
                        break
                    line = line.strip("\n").split("\t")
                    pname = line[0]
                    gos = []
                    for go in line[2].split("; "):
                        if go in self.vocab_go:
                            gos.append(self.vocab_go[go])
                    pro_gos = propagate(gos,self.go_graph_nx)
                    self.sp_gos[pname] = list(pro_gos)
            self.save_sp_data()
        
        print("building go2prot...")
        go2prot = {}
        for pname in self.sp_gos:
            for go in self.sp_gos[pname]:
                if go not in go2prot:
                    go2prot[go] = []
                go2prot[go].append(pname)
        
        print("sampling positive and negative GO terms...")
        self.data_info = [()]*n
        for go in tqdm(self.go_graph_nx.nodes()):
            vis = {}
            if go in go2prot:
                for pname in go2prot[go]:
                    for sub_go in self.sp_gos[pname]:
                        if sub_go not in vis:
                            vis[sub_go] = 0
                        vis[sub_go] += 1
            pos = []
            neg = []
            ansestor = propagate([go],self.go_graph_nx)
            u = go
            # positive samples
            for an_go in ansestor:
                if an_go != go and len(pos) < self.pos_max:
                    pos.append(an_go)
            
            # negative samples
            for edge in self.go_graph_nx.out_edges(u):
                fa = edge[1]
                sib = set()
                for edge in self.go_graph_nx.in_edges(fa):
                    v = edge[0]
                    sib.add(v)
                for v in sib:
                    if v not in vis and len(neg) < self.neg_max:
                        neg.append(v)
            
            for edge in self.go_graph_nx.in_edges(u):
                son = edge[0]
                if son in vis and vis[son]/len(go2prot[go]) > pos_rate and len(pos) < self.pos_max:
                    pos.append(son)
            while len(neg) < self.neg_max:
                v = np.random.randint(0,n)
                if v not in vis:
                    neg.append(v)
            self.data_info[u]=[pos,neg]
        self.save_dataset()
    
    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pos, neg = self.data_info[idx]
        pos = torch.tensor(np.random.choice(pos,self.pos_k))
        neg = torch.tensor(np.random.choice(neg,self.neg_k))
        return torch.tensor([idx]), pos, neg

    def save_dataset(self):
        save_path = os.path.join("dataset","go_cache.pkl")
        with open(save_path,"wb") as f:
            pickle.dump(self.data_info,f)
    
    def load_dataset(self):
        load_path = os.path.join("dataset","go_cache.pkl")
        with open(load_path,"rb") as f:
            self.data_info = pickle.load(f)
    
    def has_cache(self):
        cache_path = os.path.join("dataset","go_cache.pkl")
        if not os.path.exists(cache_path):
            return False
        return True
    
    # for swpissprot propagate data
    def save_sp_data(self):
        save_path = os.path.join("dataset","swissprot_go_cache.pkl")
        with open(save_path,"wb") as f:
            pickle.dump(self.sp_gos,f)
    
    def load_sp_data(self):
        load_path = os.path.join("dataset","swissprot_go_cache.pkl")
        with open(load_path,"rb") as f:
            self.sp_gos = pickle.load(f)
    
    def has_sp_cache(self):
        cache_path = os.path.join("dataset","swissprot_go_cache.pkl")
        if not os.path.exists(cache_path):
            return False
        return True
