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
    CACHE_VERSION = 2

    def __init__(self, pos_k, pos_max, neg_k, neg_max, pos_rate, use_cache = True):
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        assert 0 < pos_k <= pos_max
        assert 0 < neg_k <= neg_max
        
        self.pos_k = pos_k
        self.pos_max = pos_max
        self.neg_k = neg_k
        self.neg_max = neg_max
        self.pos_rate = pos_rate
        
        with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
            self.go_graph_nx = pickle.load(f)
        with open(os.path.join("processed_data","go_id.json"), 'r') as f:
            self.vocab_go = json.load(f)
        n = self.go_graph_nx.number_of_nodes()
        
        if self.has_cache() and use_cache:
            if self.load_dataset():
                print("Using Cached GO Dataset!")
                return
            print("GO sample cache is outdated; rebuilding positive and negative pools...")
        
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
                    go2prot[go] = set()
                go2prot[go].add(pname)
        
        print("sampling positive and negative GO terms...")
        self.data_info = [()]*n
        for go in tqdm(self.go_graph_nx.nodes()):
            self.data_info[go] = self.build_sample_pools(go, go2prot)
        self.save_dataset()

    def build_sample_pools(self, go, go2prot):
        # 先构建完整正样本池；祖先即使未进入截断后的正样本池，也不能作为负样本。
        ancestors = list(propagate([go], self.go_graph_nx))
        pos = [v for v in ancestors if v != go][:self.pos_max]
        positive_set = set(pos)
        proteins = go2prot.get(go, set())
        if proteins:
            for son in self.go_graph_nx.predecessors(go):
                if len(pos) >= self.pos_max:
                    break
                cooccurrence = len(proteins.intersection(go2prot.get(son, ())))
                if son not in positive_set and cooccurrence / len(proteins) > self.pos_rate:
                    pos.append(son)
                    positive_set.add(son)
        if not pos:
            raise ValueError(f"GO term {go} has no positive candidates")

        excluded = set(ancestors) | positive_set | {go}
        neg = []
        negative_set = set()

        def valid_negative(v):
            # 保留原有的共现排除规则，同时排除自身、所有祖先、正样本及重复候选。
            return (v not in excluded and v not in negative_set
                    and proteins.isdisjoint(go2prot.get(v, ())))

        # 优先选取合法的兄弟节点作为负样本。
        for parent in self.go_graph_nx.successors(go):
            for sibling in self.go_graph_nx.predecessors(parent):
                if len(neg) >= self.neg_max:
                    break
                if valid_negative(sibling):
                    neg.append(sibling)
                    negative_set.add(sibling)
            if len(neg) >= self.neg_max:
                break

        # 随机补采；设置上限，合法候选不足时不无限重试。
        n = self.go_graph_nx.number_of_nodes()
        for _ in range(max(100, 10 * self.neg_max)):
            if len(neg) >= self.neg_max:
                break
            v = int(np.random.randint(0, n))
            if valid_negative(v):
                neg.append(v)
                negative_set.add(v)
        if len(neg) < self.neg_max:
            remaining = [v for v in self.go_graph_nx.nodes() if valid_negative(v)]
            count = min(self.neg_max - len(neg), len(remaining))
            if count:
                neg.extend(np.random.choice(remaining, count, replace=False).tolist())
        if not neg:
            raise ValueError(f"GO term {go} has no valid negative candidates")
        return [pos, neg]
    
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
            pickle.dump({
                "version": self.CACHE_VERSION,
                "sampling_params": (self.pos_max, self.neg_max, self.pos_rate),
                "data_info": self.data_info,
            }, f)
    
    def load_dataset(self):
        load_path = os.path.join("dataset","go_cache.pkl")
        with open(load_path,"rb") as f:
            cache = pickle.load(f)
        if (not isinstance(cache, dict)
                or cache.get("version") != self.CACHE_VERSION
                or cache.get("sampling_params") != (self.pos_max, self.neg_max, self.pos_rate)
                or len(cache.get("data_info", ())) != self.go_graph_nx.number_of_nodes()):
            return False
        self.data_info = cache["data_info"]
        return True
    
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
