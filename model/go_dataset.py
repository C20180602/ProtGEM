import os
import json
import queue
import pickle
import networkx as nx
import torch
import dgl
from torch.utils.data import Dataset
from tqdm import tqdm
from dataset import propagate

class GoDataset(Dataset):
    def __init__(self, use_cache = True):
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        
        with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
            self.go_graph_nx = pickle.load(f)
        with open(os.path.join("processed_data","go_id.json"), 'r') as f:
            self.vocab_go = json.load(f)
        