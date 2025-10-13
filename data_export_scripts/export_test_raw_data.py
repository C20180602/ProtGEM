import os
import sys
import json

import torch
import torch.utils.data
import numpy as np
import torch.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import MyDataset, my_collate_fn

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')

# 读取全局变量
with open("config/global.conf","r") as f:
    global_var = json.load(f)
species_name = global_var["species_name"]

# 读取test数据
ds = MyDataset(species_name)
test_indices = np.load(os.path.join("dataset","{}_test_indices.npy".format(species_name))).tolist()
test_dataset = Subset(ds, test_indices)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)
with open(os.path.join("processed_data",species_name+"_protein_info.json"), 'r') as f:
    protein_info = json.load(f)

test_protein_info = {}
for pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
    pid = pids[0]
    test_protein_info[pid] = {
        "seq": protein_info[pid]["seq"],
        "struct": os.path.abspath(os.path.join("raw_data",species_name,"pdb","AF-"+pid+"-F1-model_v4.pdb")),
        "domain": protein_info[pid]["domain"],
        "go": protein_info[pid]["go"]
    }

with open(os.path.join("processed_data",species_name+"_test_protein_info.json"), 'w') as f:
    json.dump(test_protein_info, f, indent=4)

