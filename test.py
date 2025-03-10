import os
import json
import pickle

import torch
import torch.utils.data
import numpy as np
import torch.functional as F
import networkx as nx

from torch import nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn, probablity_propagate
from model.go_model import GoEncoder
from model.protein_model import ProteinEncoder, ContrastLoss

from utils import init_logger, cal_metrics 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logger = init_logger(os.path.join('logs','test_protein.log'))

# 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
device = torch.device("cuda:0")

# 读取模型超参数
protein_model_conf_path = os.path.join("config","ProteinEncoder.conf")
go_model_conf_path = os.path.join("config","GoEncoder.conf")
with open(protein_model_conf_path,"r") as f:
    pm_param = json.load(f)
with open(go_model_conf_path,"r") as f:
    gm_param = json.load(f)
    
# 读取测试设置的参数
test_conf_path = os.path.join("config","test.conf")
with open(test_conf_path,"r") as f:
    test_param = json.load(f)
batch_size = test_param["batch_size"]

# 读取必要的数据
with open(os.path.join("processed_data","go_terms.json"),"r") as f:
    go_terms = json.load(f)
ns_map = {"biological_process":"bp","cellular_component":"cc","molecular_function":"mf"}
ns_ids = {"bp":[],"cc":[],"mf":[]}
for i,go in enumerate(go_terms):
    ns_ids[ns_map[go["namespace"]]].append(i)
for ns in ns_ids:
    print(ns," cnt:",len(ns_ids[ns]))
actual_ns = {"bp":[],"cc":[],"mf":[]}
pred_ns = {"bp":[],"cc":[],"mf":[]}
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)
with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
    go_graph_nx = pickle.load(f)
topo_order = list(nx.topological_sort(go_graph_nx))
go_graph = go_graph.to(device)

# 初始化模型
print("Use model param in config directory? ([y]/n)")
config_flag = input()
if config_flag != 'n':
    model_protein = ProteinEncoder(**pm_param)
    model_go = GoEncoder(**gm_param)
else:
    model_protein = ProteinEncoder()
    model_go = GoEncoder()
model_protein.load_model("model_params/protein_model.pth")
model_protein.to(device)
model_go.load_model("model_params/go_model.pth")
model_go.to(device)

# 读取test数据
ds = MyDataset("HUMAN")
test_indices = np.load(os.path.join("dataset","test_indices.npy")).tolist()
test_dataset = Subset(ds, test_indices)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

ds.show_dataset_info()
summary(model_protein)
summary(model_go)

logger.info("Parameters INFO:")
logger.info(test_param)
logger.info(pm_param)
logger.info(gm_param)
logger.info("Start Testing!")
go_idx = torch.arange(gm_param["vocab_dim"]).to(device)
model_protein.eval()
model_go.eval()
ablation_list = [[],["seq"],["struct"],["domain"],["seq","struct"],["seq","domain"],["struct","domain"],["glob"],["part"]]
for ablation_feature in ablation_list:
    logger.info("ablation_feature: "+str(ablation_feature))
    total = 0
    correct = 0
    actual = []
    pred = []
    with torch.no_grad():
        for pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
            
            protein_emb = model_protein(seq_embs, batched_graph, domain_info, ablation_feat=ablation_feature)
            # protein_emb: [batch_size, hid_dim]
            go_emb = model_go(go_idx, go_graph)
            # go_emb: [go_nums, hid_dim]
            predict_matrix = protein_emb @ go_emb.T
            predict_matrix = torch.sigmoid(predict_matrix/0.2)
            # predict_matrix: [batch_size, go_nums]
            predict_matrix = predict_matrix.transpose(1,0)
            predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order)
            predict_matrix = predict_matrix.transpose(1,0)
            pred += predict_matrix.tolist()
            actual += labels.tolist()
            if ablation_feature == []:
                for ns in ns_ids:
                    ids = ns_ids[ns]
                    pred_ns[ns] += predict_matrix[:,ids].tolist()
                    actual_ns[ns] += labels[:,ids].tolist()
    cal_metrics(pred, actual)
    if ablation_feature == []:
        for ns in ns_ids:
            logger.info("Namespace: "+ns)
            cal_metrics(pred_ns[ns], actual_ns[ns])
