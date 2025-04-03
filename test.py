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
from model.framework import ContrastLoss, ProteinGoModel

from utils import init_logger, cal_metrics 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logger = init_logger(os.path.join('logs','test_protein.log'))

# 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
device = torch.device("cuda:0")

# 读取全局变量
with open("config/global.conf","r") as f:
    global_var = json.load(f)
species_name = global_var["species_name"]

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

# 读取train中的数据集设置
train_conf_path = os.path.join("config","train.conf")
with open(train_conf_path,"r") as f:
    train_param = json.load(f)
go_max_percent = train_param["go_max_percent"]
go_min_count = train_param["go_min_count"]

# 读取必要的数据
with open(os.path.join("processed_data","go_terms.json"),"r") as f:
    go_terms = json.load(f)
ns_map = {"biological_process":"bp","cellular_component":"cc","molecular_function":"mf"}
ns_ids = {"bp":[],"cc":[],"mf":[]}
for i,go in enumerate(go_terms):
    ns_ids[ns_map[go["namespace"]]].append(i)
actual_ns = {"bp":[],"cc":[],"mf":[]}
pred_ns = {"bp":[],"cc":[],"mf":[]}
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)
with open(os.path.join("dataset","go_graph_nx.pkl"),"rb") as f:
    go_graph_nx = pickle.load(f)
topo_order = list(nx.topological_sort(go_graph_nx))
go_graph = go_graph.to(device)

# 初始化模型
for ont in ["bp","cc","mf"]:
    pg_model = ProteinGoModel(pm_param, gm_param)
    pg_model.load_model("model_params/pg_model_{}.pth".format(ont))
    pg_model.to(device)

    ds = MyDataset(species_name, device, go_max_percent, go_min_count, use_cache=True)
    test_indices = np.load(os.path.join("dataset","{}_{}_test_indices.npy".format(species_name,ont))).tolist()
    for idx in test_indices:
        if idx > len(ds):
            print(idx,len(ds))
            print("Error: test_indices out of range!")
            exit(1)
    test_dataset = Subset(ds, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    go_idx = torch.tensor(ds.balanced_go_list).to(device)
    balanced_go_map_inv = ds.balanced_go_map_inv
    for ns in ns_ids:
        new_ids = []
        for go in ns_ids[ns]:
            if go in balanced_go_map_inv:
                new_ids.append(balanced_go_map_inv[go])
        ns_ids[ns] = new_ids

    ds.show_dataset_info(logger)
    summary(pg_model)

    logger.info("Parameters INFO:")
    logger.info(test_param)
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info(f"bp_cnt:{len(ns_ids['bp'])}")
    logger.info(f"cc_cnt:{len(ns_ids['cc'])}")
    logger.info(f"mf_cnt:{len(ns_ids['mf'])}")
    logger.info("Start Testing!")

    pg_model.eval()
    ds.valid_mode()
    ablation_list = [[],["seq"],["struct"],["domain"],["seq","struct"],["seq","domain"],["struct","domain"],["glob"],["part"]]
    for ablation_feature in ablation_list:
        logger.info("ablation_feature: "+str(ablation_feature))
        total = 0
        correct = 0
        actual = []
        pred = []
        with torch.no_grad():
            for pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
                
                predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=ablation_feature)
                # predict_matrix: [batch_size, go_nums]
                predict_matrix = predict_matrix.transpose(1,0)
                predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, ds.balanced_go_map_inv)
                predict_matrix = predict_matrix.transpose(1,0)
                pred += predict_matrix.tolist()
                actual += labels.tolist()
                if ablation_feature == []:
                    for ns in ns_ids:
                        ids = ns_ids[ns]
                        pred_ns[ns] += predict_matrix[:,ids].tolist()
                        actual_ns[ns] += labels[:,ids].tolist()
        cal_metrics(pred, actual, logger)
