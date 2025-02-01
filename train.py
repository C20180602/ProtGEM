import os
import logging
import json
import pickle

import torch
import torch.utils.data
import numpy as np
import torch.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn
from model.go_model import GoEncoder
from model.protein_model import ProteinEncoder, ContrastLoss

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score    

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def init_logger(log_path = os.path.join('logs','train_protein.log')):
    logger = logging.getLogger("global")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=log_path)
    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

logger = init_logger(os.path.join('logs','train_protein.log'))

# 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
device = torch.device("cuda:0")

# 读取模型超参数
protein_model_conf_path = os.path.join("config","ProteinEncoder.conf")
go_model_conf_path = os.path.join("config","GoEncoder.conf")
with open(protein_model_conf_path,"r") as f:
    pm_param = json.load(f)
with open(go_model_conf_path,"r") as f:
    gm_param = json.load(f)
    
# 读取训练设置的参数
train_conf_path = os.path.join("config","train.conf")
with open(train_conf_path,"r") as f:
    train_param = json.load(f)
epoch_num = train_param["epoch_num"]
batch_size = train_param["batch_size"]
init_lr = train_param["init_lr"]
contrast_temp = train_param["contrast_temp"]

# 读取必要的数据
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)
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
model_protein.to(device)
model_go.load_model("model_params/go_model_1.pth")
# model_go.requires_grad_ = False
model_go.to(device)
model_go.eval()

# 初始化数据集并进行划分
ds = MyDataset("HUMAN")
train_size = int(0.7*len(ds))
valid_size = int(0.2*len(ds))
test_size = len(ds) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, valid_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

# 设置优化器、学习率规划器以及损失函数
optimizer_protein = torch.optim.Adam(model_protein.parameters(), lr=init_lr)
lr_scheduler_protein = get_cosine_schedule_with_warmup(optimizer_protein, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))
optimizer_go = torch.optim.Adam(model_go.parameters(), lr=init_lr)
lr_scheduler_go = get_cosine_schedule_with_warmup(optimizer_go, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000.0],device=device))
criterion1 = ContrastLoss()
criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1000.0],device=device))

logger.info("Parameters INFO:")
logger.info(train_param)
logger.info(pm_param)
logger.info(gm_param)
logger.info("Start Training!")
go_idx = torch.arange(gm_param["vocab_dim"]).to(device)
for epoch in range(epoch_num):
    mean_loss = 0.0
    for i, (pids, seq_embs, batched_graph, domain_info, labels) in enumerate(train_loader):
        seq_embs = seq_embs.to(device)
        batched_graph = batched_graph.to(device)
        domain_info[0] = domain_info[0].to(device)
        domain_info[1] = domain_info[1].to(device)
        labels = labels.to(device)
        
        optimizer_protein.zero_grad()
        optimizer_go.zero_grad()
        
        protein_emb = model_protein(seq_embs, batched_graph, domain_info)
        # protein_emb: [batch_size, hid_dim]
        go_emb = model_go(go_idx, go_graph)
        # go_emb: [go_nums, hid_dim]
        predict_matrix = protein_emb @ go_emb.T
        predict_matrix = torch.sigmoid(predict_matrix/0.2)
        # predict_matrix = (predict_matrix + 1)/2
        # print(predict_matrix)
        # print(labels)
        loss1 = criterion1(predict_matrix, labels, contrast_temp)
        loss2 = criterion2(predict_matrix, labels)
        loss = loss1 + loss2
        loss.backward()
        optimizer_protein.step()
        optimizer_go.step()
        lr_scheduler_protein.step()
        lr_scheduler_go.step()
        mean_loss += loss.item()
        if i % 400 == 399:
            logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {mean_loss/400}')
            mean_loss = 0.0
    model_protein.eval()
    model_go.eval()
    total = 0
    correct = 0
    actual = []
    pred = []
    with torch.no_grad():
        for pids, seq_embs, batched_graph, domain_info, labels in tqdm(val_loader):
            seq_embs = seq_embs.to(device)
            batched_graph = batched_graph.to(device)
            domain_info[0] = domain_info[0].to(device)
            domain_info[1] = domain_info[1].to(device)
            labels = labels.to(device)
            
            protein_emb = model_protein(seq_embs, batched_graph, domain_info)
            # protein_emb: [batch_size, hid_dim]
            go_emb = model_go(go_idx, go_graph)
            # go_emb: [go_nums, hid_dim]
            predict_matrix = protein_emb @ go_emb.T
            predict_matrix = (predict_matrix + 1)/2
            pred += predict_matrix.tolist()
            actual += labels.tolist()
    actual = np.array(actual).flatten()
    pred = np.array(pred).flatten()
    fpr, tpr, th = roc_curve(actual, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    precision, recall, pr_thresh = metrics.precision_recall_curve(actual, pred)
    aupr_score = metrics.auc(recall, precision)
    logger.info("AUC: "+str(auc_score))
    logger.info("AUPR: "+str(aupr_score))
    f1 = 2*precision*recall/(precision+recall+1e-8)
    thresh = pr_thresh[np.argmax(f1)]
    fmax = np.max(f1)
    logger.info("Fmax: "+str(fmax))
    logger.info("thresh: "+str(thresh))
        
    model_protein.save_model("model_params/protein_model.pth")
    model_protein.train()
    # model_go.train()
        