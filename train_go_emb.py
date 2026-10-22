import os
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
from model.go_dataset import GoDataset
from model.go_model import GoEncoder
from model.loss import GoContrastLoss, GoContrastLoss2
from model.protein_model import ProteinEncoder
from utils import init_logger

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score    

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logger = init_logger(os.path.join('logs','train_go.log'))

# 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
device = torch.device("cuda:0")

# 读取模型超参数
go_model_conf_path = os.path.join("config","GoEncoder.conf")
with open(go_model_conf_path,"r") as f:
    gm_param = json.load(f)
    
# 读取训练设置的参数
train_conf_path = os.path.join("config","train_go.conf")
with open(train_conf_path,"r") as f:
    train_param = json.load(f)
epoch_num = train_param["epoch_num"]
batch_size = train_param["batch_size"]
init_lr = train_param["init_lr"]
pos_k = train_param["pos_k"]
pos_max = train_param["pos_max"]
neg_k = train_param["neg_k"]
neg_max = train_param["neg_max"]
pos_rate = train_param["pos_rate"]
contrast_temp = train_param["contrast_temp"]

# 读取必要的数据
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)
go_graph = go_graph.to(device)

# 初始化模型
print("Use biobert embeddings to initialize GO model? ([y]/n)")
go_init_flag = input()
if go_init_flag != 'n':
    # 读取GO模型的预设embedding
    with open(os.path.join("dataset","go_embedding_biobert.pkl"),"rb") as f:
        go_embedding_biobert = pickle.load(f)
if go_init_flag != 'n':
    model_go = GoEncoder(**gm_param,manual_init=True,init_tensor=go_embedding_biobert)
else:
    model_go = GoEncoder(**gm_param)
model_go.to(device)

# 初始化数据集
ds = GoDataset(pos_k, pos_max, neg_k, neg_max, pos_rate)

train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# # 设置优化器、学习率规划器以及损失函数
optimizer_go = torch.optim.Adam(model_go.parameters(), lr=init_lr)
lr_scheduler_go = get_cosine_schedule_with_warmup(optimizer_go, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

criterion = GoContrastLoss(pos_k, neg_k) # cosine similarity
# criterion = GoContrastLoss2(pos_k, neg_k) # negative euclidean distance similarity

print("Start Training!")
save_path = os.path.join("model_params","go_pretrain_model.pth")
for epoch in range(epoch_num):
    mean_loss = 0.0
    for i, (idx, pos, neg) in enumerate(train_loader):
        x = torch.cat([idx,pos,neg],dim=1)
        x = x.to(device)
        
        x = model_go(x)
        loss = criterion(x, contrast_temp)
        
        optimizer_go.zero_grad()
        loss.backward()
        optimizer_go.step()
        lr_scheduler_go.step()
        mean_loss += loss.item()
        if i % 80 == 79:
            logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {mean_loss/80}')
            mean_loss = 0.0
    
    model_go.save_model(save_path)