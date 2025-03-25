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
from model.dataset import MyDataset, my_collate_fn
from model.go_model import GoEncoder
from model.protein_model import ProteinEncoder, ContrastLoss
from utils import init_logger, cal_metrics 

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def generate_mask(batch_size, seq_len, rate):
    mask = ((torch.rand([batch_size,seq_len])) > rate).float()
    return mask

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
seq_epoch_num = train_param["seq_epoch_num"]
struct_epoch_num = train_param["struct_epoch_num"]
domain_epoch_num = train_param["domain_epoch_num"]
fusion_epoch_num = train_param["fusion_epoch_num"]

dropout_feat_list = [["struct","domain"]]*seq_epoch_num + \
                    [["seq","domain"]]*struct_epoch_num + \
                    [["seq","struct"]]*domain_epoch_num + \
                    [[]]*fusion_epoch_num
epoch_num = seq_epoch_num + struct_epoch_num + domain_epoch_num + fusion_epoch_num

batch_size = train_param["batch_size"]
init_lr = train_param["init_lr"]
contrast_temp = train_param["contrast_temp"]
go_max_percent = train_param["go_max_percent"]
go_min_count = train_param["go_min_count"]
mask_rate = train_param["mask_rate"]
alpha = train_param["alpha"]
log_batch_count = train_param["log_batch_count"]
bce_pos_weight = train_param["bce_pos_weight"]

# 读取必要的数据
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)
go_graph = go_graph.to(device)

# 初始化模型 （所有模型都默认使用参数初始化）
model_protein = ProteinEncoder(**pm_param)
model_go = GoEncoder(**gm_param)
model_protein.to(device)
model_go.load_model("model_params/v1/go_pretrain_model.pth")
model_go.to(device)

# 初始化数据集并进行划分
ds = MyDataset("CAFA3_train", target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count)
ds.show_dataset_info(logger)
train_size = int(0.7*len(ds))
valid_size = int(0.2*len(ds))
test_size = len(ds) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, valid_size, test_size])
np.save(os.path.join("dataset","test_indices.npy"), test_dataset.indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

# 设置优化器、学习率规划器以及损失函数
optimizer_protein = torch.optim.Adam(model_protein.parameters(), lr=init_lr)
lr_scheduler_protein = get_cosine_schedule_with_warmup(optimizer_protein, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))
optimizer_go = torch.optim.Adam(model_go.parameters(), lr=init_lr)
lr_scheduler_go = get_cosine_schedule_with_warmup(optimizer_go, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

criterion1 = ContrastLoss()
criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight],device=device))

logger.info("Parameters INFO:")
logger.info(train_param)
logger.info(pm_param)
logger.info(gm_param)
logger.info("Start Training!")
go_idx = torch.tensor(ds.balanced_go_list).to(device)
for epoch in range(epoch_num):
    # train
    if epoch==0 or dropout_feat_list[epoch] != dropout_feat_list[epoch-1]:
        logger.info(f"Dropout_feat: {dropout_feat_list[epoch]}")
    ds.train_mode()
    mean_loss = 0.0
    mean_loss1 = 0.0
    mean_loss2 = 0.0
    for i, (pids, seq_embs, batched_graph, domain_info, labels) in enumerate(train_loader):
        optimizer_protein.zero_grad()
        optimizer_go.zero_grad()
        
        mask = generate_mask(seq_embs.shape[0], seq_embs.shape[1], mask_rate)
        mask = mask.to(device)
        protein_emb = model_protein(seq_embs, batched_graph, domain_info, mask=mask, ablation_feat=dropout_feat_list[epoch])
        # protein_emb: [batch_size, hid_dim]
        go_emb = model_go(go_idx)
        # go_emb: [go_nums, hid_dim]
        predict_matrix = protein_emb @ go_emb.T
        predict_matrix = torch.sigmoid(predict_matrix/0.2)
        # predict_matrix = (predict_matrix + 1)/2
        
        loss1 = criterion1(predict_matrix, labels, contrast_temp)
        loss2 = criterion2(predict_matrix, labels)
        
        loss = alpha*loss1 + loss2
        loss.backward()
        optimizer_protein.step()
        optimizer_go.step()
        lr_scheduler_protein.step()
        lr_scheduler_go.step()
        mean_loss += loss.item()
        mean_loss1 += loss1.item()
        mean_loss2 += loss2.item()
        if i % log_batch_count == log_batch_count-1:
            logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {mean_loss/log_batch_count}, Loss1: {mean_loss1/log_batch_count}, Loss2: {mean_loss2/log_batch_count}')
            mean_loss = 0.0
            mean_loss1 = 0.0
            mean_loss2 = 0.0
            
    # validate
    ds.valid_mode()
    model_protein.eval()
    model_go.eval()
    total = 0
    correct = 0
    actual = []
    pred = []
    with torch.no_grad():
        for pids, seq_embs, batched_graph, domain_info, labels in tqdm(val_loader):
            protein_emb = model_protein(seq_embs, batched_graph, domain_info, ablation_feat=dropout_feat_list[epoch])
            # protein_emb: [batch_size, hid_dim]
            go_emb = model_go(go_idx, go_graph)
            # go_emb: [go_nums, hid_dim]
            predict_matrix = protein_emb @ go_emb.T
            predict_matrix = torch.sigmoid(predict_matrix/0.2)
            # predict_matrix = (predict_matrix + 1)/2
            pred += predict_matrix.tolist()
            actual += labels.tolist()
            
    cal_metrics(pred, actual, logger)
    
    model_protein.save_model("model_params/protein_model.pth")
    model_go.save_model("model_params/go_model.pth")
    model_protein.train()
    model_go.train()
        