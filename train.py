import os
import json
import pickle
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn
from model.go_model import GoEncoder
from model.protein_model import ProteinEncoder

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

print("Use train param in config directory? ([y]/n)")
train_conf_flag = input()
if train_conf_flag != 'n':
    epoch_num = train_param["epoch_num"]
    batch_size = train_param["batch_size"]
    init_lr = train_param["init_lr"]
else:
    epoch_num = 20
    batch_size = 8
    init_lr = 1e-3

# 读取必要的数据
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)
go_graph = go_graph.to(device)

# 初始化模型
print("Use model param in config directory? ([y]/n)")
config_flag = input()
print("Use biobert embeddings to initialize GO model? ([y]/n)")
go_init_flag = input()
if go_init_flag != 'n':
    # 读取GO模型的预设embedding
    with open(os.path.join("dataset","go_embedding_biobert.pkl"),"rb") as f:
        go_embedding_biobert = pickle.load(f)
if config_flag != 'n':
    model_protein = ProteinEncoder(**pm_param)
    if go_init_flag != 'n':
        model_go = GoEncoder(**gm_param,manual_init=True,init_tensor=go_embedding_biobert)
    else:
        model_go = GoEncoder(**gm_param)
else:
    model_protein = ProteinEncoder()
    if go_init_flag != 'n':
        model_go = GoEncoder(manual_init=True,init_tensor=go_embedding_biobert)
    else:
        model_go = GoEncoder()
model_protein.to(device)
model_go.to(device)

# 初始化数据集并进行划分
ds = MyDataset("HUMAN")
train_size = int(0.7*len(ds))
valid_size = int(0.2*len(ds))
test_size = len(ds) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, valid_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

# 设置优化器、学习率规划器以及损失函数
optimizer_protein = torch.optim.Adam(model_protein.parameters(), lr=init_lr)
lr_scheduler_protein = get_cosine_schedule_with_warmup(optimizer_protein, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))
optimizer_go = torch.optim.Adam(model_go.parameters(), lr=init_lr)
lr_scheduler_go = get_cosine_schedule_with_warmup(optimizer_go, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

criterion = nn.CrossEntropyLoss()

for epoch in range(epoch_num):
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
        go_emb = model_go(go_graph)
        # go_emb: [go_nums, hid_dim]
        predict_matrix = protein_emb @ go_emb.T
        
        loss = criterion(predict_matrix, labels)
        loss.backward()
        optimizer_protein.step()
        optimizer_go.step()
        lr_scheduler_protein.step()
        lr_scheduler_go.step()
        
        if i % 50 == 49:
            print(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {loss.item()}')
            model_protein.eval()
            model_go.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for pids, seq_embs, batched_graph, domain_info, labels in val_loader:
                    seq_embs = seq_embs.to(device)
                    batched_graph = batched_graph.to(device)
                    domain_info[0] = domain_info[0].to(device)
                    domain_info[1] = domain_info[1].to(device)
                    labels = labels.to(device)
                    
                    protein_emb = model_protein(seq_embs, batched_graph, domain_info)
                    # protein_emb: [batch_size, hid_dim]
                    go_emb = model_go(go_graph)
                    # go_emb: [go_nums, hid_dim]
                    predict_matrix = protein_emb @ go_emb.T
                    predict_matrix = torch.sign(predict_matrix)
                    
                    total += labels.size(0)
                    correct += (predict_matrix == labels).sum().item()

            acc = correct / total
            print(f'Val Acc: {acc}')
            model_protein.train()
            model_go.train()
        