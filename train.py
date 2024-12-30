import os
import json
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn
from model.go_model import GoEncoder
from model.protein_model import ProteinEncoder

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置设备编号
device = torch.device("cuda:2")

# 读取模型超参数
protein_model_conf_path = os.path.join("config","ProteinEncoder.conf")
go_model_conf_path = os.path.join("config","GoEncoder.conf")
with open(protein_model_conf_path,"r") as f:
    pm_param = json.load(f)
with open(go_model_conf_path,"r") as f:
    gm_param = json.load(f)
    
# 读取训练设置的参数
train_conf_path = os.path.join("config","train.conf")
with open(go_model_conf_path,"r") as f:
    train_param = json.load(f)

print("Use train param in config directory? ([y]/n)")
train_conf_flag = input()
if train_conf_flag != 'n':
    epoch_num = train_param["epoch_num"]
    batch_size = train_param["batch_size"]
    init_lr = train_param["init_lr"]
else:
    epoch_num = 20
    batch_size = 64
    init_lr = 1e-3

# 读取必要的数据
with open(os.path.join("dataset","go_graph_dgl.pkl"),"rb") as f:
    go_graph = pickle.load(f)

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

# 初始化数据集并进行划分
ds = MyDataset("HUMAN")
dl = DataLoader(ds,batch_size=batch_size,shuffle=False, collate_fn=my_collate_fn)

# 设置优化器与学习率规划器
optimizer_protein = torch.optim.Adam(model_protein.parameters(), lr=init_lr)
lr_scheduler_protein = get_cosine_schedule_with_warmup(optimizer_protein, num_warmup_steps=100, num_training_steps=epoch_num*len(dl))
optimizer_go = torch.optim.Adam(model_go.parameters(), lr=init_lr)
lr_scheduler_go = get_cosine_schedule_with_warmup(optimizer_go, num_warmup_steps=100, num_training_steps=epoch_num*len(dl))

loss = nn.CrossEntropyLoss()

for epoch in range(epoch_num):
    for pids, seq_embs, graph_list, domain_info, labels in dl:
        protein_emb = model_protein(seq_embs, graph_list, domain_info)
        # protein_emb: [batch_size, hid_dim]
        go_emb = model_go(go_graph)
        # go_emb: [go_nums, hid_dim]
        
        predict_matrix = protein_emb @ go_emb.T
        