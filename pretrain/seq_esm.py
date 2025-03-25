'''
用于生成序列中每个氨基酸的特征向量
'''
import json
import os
import pickle
import tqdm
import pandas as pd
from pandas import DataFrame as df

import torch
import esm

# 设置CUDA设备编号
device = "cuda:2"
# 设置工作目录为项目文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir("..")

# ESM2预训练模型初始化,取消梯度
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
converter = alphabet.get_batch_converter()
model.to(device)
for p in model.parameters():
    p.requires_grad = False
model.eval()

# 读取预处理后的文件
species_name = "CAFA3_train"
with open(os.path.join("processed_data",species_name+"_protein_info.json"),'r') as f:
    protein_info = json.load(f)

print("start ESM2, get sequence embeddings!")
cnt = 0
for uid in protein_info.keys():
    cnt += 1
    print(cnt,'/',len(protein_info))
    val = protein_info[uid]
    labels, strs, tokens = converter([(uid,val["seq"])])
    tokens = tokens.to(device)
    seq_emb = model(tokens, repr_layers=[33])['representations'][33]
    seq_emb = seq_emb.squeeze(0)
    seq_emb = seq_emb.cpu().numpy()
    
    dir_path = os.path.join("dataset",species_name,uid)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path,"seq_emb.pkl"), 'wb') as f:
        pickle.dump(seq_emb, f)
    del tokens,seq_emb
    torch.cuda.empty_cache()