import os
import json
import pickle
import click
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import torch.functional as F
import dgl
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn, my_collate_fn_v2
from model.go_dataset import GoDataset
from model.go_model import GoContrastLoss
from model.predictor import ProteinGoModel, ProteinGoModelv2, ContrastLoss
from utils import init_logger, cal_metrics, load_config_clone, get_split_dataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))
default_pm_conf = {"v1.1":Path("config","ProteinEncoder.conf"), "v2":Path("config","ProteinEncoder2.conf")}
default_gm_conf = {"v1.1":Path("config","GoEncoder.conf"), "v2":Path("config","GoEncoder2.conf")}
default_pred_conf = {"v1.1":Path("config","Predictor.conf"), "v2":Path("config","Predictor.conf")}

def generate_mask(batch_size, seq_len, rate):
    mask = ((torch.rand([batch_size,seq_len])) > rate).float()
    return mask

def generate_graph_mask(node_num, rate):
    mask = ((torch.rand([node_num])) > rate).float()
    return mask

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2']), default='v1.1')
def main(ont, species_name, model_name, model_type):
    os.makedirs(Path("workspace"),exist_ok=True)
    workspace_dir = Path("workspace",species_name,model_name)
    os.makedirs(Path("workspace",species_name),exist_ok=True)
    os.makedirs(workspace_dir,exist_ok=True)
    log_dir = Path(workspace_dir,"log")
    idx_dir = Path(workspace_dir,"idx")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    model_param_dir = Path(workspace_dir,"model_param")
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(idx_dir,exist_ok=True)
    os.makedirs(cache_dir,exist_ok=True)
    os.makedirs(config_dir,exist_ok=True)
    os.makedirs(model_param_dir,exist_ok=True)
    
    # 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
    device = torch.device("cuda:0")

    # 初始化日志
    logger = init_logger(Path(log_dir, 'train_{}.log'.format(ont)))

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    if model_type == "v1.1":
        pg_model = ProteinGoModel(pm_param, gm_param, **pred_param)
        local_collate_fn = my_collate_fn
    elif model_type == "v2":
        pg_model = ProteinGoModelv2(pm_param, gm_param)
        local_collate_fn = my_collate_fn_v2
    else:
        logger.error("unknown model type")
        return
    pg_model.gm.load_model(Path("model_params","go_pretrain_model.pth"))
    pg_model.to(device)
        
    # 读取训练参数
    train_default_conf_path = Path("config","train.conf")
    train_param = load_config_clone(train_default_conf_path, config_dir)
    train_proportion = train_param["train_proportion"]
    valid_proportion = train_param["valid_proportion"]
    batch_size = train_param["batch_size"]
    init_lr = train_param["init_lr"]
    contrast_temp = train_param["contrast_temp"]
    go_max_percent = train_param["go_max_percent"]
    go_min_count = train_param["go_min_count"]
    mask_rate = train_param["mask_rate"]
    alpha1 = train_param["alpha1"]
    alpha2 = train_param["alpha2"]
    alpha3 = train_param["alpha3"]
    log_batch_count = train_param["log_batch_count"]
    bce_pos_weight = train_param["bce_pos_weight"]
    seq_epoch_num = train_param["seq_epoch_num"]
    struct_epoch_num = train_param["struct_epoch_num"]
    domain_epoch_num = train_param["domain_epoch_num"]
    fusion_epoch_num = train_param["fusion_epoch_num"]
    epoch_num = seq_epoch_num + struct_epoch_num + domain_epoch_num + fusion_epoch_num
    dropout_feat_list = [["struct","domain"]]*seq_epoch_num + \
                        [["seq","domain"]]*struct_epoch_num + \
                        [["seq","struct"]]*domain_epoch_num + \
                        [[]]*fusion_epoch_num
    go_loss_switch = train_param["go_loss_switch"]
                        
    train_go_default_conf_path = Path("config","train_go.conf")
    train_go_param = load_config_clone(train_go_default_conf_path, config_dir)
    pos_k = train_go_param["pos_k"]
    pos_max = train_go_param["pos_max"]
    neg_k = train_go_param["neg_k"]
    neg_max = train_go_param["neg_max"]
    pos_rate = train_go_param["pos_rate"]
    contrast_temp_go = train_go_param["contrast_temp"]
    
    # 初始化数据集并进行划分
    best_fmax = 0.0
    best_epoch = -1
    corr_auc = 0.0
    corr_aupr = 0.0
    corr_thresh = 0.0

    ds = MyDataset(species_name, ont, cache_dir, target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count)
    ds.show_dataset_info(logger)
    train_dataset, valid_dataset, test_dataset = get_split_dataset(idx_dir,ont,ds,logger,train_proportion,valid_proportion,use_cache=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=local_collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)

    go_ds = GoDataset(pos_k, pos_max, neg_k, neg_max, pos_rate)

    # 设置优化器、学习率规划器以及损失函数
    optimizer = torch.optim.Adam(pg_model.parameters(), lr=init_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

    criterion1 = ContrastLoss()
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight],device=device))
    criterion3 = GoContrastLoss(pos_k, neg_k)

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
        mean_loss3 = 0.0
        pred = []
        actual = []
        for i, (idxs, pids, seq_embs, batched_graph, domain_info, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            mask = generate_mask(seq_embs.shape[0], seq_embs.shape[1], mask_rate)
            graph_mask = generate_graph_mask(batched_graph.num_nodes(), mask_rate)
            mask = mask.to(device)
            graph_mask = graph_mask.to(device)
            
            predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx, mask, graph_mask, ablation_feat=dropout_feat_list[epoch])
            
            loss1 = criterion1(predict_matrix, labels, contrast_temp)
            loss2 = criterion2(predict_matrix, labels)
            loss3 = torch.tensor(0.0,device=device)
            if go_loss_switch:
                go_id_set = set()
                for idx in idxs:
                    for go in ds.data_info[idx][5]:
                        go_id_set.add(go)
                go_subset = Subset(go_ds, list(go_id_set))
                go_dl = DataLoader(go_subset, batch_size=batch_size, shuffle=True)
                for (idx, pos, neg) in go_dl:
                    x = torch.cat([idx,pos,neg],dim=1)
                    x = x.to(device)
                    x = pg_model.gm(x)
                    loss3 += criterion3(x, contrast_temp_go)
                loss3 /= len(go_id_set)
            
            loss = alpha1*loss1 + alpha2*loss2 + alpha3*loss3
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            mean_loss += loss.item()
            mean_loss1 += loss1.item()
            mean_loss2 += loss2.item()
            mean_loss3 += loss3.item()
            if i % log_batch_count == log_batch_count-1:
                logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {mean_loss/log_batch_count}, Loss1: {mean_loss1/log_batch_count}, Loss2: {mean_loss2/log_batch_count}, Loss3: {mean_loss3/log_batch_count}')
                mean_loss = 0.0
                mean_loss1 = 0.0
                mean_loss2 = 0.0
                mean_loss3 = 0.0
                pred += predict_matrix.tolist()
                actual += labels.tolist()
        logger.info("train set:")
        auc_score, aupr_score, fmax, thresh = cal_metrics(pred, actual)
        logger.info("AUC: {:.4f} AUPR: {:.4f} Fmax: {:.4f} thresh: {:.4f}".format(auc_score, aupr_score, fmax, thresh))
                
        # validate
        ds.valid_mode()
        pg_model.eval()
        total = 0
        correct = 0
        actual = []
        pred = []
        with torch.no_grad():
            for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(val_loader):
                predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=dropout_feat_list[epoch])
                # predict_matrix = (predict_matrix + 1)/2
                pred += predict_matrix.tolist()
                actual += labels.tolist()
        logger.info("valid set:")
        auc_score, aupr_score, fmax, thresh = cal_metrics(pred, actual)
        logger.info("AUC: {:.4f} AUPR: {:.4f} Fmax: {:.4f} thresh: {:.4f}".format(auc_score, aupr_score, fmax, thresh))
        if fmax > best_fmax:
            best_fmax = fmax
            best_epoch = epoch
            corr_auc = auc_score
            corr_aupr = aupr_score
            corr_thresh = thresh
            pg_model.save_model(Path(model_param_dir,"{}_model.pt".format(ont)))
        pg_model.train()
    
    logger.info(f"{species_name} {model_name} {ont} dataset training result: Valid")
    logger.info(f"best_epoch:{best_epoch} AUC:{corr_auc} best_Fmax:{best_fmax} AUPR:{corr_aupr} thresh:{corr_thresh}")

if __name__ == '__main__':
    main()