'''
将某数据集中某模型运用到另一数据集测试准确率
只包含GO的微调测试
e.g.
CUDA_VISIBLE_DEVICES=? python fine_tuning_test.py -o mf -s xxx -n model_v1.1_back -ds yyy -fn 1000
'''

import os
import json
import pickle
import click
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import torch.functional as F
import networkx as nx

from torch import nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn, my_collate_fn_v2, probablity_propagate
from model.predictor import ContrastLoss, ProteinGoModel, ProteinGoModelv2

from register import *
from utils import init_logger, cal_metrics, load_config_clone, generate_mask, generate_graph_mask

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2','v3']), default='v1.1')
@click.option('-thresh', '--thresh', type=click.FLOAT, default=None)
@click.option('-ds', '--dest-species', type=click.STRING)
@click.option('-fn', '--fine-tuning-nums', type=click.INT, default=100)
def main(ont, species_name, model_name, model_type, thresh, dest_species, fine_tuning_nums):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    idx_dir = Path(workspace_dir,"idx")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    model_param_dir = Path(workspace_dir,"model_param")
    
    dest_workspace_dir = Path("workspace",dest_species,"transfer_tmp")
    os.makedirs(dest_workspace_dir,exist_ok=True)
    dest_cache_dir = Path(dest_workspace_dir,"cache")
    os.makedirs(dest_cache_dir,exist_ok=True)

    # 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
    device = torch.device("cuda:0")
    
    # 初始化日志
    logger = init_logger(Path(log_dir, 'transfer_test_{}.log'.format(ont)))

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    pg_model, local_collate_fn = model_init(model_type,pm_param,gm_param,pred_param)
    pg_model.load_model(Path(model_param_dir,"{}_model.pt".format(ont)))
    pg_model.to(device)

    # 测试集参数
    test_default_conf_path = Path("config","test.conf")
    test_param = load_config_clone(test_default_conf_path, config_dir)
    batch_size = test_param["batch_size"]

    # 读取train中的数据集设置（保持数据集读取结构与预训练模型一致）
    train_default_conf_path = Path("config","train.conf")
    train_param = load_config_clone(train_default_conf_path, config_dir)
    go_max_percent = train_param["go_max_percent"]
    go_min_count = train_param["go_min_count"]
    
    # 微调训练参数
    fine_tuning_default_conf_path = Path("config","fine-tuning.conf")
    fine_tuning_param = load_config_clone(fine_tuning_default_conf_path, config_dir)
    epoch_num = fine_tuning_param["epoch_num"]
    batch_size = fine_tuning_param["batch_size"]
    init_lr = fine_tuning_param["init_lr"]
    mask_rate = fine_tuning_param["mask_rate"]
    contrast_temp = fine_tuning_param["contrast_temp"]
    alpha1 = fine_tuning_param["alpha1"]
    alpha2 = fine_tuning_param["alpha2"]
    bce_pos_weight = fine_tuning_param["bce_pos_weight"]

    # 读取必要的数据
    with open(Path("dataset","go_graph_nx.pkl"),"rb") as f:
        go_graph_nx = pickle.load(f)
    topo_order = list(nx.topological_sort(go_graph_nx))
    
    # 读取数据集
    ds = MyDataset(species_name, ont, cache_dir, target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count, use_cache=True)
    dest_ds = MyDataset(dest_species, ont, dest_cache_dir, target_device=device,
                        force_balance_go_list=ds.balanced_go_list, use_cache=False)
    dest_ds.show_dataset_info(logger)
    go_idx = torch.tensor(ds.balanced_go_list).to(device)
    balanced_go_map_inv = ds.balanced_go_map_inv
    
    # 分割fine-tuning数据集和测试集
    if fine_tuning_nums >= len(dest_ds):
        print("Warning: fine_tuning_nums >= dest_dataset_size")
        return
    indices = np.arange(len(dest_ds))
    np.random.shuffle(indices)
    fine_tuning_indices = indices[:fine_tuning_nums]
    test_indices = indices[fine_tuning_nums:]
    
    fine_tuning_ds = Subset(dest_ds, fine_tuning_indices)
    test_ds = Subset(dest_ds, test_indices)
    train_loader = DataLoader(fine_tuning_ds, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)
    logger.info("fine-tuning train size:{} , fine-tuning test size:{}".format(len(train_loader),len(test_loader)))

    logger.info("Parameters INFO:")
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info("GO_term num:"+str(len(ds.balanced_go_list)))
    
    logger.info("Start Fine-tuning")
    # 冻结Protein_Model和GO_Model的参数
    for p in pg_model.pm.parameters():
        p.requires_grad = False
    for p in pg_model.gm.parameters():
        p.requires_grad = False
    
    # 设置优化器、学习率规划器以及损失函数
    optimizer = torch.optim.Adam(pg_model.parameters(), lr=init_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=epoch_num*len(train_loader))
    criterion1 = ContrastLoss()
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight],device=device))
    
    # 各项评估指标初始化
    best_fmax = 0.0
    best_epoch = -1
    corr_auc = 0.0
    corr_aupr = 0.0
    corr_thresh = 0.0
    
    for epoch in range(epoch_num):
        pg_model.train()
        dest_ds.train_mode()
        mean_loss = 0.0
        mean_loss1 = 0.0
        mean_loss2 = 0.0
        pred = []
        actual = []
        for i, (idxs, pids, seq_embs, batched_graph, domain_info, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            mask = generate_mask(seq_embs.shape[0], seq_embs.shape[1], mask_rate)
            graph_mask = generate_graph_mask(batched_graph.num_nodes(), mask_rate)
            mask = mask.to(device)
            graph_mask = graph_mask.to(device)
            
            predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx, mask, graph_mask)
            
            loss1 = criterion1(predict_matrix, labels, contrast_temp)
            loss2 = criterion2(predict_matrix, labels)
            
            loss = alpha1*loss1 + alpha2*loss2
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            mean_loss += loss.item()
            mean_loss1 += loss1.item()
            mean_loss2 += loss2.item()
            pred += predict_matrix.tolist()
            actual += labels.tolist()
        logger.info(f'Epoch: {epoch} / {epoch_num}, Loss: {mean_loss/len(train_loader)}, Loss1: {mean_loss1/len(train_loader)}, Loss2: {mean_loss2/len(train_loader)}')
        mean_loss = 0.0
        mean_loss1 = 0.0
        mean_loss2 = 0.0
            
        logger.info("fine-tuning set:")
        auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual)
        logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
    
        # 测试   
        logger.info("Start Testing!")
        pg_model.eval()
        dest_ds.valid_mode()
        actual = []
        pred = []
        with torch.no_grad():
            for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
                predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx)
                # predict_matrix: [batch_size, go_nums]
                predict_matrix = predict_matrix.transpose(1,0)
                predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, ds.balanced_go_map_inv)
                predict_matrix = predict_matrix.transpose(1,0)
                pred += predict_matrix.tolist()
                actual += labels.tolist()
        auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual, thresh)
        logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
        
        if fmax > best_fmax:
            best_fmax = fmax
            best_epoch = epoch
            corr_auc = auc_score
            corr_aupr = aupr_score
            corr_thresh = tmp_thresh
            pg_model.save_model(Path(model_param_dir,"{}_model_FTon_{}.pt".format(ont, dest_species)))
        pg_model.train()
    
    logger.info(f"{species_name} {model_name} {ont} dataset training result: Valid")
    logger.info(f"best_epoch:{best_epoch} AUC:{corr_auc} best_Fmax:{best_fmax} AUPR:{corr_aupr} thresh:{corr_thresh}")

if __name__ == '__main__':
    main()