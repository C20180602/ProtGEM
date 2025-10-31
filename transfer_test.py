'''
将某数据集中某模型运用到另一数据集测试准确率
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
from tqdm import tqdm
from model.dataset import MyDataset, my_collate_fn, my_collate_fn_v2, probablity_propagate
from model.predictor import ContrastLoss, ProteinGoModel, ProteinGoModelv2

from register import *
from utils import init_logger, cal_metrics, load_config_clone, set_seeds

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2']), default='v1.1')
@click.option('-thresh', '--thresh', type=click.FLOAT, default=None)
@click.option('-ds', '--dest-species', type=click.STRING)
def main(ont, species_name, model_name, model_type, thresh, dest_species):
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

    # 初始化随机种子以便复现
    seed = np.random.randint(1e9)
    set_seeds(seed)
    
    # 输出输入的参数信息
    logger.info("----------------------------A new test process----------------------------")
    logger.info("random seed: {}".format(seed))
    logger.info("Input args:")
    logger.info("GO type: {}".format(ont))
    logger.info("species_name: {}".format(species_name))
    logger.info("model_name: {}".format(model_name))
    logger.info("model_type: {}".format(model_type))
    logger.info("thersh: {}".format(thresh))
    logger.info("dest_species: {}".format(dest_species))
    logger.info("----------------------------")
    
    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    pg_model, local_collate_fn = model_init(model_type,pm_param,gm_param,pred_param)
    pg_model.load_model(Path(model_param_dir,"{}_model.pt".format(ont)))
    pg_model.to(device)
    for p in pg_model.parameters():
        p.requires_grad = False
    pg_model.eval()

    # 测试集参数
    test_default_conf_path = Path("config","test.conf")
    test_param = load_config_clone(test_default_conf_path, config_dir)
    batch_size = test_param["batch_size"]

    # 读取train中的数据集设置
    train_default_conf_path = Path("config","train.conf")
    train_param = load_config_clone(train_default_conf_path, config_dir)
    go_max_percent = train_param["go_max_percent"]
    go_min_count = train_param["go_min_count"]

    # 读取必要的数据
    with open(Path("dataset","go_graph_nx.pkl"),"rb") as f:
        go_graph_nx = pickle.load(f)
    topo_order = list(nx.topological_sort(go_graph_nx))
    
    # 读取测试集
    ds = MyDataset(species_name, ont, cache_dir, target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count, use_cache=True)
    dest_ds = MyDataset(dest_species, ont, dest_cache_dir, target_device=device, inference_flag=True,
                        force_balance_go_list=ds.balanced_go_list, exclude_pid_list=ds.get_pid_list(), use_cache=False)
    dest_ds.show_dataset_info(logger)
    test_loader = DataLoader(dest_ds, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)

    go_idx = torch.tensor(ds.balanced_go_list).to(device)
    balanced_go_map_inv = ds.balanced_go_map_inv

    summary(pg_model)
    logger.info("Parameters INFO:")
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info("GO_term num:"+str(len(ds.balanced_go_list)))
    logger.info("Start Testing!")

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
    auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual)
    logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))

if __name__ == '__main__':
    main()