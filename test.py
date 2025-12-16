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

from register import *
from model.dataset import MyDataset, probablity_propagate
from model.predictor import ContrastLoss, ProteinGoModel, ProteinGoModelv2
from utils import init_logger, cal_metrics, cal_metrics_special, cal_Pmin, cal_metrics_all, cal_effect, cal_pert_effect, load_config_clone

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2','v3']), default='v1.1')
@click.option('-th', '--thresh', type=click.FLOAT, default=None)
def main(ont, species_name, model_name, model_type, thresh):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    idx_dir = Path(workspace_dir,"idx")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    model_param_dir = Path(workspace_dir,"model_param")

    # 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
    device = torch.device("cuda:0")
    
    # 初始化日志
    logger = init_logger(Path(log_dir, 'test_{}.log'.format(ont)))
    
    # 输出输入的参数信息
    logger.info("----------------------------A new test process----------------------------")
    logger.info("Input args:")
    logger.info("GO type: {}".format(ont))
    logger.info("species_name: {}".format(species_name))
    logger.info("model_name: {}".format(model_name))
    logger.info("model_type: {}".format(model_type))
    logger.info("thresh: {}".format(thresh))
    logger.info("----------------------------")
    
    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    pg_model, local_collate_fn = model_init(model_type, pm_param, gm_param, pred_param)
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
    with open(Path("dataset","go_term_dis.json"),"r") as f:
        go_dis = json.load(f)
    with open(Path("processed_data","go_id.json"),"r") as f:
        go_id = json.load(f)
    with open(Path("processed_data","interpro2go.json"),"r") as f:
        interpro2go = json.load(f)
    with open(Path("processed_data","interpro_id.json"),"r") as f:
        interpro_id = json.load(f)
    topo_order = list(nx.topological_sort(go_graph_nx))
    
    # 读取测试集
    ds = MyDataset(species_name, ont, cache_dir, target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count, use_cache=True)
    ds.show_dataset_info(logger)
    test_indices = np.load(Path(idx_dir,f"{ont}_test_indices.npy")).tolist()
    if max(test_indices) > len(ds):
        logger.error(str(max(test_indices))+str(len(ds)))
        logger.error("Error: test_indices out of range!")
        exit(1)
    test_dataset = Subset(ds, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)

    go_idx = torch.tensor(ds.balanced_go_list).to(device)
    go_idx_special = []
    for i, go_id in enumerate(ds.balanced_go_list):
        if go_dis[str(go_id)] >= 5:
            go_idx_special.append(i)

    go_punish = []
    for go_id in ds.balanced_go_list:
        go_punish.append(go_dis[str(go_id)])
    go_punish = -torch.log(1.0/torch.tensor(go_punish))
    go_punish = go_punish.tolist()
    balanced_go_map_inv = ds.balanced_go_map_inv

    summary(pg_model)
    logger.info("Parameters INFO:")
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info("GO_term num:"+str(len(ds.balanced_go_list)))
    logger.info("Start Testing!")

    ds.valid_mode()
    actual = []
    pred = []
    with torch.no_grad():
        for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
            
            predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx)
            # predict_matrix: [batch_size, go_nums]
            predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, ds.balanced_go_map_inv)
            pred += predict_matrix.tolist()
            actual += labels.tolist()
    cal_metrics_all(pred, actual, go_idx_special, go_punish, logger, thresh)

    logger.info("-----Global Local Feature Effectivity Study-----")
    glob_pred_list = []
    part_pred_list = []
    sum_effect = torch.zeros(4)
    sum_pert_effect = torch.zeros(2)
    with torch.no_grad():
        for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
            
            ori_pred = pg_model(seq_embs, batched_graph, domain_info, go_idx)
            glob_pred = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=["part"])
            glob_pert = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=["glob"], pert_mode=True)
            glob_pred_prop = probablity_propagate(glob_pred, go_graph_nx, topo_order, ds.balanced_go_map_inv)
            glob_pred_list += glob_pred_prop.tolist()

            part_pred = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=["glob"])
            part_pert = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=["part"], pert_mode=True)
            part_pred_prop = probablity_propagate(part_pred, go_graph_nx, topo_order, ds.balanced_go_map_inv)
            part_pred_list += part_pred_prop.tolist()
            effect = cal_effect(ori_pred, glob_pred, part_pred)
            pert_effect = cal_pert_effect(ori_pred, glob_pert, part_pert)
            sum_effect = sum_effect + effect * len(pids)
            sum_pert_effect = sum_pert_effect + pert_effect * len(pids)
    logger.info("Only Global Feature:")
    cal_metrics_all(glob_pred_list, actual, go_idx_special, go_punish, logger, thresh)
    
    logger.info("Only Local Feature:")
    cal_metrics_all(part_pred_list, actual, go_idx_special, go_punish, logger, thresh)

    sum_effect /= len(test_dataset)
    logger.info("Static Analysis Results")
    logger.info("Global Effect: {:.6f}, Global Unique Effect: {:.6f}".format(sum_effect[0].item(), sum_effect[1].item()))
    logger.info(" Local Effect: {:.6f},  Local Unique Effect: {:.6f}".format(sum_effect[2].item(), sum_effect[3].item()))

    sum_pert_effect /= len(test_dataset)
    logger.info("Perturbation Analysis Results")
    logger.info("Global Var: {:.6f}".format(sum_effect[0].item()))
    logger.info(" Local Var: {:.6f}".format(sum_effect[1].item()))

    logger.info("-----Zero Mask Abltion Study-----")
    for abl_feat in [["seq"],["struct"],["domain"]]:
        logger.info(f"ablation feature (without): {abl_feat}")
        pred = []
        with torch.no_grad():
            for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
                
                predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx, ablation_feat=abl_feat)
                # predict_matrix: [batch_size, go_nums]
                predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, ds.balanced_go_map_inv)
                pred += predict_matrix.tolist()
        cal_metrics_all(pred, actual, go_idx_special, go_punish, logger, thresh)

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()