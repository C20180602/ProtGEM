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

from utils import init_logger, cal_metrics, load_config_clone

os.chdir(os.path.dirname(os.path.abspath(__file__)))
default_pm_conf = {"v1.1":Path("config","ProteinEncoder.conf"), "v2":Path("config","ProteinEncoder2.conf")}
default_gm_conf = {"v1.1":Path("config","GoEncoder.conf"), "v2":Path("config","GoEncoder2.conf")}
default_pred_conf = {"v1.1":Path("config","Predictor.conf"), "v2":Path("config","Predictor.conf")}

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2']), default='v1.1')
@click.option('-thresh', '--thresh', type=click.FLOAT, default=None)
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

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    if model_type == "v1.1":
        pg_model = ProteinGoModel(pm_param, gm_param, **pred_param)
        local_collate_fn = my_collate_fn
    else:
        pg_model = ProteinGoModelv2(pm_param, gm_param)
        local_collate_fn = my_collate_fn_v2
    ablation_list = [[]]
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
            predict_matrix = predict_matrix.transpose(1,0)
            predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, ds.balanced_go_map_inv)
            predict_matrix = predict_matrix.transpose(1,0)
            pred += predict_matrix.tolist()
            actual += labels.tolist()
    auc_score, aupr_score, fmax, tmp_thresh = cal_metrics(pred, actual)
    logger.info("AUC: {:.4f} AUPR: {:.4f} Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, fmax, tmp_thresh))

if __name__ == '__main__':
    main()