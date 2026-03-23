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
import matplotlib.pyplot as plt

from torch import nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from register import *
from model.dataset import MyDataset, probablity_propagate
from model.predictor import ContrastLoss, ProteinGoModel, ProteinGoModelv2
from utils import init_logger, cal_metrics, cal_metrics_special, cal_Pmin, cal_metrics_all, cal_effect, cal_pert_effect, load_config_clone

import mlflow

# python pca_test.py -o mf -s AF2DB -n model_v1.1 -t v1.1
@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='AF2DB')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2','v3']), default='v1.1')
def main(ont, species_name, model_name, model_type):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    idx_dir = Path(workspace_dir,"idx")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    model_param_dir = Path(workspace_dir,"model_param")
    result_dir = Path(workspace_dir,"result_pca")
    os.makedirs(result_dir,exist_ok=True)

    # 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
    device = torch.device("cuda:0")
    
    # 初始化日志
    logger = init_logger(Path(log_dir, 'draw_pca_{}.log'.format(ont)))
    
    # 输出输入的参数信息
    logger.info("----------------------------A new test process----------------------------")
    logger.info("Input args:")
    logger.info("GO type: {}".format(ont))
    logger.info("species_name: {}".format(species_name))
    logger.info("model_name: {}".format(model_name))
    logger.info("model_type: {}".format(model_type))
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
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)

    go_idx = torch.tensor(ds.balanced_go_list).to(device)
    go_emb = pg_model.gm(go_idx).cpu().numpy()
    balanced_go_map_inv = ds.balanced_go_map_inv

    logger.info("Parameters INFO:")
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info("GO_term num:"+str(len(ds.balanced_go_list)))
    logger.info("Start Testing!")

    ds.valid_mode()
    prot_emb = []
    with torch.no_grad():
        for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(data_loader):
            prot_emb.append(pg_model.pm(seq_embs, batched_graph, domain_info))
    prot_emb = torch.cat(prot_emb, dim=0).cpu().numpy()

    scaler = StandardScaler()

    # 增量法PCA分析
    pca = IncrementalPCA(n_components=2, batch_size=5*batch_size)
    prot_emb_pca = pca.fit_transform(scaler.fit_transform(prot_emb))
    go_emb_pca = pca.fit_transform(scaler.fit_transform(go_emb))

    # 绘制PCA图
    plt.figure(figsize=(10, 10))
    plt.scatter(prot_emb_pca[:, 0], prot_emb_pca[:, 1], label='Proteins', alpha=0.5, color='blue')
    plt.title('PCA of Protein Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(Path(result_dir, 'protein_pca_plot_{}.png'.format(ont)))

    plt.figure(figsize=(10, 10))
    plt.scatter(go_emb_pca[:, 0], go_emb_pca[:, 1], label='GO Terms', alpha=0.5, color='orange')
    plt.title('PCA of GO Term Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.savefig(Path(result_dir, 'go_pca_plot_{}.png'.format(ont)))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()