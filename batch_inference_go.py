'''
将某数据集中某GO模型运用到另一数据集
预测所有标签并导出为json文件
e.g.
CUDA_VISIBLE_DEVICES=? python batch_inference_go.py -o bp -s xxx -ds yyy -th 0.68
CUDA_VISIBLE_DEVICES=? python batch_inference_go.py -o bp -s xxx -n model_v1.1_back -pn ??_model_FTon_yyy.pt -ds yyy -th 0.68
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
from model.dataset import MyDataset, my_collate_fn, my_collate_fn_v2, probablity_propagate, bfs_get_dis_go
from model.predictor import ProteinGoModel, ProteinGoModelv2
from model.loss import ContrastLoss

from register import *
from utils import init_logger, cal_metrics, load_config_clone, set_seeds

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-pn', '--precise-model-name', type=click.STRING, default=None)
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2','v3']), default='v1.1')
@click.option('-th', '--thresh', type=click.FLOAT, default=None)
@click.option('-ds', '--dest-species', type=click.STRING, default=None)
def main(ont, species_name, model_name, precise_model_name, model_type, thresh, dest_species):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    result_dir = Path(workspace_dir,"result_{}".format(dest_species))
    os.makedirs(result_dir,exist_ok=True)
    model_param_dir = Path(workspace_dir,"model_param")
    
    dest_workspace_dir = Path("workspace",dest_species,"infer_tmp")
    os.makedirs(dest_workspace_dir,exist_ok=True)
    dest_cache_dir = Path(dest_workspace_dir,"cache")
    os.makedirs(dest_cache_dir,exist_ok=True)

    # 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
    device = torch.device("cuda:0")
    
    # 初始化日志
    logger = init_logger(Path(log_dir, 'go_infer_{}.log'.format(ont)))
    
    # 初始化随机种子以便复现
    seed = np.random.randint(1e9)
    set_seeds(seed)
    
    # 输出输入的参数信息
    logger.info("----------------------------A new inference process----------------------------")
    logger.info("random seed: {}".format(seed))
    logger.info("Input args:")
    logger.info("GO type: {}".format(ont))
    logger.info("species_name: {}".format(species_name))
    logger.info("model_name: {}".format(model_name))
    logger.info("precise_model_name: {}".format(precise_model_name))
    logger.info("model_type: {}".format(model_type))
    logger.info("thresh: {}".format(thresh))
    logger.info("dest_species: {}".format(dest_species))
    logger.info("----------------------------")

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    pg_model, local_collate_fn = model_init(model_type,pm_param,gm_param,pred_param)
    if precise_model_name is not None:
        pg_model.load_model(Path(model_param_dir,precise_model_name))
    else:
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
    with open(Path("processed_data","go_terms.json"),"r") as f:
        go_terms = json.load(f)
    with open(Path("processed_data","go_id.json"),"r") as f:
        go_id_map = json.load(f)
    topo_order = list(nx.topological_sort(go_graph_nx))
    
    # 读取测试集
    ds = MyDataset(species_name, ont, cache_dir, target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count, use_cache=True)
    dest_ds = MyDataset(dest_species, ont, dest_cache_dir, target_device=device, inference_flag=True,
                        force_balance_go_list=ds.balanced_go_list, use_cache=False)
    dest_ds.show_dataset_info(logger)
    test_loader = DataLoader(dest_ds, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)

    go_idx = torch.tensor(ds.balanced_go_list).to(device)

    summary(pg_model)
    logger.info("Parameters INFO:")
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info("GO_term num:"+str(len(ds.balanced_go_list)))
    logger.info("Start Testing!")

    existed_result = dest_ds.propagated_go
    predict_result = {}
    dest_ds.valid_mode()
    actual = []
    pred = []
    with torch.no_grad():
        for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(test_loader):
            predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx)
            # predict_matrix: [batch_size, go_nums]
            predict_matrix = probablity_propagate(predict_matrix, go_graph_nx, topo_order, ds.balanced_go_map_inv)
            pred += predict_matrix.tolist()
            actual += labels.tolist()
            
            predict_matrix = predict_matrix.cpu().numpy()
            filtered_go = np.argwhere(predict_matrix > thresh)
            if len(filtered_go) == 0:
                continue
            assert len(filtered_go[0]) == 2
            for pos in filtered_go:
                i,j = pos
                pid = pids[i]
                go = go_terms[ds.balanced_go_list[j]]['id']
                if pid not in predict_result:
                    predict_result[pid] = []
                predict_result[pid].append((predict_matrix[i][j], go))
    auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual, thresh)
    logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
    
    # 计算交, 统计假阴性率, 平均新标签数, 综合预测概率与标签深度, 推荐新预测出的标签
    iter_rate = 0.8
    real_precision_score = 0
    real_precision_cnt = 0
    real_recall_score = 0
    real_recall_cnt = 0
    Ave_new_cnt = 0
    recommend_cnt = 10
    recommended_result = {}
    for pid in tqdm(existed_result,desc="calculating recommended go"):
        A_go = set(existed_result[pid])
        if pid not in predict_result:
            logger.error("Protein {} not in prediction".format(pid))
            continue
        ITS_go = A_go.intersection(set([x[1] for x in predict_result[pid]]))
        ITS_go_id = [go_id_map[go] for go in ITS_go]
        go_id_dis = bfs_get_dis_go(ITS_go_id, go_graph_nx)
        go_dis = {go_terms[go]['id']:dis for go,dis in go_id_dis.items()}
        rec_go_list = []
        for raw_score, go in predict_result[pid]:
            if go not in ITS_go:
                rec_go_list.append((raw_score*pow(iter_rate,go_dis[go]), go))
        if len(A_go) > 0:
            real_recall_score += len(ITS_go)/len(A_go)
            real_recall_cnt += 1
        if len(predict_result[pid]) > 0:
            real_precision_score += len(ITS_go)/len(predict_result[pid])
            real_precision_cnt += 1
        Ave_new_cnt += len(rec_go_list)
        rec_go_list.sort(reverse=True)
        recommended_result[pid] = rec_go_list[:recommend_cnt]
    real_recall_score /= real_recall_cnt
    real_precision_score /= real_precision_cnt
    Ave_new_cnt /= len(recommended_result)
    logger.info("Real Average precision: {:.6f} , Real Average recall: {:.6f} , Average New GO: {:.6f}".format(real_precision_score, real_recall_score, Ave_new_cnt))
        
    with open(Path(result_dir,"{}_all_result.pkl".format(ont)),"wb") as f:
        pickle.dump(predict_result,f)
    predict_result = {pid:[go for p,go in pgos] for pid,pgos in predict_result.items()}
    with open(Path(result_dir,"{}_all_result.json".format(ont)),"w") as f:
        json.dump(predict_result,f,indent=4)
    with open(Path(result_dir,"{}_recommand_result.pkl".format(ont)),"wb") as f:
        pickle.dump(recommended_result,f)

if __name__ == '__main__':
    main()