'''
预测所有可能的PPI对并导出为json文件
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
from model.dataset import UpstreamDataset, upstream_collate_fn, PPIDataset
from model.predictor import PPIModel

from register import *
from utils import init_logger, cal_metrics, load_config_clone, set_seeds

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@click.command()
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_ppi_v1')
@click.option('-pn', '--precise-model-name', type=click.STRING, default=None)
@click.option('-t', '--model-type', type=click.Choice(['ppi_v1']), default='ppi_v1')
@click.option('-thresh', '--thresh', type=click.FLOAT, default=None)
def main(species_name, model_name, precise_model_name, model_type, thresh):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    model_param_dir = Path(workspace_dir,"model_param")
    result_dir = Path(workspace_dir,"result")
    os.makedirs(result_dir, exist_ok=True)

    # 设置设备编号(需要使用CUDA_VISIBLE_DEVICES设置)
    device = torch.device("cuda:0")
    
    # 初始化日志
    logger = init_logger(Path(log_dir, 'ppi_infer.log'))
    
    # 初始化随机种子以便复现
    seed = np.random.randint(1e9)
    set_seeds(seed)
    
    # 输出输入的参数信息
    logger.info("----------------------------A new inference process----------------------------")
    logger.info("random seed: {}".format(seed))
    logger.info("Input args:")
    logger.info("species_name: {}".format(species_name))
    logger.info("model_name: {}".format(model_name))
    logger.info("precise_model_name: {}".format(precise_model_name))
    logger.info("model_type: {}".format(model_type))
    logger.info("thresh: {}".format(thresh))
    logger.info("----------------------------")

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    ppi_model, _ = model_init(model_type,pm_param,None,pred_param)
    if precise_model_name is not None:
        ppi_model.load_model(Path(model_param_dir,precise_model_name))
    else:
        ppi_model.load_model(Path(model_param_dir,"ppi_model.pt"))
    ppi_model.to(device)
    for p in ppi_model.parameters():
        p.requires_grad = False
    ppi_model.eval()
    
    # 读取测试集
    ds = UpstreamDataset(species_name, cache_dir, target_device=device, inference_flag=True, use_cache=False)
    ppi_ds = PPIDataset(species_name, cache_dir, target_device=device, ignore_go=True, use_cache=True)
    ds.show_dataset_info(logger)
    test_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=upstream_collate_fn)

    logger.info("Parameters INFO:")
    logger.info(pm_param)
    logger.info("Start Testing!")
    
    # 分两步，先计算上游的蛋白质嵌入，再计算下游的PPI预测
    pid_list = []
    emb_list = []
    with torch.no_grad():
        for idxs, pids, seq_embs, batched_graph, domain_info in tqdm(test_loader, desc="Step 1: get protein embeddings"):
            emb = ppi_model.upstream_infer(seq_embs, batched_graph, domain_info)
            # emb: [batch_size, go_nums]
            i = 0
            for pid in pids:
                pid_list.append(pid)
                emb_list.append(emb[i].squeeze())
                i += 1
    
    pid_id_map = {pid:i for i,pid in enumerate(pid_list)}
    existed_result = ppi_ds.export_all_posivite_pairs()
    actual_np = np.zeros((len(pid_list),len(pid_list)))
    for p1 in existed_result:
        if p1 not in pid_id_map:
            continue
        id1 = pid_id_map[p1]
        for p2 in existed_result[p1]:
            if p2 not in pid_id_map:
                continue
            id2 = pid_id_map[p2]
            actual_np[id1][id2] = 1.0
    pred_np = []
    predict_result = {}
    all_emb = torch.stack(emb_list,dim=0)
    with torch.no_grad():
        for pid,emb in tqdm(zip(pid_list,emb_list),desc="Step 2: inference all pairs"):
            one_emb = emb.unsqueeze(0).repeat(len(pid_list),1)
            predict_matrix = ppi_model.downstream_infer(one_emb,all_emb)
            pred_np.append(predict_matrix.tolist())
            
            predict_matrix = predict_matrix.cpu().numpy()
            filtered_ppi = np.argwhere(predict_matrix > thresh)
            if len(filtered_ppi) == 0:
                continue
            assert len(filtered_ppi[0]) == 1
            predict_result[pid] = []
            for pos in filtered_ppi:
                pid2 = pid_list[int(pos[0])]
                predict_result[pid].append((predict_matrix[pos],pid2))
    pred_np = np.array(pred_np)
    
    pred = []
    actual = []
    pair_list = ppi_ds.export_all_pairs()
    for p1,p2 in pair_list:
        if p1 not in pid_id_map or p2 not in pid_id_map:
            continue
        id1,id2 = pid_id_map[p1],pid_id_map[p2]
        pred.append(pred_np[id1][id2])
        actual.append(actual_np[id1][id2])
    
    auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual, thresh)
    logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
    
    # 计算交, 统计假阴性率, 平均新标签数, 综合预测概率与标签深度, 推荐新预测出的标签
    iter_rate = 0.8
    real_precision_score = 0
    real_precision_cnt = 0
    real_recall_score = 0
    real_recall_cnt = 0
    Ave_new_cnt = 0
    recommend_cnt = 500
    recommended_result = {}
    for pid1 in tqdm(predict_result,desc="calculating recommended ppi"):
        predict_result[pid1].sort(reverse=True)
        if pid1 in existed_result:
            A_ppi = existed_result[pid1]
        else:
            A_ppi = set()
        ITS_ppi = A_ppi.intersection(set([x[1] for x in predict_result[pid1]]))
        rec_ppi_list = []
        for raw_score,pid2 in predict_result[pid1]:
            if pid2 not in ITS_ppi:
                rec_ppi_list.append((raw_score, pid2))
        rec_ppi_list.sort(reverse=True)
        if len(A_ppi) > 0:
            real_recall_score += len(ITS_ppi)/len(A_ppi)
            real_recall_cnt += 1
        if len(predict_result[pid1]) > 0:
            real_precision_score += len(ITS_ppi)/len(predict_result[pid1])
            real_precision_cnt += 1
        Ave_new_cnt += len(rec_ppi_list)
        recommended_result[pid1] = rec_ppi_list[:recommend_cnt]
    real_recall_score /= real_recall_cnt
    real_precision_score /= real_precision_cnt
    Ave_new_cnt /= len(recommended_result)
    logger.info("Real Average recall: {:.6f} , Real Average precision: {:.6f} , Average New PPI: {:.6f}".format(real_recall_score, real_precision_score, Ave_new_cnt))
        
    with open(Path(result_dir,"ppi_all_result.pkl"),"wb") as f:
        pickle.dump(predict_result,f)
    predict_result = {pid1:[pid2 for score,pid2 in ppis] for pid1,ppis in predict_result.items()}
    with open(Path(result_dir,"ppi_all_result.json"),"w") as f:
        json.dump(predict_result,f,indent=4)
    with open(Path(result_dir,"ppi_recommand_result.pkl"),"wb") as f:
        pickle.dump(recommended_result,f)

if __name__ == '__main__':
    main()