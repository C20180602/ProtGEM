'''
PPI预测模型训练
e.g.
python train_ppi.py -s xxx -prep workspace/Pretrain_dataset/xxx_model/model_param/all_pm_model.pt
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
import dgl
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from register import *
from model.dataset import PPIDataset
from utils import init_logger, cal_metrics, load_config_clone, get_split_dataset, generate_mask, generate_graph_mask, set_seeds

import mlflow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@click.command()
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_ppi_v1')
@click.option('-t', '--model-type', type=click.Choice(['ppi_v1']), default='ppi_v1')
@click.option('-c', '--continue-flag', type=click.BOOL, default=False) # whether to use the previous param to train
@click.option('-f', '--freeze', type=click.BOOL, default=False) # freeze the pretrained protein model
@click.option('-test', '--test_flag', type=click.BOOL, default=False) # only test
@click.option('-trs', '--transfer-ppi-flag', type=click.BOOL, default=False) # use raw/SPECIES/ppi/transferred_ppi_list.json to build ppi dataset
@click.option('-tpl', '--forced-test-pid-list-path', type=click.STRING, default=None) # set the forced_test_pid_list in dataset, to specify the test set
@click.option('-prep', '--pretrain-model-path', type=click.STRING, default=None)
@click.option('-th', '--thresh', type=click.FLOAT, default=None) # only test
def main(species_name, model_name, model_type,
         continue_flag, freeze, test_flag, transfer_ppi_flag, forced_test_pid_list_path, pretrain_model_path, thresh):
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
    logger = init_logger(Path(log_dir, 'train_ppi.log'))
    mlflow.set_experiment("PPI_Prediction")
    mlflow.start_run(run_name=f"ppi_{species_name}_{model_name}")
    
    # 初始化随机种子以便复现
    seed = np.random.randint(1e9)
    set_seeds(seed)
    
    # 输出输入的参数信息
    logger.info("----------------------------A new train process----------------------------")
    global_params = {
        "random seed": seed,
        "species_name": species_name,
        "model_name": model_name,
        "model_type": model_type,
        "continue_flag": continue_flag,
        "freeze": freeze,
        "test_flag": test_flag,
        "transfer_ppi_flag": transfer_ppi_flag,
        "forced_test_pid_list_path": forced_test_pid_list_path,
        "pretrain_model_path": pretrain_model_path,
        "thresh": thresh
    }
    logger.info("Input Parameters:")
    logger.info(global_params)
    mlflow.log_params(global_params)
    logger.info("----------------------------")

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    ppi_model, local_collate_fn = model_init(model_type, pm_param, None, pred_param)
    if ppi_model is None or local_collate_fn is None:
        logger.error("Model type not supported!")
        return
    if pretrain_model_path is not None:
        if not os.path.exists(pretrain_model_path):
            logger.error("Pretrained protein model path doesn't exist! "+pretrain_model_path)
        else:
            ppi_model.pm.load_model(pretrain_model_path)
    # 在原模型基础上继续训练（用于欠拟合的情况）
    if continue_flag:
        if os.path.exists(Path(model_param_dir,"ppi_model.pt")):
            ppi_model.load_model(Path(model_param_dir,"ppi_model.pt"))
            logger.info("Continue training with previous model parameters!")
        else:
            # 如果不存在, 则使用新模型训练, 不改变原有数据集cache
            logger.warning("Previous model parameters not found!")
    # 冻结参数
    if freeze:
        for param in ppi_model.pm.parameters():
            param.requires_grad = False
    ppi_model.to(device)
    # 读取要求的test蛋白质列表
    forced_test_pid_list = []
    if forced_test_pid_list_path is not None:
        with open(forced_test_pid_list_path,'r') as f:
            forced_test_pid_list = json.load(f)
        
    # 读取训练参数
    train_default_conf_path = Path("config","train_ppi.conf")
    train_param = load_config_clone(train_default_conf_path, config_dir)
    train_proportion = train_param["train_proportion"]
    valid_proportion = train_param["valid_proportion"]
    batch_size = train_param["batch_size"]
    init_lr = train_param["init_lr"]
    contrast_temp = train_param["contrast_temp"]
    mask_rate = train_param["mask_rate"]
    go_diff_weight = train_param["go_diff_weight"]
    ppi_weight = train_param["ppi_weight"]
    go_weight = train_param["go_weight"]
    pick_neg_rate = train_param["pick_neg_rate"]
    random_neg_rate = train_param["random_neg_rate"]
    alpha1 = train_param["alpha1"]
    alpha2 = train_param["alpha2"]
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

    # 各项评估指标初始化
    best_fmax = 0.0
    best_epoch = -1
    corr_auc = 0.0
    corr_aupr = 0.0
    corr_thresh = 0.0

    # 初始化数据集并进行划分
    cache_flag = continue_flag or test_flag
    ppi_file_name = "transferred_ppi_list.json" if transfer_ppi_flag else "ppi_list.json"
    ds = PPIDataset(species_name, cache_dir, target_device=device, specified_ppi_file_name=ppi_file_name,
                    go_diff_weight=go_diff_weight, ppi_weight=ppi_weight, go_weight=go_weight,
                    pick_neg_rate=pick_neg_rate, random_neg_rate=random_neg_rate,
                    use_cache=cache_flag)
    ds.show_dataset_info(logger)
    # 对forced test的蛋白质，先驱逐出整个数据集,cold start
    train_dataset, valid_dataset, test_dataset = get_split_dataset(idx_dir,"ppi",ds,logger, forced_test_pid_list=forced_test_pid_list,
                                                                   train_p=train_proportion,valid_p=valid_proportion,use_cache=cache_flag)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=local_collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)
    logger.info("Dataset Size:")
    logger.info("train:{} val:{} test:{}".format(len(train_loader),len(val_loader),len(test_loader)))
    # 设置优化器、学习率规划器以及损失函数
    optimizer = torch.optim.Adam(ppi_model.parameters(), lr=init_lr)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

    criterion1 = ContrastLoss()
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight],device=device))

    logger.info("Parameters INFO:")
    logger.info(train_param)
    logger.info(pm_param)
    logger.info(pred_param)
    mlflow.log_params(train_param)
    mlflow.log_params(pm_param)
    mlflow.log_params(pred_param)
    if not test_flag:
        logger.info("Start Training!")
        for epoch in range(epoch_num):
            # train
            if epoch==0 or dropout_feat_list[epoch] != dropout_feat_list[epoch-1]:
                logger.info(f"Dropout_feat: {dropout_feat_list[epoch]}")
            ds.train_mode()
            mean_loss = 0.0
            mean_loss1 = 0.0
            mean_loss2 = 0.0
            pred = []
            actual = []
            for i, (idxs,
                    (pids1, seq_embs1, batched_graph1, domain_info1),
                    (pids2, seq_embs2, batched_graph2, domain_info2),
                    labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                mask1 = generate_mask(seq_embs1.shape[0], seq_embs1.shape[1], mask_rate)
                graph_mask1 = generate_graph_mask(batched_graph1.num_nodes(), mask_rate)
                mask1 = mask1.to(device)
                graph_mask1 = graph_mask1.to(device)
                
                mask2 = generate_mask(seq_embs2.shape[0], seq_embs2.shape[1], mask_rate)
                graph_mask2 = generate_graph_mask(batched_graph2.num_nodes(), mask_rate)
                mask2 = mask2.to(device)
                graph_mask2 = graph_mask2.to(device)
                
                predict_matrix = ppi_model(seq_embs1, batched_graph1, domain_info1,
                                        seq_embs2, batched_graph2, domain_info2,
                                        mask1, graph_mask1, mask2, graph_mask2)
                if int(torch.sum(labels).item()) != 0:
                    loss1 = criterion1(predict_matrix, labels, contrast_temp)
                else:
                    loss1 = torch.tensor([0.0], device=device)
                loss2 = criterion2(predict_matrix, labels)
                loss = alpha1 * loss1 + alpha2 * loss2
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                mean_loss += loss.item()
                mean_loss1 += loss1.item()
                mean_loss2 += loss2.item()
                if i % log_batch_count == log_batch_count-1:
                    logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {mean_loss/log_batch_count}, Loss1: {mean_loss1/log_batch_count}, Loss2: {mean_loss2/log_batch_count}')
                    mlflow.log_metric("train_loss", mean_loss/log_batch_count, step=epoch*len(train_loader)+i)
                    mlflow.log_metric("train_loss1", mean_loss1/log_batch_count, step=epoch*len(train_loader)+i)
                    mlflow.log_metric("train_loss2", mean_loss2/log_batch_count, step=epoch*len(train_loader)+i)
                    
                    mean_loss = 0.0
                    mean_loss1 = 0.0
                    mean_loss2 = 0.0
                
                # 对于训练集预测结果，采样进行指标计算，节省时间
                if i % 50 == 49:
                    pred += predict_matrix.tolist()
                    actual += labels.tolist()
            logger.info("train set:")
            auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual)
            logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
            mlflow.log_metric("train_AUC", auc_score, step=epoch)
            mlflow.log_metric("train_AUPR", aupr_score, step=epoch)
            mlflow.log_metric("train_Fmax", fmax, step=epoch)

            # validate
            ds.valid_mode()
            ppi_model.eval()
            actual = []
            pred = []
            with torch.no_grad():
                for i, (idxs,
                    (pids1, seq_embs1, batched_graph1, domain_info1),
                    (pids2, seq_embs2, batched_graph2, domain_info2),
                    labels) in enumerate(val_loader):
                    
                    predict_matrix = ppi_model(seq_embs1, batched_graph1, domain_info1,
                                            seq_embs2, batched_graph2, domain_info2)
                    pred += predict_matrix.tolist()
                    actual += labels.tolist()
            logger.info("valid set:")
            auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual)
            logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
            mlflow.log_metric("valid_AUC", auc_score, step=epoch)
            mlflow.log_metric("valid_AUPR", aupr_score, step=epoch)
            mlflow.log_metric("valid_Fmax", fmax, step=epoch)
            
            if fmax > best_fmax:
                best_fmax = fmax
                best_epoch = epoch
                corr_auc = auc_score
                corr_aupr = aupr_score
                corr_thresh = tmp_thresh
                ppi_model.save_model(Path(model_param_dir,"ppi_model.pt"))
            ppi_model.train()
        
        logger.info(f"{species_name} {model_name} ppi dataset training result: Valid")
        logger.info(f"best_epoch:{best_epoch} AUC:{corr_auc} best_Fmax:{best_fmax} AUPR:{corr_aupr} thresh:{corr_thresh}")
    
    # test
    logger.info("Start Testing!")
    ds.valid_mode()
    ppi_model.eval()
    actual = []
    pred = []
    with torch.no_grad():
        for i, (idxs,
            (pids1, seq_embs1, batched_graph1, domain_info1),
            (pids2, seq_embs2, batched_graph2, domain_info2),
            labels) in enumerate(test_loader):
            
            predict_matrix = ppi_model(seq_embs1, batched_graph1, domain_info1,
                                       seq_embs2, batched_graph2, domain_info2)
            # predict_matrix = (predict_matrix + 1)/2
            pred += predict_matrix.tolist()
            actual += labels.tolist()
    logger.info("test set:")
    if test_flag:
        auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual, thresh)
    else:
        auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual, corr_thresh)
    logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
    mlflow.log_metric("test_AUC", auc_score)
    mlflow.log_metric("test_AUPR", aupr_score)
    mlflow.log_metric("test_Fmax", fmax)

    mlflow.end_run()

if __name__ == '__main__':
    main()