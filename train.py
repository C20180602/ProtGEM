'''
GO预测模型训练
e.g.
python train.py -o bp -s xxx
'''

import os
import json
import math
import pickle
import click
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import dgl
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from register import *
from model.dataset import MyDataset
from model.go_dataset import GoDataset
from model.predictor import ReconstructDecoder
from model.loss import ContrastLoss, GoContrastLoss, SequenceLoss, GeometricStructureLoss
from model.training_stability import backward_and_step, numerical_error_context, require_finite
from utils import init_logger, cal_metrics, load_config_clone, get_split_dataset, generate_mask, generate_graph_mask, set_seeds, create_optimizer

import mlflow

os.chdir(os.path.dirname(os.path.abspath(__file__)))

@click.command()
@click.option('-o', '--ont', type=click.Choice(['bp', 'cc', 'mf', 'all']), default='mf')
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
@click.option('-t', '--model-type', type=click.Choice(['v1.1','v2','v3','seq_abl','struct_abl','domain_abl','enhanced_v1','enhanced_v2']), default='v1.1')
@click.option('-c', '--continue-flag', type=click.BOOL, default=False) # whether to use the previous param to train
@click.option('-f', '--freeze', type=click.BOOL, default=False) # freeze the pretrained go model (only active while preg exist)
@click.option('-li', '--light-mode', type=click.BOOL, default=False) # ESM preprocess free, but costs longer time and larger dedicated memory
@click.option('-tpl', '--forced-test-pid-list-path', type=click.STRING, default=None) # set the forced_test_pid_list in dataset, to specify the test set
@click.option('-preg', '--pretrain-model-path', type=click.STRING, default=None) # model_params/go_pretrain_model_1024.pth
@click.option('--max-grad-norm', type=float, default=None, help='Gradient clipping threshold; config value or 1.0 by default.')
@click.option('--detect-anomaly', is_flag=True, help='Trace the forward operator responsible for an invalid backward result (slower).')
def main(ont, species_name, model_name, model_type,
         continue_flag, freeze, light_mode, forced_test_pid_list_path, pretrain_model_path,
         max_grad_norm, detect_anomaly):
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
    mlflow.set_experiment("GO_Prediction")
    mlflow.start_run(run_name=f"{ont}_{species_name}_{model_name}")
    
    # 初始化随机种子以便复现
    seed = np.random.randint(1e9)
    set_seeds(seed)
    
    # 输出输入的参数信息
    logger.info("----------------------------A new train process----------------------------")
    global_param = {
        "random seed":seed,
        "GO_type":ont,
        "species_name":species_name,
        "model_name":model_name,
        "model_type":model_type,
        "continue_flag":continue_flag,
        "freeze":freeze,
        "light_mode":light_mode,
        "forced_test_pid_list_path":forced_test_pid_list_path,
        "pretrain_model_path":pretrain_model_path,
        "detect_anomaly":detect_anomaly
    }
    logger.info("Input Parameters:")
    logger.info(global_param)
    mlflow.log_params(global_param)
    logger.info("----------------------------")

    # 读取模型配置参数
    pm_param = load_config_clone(default_pm_conf[model_type], config_dir)
    gm_param = load_config_clone(default_gm_conf[model_type], config_dir)
    pred_param = load_config_clone(default_pred_conf[model_type], config_dir)
    # 初始化模型
    pg_model, local_collate_fn = model_init(model_type, pm_param, gm_param, pred_param)
    if pg_model is None or local_collate_fn is None:
        logger.error("Model type not supported!")
        return
    if model_type not in ["enhanced_v1","enhanced_v2"]:
        if pretrain_model_path is not None:
            if not os.path.exists(pretrain_model_path):
                logger.error("Pretrained protein model path doesn't exist! "+pretrain_model_path)
            else:
                pg_model.gm.load_model(pretrain_model_path)
    # 在原模型基础上继续训练（用于欠拟合的情况）
    if continue_flag:
        if os.path.exists(Path(model_param_dir,"{}_model.pt".format(ont))):
            pg_model.load_model(Path(model_param_dir,"{}_model.pt".format(ont)))
            logger.info("Continue training with previous model parameters!")
        else:
            # 如果不存在, 则使用新模型训练, 不改变原有数据集cache
            logger.warning("Previous model parameters not found!")
    # 冻结GO模型
    if freeze:
        for param in pg_model.gm.parameters():
            param.requires_grad = False
    pg_model.to(device)

    # 读取要求的test蛋白质列表
    forced_test_pid_list = []
    if forced_test_pid_list_path is not None:
        with open(forced_test_pid_list_path,'r') as f:
            forced_test_pid_list = json.load(f)
        
    # 读取训练参数
    train_default_conf_path = Path("config","train.conf")
    train_param = load_config_clone(train_default_conf_path, config_dir)
    if max_grad_norm is None:
        max_grad_norm = train_param.get("max_grad_norm", 1.0)
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    train_param["max_grad_norm"] = max_grad_norm
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
    alpha4 = train_param["alpha4"]
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
    go_batch_size = train_go_param["batch_size"]
    pos_k = train_go_param["pos_k"]
    pos_max = train_go_param["pos_max"]
    neg_k = train_go_param["neg_k"]
    neg_max = train_go_param["neg_max"]
    pos_rate = train_go_param["pos_rate"]
    contrast_temp_go = train_go_param["contrast_temp"]
    
    # 初始化重构损失计算模型（解码器+损失计算）
    rd_model = None
    if model_type == "v1.1" and alpha4 > 0.0:
        rd_model = ReconstructDecoder(pg_model.pm.final_out_dim, pg_model.pm.part_dims)
        rd_model.to(device)
    # 各项评估指标初始化
    best_fmax = 0.0
    best_epoch = -1
    corr_auc = 0.0
    corr_aupr = 0.0
    corr_thresh = 0.0

    # 初始化数据集并进行划分
    ds = MyDataset(species_name, ont, cache_dir, target_device=device,
                   go_max_percent=go_max_percent, go_min_count=go_min_count,
                   use_cache=continue_flag, light_mode=light_mode)
    ds.show_dataset_info(logger)
    train_dataset, valid_dataset, test_dataset = get_split_dataset(idx_dir,ont,ds,logger, forced_test_pid_list=forced_test_pid_list,
                                                                   train_p=train_proportion, valid_p=valid_proportion, use_cache=continue_flag)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=local_collate_fn)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=local_collate_fn)

    go_ds = GoDataset(pos_k, pos_max, neg_k, neg_max, pos_rate)

    # 设置优化器、学习率规划器以及损失函数
    optimizer = create_optimizer(pg_model, rd_model, init_lr)
    named_train_parameters = [(f"pg_model.{name}", param) for name, param in pg_model.named_parameters()]
    if rd_model is not None:
        named_train_parameters += [(f"rd_model.{name}", param) for name, param in rd_model.named_parameters()]
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epoch_num*len(train_loader))

    criterion1 = ContrastLoss()
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_pos_weight],device=device))
    criterion3 = GoContrastLoss(pos_k, neg_k)
    criterion4 = SequenceLoss()
    criterion5 = GeometricStructureLoss()

    logger.info("Parameters INFO:")
    logger.info(train_param)
    logger.info(pm_param)
    logger.info(gm_param)
    logger.info(pred_param)
    
    mlflow.log_params(train_param)
    mlflow.log_params(pm_param)
    mlflow.log_params(gm_param)
    mlflow.log_params(pred_param)
    logger.info("----------------------------")
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
        max_observed_grad_norm = 0.0
        pred = []
        actual = []
        for i, (idxs, pids, seq_embs, batched_graph, domain_info, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            this_batch_size = seq_embs.shape[0]
            max_seq_len = seq_embs.shape[1]
            lens = batched_graph.batch_num_nodes().tolist()

            mask = generate_mask(this_batch_size, max_seq_len, mask_rate)
            graph_mask = generate_graph_mask(batched_graph.num_nodes(), mask_rate)
            mask = mask.to(device)
            graph_mask = graph_mask.to(device)

            with numerical_error_context(logger, f"train epoch={epoch} batch={i} pids={list(pids)}"), \
                    torch.autograd.set_detect_anomaly(detect_anomaly):
                # 处理不同model type的数据格式, 如果有重构损失(alpha4)则导出局部特征
                if model_type == "v1.1" and alpha4 > 0.0:
                    predict_matrix, protein_emb, protein_emb_token = pg_model(seq_embs, batched_graph, domain_info, go_idx, mask, graph_mask,
                                                ablation_feat=dropout_feat_list[epoch], get_protein_feat=True)
                else:
                    predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx, mask, graph_mask, ablation_feat=dropout_feat_list[epoch])
            
                require_finite({"predictions": predict_matrix})
                loss1 = criterion1(predict_matrix, labels, contrast_temp)
                loss2 = criterion2(predict_matrix, labels)
                loss3 = torch.tensor(0.0,device=device)
                if go_loss_switch and alpha3 > 0:
                    # 加入GO loss计算开销会变大, 只使用当前epoch中覆盖的标签进行对比学习
                    go_id_set = set()
                    for idx in idxs:
                        for go in ds.data_info[idx][5]:
                            go_id_set.add(go)
                    go_subset = Subset(go_ds, list(go_id_set))
                    go_dl = DataLoader(go_subset, batch_size=go_batch_size, shuffle=True)
                    for (idx, pos, neg) in go_dl:
                        x = torch.cat([idx,pos,neg],dim=1)
                        x = x.to(device)
                        x = pg_model.gm(x)
                        loss3 += criterion3(x, contrast_temp_go)
                    loss3 /= len(go_id_set)
            
                loss = alpha1*loss1 + alpha2*loss2 + alpha3*loss3

                if model_type == "v1.1" and alpha4 > 0.0:
                    # 重构任务的目标生成
                    re_seq_tgt = ds.generate_re_seq(pids, max_seq_len, device)
                    re_struct_tgt = ds.generate_re_struct(pids, max_seq_len, device)
                
                    seq_logits, coords_logits = rd_model(protein_emb, protein_emb_token, lens)
                    # 计算序列重构损失
                    re_loss_seq = criterion4(seq_logits, re_seq_tgt, mask)
                    # 计算结构重构损失
                    re_loss_struct = torch.tensor(0.0, device=device)
                    for coords_logit, coords_real in zip(coords_logits, re_struct_tgt):
                        re_loss_struct += criterion5(coords_logit, coords_real)
                
                    loss4 = re_loss_seq + re_loss_struct
                    loss += alpha4 * loss4
            
                require_finite({"contrastive loss": loss1, "BCE loss": loss2, "GO loss": loss3})
                grad_norm = backward_and_step(loss, optimizer, named_train_parameters, max_grad_norm)
                lr_scheduler.step()
                max_observed_grad_norm = max(max_observed_grad_norm, grad_norm.item())
            mean_loss += loss.item()
            mean_loss1 += loss1.item()
            mean_loss2 += loss2.item()
            mean_loss3 += loss3.item()
            if i % log_batch_count == log_batch_count-1:
                logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_loader)}, Loss: {mean_loss/log_batch_count}, Loss1: {mean_loss1/log_batch_count}, Loss2: {mean_loss2/log_batch_count}, Loss3: {mean_loss3/log_batch_count}')
                mlflow.log_metric("train_loss", mean_loss/log_batch_count, step=epoch*len(train_loader)+i)
                mlflow.log_metric("train_loss1", mean_loss1/log_batch_count, step=epoch*len(train_loader)+i)
                mlflow.log_metric("train_loss2", mean_loss2/log_batch_count, step=epoch*len(train_loader)+i)
                mlflow.log_metric("train_loss3", mean_loss3/log_batch_count, step=epoch*len(train_loader)+i)
                logger.info(f'Max gradient norm before clipping: {max_observed_grad_norm}, clip threshold: {max_grad_norm}')
                mlflow.log_metric("train_grad_norm_max", max_observed_grad_norm, step=epoch*len(train_loader)+i)
                
                mean_loss = 0.0
                mean_loss1 = 0.0
                mean_loss2 = 0.0
                mean_loss3 = 0.0
                max_observed_grad_norm = 0.0
            
            # 对于训练集预测结果，采样进行指标计算，节省时间
            if i % 30 == 29:
                pred += predict_matrix.tolist()
                actual += labels.tolist()
        logger.info("train set:")
        auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual)
        logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
        mlflow.log_metric("train_auc", auc_score, step=epoch)
        mlflow.log_metric("train_aupr", aupr_score, step=epoch)
        mlflow.log_metric("train_fmax", fmax, step=epoch)

        # validate
        ds.valid_mode()
        pg_model.eval()
        actual = []
        pred = []
        with torch.no_grad():
            for idxs, pids, seq_embs, batched_graph, domain_info, labels in tqdm(val_loader):
                with numerical_error_context(logger, f"valid epoch={epoch} pids={list(pids)}"):
                    predict_matrix = pg_model(seq_embs, batched_graph, domain_info, go_idx)
                    require_finite({"validation predictions": predict_matrix})
                # predict_matrix = (predict_matrix + 1)/2
                pred += predict_matrix.tolist()
                actual += labels.tolist()
        logger.info("valid set:")
        auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual)
        logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
        mlflow.log_metric("valid_auc", auc_score, step=epoch)
        mlflow.log_metric("valid_aupr", aupr_score, step=epoch)
        mlflow.log_metric("valid_fmax", fmax, step=epoch)

        if fmax > best_fmax:
            best_fmax = fmax
            best_epoch = epoch
            corr_auc = auc_score
            corr_aupr = aupr_score
            corr_thresh = tmp_thresh
            pg_model.save_model(Path(model_param_dir,"{}_model.pt".format(ont)))
        pg_model.train()
    
    logger.info(f"{species_name} {model_name} {ont} dataset training result: Valid")
    logger.info(f"best_epoch:{best_epoch} AUC:{corr_auc} best_Fmax:{best_fmax} AUPR:{corr_aupr} thresh:{corr_thresh}")

    mlflow.end_run()

if __name__ == '__main__':
    main()
