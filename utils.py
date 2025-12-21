import os
import json
import logging
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

import mlflow

def init_logger(log_path):
    logger = logging.getLogger("global")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=log_path)
    logger.setLevel(logging.INFO)
    handler1.setLevel(logging.INFO)
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def cal_metrics(pred, actual, thresh=None):
    actual = np.array(actual).flatten()
    pred = np.array(pred).flatten()
    fpr, tpr, th = roc_curve(actual, pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    precision, recall, pr_thresh = metrics.precision_recall_curve(actual, pred)
    aupr_score = metrics.auc(recall, precision)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    if thresh is None:
        thresh = pr_thresh[np.argmax(f1)]
        precision = precision[np.argmax(f1)]
        recall = recall[np.argmax(f1)]
        fmax = np.max(f1)
    else:
        index = np.searchsorted(pr_thresh, thresh)
        precision = precision[index]
        recall = recall[index]
        fmax = f1[index]
    return auc_score, aupr_score, precision, recall, fmax, thresh

def cal_metrics_special(pred, actual, special_idx, thresh=None):
    pred = np.array(pred)[:,special_idx]
    actual = np.array(actual)[:,special_idx]
    return cal_metrics(pred, actual, thresh)

# 注意：需要传入已经进行标签传递后的pred值
# pred和actual都是二维list，punishment是一维list
def cal_Pmin(pred, actual, punishment, thresh_num=100):

    thresh_num += 1
    # 假阳性与假阴性标签的惩罚值之和, 按照thresh统计
    fp_sum_list = [0] * thresh_num
    fn_sum_list = [0] * thresh_num
    for pred_i, actual_i in tqdm(list(zip(pred,actual)),desc="Calculating Pmin..."):
        tmp_list = list(zip(pred_i,actual_i,punishment))
        tmp_list.sort()
        # 初始阈值为 0，此时所有阴性标签均为假阳性, 所有阳性标签都为真阳性
        fp_sum = 0
        fn_sum = 0
        for pred_ij, actual_ij, p_j in tmp_list:
            if actual_ij <= 0.5:
                fp_sum += p_j
        j = 0
        for t in range(thresh_num):
            thresh = 1.0*t/thresh_num
            while j < len(tmp_list) and tmp_list[j][0]<=thresh:
                # 根据当前阈值，新加入标签到预测阴性集合
                # 若该标签实际为阳性，则对fn造成贡献
                # 若该标签实际为阴性，则去除它在fp中的贡献
                if tmp_list[j][1] <= 0.5:
                    fp_sum -= tmp_list[j][2]
                else:
                    fn_sum += tmp_list[j][2]
                j += 1
            fp_sum_list[t] += fp_sum
            fn_sum_list[t] += fn_sum
    
    fp_ave = np.array(fp_sum_list)/len(actual)
    fn_ave = np.array(fn_sum_list)/len(actual)
    Pmin = np.min(np.sqrt(fp_ave * fp_ave + fn_ave * fn_ave))

    return Pmin

def cal_metrics_all(pred, actual, go_idx_special, go_punish, logger, mlf_name, thresh=None):
    auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics(pred, actual, thresh)
    logger.info("Normal Metrics")
    logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
    mlflow.log_metric(f"{mlf_name}_AUC", auc_score)
    mlflow.log_metric(f"{mlf_name}_AUPR", aupr_score)
    mlflow.log_metric(f"{mlf_name}_Fmax", fmax)
    
    auc_score, aupr_score, precision, recall, fmax, tmp_thresh = cal_metrics_special(pred, actual, go_idx_special, thresh)
    logger.info("Special GO terms Metrics")
    logger.info("AUC: {:.4f} AUPR: {:.4f} PRC: {:.4f} REC: {:.4f}  Fmax: {:.4f} tmp_thresh: {:.4f}".format(auc_score, aupr_score, precision, recall, fmax, tmp_thresh))
    mlflow.log_metric(f"{mlf_name}_spec_AUC", auc_score)
    mlflow.log_metric(f"{mlf_name}_spec_AUPR", aupr_score)
    mlflow.log_metric(f"{mlf_name}_spec_Fmax", fmax)

    Pmin = cal_Pmin(pred, actual, go_punish)
    logger.info("Pmin: {:.8f}".format(Pmin))
    mlflow.log_metric(f"{mlf_name}_Pmin", Pmin)

def cal_effect(ori_pred, glob_pred, part_pred):
    # 计算消除glob后，结果的变化
    glob_diff = torch.abs(part_pred - ori_pred)
    # 计算消除part后，结果的变化
    part_diff = torch.abs(glob_pred - ori_pred)
    # 计算两者分别的结果的总影响
    glob_effect = glob_diff.mean().item()
    part_effect = part_diff.mean().item()
    # 计算两者对结果的特有影响
    glob_unique = F.relu(glob_diff - part_diff).mean().item()
    part_unique = F.relu(part_diff - glob_diff).mean().item()
    return torch.tensor([glob_effect, glob_unique, part_effect, part_unique])

def cal_pert_effect(ori_pred, glob_pert, part_pert):
    # 计算扰动glob后，结果的变化
    glob_diff = glob_pert - ori_pred
    # 计算扰动part后，结果的变化
    part_diff = part_pert - ori_pred
    # 计算两者分别变化的方差
    glob_effect = glob_diff.var().item()
    part_effect = part_diff.var().item()
    return torch.tensor([glob_effect, part_effect])

# 克隆config之后，只读取workspace中本地的config，便于记录每次的参数以单独调试
def load_config_clone(default_conf_path, clone_dir_path):
    clone_conf_path = Path(clone_dir_path,default_conf_path.name)
    if not Path.exists(clone_conf_path):
        os.system(f"cp {default_conf_path.resolve()} {clone_conf_path.resolve()}")
    with open(clone_conf_path,"r") as f:
        param = json.load(f)
    return param

def generate_mask(batch_size, seq_len, rate):
    mask = ((torch.rand([batch_size,seq_len])) > rate).float()
    return mask

def generate_graph_mask(node_num, rate):
    mask = ((torch.rand([node_num])) > rate).float()
    return mask

def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_split_dataset(idx_dir, ont, ds, logger, forced_test_pid_list=[], train_p=0.9, valid_p=0.05, use_cache=True):
    train_idx_path = Path(idx_dir,f"{ont}_train_indices.npy")
    valid_idx_path = Path(idx_dir,f"{ont}_valid_indices.npy")
    test_idx_path = Path(idx_dir,f"{ont}_test_indices.npy")
    if (not Path.exists(train_idx_path)) or (not Path.exists(valid_idx_path)) or (not Path.exists(test_idx_path)) or use_cache==False:
        logger.warning("resplit the dataset")
        # 先将强制测试的样本先划分到测试集中，在按照比例划分剩余样本
        if len(forced_test_pid_list) > 0:
            forced_test_set = set(forced_test_pid_list)
            forced_test_indices = []
            remaining_indices = []
            if ont == "ppi":
                # 对ppi数据集划分(cold start) 
                for i, data in enumerate(ds):
                    if data[1][0] in forced_test_set or data[2][0] in forced_test_set:
                        forced_test_indices.append(i)
                    else:
                        remaining_indices.append(i)
            else:
                for i, data in enumerate(ds):
                    if data[1] in forced_test_set:
                        forced_test_indices.append(i)
                    else:
                        remaining_indices.append(i)

            remaining_size = len(remaining_indices)
            adjusted_train_size = int(train_p * remaining_size)
            adjusted_valid_size = int(valid_p * remaining_size)
            adjusted_test_size = remaining_size - adjusted_train_size - adjusted_valid_size
            random.shuffle(remaining_indices)
            train_indices = remaining_indices[:adjusted_train_size]
            valid_indices = remaining_indices[adjusted_train_size : adjusted_train_size+adjusted_valid_size]
            test_indices = remaining_indices[adjusted_train_size+adjusted_valid_size : ] + forced_test_indices
            train_dataset = Subset(ds, train_indices)
            valid_dataset = Subset(ds, valid_indices)
            test_dataset = Subset(ds, test_indices)
            np.save(train_idx_path, train_indices)
            np.save(valid_idx_path, valid_indices)
            np.save(test_idx_path, test_indices)
        else:
            train_size = int(train_p*len(ds))
            valid_size = int(valid_p*len(ds))
            test_size = len(ds) - train_size - valid_size
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(ds, [train_size, valid_size, test_size])
            np.save(train_idx_path, train_dataset.indices)
            np.save(valid_idx_path, valid_dataset.indices)
            np.save(test_idx_path, test_dataset.indices)
    else:
        train_indices = np.load(train_idx_path).tolist()
        valid_indices = np.load(valid_idx_path).tolist()
        test_indices = np.load(test_idx_path).tolist()
        if max(train_indices) > len(ds):
            logger.error(str(max(train_indices))+str(len(ds)))
            logger.error("Error: train_indices out of range!")
            exit(1)
        if max(valid_indices) > len(ds):
            logger.error(str(max(valid_indices))+str(len(ds)))
            logger.error("Error: valid_indices out of range!")
            exit(1)
        if max(test_indices) > len(ds):
            logger.error(str(max(test_indices))+str(len(ds)))
            logger.error("Error: test_indices out of range!")
            exit(1)
        train_dataset = Subset(ds, train_indices)
        valid_dataset = Subset(ds, valid_indices)
        test_dataset = Subset(ds, test_indices)
    return train_dataset, valid_dataset, test_dataset