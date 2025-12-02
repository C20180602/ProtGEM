import os
import json
import logging
import numpy as np
import random
from pathlib import Path

import torch
import torch.utils.data
from torch.utils.data import Subset
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

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