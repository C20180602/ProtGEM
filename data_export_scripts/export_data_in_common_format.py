import os
import sys
import json
import pickle
import click
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import torch.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.dataset import MyDataset, my_collate_fn, my_collate_fn_v2
from model.predictor import ProteinGoModel, ProteinGoModelv2, ContrastLoss
from utils import init_logger, cal_metrics, load_config_clone, get_split_dataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('..')
default_pm_conf = {"v1.1":Path("config","ProteinEncoder.conf"), "v2":Path("config","ProteinEncoder2.conf")}
default_gm_conf = {"v1.1":Path("config","GoEncoder.conf"), "v2":Path("config","GoEncoder2.conf")}

def generate_mask(batch_size, seq_len, rate):
    mask = ((torch.rand([batch_size,seq_len])) > rate).float()
    return mask

@click.command()
@click.option('-s', '--species-name', type=click.STRING, default='HUMAN')
@click.option('-n', '--model-name', type=click.STRING, default='model_v1.1')
def main(species_name, model_name):
    workspace_dir = Path("workspace",species_name,model_name)
    log_dir = Path(workspace_dir,"log")
    idx_dir = Path(workspace_dir,"idx")
    cache_dir = Path(workspace_dir,"cache")
    config_dir = Path(workspace_dir,"config")
    export_dir = Path(workspace_dir,"export")
    os.makedirs(export_dir,exist_ok=True)
    
    device = torch.device("cpu")

    # 初始化日志
    logger = init_logger(Path(log_dir, 'tmp.log'))

    # 初始化模型
    local_collate_fn = my_collate_fn
        
    # 读取数据集参数
    train_default_conf_path = Path("config","train.conf")
    train_param = load_config_clone(train_default_conf_path, config_dir)
    go_max_percent = train_param["go_max_percent"]
    go_min_count = train_param["go_min_count"]
    
    with open(Path("processed_data","go_terms.json"),"r") as f:
        go_terms = json.load(f)
    
    all_pid_set = set()
    for ont in ['bp','cc','mf']:
        ds = MyDataset(species_name, ont, cache_dir, target_device=device, go_max_percent=go_max_percent, go_min_count=go_min_count, use_cache=True)
        ds.show_dataset_info(logger)
        ds.valid_mode()
        train_dataset, valid_dataset, test_dataset = get_split_dataset(idx_dir,ont,ds,logger)
        dl = {}
        dl['train'] = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=local_collate_fn)
        dl['valid'] = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=local_collate_fn)
        dl['test'] = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=local_collate_fn)

        go_idx = torch.tensor(ds.balanced_go_list).to(device)
        balanced_go_list = ds.balanced_go_list
        for tag in dl:
            pid_set = set()
            pid_go_dict = {}
            with open(Path(export_dir,f"{ont}_{tag}_go.txt"),"w") as f:
                for _, pids, _, _, _, labels in tqdm(dl[tag]):
                    pid = pids[0]
                    pid_set.add(pid)
                    all_pid_set.add(pid)
                    indices = torch.nonzero(labels[0] == 1.0)
                    for i in indices:
                        go = go_terms[balanced_go_list[i]]['id']
                        f.write(f"{pid}\t{go}\t{ont}\n")
            with open(Path(export_dir,f"{ont}_{tag}_pid_list.pkl"),"wb") as f:
                pickle.dump(list(pid_set),f)
    with open(Path(export_dir,"pid_list.pkl"),"wb") as f:
        pickle.dump(list(all_pid_set),f)
    with open(Path(export_dir,"pid_list.json"),"w") as f:
        json.dump(list(all_pid_set),f)
    pid2esm = {}
    for pid in all_pid_set:
        esm_path = os.path.abspath(os.path.join("dataset",species_name,pid,"seq_emb.pkl"))
        if os.path.exists(esm_path):
            pid2esm[pid] = esm_path
    with open(Path(export_dir,"pid2esm.pkl"),"wb") as f:
        pickle.dump(pid2esm,f)


if __name__ == '__main__':
    main()