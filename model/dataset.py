import os
import json

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, species_name):
        # GO数据与interpro数据, 较小直接读入
        # seq_emb, struct较大，采用懒加载
        
        # 设置工作目录为项目文件夹
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
        with open(os.path.join("processed_data",species_name+"_protein_info.json"), 'r') as f:
            self.protein_info = json.load(f)
        with open(os.path.join("processed_data","go_map.json"), 'r') as f:
            self.vocab_go = json.load(f)
            
        with open(os.path.join("processed_data","interpro_id.json"), 'r') as f:
            interpro_list = json.load(f)
            self.vocab_interpro = {}
            for i,interpro in enumerate(interpro_list):
                self.vocab_interpro[interpro] = i
        
        species_path = os.path.join("dataset",species_name)
        protein_list = os.listdir(species_path)
        self.data_info = []
        for pname in protein_list:
            seq_path = os.path.join(species_path,pname,"seq_emb.pkl")
            struct_path = os.path.join(species_path,pname,"struct.dgl")
            
            # 当一个数据在序列, 结构, GO标签都存在时, 将其加入数据集
            if os.path.exists(seq_path) and os.path.exists(struct_path) and self.protein_info["pname"]["go"] != []:
                plen = len(self.protein_info["pname"]["seq"])
                # 保存数据为: (序列长度, 蛋白质名, 序列嵌入路径, 结构图路径, interpro标签, GO标签(TODO 传播后))
                self.data_info.append((plen, pname,
                                       os.path.abspath(seq_path), os.path.abspath(struct_path),
                                       self.protein_info["pname"]["domain"], self.protein_info["pname"]["go"]))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = self.load_data(self.file_paths[idx])
        label = self.load_label(self.file_paths[idx])
        return data, label

    def load_data(self, file_path):
        
        return 

    def load_label(self, file_path):
        
        return 

    
