# 复现流程
## Pipeline
![本地图片](../asset/pipeline.drawio.png)
## 数据下载以及准备流程
- 在GeneOntology [https://geneontology.org/docs/download-ontology/]数据库上下载go-basic.obo，取得GO标签的基础信息与关系。保存至raw_data文件夹下。

- 在InterPro [https://www.ebi.ac.uk/interpro/download/]数据库上下载InterPro entry list和InterPro-GO mapping，获取InterPro标签的基础信息。保存至raw_data文件夹下。

- 在raw_data下建立物种文件夹SPECIES_NAME，并在下方建立四个子文件夹domain, go, pdb, ppi

- 在物种文件夹下，新建pid_list.json，内容格式为：(Uniprot ID 1表示蛋白质的Uniprot数据库编号)
``` json
[
    "Uniprot ID 1",
    "Uniprot ID 2",
    "Uniprot ID 3",
    ...
]
```

### pdb数据准备
- 使用alphafold_pdb_downloader.py读取pid列表下载PDB文件，使用 -s参数传递物种名称。

### InterPro（domain）数据准备
- interpro_domain_downloader.py通过pid列表下载interpro文件（可同时进行）

### GO标签数据准备（TODO：自动化GO标签获取）
- 根据pid列表，使用uniprot的id_mapping功能，完成映射后，下载tsv文件，里面勾选只勾选GO标签编号，将下载的tsv文件命名为go.tsv，放置在raw_data/SPECIES_NAME/go文件夹下。

- 如果你有已经整理好的数据，可以直接使用json导入，文件名为pgo_dict.json，json格式为如下，位置也放置在raw_data/SPECIES_NAME/go文件夹下即可。
``` json
{
    "Uniprot ID": [
        "GO:XXXXXXX",
        "GO:XXXXXXX",
        "GO:XXXXXXX",
        ...
    ],
    "Uniprot ID": [
        "GO:XXXXXXX"
    ],
    ...
}
```
- 注意：若使用第二种方式读取数据，会自动忽略第一种格式的数据。

### PPI数据准备
- 为了避免数据处理逻辑过于复杂，建议在外部处理好数据再转为json格式导入，json文件名为ppi_list.json，放置在raw_data/SPECIES_NAME/ppi文件夹下，格式如下：

- 注意：文件中只保留ppi的正样本数据。
``` json
[
    [
        "PPI1-Uniprot ID 1",
        "PPI1-Uniprot ID 2"
    ],
    [
        "PPI2-Uniprot ID 1",
        "PPI2-Uniprot ID 2"
    ],
    ...
]
```



## 数据处理
1、运行data_process_scripts下的process_go_basic.py、process_interpro.py、process_protein_new.py

2、运行pretrain/seq_esm_new.py和data_process_scripts/build_pdb_graph_new.py，顺序可任意
## 模型训练
### GO模型训练与测试
1、运行train_go_emb.py，预训练go的初始嵌入，得到go_pretrain_model模型，保存在model_params文件夹中

2、运行train.py，训练protein_model模型和go_model模型，输出内容在workspace_{species_name}_{model_name}中
```
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -s HUMAN
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -s HUMAN
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -s HUMAN
```
3、运行test.py，得到测试集结果。
```
CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s HUMAN
CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s HUMAN
CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s HUMAN
```