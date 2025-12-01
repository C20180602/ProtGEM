# 2025.12.2 新增功能：PPI迁移学习
- 对原物种建库，在新物种上查询

# 2025.11.21 新增功能：GO训练磁盘空间节省模式
- 使用ESM在线推理获取ESM embedding，但训练时间会更长，显存占用更高
- 由于PPI训练规模远大于GO训练，最好还是提前预处理ESM嵌入，故不支持磁盘节省模式。
- 使用"-li True"参数开启该模式
```
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -t v1.1 -n model_v1.1 -s 7120 -li True
```

# 2025.10.27 新增功能：GO、PPI批量预测
- python batch_inference_go.py -o bp -s AF2DB -n model_v1.1 -t v1.1 -thresh 0.75 -ds TARGET_DATASET
- python batch_inference_ppi.py -s DATASET -n model_ppi_v1 -t ppi_v1 -thresh 0.9

# 2025.10.12 新增功能：PPI预测
- 运行data_export_scripts/export_pm_model.py，导出任意一个模型的protein model部分
```
python data_export_scripts/export_pm_model.py -o all -s AF2DB -n model_v1.1_back -t v1.1
```
- 运行train_ppi.py，可选择以刚才导出的protein model模型为基础进行训练+测试
```
python train_ppi.py -s TARGET_DATASET -prep workspace/AF2DB/model_v1.1/model_param/all_pm_model.pt
```

# 2025.9.30 新增功能：GO微调+迁移测试
- 运行fine_tuning_test.py即可测试
- 使用基于PRETRAIN_DATASET训练的模型，对TARGET_DATASET进行预测
- fn参数表示从TARGET_DATASET中选用于微调的数据组数
- 测试日志与微调后模型保存在原模型的相应文件夹中

```
CUDA_VISIBLE_DEVICES=0 python fine_tuning_test.py -o bp -s PRETRAIN_DATASET -n model_v1.1 -t v1.1 -ds TARGET_DATASET -fn 200
```

# 2025.9.29 新增功能：GO迁移预测
- 运行transer_test.py即可测试
- 使用基于PRETRAIN_DATASET训练的模型，对TARGET_DATASET进行预测
- 使用的GO标签集合来自于训练集筛选后的标签
- 会剔除掉已经处于PRETRAIN_DATASET中的蛋白质（可优化为只剔除PRETRAIN_DATASET训练集中的蛋白质）
- 测试日志保存在原模型的log文件夹中

```
python transfer_test.py -o cc -s PRETRAIN_DATASET -n model_v1.1 -t v1.1 -ds TARGET_DATASET
```

# 2025.9.26 数据Pipeline优化
- 可以只根据蛋白质的uniprot_id（后文简称pid）进行数据集构建，不依赖物种分类（物种名称可视为数据集名称）

- 下载流程进行了多线程优化

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
运行data_process_scripts下的process_go_basic.py、process_interpro.py、process_protein_new.py

运行pretrain/seq_esm_new.py和data_process_scripts/build_pdb_graph_new.py

顺序可任意

# 运行流程
## 数据下载（该流程已弃用）
### 序列数据：

[Uniprot](https://www.uniprot.org/)

搜索物种，下载reviewed的蛋白质的fasta

*在下载栏中找到TSV选项，开启之后可以添加各类数据，例如GO标签、PPI相互作用

### 结构数据：

[AlphaFoldDB](https://alphafold.com/download)

下载整合好的物种蛋白质结构包

### 结构域数据：

[interpro](https://www.ebi.ac.uk/interpro/)

使用data_process_scripts/interpro_domain_downloader.py下载Interpro数据库中的内容

（TODO：改为使用InterProScan处理缺少结构域数据的蛋白质）

### GO标签数据：

[Gene Ontology](https://current.geneontology.org/products/pages/downloads.html)

下载对应物种的GO标签的.gaf文件

*当然，你也可以使用Uniprot汇总的GO标签数据，不过后续的数据处理需要"魔改"一下

在GO预训练过程中，使用了Uniprot的swissprot数据，标签排序方式为：

Entry	Length	Gene Ontology IDs

### PPI数据：

[STRING](https://cn.string-db.org/cgi/download)

选择物种，下载physical_link和link即可

由于下载之后的蛋白质ID与uniprot不一致，所以需要在uniprot tsv下载界面的Protein-protein Interaction中添加STRING数据库的ID，用以获取对应关系

*当然，你也可以使用Uniprot中已经汇总好的Interaction信息，以避免过于繁琐的数据处理。


### 数据整理以及其他数据

raw_data的文件夹结构：

![图一](asset/image.png)

图一中的export.tsv只用于interpro_domain_downloader.py中获取蛋白质列表的步骤，也可以通过读取其他文件来获取蛋白质列表
        
![图二](asset/image-1.png)

图二中的go-basic.obo, interpro_entry.txt, interpro2go.txt需要下载

[go-basic.obo](https://geneontology.org/docs/download-ontology/)（版本为2024-06-17）

[interpro_entry.txt](https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list)

[interpro2go.txt](https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/interpro2go)

其余文件均为调试文件，不用下载


## 数据处理（该流程已弃用）

总数据流：raw_data -> processed_data -> dataset

使用data_process_scripts中的代码

1、解压结构数据压缩包

2、使用delete_cif.py删除cif数据，我们只用PDB文件

3、使用unzip_pdb.py批量解压PDB文件。

4、运行process_protein.py, process_go_basic.py, process_interpro.py, 可以同时进行, 完成此步骤之后processed_data文件夹处理完毕

5、build_pdb_graph.py, pretrain/seq_esm.py, 可以同时进行, 完成此步骤之后dataset文件夹处理完毕

5.5、按照pretrain/biobert/readme.md的指示，完成运行biobert获取go标签的初始嵌入（不运行也可以，使用随机向量初始化）

6、运行build_go_graph.py，建立GO图

## 模型训练

1、运行train_go_emb.py，预训练go的初始嵌入，得到go_pretrain_model模型，保存在model_params文件夹中

2、运行train.py，训练protein_model模型和go_model模型，输出内容在workspace_{species_name}_{model_name}中
```
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -t v1.1 -n model_v1.1 -s HUMAN
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -t v1.1 -n model_v1.1 -s HUMAN
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -t v1.1 -n model_v1.1 -s HUMAN

CUDA_VISIBLE_DEVICES=0 python train.py -o bp -t v1.1 -n model_v1.1_wo_pre -s HUMAN
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -t v1.1 -n model_v1.1_wo_pre -s HUMAN
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -t v1.1 -n model_v1.1_wo_pre -s HUMAN

CUDA_VISIBLE_DEVICES=0 python train.py -o bp -t v1.1 -n model_v1.1 -s CAFA3
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -t v1.1 -n model_v1.1 -s CAFA3
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -t v1.1 -n model_v1.1 -s CAFA3

CUDA_VISIBLE_DEVICES=4 python train.py -o bp -t v1.1 -n model_v1.1_large -s CAFA3
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -t v1.1 -n model_v1.1_large -s CAFA3
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -t v1.1 -n model_v1.1_large -s CAFA3
```
3、运行test.py，得到测试集结果与消融实验结果。
```
CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s HUMAN -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s HUMAN -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s HUMAN -n model_v1.1 -t v1.1

CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s HUMAN -n model_v1.1_wo_pre -t v1.1
CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s HUMAN -n model_v1.1_wo_pre -t v1.1
CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s HUMAN -n model_v1.1_wo_pre -t v1.1

CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s CAFA3 -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s CAFA3 -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s CAFA3 -n model_v1.1 -t v1.1

CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s CAFA3 -n model_v1.1_large -t v1.1
CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s CAFA3 -n model_v1.1_large -t v1.1
CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s CAFA3 -n model_v1.1_large -t v1.1
```