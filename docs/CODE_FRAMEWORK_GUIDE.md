用CodeX总结了一下整体框架，可以用于阅读代码的参考

# ProtGEM 代码框架阅读说明

本文档基于当前仓库代码整理，目标是说明项目各部分职责、数据流、常用脚本运行方式以及主要参数含义。

## 1. 项目定位

本项目用于蛋白质功能多标签分类，核心任务是预测蛋白质对应的 Gene Ontology（GO）标签。GO 标签之间存在 DAG 依赖关系，因此训练和测试时会考虑 GO 标签传播。项目还包含 PPI（protein-protein interaction）预测、跨物种迁移测试、微调测试、批量推理、消融实验和数据导出脚本。

模型总体思路：

1. 蛋白质侧编码：融合序列、结构、InterPro domain 三类信息。
2. GO 侧编码：把 GO term 映射为可训练 embedding，可选 BioBERT 初始化。
3. 预测器：将蛋白质 embedding 与 GO embedding 交互，输出每个 GO term 的预测概率。
4. 后处理：根据 GO DAG 对预测概率做传播，使父子标签关系更一致。

## 2. 目录结构

| 路径 | 含义 |
| --- | --- |
| `README.md` | 环境配置和复现入口说明。 |
| `docs/` | 项目说明、实验记录、baseline 和本文档。 |
| `config/` | 默认配置文件。运行时会克隆到 `workspace/{species}/{model}/config`，后续调参应优先改 workspace 内配置。 |
| `model/` | 核心模型、数据集、损失函数、Transformer/GVP 等模块。 |
| `ablation/` | 只使用序列、结构或 domain 的消融模型。 |
| `data_process_scripts/` | 原始数据下载、清洗、GO/InterPro/蛋白质处理、结构图构建。 |
| `pretrain/` | ESM 序列 embedding 生成、BioBERT GO 文本 embedding 相关脚本。 |
| `data_export_scripts/` | 将数据导出为通用格式、PO2GO 格式、FASTA、蛋白质编码器权重等。 |
| `raw_data/` | 原始数据目录，按物种保存 PDB、GO、domain、PPI 等。 |
| `processed_data/` | 处理后的统一 JSON/PKL 文件，如 GO id、蛋白质信息、坐标等。 |
| `dataset/` | 模型直接读取的数据，如 ESM embedding、DGL 结构图、GO 图、缓存。 |
| `workspace/` | 每次训练/推理的输出目录，含日志、配置克隆、缓存、划分索引、模型参数、结果。 |
| `model_params/` | 通用预训练权重，如 GO 预训练模型。 |
| `logs/` | 部分旧脚本或全局脚本的日志。 |

## 3. 数据流

推荐的数据准备流程：

1. 准备公共原始文件：
   - `raw_data/go-basic.obo`
   - `raw_data/interpro_entry.txt`
   - `raw_data/interpro2go.txt`
   - 可选：`raw_data/swissprot_go.tsv`，用于 GO embedding 预训练采样。
2. 准备物种目录：
   - `raw_data/{SPECIES}/pid_list.json`
   - `raw_data/{SPECIES}/pdb/`
   - `raw_data/{SPECIES}/domain/`
   - `raw_data/{SPECIES}/go/`
   - `raw_data/{SPECIES}/ppi/`
3. 下载或放入 PDB、InterPro、GO、PPI 原始数据。
4. 生成 `processed_data`：
   - `process_go_basic.py`
   - `process_interpro.py`
   - `process_protein_new.py`
5. 生成模型读取的 `dataset`：
   - `build_go_graph.py`
   - `pretrain/seq_esm_new.py`
   - `build_pdb_graph_new.py`
6. 训练 GO 预训练 embedding、GO 预测模型或 PPI 模型。

关键文件：

| 文件 | 生成者 | 用途 |
| --- | --- | --- |
| `processed_data/go_terms.json` | `process_go_basic.py` | GO term 基本信息列表。 |
| `processed_data/go_id.json` | `process_go_basic.py` | GO id 到内部整数编号的映射，包含 alt/replaced 映射。 |
| `processed_data/interpro2go.json` | `process_interpro.py` | InterPro 到 GO 的映射。 |
| `processed_data/interpro_id.json` | `process_interpro.py` | InterPro vocabulary。 |
| `processed_data/{SPECIES}_protein_info.json` | `process_protein_new.py` | 蛋白质序列、GO、domain 信息。 |
| `processed_data/{SPECIES}_protein_coords.pkl` | `process_protein_new.py` | 蛋白质 CA 坐标。 |
| `dataset/go_graph_dgl.pkl` | `build_go_graph.py` | GO DGL 图，用于 GO 预训练。 |
| `dataset/go_graph_nx.pkl` | `build_go_graph.py` | GO NetworkX DAG，用于标签传播。 |
| `dataset/go_term_dis.json` | `build_go_graph.py` | GO term 深度/具体程度。 |
| `dataset/{SPECIES}/{PID}/seq_emb.pkl` | `seq_esm_new.py` | ESM2 token embedding。 |
| `dataset/{SPECIES}/{PID}/struct_gvp.dgl` | `build_pdb_graph_new.py` | GVP-GNN 使用的结构图。 |

## 4. 核心模型模块

### 4.1 `model/protein_model.py`

`ProteinEncoder` 是主蛋白质编码器，融合三类特征：

- `SeqEncoder`：读取 ESM2 token embedding，输出全局序列特征和残基级局部特征。
- `StructEncoder`：读取 DGL 结构图，基于 GVP/GVPConv 输出结构全局和局部特征。
- `DomainEncoder`：读取 InterPro domain id 和区间，输出 domain 全局和残基位置局部特征。

融合方式：

1. 三类全局特征 concat 后经过线性层得到 global embedding。
2. 三类局部特征按残基位置 padding 后 concat。
3. 局部特征前加入可学习 `SOS_emb`，再过 Transformer encoder。
4. 取 `SOS` 位置作为 part embedding。
5. global embedding 与 part embedding concat，得到最终蛋白质 embedding。

`ProteinEncoderv2` 和 `ProteinEncoderv3` 是实验版本；`v2` 将 ESM embedding 并入结构图，`v3` 类似主结构但使用更多 attention heads。

### 4.2 `model/predictor.py`

主要类：

- `ProteinGoModel`：GO 预测主模型，对应 `model_type=v1.1`。
- `ProteinGoModelv2` / `ProteinGoModelv3`：实验版本。
- `PPIModel`：PPI 预测模型。
- `ReconstructDecoder`：可选重构分支，用于序列/结构重构损失。

GO 预测流程：

1. `ProteinEncoder` 输出 `[batch, dim]`。
2. `GoEncoder` 根据 GO index 输出 `[go_num, dim]`。
3. 蛋白质 embedding 与 GO embedding 逐元素相乘。
4. MLP 输出 `[batch, go_num]` 概率。

PPI 预测流程：

1. 同一个 `ProteinEncoder` 分别编码两个蛋白质。
2. concat 两个 embedding。
3. MLP 输出交互概率。

### 4.3 `model/dataset.py`

主要数据集：

- `MyDataset`：GO 预测数据集。读取 `processed_data`、ESM embedding、结构图、domain、GO graph，生成多标签样本。
- `PPIDataset`：PPI 数据集。读取 PPI 正样本并进行负采样。
- `UpstreamDataset`：只加载蛋白质信息，常用于 PPI 批量推理时先得到蛋白质 embedding。

关键函数：

- `propagate`：根据 GO DAG 将标签向祖先传播。
- `probablity_propagate`：测试/推理后根据 GO DAG 传播预测概率。
- `get_split_dataset`：在 `utils.py` 中，根据比例划分 train/valid/test，并把索引保存到 workspace。
- `my_collate_fn` / `my_collate_fn_v2` / `ppi_collate_fn`：对序列 padding、DGL graph batch、domain offset 和标签做 batch 组装。

### 4.4 其他模型文件

| 文件 | 含义 |
| --- | --- |
| `model/seq_model.py` | ESM2 token embedding 编码器。 |
| `model/struct_model.py` | GVP/GVPConv 结构编码器。注意该模块注释说明内部需固定使用 `cuda:0`，换卡应通过 `CUDA_VISIBLE_DEVICES`。 |
| `model/domain_model.py` | InterPro domain 编码器。 |
| `model/go_model.py` | GO term embedding 编码器。 |
| `model/go_dataset.py` | GO 预训练采样数据集。 |
| `model/loss.py` | 对比损失、GO 对比损失、DPO 损失、序列/结构重构损失。 |
| `model/enhanced_predictor.py` | enhanced 版本模型，包含 cross attention、自适应 GO、层级 pooling 等实验结构。 |
| `ablation/*.py` | 只保留单一模态的消融模型。 |

## 5. 模型注册与版本

模型入口在 `register.py`。

| `model_type` | 蛋白质配置 | GO 配置 | 预测器配置 | 模型类 |
| --- | --- | --- | --- | --- |
| `v1.1` | `ProteinEncoder.conf` | `GoEncoder.conf` | `Predictor.conf` | `ProteinGoModel` |
| `v2` | `ProteinEncoder2.conf` | `GoEncoder2.conf` | `Predictor.conf` | `ProteinGoModelv2` |
| `v3` | `ProteinEncoder3.conf` | `GoEncoder.conf` | `Predictor.conf` | `ProteinGoModelv3` |
| `seq_abl` | `ProteinEncoder.conf` | `GoEncoder.conf` | `Predictor.conf` | 序列消融模型 |
| `struct_abl` | `ProteinEncoder.conf` | `GoEncoder.conf` | `Predictor.conf` | 结构消融模型 |
| `domain_abl` | `ProteinEncoder.conf` | `GoEncoder.conf` | `Predictor.conf` | domain 消融模型 |
| `ppi_v1` | `ProteinEncoder.conf` | 无 | `Predictor_ppi.conf` | `PPIModel` |
| `enhanced_v1` | `EnhancedProteinEncoder.conf` | `EnhancedGoEncoder.conf` | `EnhancedPredictor.conf` | `EnhancedProteinGoModel` |
| `enhanced_v2` | `EnhancedProteinEncoder.conf` | `EnhancedGoEncoder.conf` | `EnhancedPredictor.conf` | `EnhancedProteinGoModelv2` |

## 6. 配置文件参数

### 6.1 训练配置 `config/train.conf`

| 参数 | 含义 |
| --- | --- |
| `train_proportion` | 训练集比例。 |
| `valid_proportion` | 验证集比例，剩余为测试集。 |
| `batch_size` | GO 预测训练 batch size。 |
| `init_lr` | 初始学习率。 |
| `contrast_temp` | 蛋白质-GO 对比损失温度。 |
| `go_max_percent` | 过滤过于常见的 GO 标签，标签出现次数不能超过样本数的该比例。 |
| `go_min_count` | 过滤过少出现的 GO 标签。 |
| `mask_rate` | 训练时序列 token mask 和结构图节点 mask 比例。 |
| `alpha1` | 对比损失权重。 |
| `alpha2` | BCE 多标签损失权重。 |
| `alpha3` | GO 对比损失权重。 |
| `alpha4` | 重构损失权重，`v1.1` 下大于 0 时启用 `ReconstructDecoder`。 |
| `go_loss_switch` | 是否启用 GO loss 分支。 |
| `log_batch_count` | 每多少 batch 打一次日志。 |
| `bce_pos_weight` | BCE 正样本权重。 |
| `seq_epoch_num` | 只训练序列相关阶段 epoch 数，此阶段 dropout 结构和 domain。 |
| `struct_epoch_num` | 只训练结构相关阶段 epoch 数，此阶段 dropout 序列和 domain。 |
| `domain_epoch_num` | 只训练 domain 相关阶段 epoch 数，此阶段 dropout 序列和结构。 |
| `fusion_epoch_num` | 全模态融合训练 epoch 数。 |

总 epoch 为上述四个阶段之和。

### 6.2 GO 预训练配置 `config/train_go.conf`

| 参数 | 含义 |
| --- | --- |
| `epoch_num` | GO embedding 预训练 epoch 数。 |
| `batch_size` | GO 采样 batch size。 |
| `init_lr` | 初始学习率。 |
| `pos_k` | 每个 GO 节点实际采样的正样本数量。 |
| `pos_max` | 构建缓存时保留的正样本候选上限。 |
| `neg_k` | 每个 GO 节点实际采样的负样本数量。 |
| `neg_max` | 构建缓存时保留的负样本候选上限。 |
| `pos_rate` | 子节点被视为正样本时的 SwissProt 共现比例阈值。 |
| `contrast_temp` | GO 对比损失温度。 |

### 6.3 PPI 训练配置 `config/train_ppi.conf`

| 参数 | 含义 |
| --- | --- |
| `train_proportion` / `valid_proportion` | PPI 训练/验证/测试划分比例。 |
| `batch_size` | PPI 训练 batch size。 |
| `init_lr` | 初始学习率。 |
| `contrast_temp` | PPI 对比损失温度。 |
| `mask_rate` | 两个蛋白质输入的 mask 比例。 |
| `go_diff_weight` | 负采样时 bp/cc/mf GO 相似度权重。 |
| `ppi_weight` | 负采样时 PPI 距离项权重。 |
| `go_weight` | 负采样时 GO 相似度项权重。 |
| `pick_neg_rate` | 按启发式距离筛选的负样本数量比例。 |
| `random_neg_rate` | 随机负样本数量比例。 |
| `alpha1` | PPI 对比损失权重。 |
| `alpha2` | PPI BCE 损失权重。 |
| `log_batch_count` | 日志间隔。 |
| `bce_pos_weight` | BCE 正样本权重。 |
| `seq_epoch_num` / `struct_epoch_num` / `domain_epoch_num` / `fusion_epoch_num` | 同 GO 训练的分阶段训练含义。 |

### 6.4 模型结构配置

`ProteinEncoder.conf`：

- `final_out_dim`：最终蛋白质 embedding 维度，需与 GO embedding 维度一致。
- `final_hid_dim`：global/part 融合前的隐藏维度。
- `fusion_layers`：局部特征融合 Transformer 层数。
- `fusion_att_heads`：局部融合 attention heads。
- `fusion_hid_dim`：局部融合 FFN 维度。
- `fusion_dropout`：融合 dropout。
- `model_args.SeqEncoder`：序列编码器参数，如 ESM 输入维度、head 数、全局/局部输出维度。
- `model_args.StructEncoder`：结构编码器参数，如 GVP 输入/输出维度、层数、dropout、是否 cross gate。
- `model_args.DomainEncoder`：InterPro vocabulary 大小与输出维度。

`GoEncoder.conf`：

- `vocab_dim`：GO vocabulary 大小。
- `emb_dim`：GO embedding 维度。

`Predictor.conf` / `Predictor_ppi.conf`：

- `layer_cnt`：预测 MLP 隐藏层数。
- `hid_dim`：预测 MLP 隐藏维度。

`test.conf`：

- `batch_size`：测试和推理 batch size。

`fine-tuning.conf`：

- `epoch_num`：微调 epoch 数。
- `batch_size`：微调 batch size。
- `init_lr`：微调学习率。
- `mask_rate`、`contrast_temp`、`alpha1`、`alpha2`、`bce_pos_weight`：含义同训练配置。

## 7. 常用脚本

所有命令默认在项目根目录执行。GPU 选择建议用 `CUDA_VISIBLE_DEVICES=N`，因为代码内部通常固定 `torch.device("cuda:0")`。

### 7.1 数据准备脚本

#### 下载 AlphaFold PDB

```bash
python data_process_scripts/alphafold_pdb_downloader.py -s HUMAN
```

参数：

- `-s, --species-name`：物种名，对应 `raw_data/{species}`。

输入：

- `raw_data/{species}/pid_list.json`

输出：

- `raw_data/{species}/pdb/{pid}.pdb`

#### 下载 InterPro domain

```bash
python data_process_scripts/interpro_domain_downloader.py -s HUMAN
```

参数：

- `-s, --species-name`：物种名。

输入：

- `raw_data/{species}/pid_list.json`

输出：

- `raw_data/{species}/domain/{pid}.tsv`

#### 处理 GO 基础信息

```bash
python data_process_scripts/process_go_basic.py
```

输入：

- `raw_data/go-basic.obo`

输出：

- `processed_data/go_terms.json`
- `processed_data/go_id.json`

#### 处理 InterPro 映射

```bash
python data_process_scripts/process_interpro.py
```

输入：

- `raw_data/interpro_entry.txt`
- `raw_data/interpro2go.txt`

输出：

- `processed_data/interpro2go.json`
- `processed_data/interpro_id.json`

#### 处理蛋白质综合信息

```bash
python data_process_scripts/process_protein_new.py -s HUMAN -ml 2048
```

参数：

- `-s, --species-name`：物种名。
- `-ml, --max-length`：过滤超过该长度的蛋白质，默认 2048。

输入：

- `raw_data/{species}/pdb/*.pdb`
- `raw_data/{species}/domain/*.tsv`
- `raw_data/{species}/go/pgo_dict.json` 或 `raw_data/{species}/go/go.tsv`

输出：

- `processed_data/{species}_protein_info.json`
- `processed_data/{species}_protein_coords.pkl`

#### 构建 GO 图

```bash
python data_process_scripts/build_go_graph.py
```

输入：

- `processed_data/go_id.json`
- `processed_data/go_terms.json`

输出：

- `dataset/go_graph_dgl.pkl`
- `dataset/go_graph_nx.pkl`
- `dataset/go_term_dis.json`

#### 生成 ESM2 序列 embedding

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain/seq_esm_new.py -s HUMAN
```

参数：

- `-s, --species-name`：物种名，默认 `HUMAN`。
- `-pp, --protein_list_file_path`：可选，只重新处理指定 PDB 文件列表；文件每行一个 PDB 文件名。

输出：

- `dataset/{species}/{pid}/seq_emb.pkl`

#### 构建 PDB 结构图

```bash
CUDA_VISIBLE_DEVICES=0 python data_process_scripts/build_pdb_graph_new.py -s HUMAN
```

参数：

- `-s, --species-name`：物种名。
- `-pp, --protein_list_file_path`：可选，只重新处理指定 PDB 文件列表。

输出：

- `dataset/{species}/{pid}/struct_gvp.dgl`

#### 清理不在 pid_list 中的数据

```bash
python data_process_scripts/clean_data.py -s HUMAN
```

会删除不在 `raw_data/{species}/pid_list.json` 中的 PDB、domain TSV 和 dataset 子目录。

### 7.2 GO embedding 预训练

```bash
CUDA_VISIBLE_DEVICES=0 python train_go_emb.py
```

脚本会询问：

```text
Use biobert embeddings to initialize GO model? ([y]/n)
```

- 输入 `n`：随机初始化 GO embedding。
- 直接回车或输入其他内容：读取 `dataset/go_embedding_biobert.pkl` 初始化。

输入：

- `dataset/go_graph_dgl.pkl`
- `dataset/go_graph_nx.pkl`
- `processed_data/go_id.json`
- `raw_data/swissprot_go.tsv`
- 可选：`dataset/go_embedding_biobert.pkl`

输出：

- `model_params/go_pretrain_model.pth`
- 日志：`logs/train_go.log`

### 7.3 GO 主模型训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -s HUMAN -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -s HUMAN -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -s HUMAN -n model_v1.1 -t v1.1
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-o, --ont` | GO ontology：`bp`、`cc`、`mf`、`all`，默认 `mf`。 |
| `-s, --species-name` | 物种名，默认 `HUMAN`。 |
| `-n, --model-name` | workspace 下的模型实验名，默认 `model_v1.1`。 |
| `-t, --model-type` | 模型类型：`v1.1`、`v2`、`v3`、`seq_abl`、`struct_abl`、`domain_abl`、`enhanced_v1`、`enhanced_v2`。 |
| `-c, --continue-flag` | 是否继续训练。为 true 时加载已有模型和缓存划分。 |
| `-f, --freeze` | 是否冻结 GO model，仅在使用预训练 GO 模型时通常有意义。 |
| `-li, --light-mode` | 轻量磁盘模式，不预读 ESM embedding，运行时在线 ESM 推理；更慢且显存占用更集中。 |
| `-tpl, --forced-test-pid-list-path` | JSON 文件路径，指定必须进入测试集的蛋白质列表。 |
| `-preg, --pretrain-model-path` | GO 预训练权重路径，如 `model_params/go_pretrain_model.pth`。 |

输出目录：

```text
workspace/{species}/{model_name}/
  config/
  cache/
  idx/
  log/
  model_param/
```

重要输出：

- `model_param/{ont}_model.pt`
- `idx/{ont}_train_indices.npy`
- `idx/{ont}_valid_indices.npy`
- `idx/{ont}_test_indices.npy`
- `cache/{ont}_dataset.pkl`
- `cache/{ont}_balanced_go_list.json`
- `log/train_{ont}.log`

### 7.4 GO 测试评估

```bash
CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s HUMAN -n model_v1.1 -t v1.1
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-o, --ont` | GO ontology。 |
| `-s, --species-name` | 物种名。 |
| `-n, --model-name` | workspace 模型名。 |
| `-t, --model-type` | 模型类型：`v1.1`、`v2`、`v3`。 |
| `-th, --thresh` | 指定评估阈值；不传时按 Fmax 自动选择。 |

测试内容：

- 常规 AUC/AUPR/Fmax。
- 深层 GO term 子集指标。
- Pmin。
- global/local feature 分析。
- zero-mask ablation：分别遮蔽 seq、struct、domain。

输出：

- `workspace/{species}/{model}/log/test_{ont}.log`
- MLflow 指标。

### 7.5 PPI 模型训练与测试

训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ppi.py -s HUMAN -n model_ppi_v1 -t ppi_v1
```

只测试已有模型：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ppi.py -s HUMAN -n model_ppi_v1 -test true -th 0.5
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-s, --species-name` | 物种名。 |
| `-n, --model-name` | PPI workspace 模型名。 |
| `-t, --model-type` | 当前仅支持 `ppi_v1`。 |
| `-c, --continue-flag` | 是否继续训练已有 `ppi_model.pt`。 |
| `-f, --freeze` | 是否冻结预训练 protein encoder。 |
| `-test, --test_flag` | 是否只测试，不训练。 |
| `-pfn, --ppi-file-name` | PPI 正样本文件名，读取 `raw_data/{species}/ppi/{file}`，默认 `ppi_list.json`。 |
| `-tpl, --forced-test-pid-list-path` | 强制进入测试集的蛋白质 JSON 列表。 |
| `-prep, --pretrain-model-path` | 预训练 protein encoder 权重路径。 |
| `-th, --thresh` | 测试阈值。 |

输出：

- `workspace/{species}/{model}/model_param/ppi_model.pt`
- `workspace/{species}/{model}/log/train_ppi.log`

### 7.6 GO 批量推理

```bash
CUDA_VISIBLE_DEVICES=0 python batch_inference_go.py -o bp -s HUMAN -n model_v1.1 -t v1.1 -ds MOUSE -th 0.68
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-o, --ont` | 要推理的 ontology。 |
| `-s, --species-name` | 训练模型所属物种。 |
| `-n, --model-name` | 训练模型所属 workspace 模型名。 |
| `-pn, --precise-model-name` | 精确指定模型参数文件名；不传则用 `{ont}_model.pt`。 |
| `-t, --model-type` | 模型类型：`v1.1`、`v2`、`v3`。 |
| `-th, --thresh` | 输出标签阈值，推理时必须给出合理值。 |
| `-ds, --dest-species` | 目标物种。 |

输出：

- `workspace/{source_species}/{model}/result_{dest_species}/{ont}_all_result.pkl`
- `workspace/{source_species}/{model}/result_{dest_species}/{ont}_all_result.json`
- `workspace/{source_species}/{model}/result_{dest_species}/{ont}_recommand_result.pkl`

### 7.7 PPI 批量推理

```bash
CUDA_VISIBLE_DEVICES=0 python batch_inference_ppi.py -s HUMAN -n model_ppi_v1 -t ppi_v1 -th 0.5
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-s, --species-name` | 物种名。 |
| `-n, --model-name` | workspace 模型名。 |
| `-pn, --precise-model-name` | 精确指定模型参数文件名；不传则用 `ppi_model.pt`。 |
| `-t, --model-type` | 当前仅支持 `ppi_v1`。 |
| `-th, --thresh` | 输出 PPI 结果阈值。 |

输出：

- `workspace/{species}/{model}/result/ppi_all_result.pkl`
- `workspace/{species}/{model}/result/ppi_all_result.json`
- `workspace/{species}/{model}/result/ppi_recommand_result.pkl`

### 7.8 跨物种 GO 迁移测试

```bash
CUDA_VISIBLE_DEVICES=0 python transfer_test.py -o mf -s HUMAN -n model_v1.1 -t v1.1 -ds MOUSE
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-o, --ont` | ontology。 |
| `-s, --species-name` | 源物种，即模型训练物种。 |
| `-n, --model-name` | 源模型名。 |
| `-t, --model-type` | 模型类型。 |
| `-thresh, --thresh` | 阈值参数；当前代码调用 `cal_metrics(pred, actual)` 未使用该参数。 |
| `-ds, --dest-species` | 目标物种。 |

特点：

- 读取源模型的 balanced GO list。
- 目标物种数据集用 `inference_flag=True`，并排除源数据集已有 pid。
- 输出迁移测试 AUC/AUPR/Fmax 到日志。

### 7.9 跨物种微调测试

```bash
CUDA_VISIBLE_DEVICES=0 python fine_tuning_test.py -o mf -s HUMAN -n model_v1.1 -t v1.1 -ds MOUSE -fn 1000
```

参数：

| 参数 | 含义 |
| --- | --- |
| `-o, --ont` | ontology。 |
| `-s, --species-name` | 源物种。 |
| `-n, --model-name` | 源模型名。 |
| `-t, --model-type` | 模型类型。 |
| `-thresh, --thresh` | 测试阈值。 |
| `-ds, --dest-species` | 目标物种。 |
| `-fn, --fine-tuning-nums` | 从目标数据集中抽取多少样本用于微调，默认 100。 |

特点：

- 冻结 `pg_model.pm` 和 `pg_model.gm`，主要微调预测头。
- 保存最佳模型到 `model_param/{ont}_model_FTon_{dest_species}.pt`。

### 7.10 PPI 迁移样本生成

```bash
python gen_ppi_transfer.py -s HUMAN -d MOUSE
```

参数：

- `-s, --species-name`：源物种。
- `-d, --dest-species`：目标物种。

功能：

1. 将源物种和目标物种导出 FASTA。
2. 使用 BLAST 比对蛋白质相似性。
3. 将源物种 PPI 按相似蛋白映射到目标物种。
4. 合并目标物种已有 PPI。

依赖：

- `makeblastdb`
- `blastp`

输出：

- `raw_data/{dest_species}/ppi/transferred_ppi_list.json`

### 7.11 本地网页推理

```bash
python inference.py
```

功能：

- 启动 Gradio/FastAPI 服务，默认 `127.0.0.1:8080`。
- 上传单个 PDB 文件后预测 GO term。

注意：

- 文件头部注释说明该 demo 较旧，不一定能运行。
- 需要 `gradio`、`gradio_molecule3d`、`xmltramp2` 等依赖。
- 会调用 `iprscan5.py` 获取 InterPro 信息。
- 代码内 ESM 推理未显式 `.to(cuda)`，如需稳定使用可能需要额外检查。

### 7.12 消融实验

训练时通过 `train.py -t` 指定：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -s HUMAN -n model_seq_abl -t seq_abl
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -s HUMAN -n model_struct_abl -t struct_abl
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -s HUMAN -n model_domain_abl -t domain_abl
```

含义：

- `seq_abl`：序列单模态。
- `struct_abl`：结构单模态。
- `domain_abl`：domain 单模态。

### 7.13 数据导出

导出 FASTA：

```bash
python data_export_scripts/export_fasta.py -s HUMAN
```

输出：

- `processed_data/HUMAN.fasta`

导出通用格式：

```bash
python data_export_scripts/export_data_in_common_format.py -s HUMAN -n model_v1.1
```

输出在：

- `workspace/{species}/{model}/export/`

导出 PO2GO 格式：

```bash
python data_export_scripts/export_data_in_po2go_format.py -s HUMAN -n model_v1.1
```

输出在：

- `workspace/{species}/{model}/export_po2go/`

注意：该脚本中存在硬编码路径 `/d/chensq/protein-annotation/data2/{ont}o_terms.pkl`，迁移环境时需要修改。

导出蛋白质编码器权重：

```bash
python data_export_scripts/export_pm_model.py -o bp -s HUMAN -n model_v1.1 -t v1.1
```

输出：

- `workspace/{species}/{model}/model_param/{ont}_pm_model.pt`

## 8. MLflow 与日志

`train.py`、`test.py`、`train_ppi.py` 会使用 MLflow：

```bash
mlflow server --host 127.0.0.1 --port 8080
```

实验名：

- GO：`GO_Prediction`
- PPI：`PPI_Prediction`

日志位置：

- GO 训练：`workspace/{species}/{model}/log/train_{ont}.log`
- GO 测试：`workspace/{species}/{model}/log/test_{ont}.log`
- PPI 训练/测试：`workspace/{species}/{model}/log/train_ppi.log`
- 批量推理：`go_infer_{ont}.log` 或 `ppi_infer.log`

## 9. 复现示例

以 `HUMAN` 为例：

```bash
python data_process_scripts/process_go_basic.py
python data_process_scripts/process_interpro.py
python data_process_scripts/process_protein_new.py -s HUMAN -ml 2048
python data_process_scripts/build_go_graph.py

CUDA_VISIBLE_DEVICES=0 python pretrain/seq_esm_new.py -s HUMAN
CUDA_VISIBLE_DEVICES=0 python data_process_scripts/build_pdb_graph_new.py -s HUMAN

CUDA_VISIBLE_DEVICES=0 python train_go_emb.py

CUDA_VISIBLE_DEVICES=0 python train.py -o bp -s HUMAN -n model_v1.1 -t v1.1 -preg model_params/go_pretrain_model.pth
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -s HUMAN -n model_v1.1 -t v1.1 -preg model_params/go_pretrain_model.pth
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -s HUMAN -n model_v1.1 -t v1.1 -preg model_params/go_pretrain_model.pth

CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s HUMAN -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s HUMAN -n model_v1.1 -t v1.1
CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s HUMAN -n model_v1.1 -t v1.1
```

PPI 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train_ppi.py \
  -s HUMAN \
  -n model_ppi_v1 \
  -t ppi_v1 \
  -prep workspace/HUMAN/model_v1.1/model_param/bp_pm_model.pt

CUDA_VISIBLE_DEVICES=0 python batch_inference_ppi.py \
  -s HUMAN \
  -n model_ppi_v1 \
  -t ppi_v1 \
  -th 0.5
```

## 10. 注意事项

1. `config/` 下是默认配置。训练脚本通过 `load_config_clone` 将默认配置复制到 workspace，之后同一实验应修改 workspace 内配置以保证可追踪。
2. 代码内部普遍使用 `cuda:0`，多卡并行一般通过多个进程加 `CUDA_VISIBLE_DEVICES` 实现。
3. `struct_model.py` 的 readme 明确提示结构模型需要固定 `cuda:0`。
4. `continue_flag=True` 会复用缓存和已有数据划分；如果修改了数据集或 GO 筛选参数，需要谨慎使用。
5. `light_mode=True` 会跳过磁盘 ESM embedding，运行时在线调用 ESM，节省存储但更慢且显存压力更明显。
6. `test.py` 会做额外 ablation 和 feature effect 分析，因此比普通验证更慢。
7. `batch_inference_go.py` 和 `batch_inference_ppi.py` 的 `thresh` 不应为 `None`，因为代码会用它直接过滤预测结果。
8. `export_data_in_po2go_format.py` 有环境相关硬编码路径，跨机器运行需先调整。
9. `alphafold_pdb_downloader.py` 使用 `wget`，`interpro_domain_downloader.py` 访问 EBI API，网络失败时需要重跑。
10. `gen_ppi_transfer.py` 依赖本机 BLAST 工具链。

