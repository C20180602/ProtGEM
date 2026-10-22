# 2026.10.22 实验阶段存档与偶发训练 NaN 排查

当前决定：后续实验分析已完成，按项目作者的实验结论，正常训练效果良好并达到 SOTA。为保持已完成实验对应的实现与配置，在投稿前暂不应用本次稳定性修改建议，不重新训练；投稿后有时间或审稿意见涉及此问题时再评估。这里记录的是待处理问题，不表示问题已修复，也不将本次排查视为对 SOTA 结论的独立验证。偶发事件的实际发生率尚未统计。

## 本次 commit 保存的已有工作

- GO 预训练增加负欧氏距离对比损失 `GoContrastLoss2`，采用 `logsumexp`，并增加测试；当前 `train_go_emb.py` 仍默认使用余弦损失，负距离版本保留为可选项。两种相似度的已有实验结果见下文。
- 重构目标按 batch 生成，结构坐标延迟加载，不再依赖整份序列重构缓存；保留已有的序列损失掩码、解码器广播及共享优化器调整和测试。
- 保存当前配置：ProteinEncoder / ProteinEncoder3 的结构层数为 15，主训练融合阶段为 30 轮，GO 预训练为 100 轮。以上均为存档前已有改动，本次仅补充记录。
- 本 commit 保存代码、默认配置、测试与记录；`workspace/` 中的运行配置、数据划分、缓存、日志和模型权重被 Git 忽略，仍保留在本地，不包含在 commit 中。实际实验配置以各 workspace 的副本和日志为准。

## 已观察到的现象与验证范围

- 日志：`workspace/AF2DB/deepstruct/log/train_cc.log`。2026-10-22 第一次运行的随机种子为 706812398，模型为 v1.1，batch size 为 8，学习率为 0.01，结构网络为 15 层，`alpha4=0`、`go_loss_switch=false`。
- 第 0 轮 Step 1999 的统计仍正常，Step 2399 开始 `Loss1`、`Loss2` 同时为 NaN，之后持续异常。首次异常位于 Step 2000～2399 之间；日志没有逐步梯度范数，不能据此确定是梯度爆炸还是前向数值异常。
- 同一日志中后一次启动（种子 558075496）在对应位置及后续多个 epoch 没有出现相同异常。两次运行不构成固定随机状态的对照实验。
- 当前 cc 缓存的 56,324 个样本均至少有两个有效正标签，未发现无正标签样本。默认 FP32、Sigmoid 概率输入、温度 0.05 下，主对比损失的指数范围不足以直接溢出；未将 `exp` 作为已确认根因。
- CPU 小规模验证：50 组随机初始化的结构分支未复现爆炸；当前已保存 checkpoint 的参数均有限，三个真实短序列样本上的结构分支经 60 组随机掩码测试未复现爆炸，完整模型三次反向传播的总梯度范数约为 0.24～0.29。该 checkpoint 并非失败前的即时状态，测试也没有覆盖所有样本，不能排除偶发问题。本次未重新训练模型或更新 checkpoint。

## 后续候选修改（本次均未应用）

1. `train.py`：在 `optimizer.step()` 前检查预测值、loss 和梯度的有限性，记录裁剪前梯度范数、参数名、pids 和随机状态；非有限梯度不得进入 Adam 更新。增加可配置梯度裁剪，阈值根据正常梯度分布确定。
2. 实际运行配置 `workspace/AF2DB/deepstruct/config/train.conf`：以学习率 0.001 进行受控对照，暂不同时调整温度和损失权重。当前 0.01 学习率与仅 100 step warmup 是风险因素，并非已确认根因。
3. `model/struct_model.py::VectorDropout`：当前只生成形状 `[vector_num, 1]` 的掩码，整批所有节点共享。3 个向量通道、dropout=0.5 时，每层有 12.5% 概率清零整批消息分支的向量输出（残差仍保留）；此行为已复现。可改为形状 `[*x.shape[:-1], 1]` 的逐节点逐向量掩码，但尚未证明该行为是本次 NaN 根因。
4. `VectorLayerNorm`：当前分母下限为 1e-4，小向量区域可产生约 1e4 的局部梯度增益。先定位各层归一化输入 RMS 与梯度，再评估较大的稳定项；不要无差别修改所有 GVP 共用的 `L2()`。
5. `model/predictor.py` / `train.py`：Sigmoid 概率被传给 `BCEWithLogitsLoss` 是确定的接口不匹配。后续应分离 logits 与概率：BCE 使用 logits，对比损失和评估可继续使用概率，以保留现有对比目标尺度。不能只删除 Sigmoid 后直接把裸 logits 送进现有 `exp(pred/t)`。
6. `model/loss.py::ContrastLoss`：后续使用等价 `logsumexp` 表达式，并检查有效正标签；这是数值加固，当前没有证据将其直接指数溢出认定为此次原因。

已关闭的重构损失和 GO 辅助损失不属于本次主训练异常路径。此前小例子中复现的退化几何 NaN、零范数 GO embedding 大梯度，仅保留为对应功能启用时的独立风险。

## 存档检查

- `git diff --check` 通过。使用 ev1 环境执行 `python -B -m unittest discover -s tests -p 'test_*.py'`：17 个测试通过，重构测试模块因下述 `model_init` 导入错误未能加载，完整测试集未通过。此前 dgo 环境的四个负距离损失测试通过，另两个测试模块缺少 `esm` 依赖；未为此安装或修改环境。
- 静态检查发现当前 `train.py` 从 `utils` 导入 `model_init`，但该函数定义在 `register.py`，`utils.py` 未定义或导入它。这是当前工作区的独立启动问题，不是此前运行日志中 NaN 的已知原因；按本次仅存档、不修改代码的决定原样保留，未来启动训练前需处理。
- 本次稳定性排查与存档未应用上述训练修改，也未覆盖、重跑已有实验结果。

# 2026.10.14 重新测试GO预训练
CUDA_VISIBLE_DEVICES=0 python train.py -o bp -n right_go_loss -s AF2DB -preg model_params/go_pretrain_model.pth
CUDA_VISIBLE_DEVICES=1 python train.py -o cc -n right_go_loss -s AF2DB -preg model_params/go_pretrain_model.pth
CUDA_VISIBLE_DEVICES=2 python train.py -o mf -n right_go_loss -s AF2DB -preg model_params/go_pretrain_model.pth

之前的错误数据：
AUC/AUPR/Fmax
bp:
& Pretrained GO embedding    & 0.9810 & 0.5094 & 0.5015 \\
cc:
& Pretrained GO embedding    & \underline{0.9919} & 0.8139 & 0.7446 \\
mf:
& Pretrained GO embedding    & \textbf{0.9919} & 0.8617 & 0.8209 \\

采用余弦相似度的实验结果
bp:
2026-10-20 19:32:28,994 global INFO AUC: 0.9822 AUPR: 0.4976 PRC: 0.5797 REC: 0.4197  Fmax: 0.4869 tmp_thresh: 0.7157
cc:
2026-10-20 19:28:50,138 global INFO AUC: 0.9934 AUPR: 0.8172 PRC: 0.8035 REC: 0.6952  Fmax: 0.7454 tmp_thresh: 0.8083
mf:
2026-10-20 19:29:57,905 global INFO AUC: 0.9922 AUPR: 0.8727 PRC: 0.9049 REC: 0.7561  Fmax: 0.8239 tmp_thresh: 0.7785

采用余弦相似度的实验结果
bp: AUC: 0.9822 AUPR: 0.4976 Fmax: 0.4869
cc: AUC: 0.9934 AUPR: 0.8172 Fmax: 0.7454
mf: AUC: 0.9922 AUPR: 0.8727 Fmax: 0.8239

采用负距离相似度的实验结果
bp: AUC: 0.9827 AUPR: 0.5315 Fmax: 0.5136
cc: AUC: 0.9922 AUPR: 0.8248 Fmax: 0.7532
mf: AUC: 0.9904 AUPR: 0.8770 Fmax: 0.8280
bp:
2026-10-22 14:15:53,652 global INFO AUC: 0.9827 AUPR: 0.5315 PRC: 0.5970 REC: 0.4507  Fmax: 0.5136 tmp_thresh: 0.7368
cc:
2026-10-22 14:05:47,039 global INFO AUC: 0.9922 AUPR: 0.8248 PRC: 0.8165 REC: 0.6990  Fmax: 0.7532 tmp_thresh: 0.8181
mf:
2026-10-22 14:07:07,968 global INFO AUC: 0.9904 AUPR: 0.8770 PRC: 0.9011 REC: 0.7659  Fmax: 0.8280 tmp_thresh: 0.7847

# 2026.3~6 添加各类绘图
- PCA、UMAP绘图
- 参数敏感性分析绘图
- GO与PPI关联的小提琴图

# 2025.12.21 添加ML-FLOW日志记录
- 之前设计失误，把GO训练测试分开了，不太好统计到一起，需要手动把train中的ML-FLOW任务名记录一下，然后传给test.py

# 2025.12.18 新增mask_prediction任务，模型结构小优化
- 对序列和结构进行mask_prediction，由alpha4控制该任务开关状态
- 在局部特征融合先加入<SOS>，再进行transformer
- 两个优化都没用

# 2025.12.2 新增功能：PPI迁移学习
- 在目标物种D上出现PPI较少，孤立点较多的情况，建议采用gen_ppi_transfer.py丰富PPI
- 将物种A的PPI通过blast比对，将高相似的ppi转移到B上
```
python gen_ppi_transfer.py -s A_species -d B_species
```
- 使用train_ppi.py训练时添加 -trs True选项，使用转移后的ppi数据进行训练

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

......

2025.6.3:
根据数学理论推导了一下，发现Predictor部分，应该把MLP放在protein_emb处，最后的运算改为点乘，才和DPFunc最后的分类器等价，结果并没有用，不收敛
使用local_scope防止原图数据受影响，没有提升
尝试将ESM2的序列token直接加入到GVP-GNN的节点特征中，没有提升

2025.6.5:
把网页输入调好了

2025.6.21:
想改一下结构提取方式，改成pointNet

2025.6.22：
最后一次尝试，把蛋白质重新建图，全部按照GVP论文建图
验证best_epoch:16 AUC:0.9207910159828104 best_Fmax:0.3997247271258178 AUPR:0.3960638075077751 thresh:0.6324151158332825
测试AUC: 0.9273 AUPR: 0.3885 F1_score: 0.3855
微弱提升，暂时保留

2025.6.23：
尝试将预测部分改为transformer类似解码器的形式


2025.6.26:
优化序列特征维数压缩，按照随机正交分解来缩小特征维数
把几个特征输出进行LayerNorm
有微弱提升

2025.7.19
把消融实验按照严格意义上进行了重做
domain > seq > struct

2025.9.19
用claude code优化了一下模型结构
结果不收敛了，还找不到原因，乐

2025.9.26~29
用原来的模型跑了一个新数据集，居然效果就变好了，bp预测的AUPR能到0.48左右
之前在18000个人类蛋白质的数据集上，无论怎么优化，bp的AUPR都只有0.40左右
裁剪标签和不裁剪标签都变好了
可能是调了一下训练参数的原因吧。（后面其实发现还是数据集的原因）

2025.9.29
把迁移学习写完了，也算是了却了自己的一桩心愿
发现测试结果一般般，也不算太差，可以考虑加点预训练的数据量。

2025.9.30
把微调写完了，稍微测了一下，有用但不多

2025.10.12
终于腾出时间搞PPI模型了，下游直接concat+MLP，发现效果还不错
正负样本数1：5的情况下，Fmax能有0.93

2025.10.15
bp预测的AUPR终于突破0.5了，可喜可贺啊可喜可贺（虽然离DPFunc的0.56还差了一段距离，ps:但后面实测只有0.46）
加了MHA融合, 开始训练20轮感觉也一般
后来发现其实是欠拟合了，继续训练还能涨(其实也就只有0.02的提升)
顺便把大数据集上的训练的模型迁移到HUMAN数据集上进行了测试，发现bp的Fmax和AUPR都直接上0.5了(当然筛选掉了在大数据集的蛋白质，剩下的测试蛋白质只有3000多个)
```
AUC: 0.9803 AUPR: 0.5132 Fmax: 0.5235 tmp_thresh: 0.7884
```
这真的是数据集大带来的提升吗？
之前CAFA3数据集也挺大，但是提升却微乎其微
可能还是挑选蛋白质的一些问题
```
2025-09-29 10:32:51,983 global INFO Species:AF2DB ontology:all
2025-09-29 10:32:51,984 global INFO Protein Num: 57593
2025-09-29 10:32:51,984 global INFO Protein Length: Max: 2048, Min: 16, Avg: 509.95
2025-09-29 10:32:51,984 global INFO Domain Num: Max: 100, Min: 1, Avg: 6.16
2025-09-29 10:32:51,984 global INFO GO Terms: Max: 998, Min: 2, Avg: 86.99
2025-09-29 10:32:51,984 global INFO Balanced GO List length: 11050

2025-04-13 15:01:10,974 global INFO Species:CAFA3 ontology:bp
2025-04-13 15:01:10,974 global INFO Protein Num: 43515
2025-04-13 15:01:10,974 global INFO Protein Length: Max: 1022, Min: 17, Avg: 429.35
2025-04-13 15:01:10,974 global INFO Domain Num: Max: 21, Min: 1, Avg: 4.50
2025-04-13 15:01:10,975 global INFO GO Terms: Max: 678, Min: 2, Avg: 51.33
2025-04-13 15:01:10,975 global INFO Balanced GO List length: 5185

2025-10-14 22:15:59,535 global INFO Species:HUMAN ontology:bp
2025-10-14 22:15:59,535 global INFO Protein Num: 15516
2025-10-14 22:15:59,535 global INFO Protein Length: Max: 1022, Min: 16, Avg: 429.54
2025-10-14 22:15:59,535 global INFO Domain Num: Max: 21, Min: 0, Avg: 4.34
2025-10-14 22:15:59,535 global INFO GO Terms: Max: 1001, Min: 2, Avg: 94.74
2025-10-14 22:15:59,535 global INFO Balanced GO List length: 4074
```
可能是筛选长度导致的？
并不是，HUMAN集合筛选长度改为2048之后，反而下降到0.36了
还是蛋白质筛选的问题

2025.10.16
早上起来看结果，发现bp Fmax有0.54了
结果是因为重新训练的时候没有用已经分好的数据集，空欢喜一场

2025.10.17
孩子们我们的PPI模型完蛋了
man, what can I say, PPI out
测试结果
ppi0.7 go0.3 pick5.0 random0.0 不冻结, 用预训练模型： 收敛极好
7   3   5   0   fr, pre: 收敛，但结果一般
7   3   4   1   nofr, pre： 不收敛
7   3   4.5 0.5 nofr, pre： 不收敛
7   3   4.9 0.1 nofr, pre:不收敛
7   3   4.95 0.05 nofr, pre:不收敛
7   3   5   0   nofr, nopre:不收敛
7   3   0   5   nofr, nopre:不收敛
10  0   5   0   nofr, pre： 不收敛
......
7   3   4   1   fr, pre:不收敛
7   3   4.9 0.1 fr, pre:不收敛
7   3   5   0.1 nofr, pre:不收敛
虽然欣喜的是, 第一次就找到了完全正确的参数组合
但居然没办法在负样本中加入随机采样, 一加就不收敛, 很令人难以接受
有点绝望, 说明之前的训练根本没有把GO信息从蛋白质模型中解耦出去，只是单纯记住了某些特征对应的标签，而不是，提取更多的特征，从而更好地完成下游任务

与此同时，另一边，GO模型的优化也失败了
加入MHA只有负优化
没有MHA：bp cc mf顺序
AUC: 0.9714 AUPR: 0.4899 Fmax: 0.4806
AUC: 0.9878 AUPR: 0.8158 Fmax: 0.7515
AUC: 0.9930 AUPR: 0.8905 Fmax: 0.8445
加入MHA：bp cc mf顺序
AUC: 0.9698 AUPR: 0.4891 Fmax: 0.4792
AUC: 0.9866 AUPR: 0.8142 Fmax: 0.7523
AUC: 0.9931 AUPR: 0.8884 Fmax: 0.8432

最后，人类数据集也再次验证了加入MHA和GO标签预训练也是负优化，只对比bp
没有MHA，不加预训练
best_epoch:18 AUC:0.9145377567660191 best_Fmax:0.42207135725467615 AUPR:0.4010367680235847
有MHA，不加预训练
best_epoch:18 AUC:0.9049381923568115 best_Fmax:0.4135325551431723 AUPR:0.38473951972702547
有MHA，加预训练
best_epoch:19 AUC:0.9069639491739451 best_Fmax:0.38770562728174274 AUPR:0.36805899080077975

2025.10.19
PPI训练不收敛的问题依旧没找到
7   3   5   0   nofr, pre：收敛了，但效果一般，AUC只有0.66，比之前差了好多，已经出现三次这种情况了
开始尝试调试loss的超参数

2025.10.20
孩子们，我们有救了
调了一下正负样本比例就好了之前1：5，调成1：1，直接收敛了
后来尝试了一下1：2，也能正常收敛
不过预测的分数都普遍偏高

2025.3.23
大概12月搞完了代码，一月份把论文写得差不多了，二月过年玩，啥事没干
今天觉得得加点图，PCA分析，后面也可以加一下Smin指标（之前加了一个自己设的Pmin指标，可惜比不出差距）
