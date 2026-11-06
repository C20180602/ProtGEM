# 2026.11.04 重新测试重构训练
bp:
2026-11-04 19:18:51,272 global INFO AUC: 0.9815 AUPR: 0.5190 PRC: 0.6186 REC: 0.4323  Fmax: 0.5089 tmp_thresh: 0.7442
cc:
2026-11-05 14:34:13,261 global INFO AUC: 0.9926 AUPR: 0.8278 PRC: 0.8225 REC: 0.7015  Fmax: 0.7572 tmp_thresh: 0.8127
mf:
2026-11-05 14:35:26,871 global INFO AUC: 0.9920 AUPR: 0.8719 PRC: 0.9062 REC: 0.7552  Fmax: 0.8238 tmp_thresh: 0.7805


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
