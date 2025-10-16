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
bp预测的AUPR终于突破0.5了，可喜可贺啊可喜可贺（虽然离DPFunc的0.56还差了一段距离）
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