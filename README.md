# dag-classify
Multi-tag classification task in which the tags has DAG-like dependencies

# 推荐环境
Miniconda 或者 Anaconda 均可

CUDA 12.1，torch 2.2.1+cu121，dgl 2.2.1

建议使用Linux

其他python依赖库：
biopython 1.84
matplotlib 3.9.2
pyyaml 6.0.2
pydantic 2.8.2

配置流程：
```
conda create -n ev1 python=3.11.7
conda activate ev1
conda install cuda-toolkit==12.1

```

# 运行流程
## data处理
raw_data -> processed_data -> dataset

使用data_process_scripts中的代码

## 模型训练