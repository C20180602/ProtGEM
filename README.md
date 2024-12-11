# dag-classify
Multi-tag classification task in which the tags has DAG-like dependencies

# 推荐环境
Miniconda 或者 Anaconda 均可

python 3.11.7

CUDA 12.1，torch 2.2.1+cu121，dgl 2.4.0

建议使用Linux环境（由于在后期可能使用InterProScan工具，该工具只提供了Linux版本）

其他python依赖库：

transformers 4.46.3

biopython 1.84

matplotlib 3.9.2

pyyaml 6.0.2

pydantic 2.8.2

fair-esm 2.0.0

h5py 3.12.1

配置流程：
```
conda create -n ev1 python=3.11.7

conda activate ev1

conda install cuda-toolkit==12.1.1

// 若报错，使用以下命令安装cuda12.1.1

// conda install nvidia/label/cuda-12.1.1::cuda-toolkit

pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

pip install transformers==4.46.3 biopython==1.84 matplotlib==3.9.2 pyyaml==6.0.2 pydantic==2.8.2 fair-esm==2.0.0
```

# 复现方法

详见Baseline.md