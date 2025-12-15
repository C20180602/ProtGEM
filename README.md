# dag-classify
Multi-tag classification task in which the tags has DAG-like dependencies

**Please use "git checkout v2" to get the newest version.**

# 推荐环境
Miniconda 或者 Anaconda 均可

python 3.11.7

CUDA 12.1，torch 2.2.1+cu121，dgl 2.4.0

最好是Linux环境（由于在后期可能使用InterProScan工具，该工具只提供了Linux版本）

其他python依赖库：(最好按顺序安装)

transformers==4.46.3

biopython==1.84

matplotlib==3.9.2

pyyaml==6.0.2

pydantic==2.8.2

fair-esm==2.0.0

h5py==3.12.1

torchinfo==1.8.0

如果需要使用在线推理，还需安装：

xmltramp2==3.1.1

gradio==5.29.0

gradio_molecule3d==0.0.7

若需使用PPI迁移推理，还需安装ncbi-blast：(最好在项目文件夹外进行)

``` bash
mkdir blast
cd blast
wget -c ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.17.0+-x64-linux.tar.gz
tar -zxvf ncbi-blast-2.17.0+-x64-linux.tar.gz    // if failed to execute, try downloading files on https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
echo 'export PATH=$PATH:~/blast/ncbi-blast-2.17.0+/bin' >> ~/.bashrc
source ~/.bashrc
```
# 配置流程

python环境配置：
```
conda create -n ev1 python=3.11.7

conda activate ev1

conda install cuda-toolkit==12.1.1 -c conda-forge

// 若报错，使用以下命令安装cuda12.1.1

// conda install nvidia/label/cuda-12.1.1::cuda-toolkit

pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121

pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

pip install transformers==4.46.3 biopython==1.84 matplotlib==3.9.2 pyyaml==6.0.2 pydantic==2.8.2 fair-esm==2.0.0 h5py==3.12.1 torchinfo==1.8.0

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```

# 复现方法

详见Baseline.md