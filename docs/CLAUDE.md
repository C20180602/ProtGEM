# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-tag protein function classification project that predicts Gene Ontology (GO) terms with DAG-like dependencies. The system combines protein sequence, structure, and domain information to classify proteins into biological process (bp), cellular component (cc), and molecular function (mf) categories.

## Key Architecture Components

### Core Models (model/ directory)
- **predictor.py**: Main prediction models (ProteinGoModel, ProteinGoModelv2) that combine protein and GO encoders
- **protein_model.py**: Multi-modal protein encoder combining sequence, structure, and domain features
- **go_model.py**: GO term encoder with graph attention networks
- **dataset.py**: Data loading and preprocessing with collate functions (my_collate_fn, my_collate_fn_v2)
- **seq_model.py**: ESM2-based sequence encoder with transformer architecture
- **struct_model.py**: GVP-GNN based protein structure encoder (fixed to cuda:0)
- **domain_model.py**: InterPro domain feature encoder

### Data Processing Pipeline
Raw data → processed_data → dataset

1. **Data Sources**: UniProt sequences, AlphaFold structures, InterPro domains, GO annotations, STRING PPI networks
2. **Processing Scripts** (data_process_scripts/): Convert raw biological data into model-ready formats
3. **Graph Construction**: PDB structures converted to graphs, GO terms organized in DAG structure

### Configuration System
- JSON-based configs in config/ directory
- Model versions: v1.1 (current stable), v2 (experimental)
- Separate configs for ProteinEncoder, GoEncoder, and Predictor components

## Development Commands

### Environment Setup
```bash
conda create -n ev1 python=3.11.7
conda activate ev1
conda install cuda-toolkit==12.1.1 -c conda-forge
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install transformers==4.46.3 biopython==1.84 matplotlib==3.9.2 pyyaml==6.0.2 pydantic==2.8.2 fair-esm==2.0.0 h5py==3.12.1 torchinfo==1.8.0
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
```

### Training Pipeline
1. **GO Embedding Pretraining**:
   ```bash
   python train_go_emb.py
   ```

2. **Main Model Training** (multi-GPU with CUDA_VISIBLE_DEVICES):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python train.py -o bp -t v1.1 -n model_v1.1 -s HUMAN
   CUDA_VISIBLE_DEVICES=1 python train.py -o cc -t v1.1 -n model_v1.1 -s HUMAN
   CUDA_VISIBLE_DEVICES=2 python train.py -o mf -t v1.1 -n model_v1.1 -s HUMAN
   ```

3. **Testing and Evaluation**:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python test.py -o bp -s HUMAN -n model_v1.1 -t v1.1
   CUDA_VISIBLE_DEVICES=1 python test.py -o cc -s HUMAN -n model_v1.1 -t v1.1
   CUDA_VISIBLE_DEVICES=2 python test.py -o mf -s HUMAN -n model_v1.1 -t v1.1
   ```

### Inference
```bash
python inference.py  # For single protein prediction
```

### Ablation Studies
Located in ablation/ directory:
- seq_ablation.py: Sequence-only model
- struct_ablation.py: Structure-only model
- domain_ablation.py: Domain-only model

## Important Notes

### GPU/CUDA Constraints
- Structure model (struct_model.py) must use cuda:0 due to implementation constraints
- Use CUDA_VISIBLE_DEVICES to assign different GPUs while maintaining cuda:0 internally
- Multi-GPU training requires separate processes for each ontology (bp/cc/mf)

### Model Versions
- **v1.1**: Current stable version with proven performance
- **v2**: Experimental version with enhanced features
- **Ablation variants**: For component analysis (seq_abl, struct_abl, domain_abl)

### Data Dependencies
- Requires significant preprocessing of biological databases
- InterProScan tool needed for domain annotation (Linux only)
- Large dataset storage requirements for protein structures

### Output Structure
Results stored in workspace_{species_name}_{model_name}/ with subdirectories:
- log/: Training and testing logs
- model_param/: Saved model checkpoints
- config/: Runtime configuration files
- cache/: Preprocessed data cache