# 基于 ESM 的蛋白-小分子结合亲和力预测

本项目面向 PDBbind 结构数据，目标是给定蛋白-小分子复合物的三维结构，预测结合亲和力（`pKd`、`pKi`、`pIC50`）。当前实现已经覆盖从原始结构切分、manifest 生成、特征缓存、batch 验证，到四种模型的训练与可视化。

## 1. 项目结构

与本实验直接相关的目录如下：

```text
final_project/
├── data/
│   ├── raw/
│   │   └── pdbbind2020/
│   │       └── complexes/P-L/
│   └── processed/
│       └── <split_name>/
│           └── pdbbind_sequence_cluster_splits.csv
├── scripts/
│   ├── create_sequence_cluster_split.py
│   ├── create_interformer_splits.py
│   └── sequence_leakage_check.py
└── str/
    ├── README.md
    ├── requirements.txt
    ├── manifest/
    │   ├── esm_affinity_manifest.csv
    │   ├── esm_affinity_trainable_manifest.csv
    │   ├── cache/
    │   │   ├── ligand_graphs/
    │   │   ├── esm_embeddings_8m/
    │   │   ├── esm_embeddings_150m/
    │   │   └── pocket_features/
    │   └── outputs/
    ├── output/
    │   ├── 8m/
    │   └── 150m/
    └── scripts/
        ├── data/
        ├── plot/
        ├── train/
        ├── prepare_until_baseline.sh
        ├── run_full_training_esm_scales.sh
        ├── run_frozen_esm_baseline.sh
        ├── run_ligand_gnn_baseline.sh
        ├── run_ligand_graph_transformer_baseline.sh
        ├── run_pocket_gnn_baseline.sh
        ├── split_raw_pdbbind.sh
        ├── build_manifest_from_split.sh
        ├── validate_after_manifest.sh
        └── ...
```

说明：

1. `data/processed/<split_name>/pdbbind_sequence_cluster_splits.csv` 是从原始 PDBbind 结构与 split 文件生成的桥接表。
2. `str/manifest/` 存放 manifest、缓存和单次实验中间产物。
3. `str/output/` 存放全量训练产物、曲线图和 TensorBoard 日志。
4. `str/scripts/` 下的脚本是当前实验的主入口，推荐优先使用这些脚本而不是手工拼命令。

## 2. 环境依赖创建与配置


### 2.1 创建环境

推荐使用 conda：

```bash
conda create -n ai4s python=3.11 -y
conda activate ai4s
python -m pip install --upgrade pip
```

### 2.2 安装依赖

```bash
pip install -r str/requirements.txt
```

如果你需要自己指定 PyTorch 版本，可先安装对应 CUDA 轮子，再装其余依赖：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r str/requirements.txt
```


### 2.4 运行约定

所有脚本默认使用仓库根目录作为 `ROOT_DIR`，并通过 `PYTHONPATH=$(pwd)` 运行。常见做法是：

```bash
PYTHONPATH=$(pwd) bash str/scripts/<script>.sh
```

如果本机没有默认 `python`，可以统一指定：

```bash
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python PYTHONPATH=$(pwd) bash str/scripts/<script>.sh
```

## 3. 数据预处理

这一部分从原始 PDB 结构开始，走到可训练的 manifest。默认采用结构级切分。

### 3.1 原始数据

需要准备 PDBbind 原始结构目录，默认脚本读取：

```text
data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst
data/raw/pdbbind2020/complexes/P-L/
```

每个复合物通常包含：

```text
{pdb_id}_protein.pdb
{pdb_id}_pocket.pdb
{pdb_id}_ligand.sdf
{pdb_id}_ligand.mol2
```

### 3.2 结构切分

先从原始结构生成 split。`iid=true` 时是随机结构级切分；`iid=false` 时会使用 40% 序列相似性先验，生成更严格的非 IID 切分。

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh \
  --split-name split \
  --iid false
```

输出包括：

```text
str/splits/split/
data/processed/split/pdbbind_sequence_cluster_splits.csv
```

如果你要换 split 名称，比如 `split_noiid`，则同时改变两个位置：

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh \
  --split-name split_noiid \
  --iid false
```

### 3.3 生成 manifest

基于结构切分结果生成 manifest：

```bash
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh \
  --split-name split \
  --split-dir str/splits/split
```

默认会读取：

```text
data/processed/split/pdbbind_sequence_cluster_splits.csv
```

并生成：

```text
str/manifest/esm_affinity_manifest.csv
str/manifest/esm_affinity_manifest_report.json
str/manifest/general_PL_2020_sequence_cluster_all_raw.csv
str/manifest/esm_affinity_manifest_validation_report.json
```

### 3.4 manifest 验证与缓存

一键完成 manifest 验证、trainable manifest 生成、ligand graph 缓存、ESM 缓存和 batch 验证：

```bash
PYTHONPATH=$(pwd) bash str/scripts/validate_after_manifest.sh
```

常见输出包括：

```text
str/manifest/esm_affinity_trainable_manifest.csv
str/manifest/cache/ligand_graphs/
str/manifest/cache/esm_embeddings/
str/manifest/esm_ligand_training_batch_report.json
```

### 3.5 pocket 缓存

7.2 版本需要额外的 pocket residue mask。缓存脚本会从 `protein.pdb` 和 `pocket.pdb` 构造 residue-level pocket 特征：

```bash
PYTHONPATH=$(pwd) bash str/scripts/run_pocket_gnn_baseline.sh
```

这一步内部会生成：

```text
str/manifest/cache/pocket_features/
```

## 4. Baseline模型与不同的改进


### 4.1 Frozen ESM Baseline

结构：

```text
protein ESM residue embedding
  -> full mean pooling
ligand atom features
  -> mean pooling
protein vector + ligand vector
  -> MLP
  -> pAffinity
```

原理与思路：

1. 固定 ESM，只把 protein 作为预计算 embedding。
2. ligand 先不做图消息传递，只做最简单的 mean pooling。
3. 这是整个项目的最低可复现闭环基线。

默认超参数：

```text
HIDDEN_DIM=256
DROPOUT=0.1
LR=1e-3
WEIGHT_DECAY=1e-4
LOSS=mse
GRAD_CLIP=5.0
EPOCHS=50
BATCH_SIZE=16
```

8M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_8m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
OUTPUT_DIR=str/output/8m/baseline_frozen_esm \
EPOCHS=50 \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

150M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_150m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
OUTPUT_DIR=str/output/150m/baseline_frozen_esm \
EPOCHS=50 \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

### 4.2 Ligand GNN

结构：

```text
protein ESM residue embedding
  -> full mean pooling
ligand atom features + bond index + bond features
  -> GNN (GCN / GraphSAGE / GINE)
  -> graph pooling
protein vector + ligand vector
  -> MLP
```

原理与思路：

1. protein 端仍然保持 frozen ESM。
2. ligand 端升级为消息传递，能够建模原子局部拓扑。
3. 这是从“只看 ligand 平均特征”升级到“看 ligand 图结构”的第一步。

默认超参数：

```text
PROTEIN_HIDDEN_DIM=256
GNN_TYPE=gine
GNN_LAYERS=3
GNN_HIDDEN_DIM=128
POOLING=mean
FUSION_HIDDEN_DIM=256
LR=1e-3
EPOCHS=50
BATCH_SIZE=16
```

8M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_8m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
OUTPUT_DIR=str/output/8m/ligand_gnn_frozen_esm \
EPOCHS=50 \
GNN_TYPE=gine \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_gnn_baseline.sh
```

150M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_150m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
OUTPUT_DIR=str/output/150m/ligand_gnn_frozen_esm \
EPOCHS=50 \
GNN_TYPE=gine \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_gnn_baseline.sh
```

### 4.3 Ligand Graph Transformer

结构：

```text
protein ESM residue embedding
  -> full mean pooling
ligand atom features + atom coordinates + bond features
  -> Graph Transformer
  -> graph pooling
protein vector + ligand vector
  -> MLP
```

原理与思路：

1. 在 GNN 的基础上，引入 pairwise attention bias。
2. 使用 ligand atom 3D 坐标和 bond features，让模型更明确地利用空间关系。
3. 适合做比 GNN 更强的 ligand encoder 对照。

默认超参数：

```text
PROTEIN_HIDDEN_DIM=256
TRANSFORMER_LAYERS=4
TRANSFORMER_HIDDEN_DIM=192
ATTENTION_HEADS=6
FFN_MULTIPLIER=4
POOLING=attention
RBF_BINS=32
RBF_MAX_DISTANCE=20.0
FUSION_HIDDEN_DIM=256
LR=5e-4
EPOCHS=50
BATCH_SIZE=16
```

8M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_8m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
OUTPUT_DIR=str/output/8m/ligand_graph_transformer_frozen_esm \
EPOCHS=50 \
TRANSFORMER_HIDDEN_DIM=192 \
ATTENTION_HEADS=6 \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_graph_transformer_baseline.sh
```

150M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_150m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
OUTPUT_DIR=str/output/150m/ligand_graph_transformer_frozen_esm \
EPOCHS=50 \
TRANSFORMER_HIDDEN_DIM=192 \
ATTENTION_HEADS=6 \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_graph_transformer_baseline.sh
```

### 4.4 Pocket-aware Ligand GNN

结构：

```text
protein ESM residue embedding + pocket mask
  -> pocket mean / pocket attention / ligand-conditioned attention
ligand atom features + bond index + bond features
  -> GNN
  -> ligand vector
pocket protein vector + ligand vector
  -> MLP
```

原理与思路：

1. 不再对整条蛋白做全局 mean pooling，而是只关注 pocket residue。
2. 如果 pocket 内 residue 足够明确，通常比 full-sequence pooling 更符合 affinity 任务。
3. `ligand_conditioned_attention` 允许不同 ligand 关注不同 pocket residue。

默认超参数：

```text
PROTEIN_POOLING=pocket_attention
FALLBACK_TO_FULL_SEQUENCE=1
GNN_TYPE=gine
GNN_LAYERS=3
GNN_HIDDEN_DIM=128
LIGAND_POOLING=mean
FUSION_HIDDEN_DIM=256
LR=1e-3
EPOCHS=50
BATCH_SIZE=16
```

8M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_8m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
POCKET_CACHE_DIR=str/manifest/cache/pocket_features \
OUTPUT_DIR=str/output/8m/pocket_gnn_frozen_esm \
EPOCHS=50 \
PROTEIN_POOLING=pocket_attention \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_pocket_gnn_baseline.sh
```

150M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_150m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
POCKET_CACHE_DIR=str/manifest/cache/pocket_features \
OUTPUT_DIR=str/output/150m/pocket_gnn_frozen_esm \
EPOCHS=50 \
PROTEIN_POOLING=pocket_attention \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_pocket_gnn_baseline.sh
```

## 5. Quick Start

如果你希望一次性跑完两个 ESM 基模大小下的四种模型，并且自动生成图与 TensorBoard，可以直接用全量脚本：

```bash
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_full_training_esm_scales.sh
```

这个脚本会执行以下流程：

1. 先检查 TensorBoard 环境。
2. 先缓存 ligand graph。
3. 先缓存两个 ESM 尺度的 embedding。
4. 再依次训练 5.5 到 5.8。
5. 每个模型结束后自动根据 `history.csv` 画图，图像写到 `str/output/plots/`。
6. 每个模型的 scalar 也会写入 `str/output/tensorboard/`。
7. 最后尝试启动 TensorBoard。

如果你只想跑某个单独模型，可以直接运行对应脚本，例如：

```bash
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_gnn_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_graph_transformer_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_pocket_gnn_baseline.sh
```

## 6. 最后结果
 TBD
