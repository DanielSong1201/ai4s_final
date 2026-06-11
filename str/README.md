# 基于 ESM 的蛋白-小分子结合亲和力预测

本项目面向 PDBbind 结构数据，目标是给定蛋白-小分子复合物的三维结构，预测结合亲和力（`pKd`、`pKi`、`pIC50`）。当前实现已经覆盖从原始结构切分、manifest 生成、特征缓存、batch 验证，到五种模型的训练与可视化。

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
    ├── train_spatial_pocket_interaction_8m_150m.sh
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
        ├── run_spatial_pocket_interaction.sh
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

### 4.5 Spatial Pocket-Ligand Interaction

结构：

```text
protein ESM residue embedding + pocket mask + residue coordinates
  -> pocket spatial Transformer with residue-residue distance bias
ligand atom features + atom coordinates + bond features
  -> ligand Graph Transformer
pocket residue nodes + ligand atom nodes
  -> distance-biased pocket-ligand cross-attention
interaction-aware pocket vector + ligand vector
  -> MLP
  -> pAffinity
```

原理与思路：

1. 在 4.4 的 pocket mask 基础上，不只对 pocket residue 做 pooling，而是在 pocket residue 内部加入基于 residue 坐标的空间 Transformer。
2. ligand 端沿用 Graph Transformer，继续使用 atom 3D 坐标、bond features 和 pairwise distance RBF bias。
3. protein-ligand 之间新增 cross-attention，attention bias 来自 pocket residue 到 ligand atom 的距离 RBF，因此模型能显式学习空间接触关系。
4. 对每个 pocket residue 额外加入最近 ligand atom 距离的 contact feature，缓解只靠 ESM 表示而不看三维结构的问题。

默认超参数：

```text
HIDDEN_DIM=192
POCKET_LAYERS=2
LIGAND_LAYERS=4
ATTENTION_HEADS=6
FFN_MULTIPLIER=4
POOLING=attention
RBF_BINS=32
RBF_MAX_DISTANCE=20.0
FUSION_HIDDEN_DIM=256
LR=5e-4
LOSS=mse
GRAD_CLIP=5.0
EPOCHS=50
BATCH_SIZE=16
FALLBACK_TO_FULL_SEQUENCE=1
```

8M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_8m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
POCKET_CACHE_DIR=str/manifest/cache/pocket_features \
OUTPUT_DIR=str/output/8m/spatial_pocket_interaction_frozen_esm \
EPOCHS=50 \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_spatial_pocket_interaction.sh
```

150M 基模启动：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv \
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings_150m \
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs \
POCKET_CACHE_DIR=str/manifest/cache/pocket_features \
OUTPUT_DIR=str/output/150m/spatial_pocket_interaction_frozen_esm \
EPOCHS=50 \
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_spatial_pocket_interaction.sh
```

如果只想把 4.5 在 8M 与 150M 上连续训练，并写入既有全量训练输出结构，可以使用临时脚本：

```bash
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/train_spatial_pocket_interaction_8m_150m.sh
```

默认输出位置：

```text
str/output/8m/spatial_pocket_interaction_frozen_esm/
str/output/150m/spatial_pocket_interaction_frozen_esm/
str/output/plots/spatial_pocket_interaction_frozen_esm_8m_history.png
str/output/plots/spatial_pocket_interaction_frozen_esm_150m_history.png
```

如果同名 4.5 输出目录已经存在，临时脚本默认会覆盖该同名模型目录；不会覆盖 4.1-4.4 的输出。

## 5. Quick Start

如果你希望一次性跑完两个 ESM 基模大小下的 4.1-4.5 五种模型，并且自动生成图与 TensorBoard，可以直接用全量脚本：

```bash
PYTHON_BIN=/opt/anaconda3/envs/ai4s/bin/python \
PYTHONPATH=$(pwd) bash str/scripts/run_full_training_esm_scales.sh
```

这个脚本会执行以下流程：

1. 先检查 TensorBoard 环境。
2. 先缓存 ligand graph。
3. 先缓存两个 ESM 尺度的 embedding。
4. 再依次训练 4.1 到 4.5。
5. 每个模型结束后自动根据 `history.csv` 画图，图像写到 `str/output/plots/`。
6. 每个模型的 scalar 也会写入 `str/output/tensorboard/`。
7. 最后尝试启动 TensorBoard。

如果你只想跑某个单独模型，可以直接运行对应脚本，例如：

```bash
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_gnn_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_ligand_graph_transformer_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_pocket_gnn_baseline.sh
PYTHONPATH=$(pwd) bash str/scripts/run_spatial_pocket_interaction.sh
```

## 6. 最后结果

当前已经完成 4.1-4.5 在 8M 与 150M ESM 基模下的训练。结果来自各实验目录中的 `metrics.json`：

```text
str/output/8m/<model>/metrics.json
str/output/150m/<model>/metrics.json
```

### 6.1 测试集结果

| ESM | Model | Test RMSE | Test MAE | Test R2 | Test Pearson | Test Spearman |
|---|---|---:|---:|---:|---:|---:|
| 8M | baseline_frozen_esm | 1.9167 | 1.5354 | 0.0372 | 0.2720 | 0.2598 |
| 8M | ligand_gnn_frozen_esm | 2.1437 | 1.7188 | -0.2044 | 0.1515 | 0.1768 |
| 8M | ligand_graph_transformer_frozen_esm | 1.7431 | 1.4179 | 0.2037 | 0.4919 | 0.4541 |
| 8M | pocket_gnn_frozen_esm | 1.8355 | 1.4871 | 0.1170 | 0.4290 | 0.4386 |
| 8M | spatial_pocket_interaction_frozen_esm | 1.7173 | 1.3715 | 0.2271 | 0.4793 | 0.4905 |
| 150M | baseline_frozen_esm | 1.8867 | 1.5250 | 0.0671 | 0.3842 | 0.3708 |
| 150M | ligand_gnn_frozen_esm | 1.7998 | 1.4508 | 0.1510 | 0.4509 | 0.4069 |
| 150M | ligand_graph_transformer_frozen_esm | 1.5833 | 1.2808 | 0.3430 | 0.5967 | 0.5950 |
| 150M | pocket_gnn_frozen_esm | 2.0361 | 1.6386 | -0.0865 | 0.2546 | 0.1480 |
| 150M | spatial_pocket_interaction_frozen_esm | 1.6644 | 1.3525 | 0.2739 | 0.5379 | 0.5650 |

### 6.2 当前结论

1. 测试集 RMSE 最低的是 `150M + ligand_graph_transformer_frozen_esm`，`Test RMSE=1.5833`，同时测试集相关性也最高，`Pearson=0.5967`，`Spearman=0.5950`。
2. 4.5 `spatial_pocket_interaction_frozen_esm` 明显优于对应尺度的 4.4 pocket-aware GNN。8M 下 RMSE 从 `1.8355` 降到 `1.7173`，150M 下 RMSE 从 `2.0361` 降到 `1.6644`，说明显式 pocket-ligand 空间交互有效。
3. 4.5 在 8M 下也优于 8M Graph Transformer 的 RMSE 和 Spearman：`RMSE=1.7173` vs `1.7431`，`Spearman=0.4905` vs `0.4541`。这说明在较小 ESM 基模下，空间 pocket-ligand cross-attention 能补充 protein 表示能力。
4. 150M Graph Transformer 仍是总体最优模型。150M 4.5 的 RMSE 为 `1.6644`，比 150M Graph Transformer 的 `1.5833` 略差，但仍是全部模型中的第二梯队强结果，且相关性达到 `Pearson=0.5379`、`Spearman=0.5650`。
5. 单纯的 4.4 pocket-aware GNN 表现不稳定，尤其 150M 下 `Test R2=-0.0865`。这说明仅做 pocket pooling 不足以稳定提升性能，必须进一步引入 ligand 3D 坐标、residue-ligand 距离和交互注意力。
6. `R2` 整体仍不高，最佳为 150M Graph Transformer 的 `0.3430`。当前模型已经能学到排序信号，但绝对亲和力回归仍有较大提升空间。

### 6.3 最终排序

按 `Test RMSE` 从低到高，当前前三个模型为：

| Rank | ESM | Model | Test RMSE | Test Pearson | Test Spearman |
|---:|---|---|---:|---:|---:|
| 1 | 150M | ligand_graph_transformer_frozen_esm | 1.5833 | 0.5967 | 0.5950 |
| 2 | 150M | spatial_pocket_interaction_frozen_esm | 1.6644 | 0.5379 | 0.5650 |
| 3 | 8M | spatial_pocket_interaction_frozen_esm | 1.7173 | 0.4793 | 0.4905 |

因此，当前推荐作为最终主结果报告的是：

```text
150M ESM + ligand_graph_transformer_frozen_esm
```

推荐作为主要改进方向展示的是：

```text
spatial_pocket_interaction_frozen_esm
```

因为它相对 4.4 pocket-aware GNN 有明确提升，并且在 8M 基模下超过了同尺度 Graph Transformer。

## 7. Interformer split 实验结果

本节结果来自 `str/output/interformer/8m/<model>/metrics.json`，使用 Interformer-compatible split 与 8M frozen ESM embedding。各模型的样本数一致：

```text
train=16037
valid=945
test=356
```

### 7.1 Metrics 对比

| ESM | Split | Model | Best Epoch | Valid RMSE | Valid MAE | Valid R2 | Valid Pearson | Valid Spearman | Test RMSE | Test MAE | Test R2 | Test Pearson | Test Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8M | Interformer | baseline_frozen_esm | 50 | 1.4424 | 1.1181 | 0.3797 | 0.6205 | 0.6268 | 1.6130 | 1.3162 | 0.2033 | 0.4850 | 0.4407 |
| 8M | Interformer | ligand_gnn_frozen_esm | 47 | 1.4368 | 1.1253 | 0.3845 | 0.6209 | 0.6267 | 1.4810 | 1.1640 | 0.3284 | 0.5940 | 0.5449 |
| 8M | Interformer | ligand_graph_transformer_frozen_esm | 48 | 1.3488 | 1.0365 | 0.4576 | 0.6771 | 0.6748 | 1.4627 | 1.1796 | 0.3448 | 0.6044 | 0.5702 |
| 8M | Interformer | pocket_gnn_frozen_esm | 33 | 1.3869 | 1.0426 | 0.4265 | 0.6675 | 0.6788 | 1.5493 | 1.2171 | 0.2650 | 0.5511 | 0.5507 |
| 8M | Interformer | spatial_pocket_interaction_frozen_esm | 40 | 1.4275 | 1.0950 | 0.3924 | 0.6428 | 0.6429 | 1.4764 | 1.1701 | 0.3325 | 0.6113 | 0.5964 |

### 7.2 按 Test RMSE 排序

| Rank | ESM | Model | Test RMSE | Test Pearson | Test Spearman |
|---:|---|---|---:|---:|---:|
| 1 | 8M | ligand_graph_transformer_frozen_esm | 1.4627 | 0.6044 | 0.5702 |
| 2 | 8M | spatial_pocket_interaction_frozen_esm | 1.4764 | 0.6113 | 0.5964 |
| 3 | 8M | ligand_gnn_frozen_esm | 1.4810 | 0.5940 | 0.5449 |
| 4 | 8M | pocket_gnn_frozen_esm | 1.5493 | 0.5511 | 0.5507 |
| 5 | 8M | baseline_frozen_esm | 1.6130 | 0.4850 | 0.4407 |

### 7.3 当前结论

1. `ligand_graph_transformer_frozen_esm` 的测试集 RMSE 最低，`Test RMSE=1.4627`，也是当前 Interformer split 下按绝对误差排序的最优模型。
2. `spatial_pocket_interaction_frozen_esm` 的测试集相关性最高，`Test Pearson=0.6113`，`Test Spearman=0.5964`，但 RMSE 略高于 Graph Transformer。
3. 相比 `baseline_frozen_esm`，所有加入 ligand graph、pocket 或空间交互的模型都有明显提升，说明结构侧信息在 Interformer-compatible split 上有效。
