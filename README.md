# Protein-Ligand Affinity Prediction

本项目面向 PDBbind 蛋白-小分子复合物数据，目标是基于蛋白序列、配体结构与复合物三维结构预测结合亲和力，例如 `pKd`、`pKi` 或 `pIC50`。

当前主要完成的是 `str/` 结构建模部分：从 PDBbind 原始结构生成 train / valid / test split，构建训练 manifest，缓存 frozen ESM embedding 与 ligand graph，并训练多种 frozen ESM + ligand / pocket 模型。未来会在 `seq/` 下补充纯序列建模部分。

## 1. 项目结构

```text
final_project/
├── README.md
├── data/
│   ├── raw/
│   │   └── pdbbind2020/
│   └── processed/
│       └── <split_name>/
│           ├── pdbbind_sequence_cluster_splits.csv
│           ├── sequence_cluster_split_report.json
│           └── sequence_leakage_check/
├── scripts/
│   ├── create_interformer_splits.py
│   ├── create_sequence_cluster_split.py
│   └── sequence_leakage_check.py
├── str/
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── split_raw_pdbbind.sh
│   │   ├── build_manifest_from_split.sh
│   │   ├── validate_after_manifest.sh
│   │   ├── run_full_training_esm_scales.sh
│   │   ├── data/
│   │   ├── train/
│   │   └── plot/
│   ├── manifest/        # 本地生成，默认不追踪
│   └── output/          # 本地训练结果，默认不追踪
└── seq/
    └── TODO: sequence-only modeling will be added later.
```

说明：

1. `str/` 是当前主要实验目录，包含结构建模、ESM embedding 缓存、ligand graph、pocket-aware 与 spatial interaction 模型。
2. `scripts/` 保存与 PDBbind split 相关的通用脚本。
3. `data/raw/`、`data/processed/`、`str/manifest/`、`str/output/` 属于数据或实验产物，通常不提交到 git。
4. `seq/` 目前预留给后续序列建模部分。

更多结构模型、训练命令、实验结果见：

```text
str/README.md
```

## 2. 环境

推荐使用 conda：

```bash
conda create -n ai4s python=3.11 -y
conda activate ai4s
python -m pip install --upgrade pip
pip install -r str/requirements.txt
```

如果需要 GPU 训练，请根据服务器 CUDA 版本先安装匹配的 PyTorch，再安装其余依赖。

## 3. 数据

### 3.1 PDBbind 数据集特征

当前项目默认使用 PDBbind 2020 原始结构数据。每个样本对应一个蛋白-小分子复合物，包含蛋白结构、结合口袋结构、配体结构与亲和力标签。模型最终需要从复合物的结构与序列信息中预测结合亲和力。

默认输入路径：

```text
data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst
data/raw/pdbbind2020/complexes/P-L/
```

每个复合物目录通常包含：

```text
{pdb_id}_protein.pdb
{pdb_id}_pocket.pdb
{pdb_id}_ligand.sdf
{pdb_id}_ligand.mol2
```

其中：

1. `INDEX_general_PL.2020R1.lst` 提供 PDB ID 与亲和力标签。
2. `{pdb_id}_protein.pdb` 提供蛋白三维结构，用于提取蛋白序列和结构信息。
3. `{pdb_id}_pocket.pdb` 提供结合口袋区域结构。
4. `{pdb_id}_ligand.sdf` / `{pdb_id}_ligand.mol2` 提供小分子结构。

### 3.2 数据切分方法

切分脚本统一入口为：

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh \
  --split-name <split_name> \
  --iid <true_or_false>
```

`--split-name` 同时决定两个输出目录：

```text
str/splits/<split_name>/
data/processed/<split_name>/
```

#### 非 IID 切分

推荐用于主要实验：

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh \
  --split-name split_noiid \
  --iid false
```

逻辑：

1. 从 `{pdb_id}_protein.pdb` 提取蛋白链序列。
2. 使用 MMseqs2 做序列相似性聚类。
3. 默认以 `40% identity` 和 `0.8 coverage` 作为相似阈值。
4. 将相似蛋白簇作为整体分配到 train / valid / test，避免同一相似簇跨 split。
5. 生成 Interformer-compatible split 文件，并检查 cross-split sequence leakage。

这个 split 更接近 cold-protein / anti-leakage 评估，指标通常更低，但泛化评估更严格。

#### IID 随机切分

用于普通随机 baseline：

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh \
  --split-name split_iid \
  --iid true \
  --seed 42
```

逻辑：

1. 直接按 PDB complex ID 随机划分 train / valid / test。
2. 尽量保持与 Interformer split 相近的样本比例。
3. 不使用 40% 序列相似性约束。

这个 split 更接近 IID baseline，但不能用于证明模型避免 protein leakage。

### 3.3 切分后的输出结构与使用方法

以 `--split-name split_noiid` 为例，切分后会生成：

```text
str/splits/split_noiid/
├── timesplit_no_lig_overlap_train
├── timesplit_no_lig_overlap_val
└── timesplit_test

data/processed/split_noiid/
├── pdbbind_sequence_cluster_splits.csv
├── sequence_cluster_split_report.json
└── sequence_leakage_check/
    └── sequence_leakage_report.json
```

主要文件用途：

1. `str/splits/<split_name>/` 保存 Interformer-compatible split 文件，后续生成 manifest 时会读取这里的 train / valid / test ID。
2. `data/processed/<split_name>/pdbbind_sequence_cluster_splits.csv` 保存每个 PDB ID 对应的 split、亲和力标签、结构路径与聚类信息。
3. `sequence_cluster_split_report.json` 保存切分统计信息。
4. `sequence_leakage_check/sequence_leakage_report.json` 保存跨 split 序列泄漏检查结果；非 IID 主实验应确认其中 `leakage_detected: false`。

## 4. 实验

本项目的实验可以通过 `str/` 目录下完成结构建模的模型, `seq/` 目录下完成序列建模的模型. 具体的实验方案见各自的README.