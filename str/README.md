# 基于 ESM 的蛋白-小分子结合亲和力预测项目

本目录记录当前 ESM affinity 项目的数据准备、缓存验证和后续训练计划。目标是基于蛋白-小分子复合物三维结构预测结合亲和力 `pKd / pKi / pIC50 / pAffinity`。

## 1. 项目结构

当前仓库中与本项目直接相关的结构如下：

```text
final_project/
├── str/
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts/
│   │   └── data/
│   │       ├── create_esm_manifest.py
│   │       ├── validate_manifest.py
│   │       ├── smoke_test_manifest_batch.py
│   │       ├── create_trainable_manifest.py
│   │       ├── cache_ligand_graphs.py
│   │       ├── smoke_test_graph_batch.py
│   │       └── cache_esm_embeddings.py
│   ├── split_sequence_cluster_all_raw/
│   │   ├── timesplit_no_lig_overlap_train
│   │   ├── timesplit_no_lig_overlap_val
│   │   └── timesplit_test
│   └── manifest/
│       ├── esm_affinity_manifest.csv
│       ├── esm_affinity_manifest_report.json
│       ├── esm_affinity_manifest_validation_report.json
│       ├── esm_affinity_trainable_manifest.csv
│       ├── esm_affinity_trainable_manifest_report.json
│       ├── ligand_parse_failures.csv
│       ├── general_PL_2020_sequence_cluster_all_raw.csv
│       ├── esm_affinity_batch_smoke_report.json
│       ├── esm_affinity_graph_batch_smoke_report.json
│       ├── esm_affinity_graph_batch_smoke_valid_report.json
│       ├── esm_affinity_graph_batch_smoke_test_report.json
│       └── cache/
│           ├── ligand_graphs/
│           ├── ligand_graphs_report.json
│           ├── esm_embeddings_debug/
│           └── esm_embeddings_report_debug.json
└── data/
    └── raw/
        └── pdbbind2020/
```

依赖文件：

```bash
pip install -r str/requirements.txt
```

当前本机已验证的主要依赖：

```text
torch 2.12.0
numpy 2.4.4
pandas 3.0.2
rdkit 2025.09.6
transformers 5.10.2
huggingface_hub 1.18.0
safetensors 0.8.0
tqdm 4.68.1
```

`scikit-learn` 当前环境尚未安装，但后续 baseline 训练和指标计算需要，已写入 `requirements.txt`。

## 2. 分步的实现思路

整体路线：

```text
PDBbind / Interformer split
  -> 生成 ESM affinity manifest
  -> 验证标签、路径、split、ligand 可解析性
  -> 过滤不可构图 ligand，生成 trainable manifest
  -> 缓存 ligand graph
  -> 缓存 ESM residue embedding
  -> 构造 graph + ESM batch
  -> 训练 frozen ESM baseline
  -> 训练 ligand GNN / pocket-aware / cross-attention 模型
```

### 2.1 数据与标签

每个样本包含：

```text
pdb_id
split
protein_sequence
protein_path
pocket_path
ligand_sdf_path
ligand_mol2_path
affinity_type
affinity_value
affinity_unit
affinity_molar
pAffinity
```

亲和力统一为：

```text
pAffinity = -log10(affinity in mol/L)
```

脚本支持解析 `Kd / Ki / IC50`，以及 `= / < / > / ~` 关系符号和 `mM / uM / nM / pM / fM` 单位。

### 2.2 数据划分

当前使用 `str/split_sequence_cluster_all_raw/`，保持 Interformer split 文件格式：

```text
timesplit_no_lig_overlap_train
timesplit_no_lig_overlap_val
timesplit_test
```

该 split 已通过序列泄漏检查：

```text
identity >= 40%
coverage >= 0.8
cross-split violation = 0
```

### 2.3 蛋白表示

使用 Hugging Face ESM：

```text
facebook/esm2_t6_8M_UR50D      本地 Mac 验证
facebook/esm2_t12_35M_UR50D    Linux GPU 轻量全量缓存
facebook/esm2_t30_150M_UR50D   后续主实验候选
```

缓存策略：

```text
protein_sequence -> ESM -> residue-level embedding -> .pt cache
```

多链序列使用 `:` 分隔，缓存脚本会按链拆分。单链超过 1022 residues 时，脚本会自动按 chunk 切分，再拼回 residue-level embedding。

### 2.4 小分子表示

使用 RDKit 从 `ligand_sdf_path` 优先读取 ligand；失败时尝试 `ligand_mol2_path`。

缓存的 ligand graph 字段：

```text
atom_features            [num_atoms, 9]
atom_coordinates         [num_atoms, 3]
bond_index               [2, num_directed_edges]
bond_features            [num_directed_edges, 4]
num_atoms
num_bonds
pAffinity
```

### 2.5 模型训练思路

第一阶段建议使用 frozen ESM baseline：

```text
protein: cached ESM residue embedding -> mean pooling
ligand: Morgan fingerprint 或 cached ligand graph
fusion: concat
head: MLP regression
loss: MSELoss 或 HuberLoss
target: pAffinity
```

后续升级路线：

```text
frozen ESM + Morgan fingerprint
frozen ESM + ligand GNN
frozen ESM + pocket pooling
pocket residue ESM embedding + ligand atom embedding cross-attention
partial fine-tuning / LoRA
```

评估指标：

```text
RMSE
MAE
R2
Pearson
Spearman
```

## 3. 当前已完成部分的使用方法

以下命令默认在仓库根目录运行。

### 3.1 生成完整 manifest

```bash
python str/scripts/data/create_esm_manifest.py \
  --split-dir str/split_sequence_cluster_all_raw
```

默认输出：

```text
str/manifest/esm_affinity_manifest.csv
str/manifest/esm_affinity_manifest_report.json
str/manifest/general_PL_2020_sequence_cluster_all_raw.csv
```

当前结果：

```text
rows: 19037
train / valid / test: 17608 / 1038 / 391
missing labels: 0
empty protein sequence: 0
```

### 3.2 验证 manifest

核心验证：

```bash
python str/scripts/data/validate_manifest.py \
  --split-dir str/split_sequence_cluster_all_raw \
  --ligand-parse-limit 0
```

全量 ligand 解析验证：

```bash
python str/scripts/data/validate_manifest.py \
  --split-dir str/split_sequence_cluster_all_raw \
  --ligand-parse-limit -1
```

当前全量验证结果：

```text
rows: 19037
errors: 0
status: PASS
RDKit ligand parse success: 19033 / 19037
RDKit ligand parse failure: 4
```

失败 ligand：

```text
2pll
3vjs
3vjt
4hrd
```

### 3.3 生成可训练 manifest

```bash
python str/scripts/data/create_trainable_manifest.py
```

输出：

```text
str/manifest/esm_affinity_trainable_manifest.csv
str/manifest/ligand_parse_failures.csv
str/manifest/esm_affinity_trainable_manifest_report.json
```

当前结果：

```text
input rows: 19037
trainable rows: 19033
filtered rows: 4
train / valid / test: 17604 / 1038 / 391
```

### 3.4 缓存 ligand graph

小样本调试：

```bash
python str/scripts/data/cache_ligand_graphs.py \
  --limit 100 \
  --report-json str/manifest/cache/ligand_graphs_report_debug.json \
  --overwrite
```

全量缓存：

```bash
python str/scripts/data/cache_ligand_graphs.py
```

输出：

```text
str/manifest/cache/ligand_graphs/
str/manifest/cache/ligand_graphs_report.json
```

当前结果：

```text
cached graph files: 19033
failures: 0
num_atoms min / max / mean: 6 / 370 / 61.37
directed_edges mean: 126.68
```

### 3.5 验证 graph batch

```bash
python str/scripts/data/smoke_test_graph_batch.py --split train --batch-size 8
python str/scripts/data/smoke_test_graph_batch.py --split valid --batch-size 8 \
  --report-json str/manifest/esm_affinity_graph_batch_smoke_valid_report.json
python str/scripts/data/smoke_test_graph_batch.py --split test --batch-size 8 \
  --report-json str/manifest/esm_affinity_graph_batch_smoke_test_report.json
```

当前结果：

```text
train graph batch: PASS
  protein token shape: [8, 591]
  ligand atom feature shape: [361, 9]
  ligand bond index shape: [2, 734]
  label shape: [8]

valid graph batch: PASS
  protein token shape: [8, 553]
  ligand atom feature shape: [641, 9]
  ligand bond index shape: [2, 1342]
  label shape: [8]

test graph batch: PASS
  protein token shape: [8, 601]
  ligand atom feature shape: [487, 9]
  ligand bond index shape: [2, 1002]
  label shape: [8]
```

### 3.6 缓存 ESM embedding

Mac 本地小样本验证：

```bash
python str/scripts/data/cache_esm_embeddings.py \
  --manifest str/manifest/esm_affinity_trainable_manifest.csv \
  --model-name facebook/esm2_t6_8M_UR50D \
  --cache-dir str/manifest/cache/esm_embeddings_debug \
  --report-json str/manifest/cache/esm_embeddings_report_debug.json \
  --limit 20 \
  --device auto \
  --overwrite
```

当前结果：

```text
rows: 20
written: 20
failures: 0
device: mps
hidden_dim: 320
sequence_length min / max / mean: 105 / 1239 / 365.4
```

样例 embedding：

```text
10gs.pt: [416, 320]
11gs.pt: [416, 320]
13gs.pt: [418, 320]
16pk.pt: [415, 320]
1a07.pt: [209, 320]
```

Linux GPU 全量缓存建议：

```bash
python str/scripts/data/cache_esm_embeddings.py \
  --manifest str/manifest/esm_affinity_trainable_manifest.csv \
  --model-name facebook/esm2_t12_35M_UR50D \
  --cache-dir str/manifest/cache/esm_embeddings \
  --report-json str/manifest/cache/esm_embeddings_report.json \
  --device cuda \
  --float16-output
```

## 4. 未来步骤

### 4.1 完成全量 ESM embedding 缓存

Mac 已完成 20 条样本验证。下一步应在 Linux GPU 上全量缓存：

```text
input:  str/manifest/esm_affinity_trainable_manifest.csv
output: str/manifest/cache/esm_embeddings/
```

建议先用：

```text
facebook/esm2_t12_35M_UR50D
```

显存和时间允许后，再尝试：

```text
facebook/esm2_t30_150M_UR50D
```

### 4.2 实现 ESM + ligand graph dataloader

需要把以下缓存合并成训练 batch：

```text
str/manifest/cache/esm_embeddings/{pdb_id}.pt
str/manifest/cache/ligand_graphs/{pdb_id}.pt
```

batch 应包含：

```text
protein_embedding
protein_mask
ligand_atom_features
ligand_bond_index
ligand_bond_features
ligand_atom_coordinates
label pAffinity
```

### 4.3 训练 frozen ESM baseline

第一版模型：

```text
protein_embedding -> mean pooling
ligand graph 或 Morgan fingerprint -> ligand vector
concat -> MLP -> pAffinity
loss: MSELoss
```

输出：

```text
str/manifest/outputs/baseline_frozen_esm/
├── metrics.json
├── predictions_valid.csv
└── predictions_test.csv
```

### 4.4 升级模型

按风险从低到高推进：

```text
frozen ESM + Morgan fingerprint
frozen ESM + ligand GNN
frozen ESM + pocket pooling
pocket residue embedding + ligand atom cross-attention
partial fine-tuning / LoRA
```

### 4.5 最终报告

固定使用当前 sequence-cluster all-raw split，并报告：

```text
train / valid / test: 17604 / 1038 / 391
sequence leakage: identity >= 40%, coverage >= 0.8, violation = 0
```

每个模型至少报告：

```text
RMSE
MAE
R2
Pearson
Spearman
```
