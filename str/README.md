# 基于 ESM 的蛋白-小分子结合亲和力预测项目

本目录记录当前 ESM affinity 项目的数据准备、缓存验证和后续训练计划。目标是基于蛋白-小分子复合物三维结构预测结合亲和力 `pKd / pKi / pIC50 / pAffinity`。

## 1. 项目结构

当前仓库中与本项目直接相关的结构如下：

```text
final_project/
├── scripts/
│   ├── create_sequence_cluster_split.py
│   ├── sequence_leakage_check.py
│   ├── create_interformer_splits.py
│   └── data_inspection.py
├── str/
│   ├── README.md
│   ├── requirements.txt
│   ├── scripts/
│   │   ├── data/
│   │   │   ├── create_esm_manifest.py
│   │   │   ├── validate_manifest.py
│   │   │   ├── smoke_test_manifest_batch.py
│   │   │   ├── create_trainable_manifest.py
│   │   │   ├── cache_ligand_graphs.py
│   │   │   ├── smoke_test_graph_batch.py
│   │   │   ├── cache_esm_embeddings.py
│   │   │   └── build_training_batch.py
│   │   └── train/
│   │       └── train_frozen_esm_baseline.py
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
│       ├── esm_ligand_training_batch_debug_report.json
│       ├── esm_ligand_training_batch_report.json
│       ├── cache/
│       │   ├── ligand_graphs/
│       │   ├── ligand_graphs_report.json
│       │   ├── esm_embeddings_debug/
│       │   └── esm_embeddings_report_debug.json
│       └── outputs/
│           └── baseline_frozen_esm/
│               ├── checkpoints/
│               ├── metrics.json
│               ├── history.csv
│               ├── predictions_valid.csv
│               └── predictions_test.csv
└── data/
    ├── raw/
    │   └── pdbbind2020/
    └── processed/
        └── sequence_cluster_split_validation/
            └── all_raw/
                ├── pdbbind_sequence_cluster_split_table.csv
                ├── pdbbind_sequence_cluster_split_table_report.json
                └── final_validation_summary.json
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

当前使用 `str/split_sequence_cluster_all_raw/`。这里的切分对象是 **PDBbind complex / PDB ID 级别的结构样本**，不是单独切分蛋白链、ligand、embedding 或表格行。

每个 PDB ID 对应一组结构文件：

```text
{pdb_id}_protein.pdb
{pdb_id}_pocket.pdb
{pdb_id}_ligand.sdf
{pdb_id}_ligand.mol2
```

这组文件必须整体进入同一个 split。实际训练时不需要把结构文件物理复制成 train/valid/test 三个目录；split 文件保存的是 PDB ID，后续 manifest 会把每个 PDB ID 解析回原始结构路径。

split 文件保持 Interformer 格式：

```text
timesplit_no_lig_overlap_train
timesplit_no_lig_overlap_val
timesplit_test
```

切分方法是：

```text
1. 从每个 PDB complex 的 *_protein.pdb 中抽取 chain sequence
2. 用 MMseqs2 做 chain-level all-vs-all 相似性搜索
3. 如果两个 chain 满足 identity >= 40% 且 coverage >= 0.8，则把它们所在的 PDB complex 连接到同一个 component
4. 在 component 级别分配 train / valid / test
5. 输出 PDB ID 列表形式的 Interformer-compatible split 文件
6. 再独立验证跨 split 是否存在 identity >= 40% 且 coverage >= 0.8 的命中
```

因此，sequence similarity 是约束条件，最终被切分的是 PDB 结构样本。当前 split 已通过序列泄漏检查：

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

### 3.0 Linux 服务器从切分数据复现到当前阶段

服务器上的项目目录假设为：

```text
/tmp/ai4s/ai4s_final/
```

当前阶段需要以下输入已经存在：

```text
data/raw/pdbbind2020/
data/processed/sequence_cluster_split_validation/all_raw/pdbbind_sequence_cluster_split_table.csv
str/split_sequence_cluster_all_raw/
```

其中：

```text
data/raw/pdbbind2020/
```

来自 PDBbind 原始数据，包含 `index/index/INDEX_general_PL.2020R1.lst` 和 `complexes/P-L/` 下的 protein、pocket、ligand 文件。

```text
str/split_sequence_cluster_all_raw/
```

是当前项目使用的 sequence-cluster split，至少应包含：

```text
timesplit_no_lig_overlap_train
timesplit_no_lig_overlap_val
timesplit_test
```

```text
data/processed/sequence_cluster_split_validation/all_raw/pdbbind_sequence_cluster_split_table.csv
```

是切分验证后生成的中间表，`create_esm_manifest.py` 默认从该表读取路径、标签、split 元信息，再生成训练 manifest。该文件不属于 PDBbind 官方原始数据；如果服务器上没有，需要从本地同步或重新运行 sequence-cluster split 生成流程。

#### 3.0.1 重新生成 PDB 结构级 split

如果服务器上还没有 `str/split_sequence_cluster_all_raw/`，需要先从 PDBbind 原始结构生成 PDB ID 级 split。输入是：

```text
data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst
data/raw/pdbbind2020/complexes/P-L/
split/
```

其中 `split/` 是原 Interformer split 文件目录，用于提供目标 train/valid/test 比例和辅助评估列表。生成 all-raw PDB 结构级 split：

```bash
python scripts/create_sequence_cluster_split.py \
  --index-path data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst \
  --complex-root data/raw/pdbbind2020/complexes/P-L \
  --source-split-dir split \
  --output-split-dir split_sequence_cluster_all_raw \
  --output-dir data/processed/sequence_cluster_split_all_raw \
  --universe all_raw \
  --min-seq-id 0.4 \
  --coverage 0.8 \
  --min-length 30 \
  --seeds 256
```

该步骤会输出：

```text
split_sequence_cluster_all_raw/
├── timesplit_no_lig_overlap_train
├── timesplit_no_lig_overlap_val
├── timesplit_test
├── coresetlist
├── diff_test+core
├── posebusters_pdb_ccd_ids.txt
├── timesplit_test_no_rec_overlap
└── timesplit_test_sanitizable

data/processed/sequence_cluster_split_all_raw/
├── chain_sequences.csv
├── pdbbind_seqid_40_all_vs_all.m8
├── pdbbind_sequence_cluster_splits.csv
├── sequence_component_assignments.csv
├── cross_split_sequence_violations.csv
└── sequence_cluster_split_report.json
```

当前 `str` 方案默认读取 `str/split_sequence_cluster_all_raw/`，因此生成后需要复制一份到 `str/` 下：

```bash
mkdir -p str/split_sequence_cluster_all_raw
cp split_sequence_cluster_all_raw/* str/split_sequence_cluster_all_raw/
```

当前已验证的结构级 split 结果：

```text
train: 17608
valid: 1038
test: 391
cross_split_violation_count: 0
leakage_detected: false
```

注意：这里没有把 `data/raw/pdbbind2020/complexes/P-L/` 下的结构文件物理拆成三个目录。切分结果是 PDB ID 列表；manifest 会将 PDB ID 映射回原始结构文件路径，保证同一个 PDB complex 的 protein、pocket、ligand 始终作为一个整体进入同一个 split。

如果需要生成 `create_esm_manifest.py` 默认使用的中间表，应保留或同步：

```text
data/processed/sequence_cluster_split_validation/all_raw/pdbbind_sequence_cluster_split_table.csv
```

该表记录每个 PDB ID 的 split、结构文件路径、亲和力标签和验证结果，是从结构级 split 到训练 manifest 的桥接文件。

如果需要从本地同步这些切分相关文件到服务器，可在本地仓库根目录执行：

```bash
rsync -avh --progress --partial \
  -e "ssh -p 10731" \
  str/split_sequence_cluster_all_raw/ \
  root@connect.bjb2.seetacloud.com:/tmp/ai4s/ai4s_final/str/split_sequence_cluster_all_raw/

rsync -avh --progress --partial \
  -e "ssh -p 10731" \
  data/processed/sequence_cluster_split_validation/all_raw/ \
  root@connect.bjb2.seetacloud.com:/tmp/ai4s/ai4s_final/data/processed/sequence_cluster_split_validation/all_raw/
```

服务器上先检查关键输入：

```bash
cd /tmp/ai4s/ai4s_final

test -f data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst
test -d data/raw/pdbbind2020/complexes/P-L
test -f data/processed/sequence_cluster_split_validation/all_raw/pdbbind_sequence_cluster_split_table.csv
test -f str/split_sequence_cluster_all_raw/timesplit_no_lig_overlap_train
test -f str/split_sequence_cluster_all_raw/timesplit_no_lig_overlap_val
test -f str/split_sequence_cluster_all_raw/timesplit_test
```

安装依赖时，如果 PyTorch 已由云服务器镜像预装，可先安装其余依赖；若 pip 源只指向 PyTorch CUDA 源，需要显式指定 PyPI 或镜像源：

```bash
python -m pip install -r str/requirements.txt \
  -i https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

从切分数据开始，到当前阶段的完整执行顺序如下：

```bash
# 1. 由 sequence-cluster split 和中间表生成完整 manifest
python str/scripts/data/create_esm_manifest.py \
  --split-dir str/split_sequence_cluster_all_raw

# 2. 验证 manifest 的标签、路径、split 一致性
python str/scripts/data/validate_manifest.py \
  --split-dir str/split_sequence_cluster_all_raw \
  --ligand-parse-limit 0

# 3. 全量验证 ligand 是否可被 RDKit 解析
python str/scripts/data/validate_manifest.py \
  --split-dir str/split_sequence_cluster_all_raw \
  --ligand-parse-limit -1

# 4. 过滤 4 个不可构图 ligand，生成可训练 manifest
python str/scripts/data/create_trainable_manifest.py

# 5. 缓存 ligand graph
python str/scripts/data/cache_ligand_graphs.py

# 6. 验证 ligand graph batch
python str/scripts/data/smoke_test_graph_batch.py --split train --batch-size 8
python str/scripts/data/smoke_test_graph_batch.py --split valid --batch-size 8 \
  --report-json str/manifest/esm_affinity_graph_batch_smoke_valid_report.json
python str/scripts/data/smoke_test_graph_batch.py --split test --batch-size 8 \
  --report-json str/manifest/esm_affinity_graph_batch_smoke_test_report.json

# 7. 在 Linux GPU 上缓存 ESM residue embedding
python str/scripts/data/cache_esm_embeddings.py \
  --manifest str/manifest/esm_affinity_trainable_manifest.csv \
  --model-name facebook/esm2_t12_35M_UR50D \
  --cache-dir str/manifest/cache/esm_embeddings \
  --report-json str/manifest/cache/esm_embeddings_report.json \
  --device cuda \
  --float16-output

# 8. 合并 ESM embedding 与 ligand graph，验证训练 batch
python str/scripts/data/build_training_batch.py \
  --manifest str/manifest/esm_affinity_trainable_manifest.csv \
  --esm-cache-dir str/manifest/cache/esm_embeddings \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --report-json str/manifest/esm_ligand_training_batch_report.json \
  --split train \
  --limit 128 \
  --batch-size 8
```

执行完成后，应得到当前阶段的核心产物：

```text
str/manifest/esm_affinity_manifest.csv
str/manifest/esm_affinity_trainable_manifest.csv
str/manifest/cache/ligand_graphs/
str/manifest/cache/esm_embeddings/
str/manifest/esm_ligand_training_batch_report.json
```

当前已知数据规模：

```text
完整 manifest: 19037
train / valid / test: 17608 / 1038 / 391
trainable manifest: 19033
过滤 ligand: 2pll, 3vjs, 3vjt, 4hrd
trainable train / valid / test: 17604 / 1038 / 391
```

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

### 3.7 构造 ESM + ligand graph 训练 batch

该步骤把以下两个缓存按 `pdb_id` 合并：

```text
str/manifest/cache/esm_embeddings/{pdb_id}.pt
str/manifest/cache/ligand_graphs/{pdb_id}.pt
```

输出 batch 字段：

```text
pdb_id
split
protein_embedding          [batch_size, max_protein_len, esm_hidden_dim]
protein_mask               [batch_size, max_protein_len]
protein_lengths            [batch_size]
ligand_atom_features       [total_atoms, 9]
ligand_atom_coordinates    [total_atoms, 3]
ligand_bond_index          [2, total_directed_edges]
ligand_bond_features       [total_directed_edges, 4]
ligand_batch               [total_atoms]
labels                     [batch_size]
```

Mac 本地 debug cache 验证：

```bash
python str/scripts/data/build_training_batch.py \
  --manifest str/manifest/esm_affinity_trainable_manifest.csv \
  --esm-cache-dir str/manifest/cache/esm_embeddings_debug \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --report-json str/manifest/esm_ligand_training_batch_debug_report.json \
  --split train \
  --limit 20 \
  --batch-size 4
```

当前本地 debug 验证结果：

```text
status: PASS
protein_embedding: [4, 418, 320]
ligand_atom_features: [223, 9]
ligand_bond_index: [2, 456]
labels: [4]
```

Linux GPU 全量 cache 验证：

```bash
python str/scripts/data/build_training_batch.py \
  --manifest str/manifest/esm_affinity_trainable_manifest.csv \
  --esm-cache-dir str/manifest/cache/esm_embeddings \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --report-json str/manifest/esm_ligand_training_batch_report.json \
  --split train \
  --limit 128 \
  --batch-size 8
```

如需随机抽样检查，可额外添加：

```bash
--sample-mode random --seed 42
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

### 4.2 验证 ESM + ligand graph dataloader

已实现：

```text
str/scripts/data/build_training_batch.py
```

该脚本会检查 ESM cache 与 ligand graph cache 是否齐全，并构造训练 batch。Linux 全量缓存完成后，应先运行：

```bash
python str/scripts/data/build_training_batch.py \
  --esm-cache-dir str/manifest/cache/esm_embeddings \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --split train \
  --limit 128 \
  --batch-size 8
```

### 4.3 训练 frozen ESM baseline

已实现：

```text
str/scripts/train/train_frozen_esm_baseline.py
```

第一版模型保持简单，作为后续 GNN / pocket pooling / cross-attention 的对照基线：

```text
protein_embedding + protein_mask -> masked mean pooling
ligand_atom_features + ligand_batch -> atom mean pooling
concat -> MLP -> pAffinity
loss: MSELoss
```

本地或服务器小规模 smoke train：

```bash
PYTHONPATH=$(pwd) python str/scripts/train/train_frozen_esm_baseline.py \
  --esm-cache-dir str/manifest/cache/esm_embeddings \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --train-limit 512 \
  --valid-limit 128 \
  --test-limit 128 \
  --epochs 3 \
  --batch-size 8 \
  --device cuda
```

如果只想用本地 debug ESM cache 验证脚本逻辑，可临时把 train split 当作 valid/test：

```bash
PYTHONPATH=$(pwd) python str/scripts/train/train_frozen_esm_baseline.py \
  --esm-cache-dir str/manifest/cache/esm_embeddings_debug \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --train-split train \
  --valid-split train \
  --test-split train \
  --train-limit 12 \
  --valid-limit 4 \
  --test-limit 4 \
  --epochs 2 \
  --batch-size 4 \
  --device auto \
  --output-dir str/manifest/outputs/baseline_frozen_esm_debug
```

当前本地 debug 训练验证结果：

```text
status: PASS
epochs: 2
train rows / valid rows / test rows: 12 / 4 / 4
best epoch: 2
valid RMSE: 4.2290
test RMSE: 4.2290
output: str/manifest/outputs/baseline_frozen_esm_debug/
```

Linux GPU 全量 baseline：

```bash
PYTHONPATH=$(pwd) python str/scripts/train/train_frozen_esm_baseline.py \
  --esm-cache-dir str/manifest/cache/esm_embeddings \
  --ligand-cache-dir str/manifest/cache/ligand_graphs \
  --epochs 30 \
  --batch-size 16 \
  --hidden-dim 256 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --device cuda
```

输出：

```text
str/manifest/outputs/baseline_frozen_esm/
├── checkpoints/
│   └── best.pt
├── metrics.json
├── history.csv
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
