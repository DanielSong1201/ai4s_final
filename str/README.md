# 基于 ESM 的蛋白-小分子结合亲和力预测

本目录用于完成一个从 PDBbind 结构数据出发的结合亲和力预测实验。当前目标是：在已有结构级划分的基础上，生成训练 manifest，缓存蛋白 ESM embedding 与小分子 graph，验证训练 batch，并训练一个 frozen ESM baseline。

## 1. 项目结构

与当前实验直接相关的文件结构如下：

```text
final_project/
├── data/
│   ├── raw/
│   │   └── pdbbind2020/
│   │       └── complexes/P-L/
│   │           └── {pdb_id}/
│   │               ├── {pdb_id}_protein.pdb
│   │               ├── {pdb_id}_pocket.pdb
│   │               ├── {pdb_id}_ligand.sdf
│   │               └── {pdb_id}_ligand.mol2
│   └── processed/
│       └── sequence_cluster_split_all_raw/
│           └── pdbbind_sequence_cluster_splits.csv
├── scripts/
│   ├── create_sequence_cluster_split.py
│   ├── sequence_leakage_check.py
│   └── create_interformer_splits.py
└── str/
    ├── README.md
    ├── requirements.txt
    ├── split_sequence_cluster_all_raw/
    │   ├── timesplit_no_lig_overlap_train
    │   ├── timesplit_no_lig_overlap_val
    │   └── timesplit_test
    ├── split_iid_all_raw/
    │   ├── timesplit_no_lig_overlap_train
    │   ├── timesplit_no_lig_overlap_val
    │   └── timesplit_test
    ├── scripts/
    │   ├── split_raw_pdbbind.sh
    │   ├── build_manifest_from_split.sh
    │   ├── validate_after_manifest.sh
    │   ├── run_frozen_esm_baseline.sh
    │   ├── data/
    │   │   ├── create_iid_structure_split.py
    │   │   ├── create_esm_manifest.py
    │   │   ├── validate_manifest.py
    │   │   ├── create_trainable_manifest.py
    │   │   ├── cache_ligand_graphs.py
    │   │   ├── cache_esm_embeddings.py
    │   │   ├── build_training_batch.py
    │   │   ├── smoke_test_manifest_batch.py
    │   │   └── smoke_test_graph_batch.py
    │   └── train/
    │       └── train_frozen_esm_baseline.py
    └── manifest/
        ├── esm_affinity_manifest.csv
        ├── esm_affinity_manifest_report.json
        ├── esm_affinity_manifest_validation_report.json
        ├── esm_affinity_trainable_manifest.csv
        ├── esm_affinity_trainable_manifest_report.json
        ├── ligand_parse_failures.csv
        ├── general_PL_2020_sequence_cluster_all_raw.csv
        ├── cache/
        │   ├── ligand_graphs/
        │   ├── ligand_graphs_report.json
        │   ├── esm_embeddings/
        │   └── esm_embeddings_report.json
        └── outputs/
            └── baseline_frozen_esm/
                ├── checkpoints/best.pt
                ├── metrics.json
                ├── history.csv
                ├── predictions_valid.csv
                └── predictions_test.csv
```

说明：

- `str/split_sequence_cluster_all_raw/` 保存的是结构级 split 文件，文件内容是 PDB ID 列表，不是物理复制后的结构目录。
- `data/processed/sequence_cluster_split_all_raw/pdbbind_sequence_cluster_splits.csv` 是从原始 PDBbind 结构和 sequence-cluster split 生成的桥接表，包含结构路径、标签、split 等信息。
- `str/manifest/` 是当前实验的工作目录，保存 manifest、缓存、验证报告和 baseline 输出。

## 2. Requirements

建议在 Linux GPU 服务器上完成全量 ESM 缓存和 baseline 训练。Mac 本地只建议做小规模 smoke test。

### 2.1 创建虚拟环境

推荐使用 conda：

```bash
conda create -n ai4s python=3.11 -y
conda activate ai4s
python -m pip install --upgrade pip
```

如果已经有可用环境，也可以直接激活已有环境，但需要保证 Python、PyTorch、RDKit、Transformers 等依赖可用。

### 2.2 安装 PyTorch

Linux GPU 机器按 CUDA 版本安装。以 CUDA 12.1 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果服务器已经预装了 CUDA 版 PyTorch，可以先检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY
```

Mac 本地验证可安装 CPU/MPS 版本：

```bash
pip install torch torchvision torchaudio
```

### 2.3 安装项目依赖

```bash
pip install -r str/requirements.txt
```

`str/requirements.txt` 覆盖当前流程需要的主要依赖：

```text
torch
numpy
pandas
rdkit
transformers
huggingface_hub
safetensors
scikit-learn
tqdm
```

如果 `rdkit` 通过 pip 安装失败，可以改用 conda-forge：

```bash
conda install -c conda-forge rdkit -y
```

### 2.4 验证依赖

```bash
python - <<'PY'
import torch
import pandas
import numpy
import tqdm
from rdkit import Chem
from transformers import AutoTokenizer, EsmModel

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("rdkit ok")
print("transformers ok")
PY
```

### 2.5 Hugging Face ESM 模型

第一次运行 ESM 缓存脚本时会自动下载模型。默认建议先用：

```text
facebook/esm2_t6_8M_UR50D
```

服务器可以联网时直接运行缓存脚本即可。如果服务器不能联网，需要提前在可联网机器下载 Hugging Face cache，再同步到服务器，并在运行时设置：

```bash
export HF_HOME=/path/to/huggingface_cache
```

如果模型已经在本地 cache 中，可以使用：

```bash
ESM_LOCAL_FILES_ONLY=1 PYTHONPATH=$(pwd) bash str/scripts/validate_after_manifest.sh
```

## 3. 当前 Baseline 思路

当前 baseline 是 frozen ESM + ligand graph mean pooling + MLP regression。它不是最终模型，而是后续结构感知模型的对照基线。

数据流：

```text
protein_sequence
  -> Hugging Face ESM
  -> residue-level embedding cache
  -> masked mean pooling

ligand_sdf / ligand_mol2
  -> RDKit
  -> atom feature + bond graph cache
  -> atom mean pooling

protein vector + ligand vector
  -> concat
  -> MLP
  -> pAffinity
```

训练目标：

```text
target = pAffinity = -log10(affinity in mol/L)
loss   = MSELoss 或 HuberLoss
metric = RMSE / MAE / R2 / Pearson / Spearman
```

当前 baseline 的主要限制：

- 蛋白只使用全序列 ESM embedding 的 mean pooling，没有显式 pocket pooling。
- 小分子 graph 只做 atom mean pooling，没有使用 GNN 消息传递。
- 没有建模蛋白 residue 与 ligand atom 之间的空间交互。
- ESM 是 frozen cache，不参与反向传播。

因此 baseline 的意义是建立一个可复现、可比较的最低训练闭环：数据切分、缓存、batch、训练、评估、预测文件都能稳定产出。

## 4. 从原始 PDBbind 结构生成 Train/Valid/Test Split

本项目的 split 单位是 **PDB complex / PDB ID**。每个样本对应一组结构文件：

```text
{pdb_id}_protein.pdb
{pdb_id}_pocket.pdb
{pdb_id}_ligand.sdf
{pdb_id}_ligand.mol2
```

这组文件必须整体进入同一个 split。这里不会把 protein sequence 片段切进不同 split；sequence 只用于计算蛋白同源关系，作为防止结构样本泄漏的约束。

### 4.1 两种切分模式

统一入口：

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh
```

该脚本通过 `IID` 控制切分策略：

```text
IID=true   完全随机切分 PDB complex，忽略 40% sequence similarity 先验
IID=false  仍然切分 PDB complex，但使用 40% sequence similarity 先验约束
```

`IID=false` 是默认值，也是当前更适合作为主要实验的严格 split。它的逻辑是：

```text
1. 从每个 PDB complex 的 *_protein.pdb 中抽取 chain sequence
2. 用 MMseqs2 计算 chain-level all-vs-all similarity
3. 若两个 chain 满足 identity >= 40% 且 coverage >= 0.8，则连接它们所属的 PDB complex
4. 在 PDB complex connected component 级别分配 train / valid / test
5. 输出 Interformer-compatible split 文件
6. 再独立运行 sequence leakage check
```

注意：第 1-3 步使用 sequence similarity，但最终被分配的是 PDB complex 结构样本，而不是 sequence 片段。

### 4.2 默认输出

`IID=false` 默认输出：

```text
str/split_sequence_cluster_all_raw/
├── timesplit_no_lig_overlap_train
├── timesplit_no_lig_overlap_val
├── timesplit_test
├── coresetlist
├── diff_test+core
├── posebusters_pdb_ccd_ids.txt
├── timesplit_test_no_rec_overlap
└── timesplit_test_sanitizable

data/processed/sequence_cluster_split_all_raw/
├── pdbbind_sequence_cluster_splits.csv
├── sequence_cluster_split_report.json
├── sequence_component_assignments.csv
├── cross_split_sequence_violations.csv
└── sequence_leakage_check/
```

`IID=true` 默认输出：

```text
str/split_iid_all_raw/
├── timesplit_no_lig_overlap_train
├── timesplit_no_lig_overlap_val
├── timesplit_test
└── ...

data/processed/iid_split_all_raw/
├── pdbbind_sequence_cluster_splits.csv
├── iid_structure_split_report.json
└── sequence_leakage_check/
```

为了复用后续 manifest 生成脚本，IID 模式下的桥接表也命名为 `pdbbind_sequence_cluster_splits.csv`。这里的文件名是为了兼容接口，不表示 IID 模式做了 sequence-cluster 约束。

### 4.3 严格结构级 Split

默认严格模式：

```bash
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh
```

等价于：

```bash
IID=false \
INDEX_PATH=data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst \
COMPLEX_ROOT=data/raw/pdbbind2020/complexes/P-L \
SOURCE_SPLIT_DIR=split \
OUTPUT_SPLIT_DIR=str/split_sequence_cluster_all_raw \
OUTPUT_DIR=data/processed/sequence_cluster_split_all_raw \
MIN_SEQ_ID=0.4 \
COVERAGE=0.8 \
MIN_LENGTH=30 \
SEEDS=128 \
UNIVERSE=all_raw \
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh
```

阶段输出：

```text
[1/3] Create sequence-cluster split with 40% similarity prior
[2/3] Verify Interformer-style split files
[3/3] Check train-vs-valid/test sequence leakage
```

长耗时步骤会显示 `tqdm` 进度条，包括结构样本的 chain 抽取、component assignment seed 搜索、leakage check chain 抽取等。

### 4.4 IID 随机结构级 Split

如果需要一个 IID/random split 作为对照：

```bash
IID=true \
OUTPUT_SPLIT_DIR=str/split_iid_all_raw \
OUTPUT_DIR=data/processed/iid_split_all_raw \
SEED=42 \
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh
```

IID 模式会随机分配 PDB complex ID。它不会阻止同源蛋白跨 split，因此 sequence leakage check 可能报告命中；这不是脚本错误，而是 IID/random split 的预期代价。该模式可用于观察随机切分下的上限表现，但不能作为严格 cold-protein 泛化结论。

### 4.5 切分结果如何接入 Manifest

严格 split 接入 manifest：

```bash
SOURCE_CSV=data/processed/sequence_cluster_split_all_raw/pdbbind_sequence_cluster_splits.csv \
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh \
  --split-dir str/split_sequence_cluster_all_raw
```

IID split 接入 manifest：

```bash
SOURCE_CSV=data/processed/iid_split_all_raw/pdbbind_sequence_cluster_splits.csv \
MANIFEST_DIR=str/manifest_iid \
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh \
  --split-dir str/split_iid_all_raw
```

## 5. 从已切分数据到 Baseline 的完整流程

以下流程假设你已经有：

```text
str/split_sequence_cluster_all_raw/timesplit_no_lig_overlap_train
str/split_sequence_cluster_all_raw/timesplit_no_lig_overlap_val
str/split_sequence_cluster_all_raw/timesplit_test
data/processed/sequence_cluster_split_all_raw/pdbbind_sequence_cluster_splits.csv
data/raw/pdbbind2020/complexes/P-L/
```

如果数据目录被移动到大容量磁盘，确保当前位置仍能通过软链接访问到 `data/`。

### 5.1 安装依赖

```bash
pip install -r str/requirements.txt
```

服务器建议使用 CUDA 版 PyTorch。Mac 本地只建议做小规模验证，不建议全量缓存 ESM。

### 5.2 生成 manifest

一键从结构级 split 生成 manifest：

```bash
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh
```

该脚本会分阶段输出：

```text
[START] [1/4] Build full ESM affinity manifest
[DONE]  [1/4] ...
[START] [2/4] Validate manifest paths, labels, sequences, and split membership
[DONE]  [2/4] ...
[START] [3/4] Create trainable manifest by filtering unparseable ligands
[DONE]  [3/4] ...
[START] [4/4] Summary
[DONE]  [4/4] Summary
```

默认输入：

```text
SOURCE_CSV=data/processed/sequence_cluster_split_all_raw/pdbbind_sequence_cluster_splits.csv
SPLIT_DIR=str/split_sequence_cluster_all_raw
MANIFEST_DIR=str/manifest
```

`SPLIT_DIR` 可以直接在 `str/scripts/build_manifest_from_split.sh` 顶部修改，也可以通过环境变量或命令行参数覆盖。优先级为：

```text
--split-dir 参数 > SPLIT_DIR 环境变量 > bash 文件中的默认值
```

README 的默认运行命令不传 `--split-dir`，因此会使用脚本默认值：

```bash
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh
```

如果要临时使用另一个同格式 split 目录，推荐显式传参：

```bash
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh \
  --split-dir str/split_iid_all_raw
```

默认输出：

```text
str/manifest/esm_affinity_manifest.csv
str/manifest/esm_affinity_manifest_report.json
str/manifest/esm_affinity_manifest_validation_report.json
str/manifest/general_PL_2020_sequence_cluster_all_raw.csv
str/manifest/esm_affinity_trainable_manifest.csv
str/manifest/ligand_parse_failures.csv
str/manifest/esm_affinity_trainable_manifest_report.json
```

如果路径不同，可用环境变量覆盖：

```bash
SOURCE_CSV=data/processed/sequence_cluster_split_all_raw/pdbbind_sequence_cluster_splits.csv \
SPLIT_DIR=str/split_sequence_cluster_all_raw \
MANIFEST_DIR=str/manifest \
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh
```

如果同时设置了 `SPLIT_DIR` 环境变量和 `--split-dir` 参数，以参数为准：

```bash
SPLIT_DIR=str/split_sequence_cluster_all_raw \
PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh \
  --split-dir str/split_iid_all_raw
```

### 5.3 生成 manifest 后的验证、缓存与 batch 检查

运行：

```bash
PYTHONPATH=$(pwd) bash str/scripts/validate_after_manifest.sh
```

这个脚本会继续完成从 manifest 到训练 batch 的全部准备工作：

```text
[1/6] Validate full manifest paths, labels, sequences, and splits
[2/6] Create trainable manifest by filtering unparseable ligands
[3/6] Cache RDKit ligand graph tensors
[4/6] Cache frozen ESM residue embeddings
[5/6] Smoke test ligand graph batch
[6/6] Validate ESM + ligand training batch
```

长耗时步骤会显示 `tqdm` 进度条，包括：

```text
Parse ligands
Filter trainable ligands
Cache ligand graphs
Cache ESM embeddings
Check ESM/ligand cache
```

常用可调参数：

```bash
MANIFEST=str/manifest/esm_affinity_manifest.csv
TRAINABLE_MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv
SPLIT_DIR=str/split_sequence_cluster_all_raw
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings
ESM_MODEL_NAME=facebook/esm2_t6_8M_UR50D
ESM_DEVICE=cuda
ESM_LIMIT=-1
ESM_FLOAT16_OUTPUT=1
BATCH_SPLIT=train
BATCH_LIMIT=128
BATCH_SIZE=8
GRAPH_BATCH_REPORT=str/manifest/esm_affinity_graph_batch_smoke_report.json
BATCH_REPORT=str/manifest/esm_ligand_training_batch_report.json
```

Linux GPU 全量缓存建议：

```bash
ESM_MODEL_NAME=facebook/esm2_t6_8M_UR50D \
ESM_DEVICE=cuda \
ESM_LIMIT=-1 \
ESM_FLOAT16_OUTPUT=1 \
PYTHONPATH=$(pwd) bash str/scripts/validate_after_manifest.sh
```

如果显存和时间允许，可将模型换为：

```text
facebook/esm2_t12_35M_UR50D
facebook/esm2_t30_150M_UR50D
```

### 5.4 训练 frozen ESM baseline

运行：

```bash
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

这个脚本会先验证训练 batch，再启动 baseline：

```text
[1/2] Validate training batch before baseline
[2/2] Train frozen ESM baseline
```

训练循环会显示 epoch 和 batch 级 `tqdm` 进度条。

常用可调参数：

```bash
MANIFEST=str/manifest/esm_affinity_trainable_manifest.csv
ESM_CACHE_DIR=str/manifest/cache/esm_embeddings
LIGAND_CACHE_DIR=str/manifest/cache/ligand_graphs
OUTPUT_DIR=str/manifest/outputs/baseline_frozen_esm

TRAIN_LIMIT=-1
VALID_LIMIT=-1
TEST_LIMIT=-1
EPOCHS=30
BATCH_SIZE=16
HIDDEN_DIM=256
DROPOUT=0.1
LR=1e-3
WEIGHT_DECAY=1e-4
LOSS=mse
DEVICE=cuda
PRETRAIN_CHECK_LIMIT=128
```

小规模 smoke train：

```bash
TRAIN_LIMIT=512 \
VALID_LIMIT=128 \
TEST_LIMIT=128 \
EPOCHS=3 \
BATCH_SIZE=8 \
DEVICE=cuda \
OUTPUT_DIR=str/manifest/outputs/baseline_frozen_esm_smoke \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

全量 baseline：

```bash
EPOCHS=30 \
BATCH_SIZE=16 \
HIDDEN_DIM=256 \
LR=1e-3 \
WEIGHT_DECAY=1e-4 \
DEVICE=cuda \
OUTPUT_DIR=str/manifest/outputs/baseline_frozen_esm \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

训练输出：

```text
str/manifest/outputs/baseline_frozen_esm/
├── pretrain_batch_check.json
├── checkpoints/
│   └── best.pt
├── metrics.json
├── history.csv
├── predictions_valid.csv
└── predictions_test.csv
```

重点查看：

```bash
cat str/manifest/outputs/baseline_frozen_esm/metrics.json
```

## 6. QuickStart

从原始 PDBbind 结构数据开始，完整跑到 baseline：

```bash
cd /tmp/ai4s/ai4s_final
git switch str
git pull

conda create -n ai4s python=3.11 -y
conda activate ai4s
python -m pip install --upgrade pip

# Linux GPU 示例；如果服务器已经有可用 CUDA 版 PyTorch，可跳过这一行
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r str/requirements.txt

IID=false \
OUTPUT_SPLIT_DIR=str/split_sequence_cluster_all_raw \
OUTPUT_DIR=data/processed/sequence_cluster_split_all_raw \
PYTHONPATH=$(pwd) bash str/scripts/split_raw_pdbbind.sh

PYTHONPATH=$(pwd) bash str/scripts/build_manifest_from_split.sh

ESM_MODEL_NAME=facebook/esm2_t6_8M_UR50D \
ESM_DEVICE=cuda \
ESM_LIMIT=-1 \
ESM_FLOAT16_OUTPUT=1 \
BATCH_SPLIT=train \
BATCH_LIMIT=128 \
BATCH_SIZE=8 \
PYTHONPATH=$(pwd) bash str/scripts/validate_after_manifest.sh

TRAIN_LIMIT=512 \
VALID_LIMIT=128 \
TEST_LIMIT=128 \
EPOCHS=3 \
BATCH_SIZE=8 \
DEVICE=cuda \
OUTPUT_DIR=str/manifest/outputs/baseline_frozen_esm_smoke \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh

EPOCHS=30 \
BATCH_SIZE=16 \
HIDDEN_DIM=256 \
LR=1e-3 \
WEIGHT_DECAY=1e-4 \
DEVICE=cuda \
OUTPUT_DIR=str/manifest/outputs/baseline_frozen_esm \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

如果只想先确认脚本参数和路径，可先运行：

```bash
ESM_LIMIT=32 \
LIGAND_CACHE_LIMIT=32 \
BATCH_LIMIT=16 \
BATCH_SIZE=4 \
TRAIN_LIMIT=32 \
VALID_LIMIT=16 \
TEST_LIMIT=16 \
EPOCHS=1 \
DEVICE=cuda \
OUTPUT_DIR=str/manifest/outputs/baseline_frozen_esm_debug \
PYTHONPATH=$(pwd) bash str/scripts/run_frozen_esm_baseline.sh
```

注意：上面的 debug 命令要求对应的 ESM cache 和 ligand graph cache 已经存在。若只缓存了前 32 条样本，batch 随机抽样可能命中未缓存样本；完整实验建议直接全量缓存。

## 7. 下一步改进方向

baseline 跑通后，后续不应只调 MLP，而应逐步加入结构建模能力。

### 7.1 Ligand GNN

当前 ligand 表示只是 atom mean pooling。下一步可以把 cached ligand graph 输入 GCN、GINE、GraphSAGE 或 Graph Transformer，得到更强的小分子表示：

```text
atom_features + bond_index + bond_features -> ligand GNN -> ligand vector
```

目标是替代简单的 atom mean pooling。

### 7.2 Pocket-aware protein pooling

当前 protein 使用全序列 mean pooling，可能稀释结合口袋信号。下一步可以根据 `pocket.pdb` 中的残基位置或 pocket chain/residue 编号，只池化 pocket residue 的 ESM embedding：

```text
protein ESM residue embedding -> pocket residue mask -> pocket mean / attention pooling
```

这一步通常比直接全序列 pooling 更符合任务目标。

### 7.3 Protein-ligand cross attention

进一步可以让 ligand atom 表示与 pocket residue 表示交互：

```text
pocket residue embeddings <-> ligand atom embeddings
cross-attention / pair bias / distance bias
```

如果加入结构距离，可从 complex 结构中构建 residue-atom 距离矩阵或接触图。

### 7.4 引入三维结构特征

目前 ligand graph 保存了 atom coordinates，但 baseline 没有使用。后续可以加入：

```text
ligand 3D distance
pocket residue coordinates
residue-atom distance
contact map
SE(3)-aware / EGNN / equivariant block
```

这会更贴近“给定蛋白-小分子复合物三维结构预测亲和力”的目标。

### 7.5 ESM 微调或参数高效微调

在 baseline 稳定后，可以尝试：

```text
frozen ESM cache
-> last-layer fine-tuning
-> LoRA / adapter tuning
```

但不建议一开始就端到端微调 ESM，因为显存、速度和过拟合风险都更高。应先保留 frozen ESM baseline 作为对照。

### 7.6 实验记录与对照

每个非 baseline 方向都应保留相同 split 和相同评估指标：

```text
valid RMSE / MAE / Pearson / Spearman
test RMSE / MAE / Pearson / Spearman
best checkpoint by valid RMSE
predictions_valid.csv
predictions_test.csv
```

这样才能判断改进来自模型结构，而不是数据划分或评估方式变化。
