# BindingDB 蛋白序列相似性约束切分管线

本目录包含 BindingDB 数据准备脚本，用于构建 sequence-based 蛋白-配体亲和力预测实验所需的 `train / valid / test` 数据集。

本管线的核心目标是：训练集、验证集和测试集之间不能存在 `identity >= 40%` 的蛋白序列相似性命中，从而降低蛋白相似性导致的数据泄露风险。

需要特别注意：**流式读取原始 TSV 本身不能保证 40% 相似性约束**。流式读取只负责处理大文件，避免一次性加载 8GB 级别的 `BindingDB_All.tsv`。真正保证切分约束的是：

1. 从清洗后的样本中提取唯一蛋白序列；
2. 使用 MMseqs2 计算蛋白序列相似性；
3. 将 `identity >= 40%` 且覆盖度满足阈值的蛋白连成 connected component；
4. 在 component 层面分配 `train / valid / test`；
5. 对最终 split 再做一次独立 cross-split MMseqs 验证。

## 原始数据来源

论文给出的 BindingDB 入口：

```text
https://www.bindingdb.org/rwd/bind/index.jsp
```

实际下载建议进入 BindingDB 下载页：

```text
https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp
```

BindingDB 下载页通常按月更新文件。当前项目使用的是 202606 版本；后续如果页面更新到新月份，文件名中的年月后缀可能会变化。

## 需要下载的文件

本项目第一阶段需要以下文件。

### 必需文件

1. `BindingDB_All_202606_tsv.zip`

   下载位置：BindingDB 下载页中的 `Ligand-Target-Affinity Datasets` -> `All data in BindingDB`。

   作用：主 ligand-target-affinity 表。该文件一行对应一次 binding measurement，包含 ligand SMILES、target 信息、Ki / Kd / IC50 / EC50、蛋白序列、PDB ID 和 UniProt 字段。

   解压后应得到：

   ```text
   BindingDB_All.tsv
   ```

2. `BindingDBTargetSequences.fasta`

   下载位置：BindingDB 下载页中的 `Lists and identifier mappings`。

   作用：BindingDB 所有 protein targets 的 FASTA 序列。当前脚本主要从 `BindingDB_All.tsv` 直接读取 target chain sequence，但该 FASTA 文件可用于后续审计和补全。

   浏览器下载后本地文件名可能变成：

   ```text
   BindingDBTargetSequences.fasta.txt
   ```

   这不影响当前项目使用。

3. `BindingDB_UniProt.txt`

   下载位置：BindingDB 下载页中的 `Lists and identifier mappings`。

   作用：BindingDB polymer target ID 到 UniProt ID 的映射。后续做 target 去重、外部验证或报告整理时会用到。

### 推荐下载的辅助文件

4. `BindingDB_Assays_202606_tsv.zip`

   作用：Entry ID + Assay ID 到 assay 文本描述的映射。若后续要分析实验条件、assay 类型或筛掉某些 assay，可以使用该文件。

5. `BindingDB_rsid_eaids_202606_tsv.zip`

   作用：BindingDB reaction set ID 到 Entry ID + Assay ID 的映射。可与 `BindingDB_Assays_202606_tsv.zip` 联合使用。

## 本地目录结构

请将下载和解压后的文件放在：

```text
seq/data/
```

期望结构为：

```text
seq/data/BindingDB_All.tsv
seq/data/BindingDB_All_202606_tsv.zip
seq/data/BindingDBTargetSequences.fasta.txt
seq/data/BindingDB_UniProt.txt
```

如果同时下载了 assay 辅助文件，可以放在同一目录：

```text
seq/data/BindingDB_Assays_202606_tsv.zip
seq/data/BindingDB_rsid_eaids_202606_tsv.zip
```

解压主 TSV 的命令示例：

```bash
unzip seq/data/BindingDB_All_202606_tsv.zip -d seq/data/
```

解压后请确认存在：

```bash
ls -lh seq/data/BindingDB_All.tsv
```

## 脚本输出目录

默认输出目录为：

```text
seq/processed/bindingdb_sequence_split/
```

主要输出文件包括：

```text
bindingdb_clean.csv
unique_proteins.csv
unique_proteins.fasta
mmseqs_all_vs_all.tsv
protein_components.csv
component_splits.csv
bindingdb_clean_with_split.csv
train.csv
valid.csv
test.csv
cross_split_leakage_hits.csv
clean_report.json
cluster_report.json
split_report.json
validation_report.json
```

其中：

- `bindingdb_clean.csv`：清洗和重复聚合后的 BindingDB 样本表；
- `unique_proteins.fasta`：参与切分的唯一蛋白序列；
- `protein_components.csv`：40% 相似性阈值下得到的蛋白 component；
- `bindingdb_clean_with_split.csv`：带有 split 标记的完整样本表；
- `train.csv`、`valid.csv`、`test.csv`：最终训练、验证、测试文件；
- `validation_report.json`：最终泄露验证报告。

## 依赖

清洗、切分和报告写出主要依赖 Python 标准库。

为了显示长任务进度条，建议安装 `tqdm`：

```bash
pip install tqdm
```

如果没有安装 `tqdm`，脚本仍然可以运行，只是不会显示进度条。

蛋白序列聚类和 cross-split 相似性验证需要 MMseqs2。脚本会优先检查 `PATH` 中的 `mmseqs`，也会检查本机常见路径：

```text
/opt/homebrew/bin/mmseqs
```

如果没有安装 MMseqs2，`cluster` 和 `validate` 步骤无法完成。

## 推荐首次运行

建议先做一个小规模调试运行，确认字段解析和输出格式正常：

```bash
python seq/scripts/data/bindingdb_sequence_split.py clean \
  --raw-tsv seq/data/BindingDB_All.tsv \
  --affinity-types Ki \
  --limit-rows 200000
```

确认清洗逻辑无误后，再运行完整 Ki-only 管线：

```bash
python seq/scripts/data/bindingdb_sequence_split.py run-all \
  --raw-tsv seq/data/BindingDB_All.tsv \
  --affinity-types Ki \
  --min-seq-id 0.4 \
  --coverage 0.8 \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 2026
```

完整数据的 `clean` 阶段会显示流式读取进度；`cluster` 和 `validate` 阶段会显示 MMseqs 输出解析和 cross-split 验证进度。

## 分步骤运行

### 1. 清洗 BindingDB 原始 TSV

```bash
python seq/scripts/data/bindingdb_sequence_split.py clean \
  --raw-tsv seq/data/BindingDB_All.tsv \
  --affinity-types Ki
```

该步骤会：

- 流式读取 `BindingDB_All.tsv`；
- 保留配体 SMILES、蛋白序列、target 信息和亲和力标签；
- 默认只保留单链 target；
- 默认排除 `>10000`、`<1` 等非精确数值；
- 将 nM 单位标签转换为 `pAffinity = 9 - log10(value_nM)`；
- 对同一 `protein_sequence + ligand_smiles + affinity_type` 的重复记录取中位数。

### 2. 构建 40% 相似性 protein component

```bash
python seq/scripts/data/bindingdb_sequence_split.py cluster \
  --min-seq-id 0.4 \
  --coverage 0.8
```

该步骤会：

- 从 `unique_proteins.fasta` 中读取唯一蛋白序列；
- 使用 MMseqs2 做 all-vs-all search；
- 使用 `identity >= 40%` 和 `coverage >= 80%` 作为相似性边；
- 将相似蛋白合并为 connected component。

### 3. 在 component 层面切分数据

```bash
python seq/scripts/data/bindingdb_sequence_split.py split \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 2026
```

该步骤不会按样本行随机切分，而是将整个 protein component 放入同一个 split。这样可以避免同一相似蛋白家族同时出现在训练集和测试集中。

### 4. 独立验证 cross-split 相似性泄露

```bash
python seq/scripts/data/bindingdb_sequence_split.py validate \
  --min-seq-id 0.4 \
  --coverage 0.8
```

该步骤会重新对 `train / valid / test` 之间的蛋白 FASTA 做 MMseqs search。最终验证结果写入：

```text
validation_report.json
cross_split_leakage_hits.csv
```

## 如何判断切分是否合格

只有当 `validation_report.json` 中出现如下结果时，才能认为该 split 满足本项目的蛋白相似性约束：

```json
{
  "hit_count": 0,
  "leakage_detected": false,
  "min_seq_id": 0.4,
  "coverage": 0.8
}
```

如果 `leakage_detected` 为 `true`，则不能声称该数据集满足 `<40%` 蛋白相似性要求。此时应基于当前清洗后的数据重新构建 component、重新切分，并再次运行验证。

## Notebook 可视化

切分完成后，可打开：

```text
seq/bindingdb_sequence_split_analysis.ipynb
```

该 notebook 会展示：

- 原始 TSV 到 clean dataset 的清洗漏斗；
- pAffinity 和蛋白长度分布；
- protein component 大小分布；
- 蛋白聚类散点图；
- `train / valid / test` 样本、蛋白、component 数量；
- cross-split 相似性验证结果。

## 重要说明

- 第一版 baseline 建议只使用一种亲和力类型，通常先使用 `Ki`，避免混合不同实验含义的标签。
- 默认清洗器只保留单链 target 和精确数值亲和力。
- 默认标签转换公式为 `pAffinity = 9 - log10(value_nM)`。
- 切分单位是 protein component，不是 binding record。
- 如果某个 component 很大，最终 `train / valid / test` 比例可能偏离 8:1:1；这是为了优先保证蛋白相似性不泄露。
- 完整数据的 MMseqs all-vs-all 可能耗时较长，建议先用 `--limit-rows` 做调试，再运行全量数据。
