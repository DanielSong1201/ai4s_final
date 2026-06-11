# PDBbind 亲和力预测项目

## 项目概述

基于 PDBbind v2020 精炼集，提取配体分子的 ECFP 指纹，使用随机森林模型预测蛋白-小分子结合亲和力 (pKd/pKi)。

**作者**: like-winter-2ck3
**日期**: 2026-06-10

---

## 技术路线

原始数据 (PDBbind v2020 Refined Set)
         ↓
    INDEX 文件解析 (提取亲和力标签)
         ↓
      数据清洗 (去除无效样本)
         ↓
   训练/验证/测试集划分 (60%/20%/20%)
         ↓
    配体特征提取 (ECFP 指纹, 1024位)
         ↓
    随机森林模型训练
         ↓
      性能评估 (RMSE, R2)

### 关键技术

- OpenBabel: 读取 MOL2 文件，转换为 SMILES
- RDKit: 从 SMILES 计算 ECFP 指纹
- scikit-learn: 随机森林回归模型
- pandas: 数据处理

---

## 实验结果

### 性能对比

模型: 基线模型
特征类型: 随机特征
测试集 R2: -0.014
测试集 RMSE: 1.691

模型: 真实特征模型
特征类型: 配体 ECFP 指纹
测试集 R2: 0.326
测试集 RMSE: 1.382

### 性能提升

- R2 提升: +0.340 (从负数到 0.326)
- RMSE 降低: -18.3%

### 各数据集表现

训练集: 样本数 3062, RMSE 0.928, R2 0.693
验证集: 样本数 1020, RMSE 1.314, R2 0.394
测试集: 样本数 1020, RMSE 1.382, R2 0.326

### 特征提取成功率

训练集: 98.5% (3062/3108)
验证集: 98.5% (1020/1036)
测试集: 98.4% (1020/1037)

---

## 项目结构

like-winter-2ck3/
├── README.md                           # 项目说明
├── config.py                           # 路径配置文件
├── data/
│   └── processed/                      # 处理后的数据集
│       ├── pdbbind_refined_2020.csv   # 完整数据 (5181条)
│       ├── train.csv                   # 训练集 (3108条)
│       ├── val.csv                     # 验证集 (1036条)
│       └── test.csv                    # 测试集 (1037条)
└── scripts/                            # Python 脚本
    ├── utils.py                        # 工具函数
    ├── 01_parse_index.py              # 解析 INDEX 文件
    ├── 02_split_dataset.py            # 数据集划分
    ├── 03_baseline_model.py           # 基线模型 (随机特征)
    └── 04b_real_features_fixed.py    # 真实特征模型 (ECFP)

---

## 环境配置

### 创建虚拟环境

uv venv --python 3.10.20 venv_openbabel
.\venv_openbabel\Scripts\activate

### 安装依赖

uv pip install scikit-learn pandas tqdm numpy rdkit-pypi openbabel pybel -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install "setuptools<82" -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install "numpy<2" --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install rdkit-pypi --no-deps --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple

---

## 运行方法

### 1. 数据准备

从 PDBbind 官网下载 v2020 Refined Set，解压到项目根目录的 refined-set/ 文件夹。

### 2. 依次运行脚本

# 解析 INDEX 文件
python scripts/01_parse_index.py

# 划分数据集
python scripts/02_split_dataset.py

# 运行基线模型 (随机特征)
python scripts/03_baseline_model.py

# 运行真实特征模型 (ECFP 指纹)
python scripts/04b_real_features_fixed.py

---

## 数据来源

数据集: PDBbind v2020 Refined Set
复合物数量: 5,316 个
有效亲和力数据: 5,181 个 (97.5%)
pK 范围: 2.16 ~ 11.92
pK 均值 ± 标准差: 6.45 ± 1.68

---

## 结论

成功构建了基于配体 ECFP 指纹的亲和力预测模型。相比随机特征基线，R2 从 -0.014 提升至 0.326，RMSE 降低 18.3%，验证了分子结构特征对亲和力预测的有效性。

---

## 后续改进方向

- ESM-2 蛋白质特征: R2 提升至 0.5-0.6 (难度: 中)
- 图神经网络 (GNN): 捕捉空间相互作用 (难度: 高)
- 特征融合 (ECFP + ESM): 进一步提升精度 (难度: 中)
- 加入 General Set 数据: 增加训练数据量 (难度: 低)

---

## 作者

- GitHub: like-winter-2ck3
- 日期: 2026-06-10