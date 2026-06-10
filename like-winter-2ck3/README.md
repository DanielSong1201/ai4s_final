\# 第五阶段：真实配体特征（ECFP指纹）



\## 实验结果



| 模型 | 测试集 R² | 测试集 RMSE |

|------|----------|------------|

| 随机特征基线 | -0.014 | 1.691 |

| \*\*真实配体特征\*\* | \*\*0.326\*\* | \*\*1.382\*\* |



\## 文件说明



\- `scripts/` - 数据处理和模型训练脚本

\- `data/processed/` - 训练/验证/测试集 CSV 文件

\- `config.py` - 路径配置文件



\## 运行方法



```bash

\# 激活虚拟环境

.\\venv\_openbabel\\Scripts\\activate



\# 运行脚本

python like-winter-2ck3/scripts/04b\_real\_features\_fixed.py

