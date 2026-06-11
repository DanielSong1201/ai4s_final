import os
from pathlib import Path

# 项目根目录（修改为新的英文路径）
PROJECT_ROOT = Path(r"E:\pdbbind_project")

# 原始数据路径
RAW_DATA_DIR = PROJECT_ROOT / "refined-set"
INDEX_DIR = RAW_DATA_DIR / "index"

# 处理后的数据路径
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# INDEX文件路径
INDEX_FILE = INDEX_DIR / "INDEX_refined_set.2020"

RANDOM_SEED = 42