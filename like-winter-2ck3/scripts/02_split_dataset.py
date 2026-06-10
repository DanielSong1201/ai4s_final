"""
脚本2: 将数据集划分为训练集、验证集、测试集
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

def main():
    print("=" * 60)
    print("步骤2: 划分数据集")
    print("=" * 60)
    
    # 1. 读取数据
    input_file = PROCESSED_DATA_DIR / "pdbbind_refined_2020.csv"
    df = pd.read_csv(input_file)
    print(f"\n1. 读取数据: {len(df)} 个复合物")
    
    # 2. 划分数据集 (60% 训练, 20% 验证, 20% 测试)
    print("\n2. 划分数据集 (60% 训练, 20% 验证, 20% 测试)...")
    
    # 先分出训练集和临时集
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.4,  # 40% 给验证+测试
        random_state=42,
        stratify=pd.cut(df['pK'], bins=10, labels=False)  # 按亲和力分层采样
    )
    
    # 再分出验证集和测试集 (各占临时集的一半 = 总数据的20%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=pd.cut(temp_df['pK'], bins=10, labels=False)
    )
    
    # 3. 保存划分结果
    train_file = PROCESSED_DATA_DIR / "train.csv"
    val_file = PROCESSED_DATA_DIR / "val.csv"
    test_file = PROCESSED_DATA_DIR / "test.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\n3. 数据已保存:")
    print(f"   训练集: {len(train_df)} 个 -> {train_file}")
    print(f"   验证集: {len(val_df)} 个 -> {val_file}")
    print(f"   测试集: {len(test_df)} 个 -> {test_file}")
    
    # 4. 统计信息
    print(f"\n4. pK值分布统计:")
    print(f"   训练集: {train_df['pK'].mean():.2f} ± {train_df['pK'].std():.2f}")
    print(f"   验证集: {val_df['pK'].mean():.2f} ± {val_df['pK'].std():.2f}")
    print(f"   测试集: {test_df['pK'].mean():.2f} ± {test_df['pK'].std():.2f}")
    
    # 显示示例
    print(f"\n训练集前5个复合物:")
    print(train_df[['pdb_id', 'pK']].head())
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    train_df, val_df, test_df = main()