"""
脚本3: 基线模型 - 使用随机森林预测亲和力
这是一个简单快速的基线，用于验证数据流程
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

def create_simple_features(df, n_features=50):
    """
    创建简单的数值特征（占位符）
    实际使用中应该用分子指纹和蛋白特征替换
    """
    np.random.seed(42)
    n_samples = len(df)
    # 生成随机特征作为演示
    features = np.random.randn(n_samples, n_features)
    return features

def main():
    print("=" * 60)
    print("步骤3: 基线模型 - 随机森林")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    print(f"   训练集: {len(train_df)}")
    print(f"   验证集: {len(val_df)}")
    print(f"   测试集: {len(test_df)}")
    
    # 2. 创建特征
    print("\n2. 创建特征 (当前使用随机特征占位符)...")
    X_train = create_simple_features(train_df)
    X_val = create_simple_features(val_df)
    X_test = create_simple_features(test_df)
    
    y_train = train_df['pK'].values
    y_val = val_df['pK'].values
    y_test = test_df['pK'].values
    
    # 3. 训练模型
    print("\n3. 训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 4. 评估
    print("\n4. 评估模型性能:")
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"   训练集 RMSE: {train_rmse:.3f}, R²: {train_r2:.3f}")
    print(f"   验证集 RMSE: {val_rmse:.3f}, R²: {val_r2:.3f}")
    print(f"   测试集 RMSE: {test_rmse:.3f}, R²: {test_r2:.3f}")
    
    # 5. 特征重要性（如果有实际特征）
    print("\n5. 模型信息:")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   树的数量: {model.n_estimators}")
    print(f"   最大深度: {model.max_depth}")
    
    print("\n" + "=" * 60)
    print("注意: 当前使用随机特征，性能应该接近随机猜测")
    print("下一步: 用真实的分子特征替换随机特征")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    model = main()