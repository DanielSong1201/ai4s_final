"""
脚本4b: 使用 OpenBabel 读取 PDBbind 分子文件，提取真实特征
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# 尝试导入必要的库
try:
    import pybel
    OPENBABEL_AVAILABLE = True
    print("✓ OpenBabel (pybel) 可用")
except ImportError:
    OPENBABEL_AVAILABLE = False
    print("✗ OpenBabel 不可用")

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
    print("✓ RDKit 可用")
except ImportError:
    RDKIT_AVAILABLE = False
    print("✗ RDKit 不可用")

def load_ligand_features(pdb_id):
    """
    使用 OpenBabel 读取分子文件，然后通过 RDKit 计算 ECFP 指纹
    """
    if not OPENBABEL_AVAILABLE or not RDKIT_AVAILABLE:
        return None
    
    mol2_path = RAW_DATA_DIR / pdb_id / f"{pdb_id}_ligand.mol2"
    sdf_path = RAW_DATA_DIR / pdb_id / f"{pdb_id}_ligand.sdf"
    
    mol = None
    
    # 方法1：使用 OpenBabel 读取 MOL2
    if mol2_path.exists():
        try:
            ob_mols = list(pybel.readfile('mol2', str(mol2_path)))
            if ob_mols:
                smiles = ob_mols[0].write('smiles').strip()
                mol = Chem.MolFromSmiles(smiles)
        except:
            pass
    
    # 方法2：如果 MOL2 失败，尝试 SDF
    if mol is None and sdf_path.exists():
        try:
            ob_mols = list(pybel.readfile('sdf', str(sdf_path)))
            if ob_mols:
                smiles = ob_mols[0].write('smiles').strip()
                mol = Chem.MolFromSmiles(smiles)
        except:
            pass
    
    if mol is None:
        return None
    
    try:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp, dtype=np.float32)
    except:
        return None

def load_protein_features(pdb_id):
    """简单的蛋白质特征（占位符）"""
    np.random.seed(hash(pdb_id) % 10000)
    return np.random.randn(128).astype(np.float32)

def main():
    print("=" * 60)
    print("步骤4b: 使用 OpenBabel + RDKit 提取真实配体特征")
    print("=" * 60)
    
    if not OPENBABEL_AVAILABLE:
        print("\n❌ OpenBabel 不可用")
        return None
    
    if not RDKIT_AVAILABLE:
        print("\n❌ RDKit 不可用")
        return None
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    print(f"   训练集: {len(train_df)}")
    print(f"   验证集: {len(val_df)}")
    print(f"   测试集: {len(test_df)}")
    
    # 2. 测试小样本
    print("\n2. 测试小样本 (20个) 提取特征...")
    test_pdbs = train_df['pdb_id'].head(20).tolist()
    success_count = 0
    
    for pdb_id in tqdm(test_pdbs, desc="测试"):
        feat = load_ligand_features(pdb_id)
        if feat is not None:
            success_count += 1
    
    print(f"\n   测试结果: {success_count}/{len(test_pdbs)} 个样本成功提取特征")
    
    if success_count == 0:
        print("\n❌ 所有样本都失败了")
        print("   可能原因: 文件格式问题，尝试检查一个样本文件")
        # 检查一个具体文件
        test_id = test_pdbs[0]
        mol2_path = RAW_DATA_DIR / test_id / f"{test_id}_ligand.mol2"
        print(f"   示例文件: {mol2_path}")
        print(f"   文件存在: {mol2_path.exists()}")
        return None
    
    # 3. 提取全部训练集特征
    print("\n3. 提取全部训练集配体特征...")
    train_lig_features = []
    valid_indices = []
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="训练集"):
        feat = load_ligand_features(row['pdb_id'])
        if feat is not None:
            train_lig_features.append(feat)
            valid_indices.append(idx)
    
    train_valid_df = train_df.loc[valid_indices].copy()
    X_train_lig = np.vstack(train_lig_features)
    
    print(f"   有效样本: {len(train_valid_df)}/{len(train_df)} ({len(train_valid_df)/len(train_df)*100:.1f}%)")
    
    # 4. 提取验证集特征
    print("\n4. 提取验证集配体特征...")
    val_lig_features = []
    val_indices = []
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="验证集"):
        feat = load_ligand_features(row['pdb_id'])
        if feat is not None:
            val_lig_features.append(feat)
            val_indices.append(idx)
    
    val_valid_df = val_df.loc[val_indices].copy()
    X_val_lig = np.vstack(val_lig_features)
    print(f"   有效样本: {len(val_valid_df)}/{len(val_df)} ({len(val_valid_df)/len(val_df)*100:.1f}%)")
    
    # 5. 提取测试集特征
    print("\n5. 提取测试集配体特征...")
    test_lig_features = []
    test_indices = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="测试集"):
        feat = load_ligand_features(row['pdb_id'])
        if feat is not None:
            test_lig_features.append(feat)
            test_indices.append(idx)
    
    test_valid_df = test_df.loc[test_indices].copy()
    X_test_lig = np.vstack(test_lig_features)
    print(f"   有效样本: {len(test_valid_df)}/{len(test_df)} ({len(test_valid_df)/len(test_df)*100:.1f}%)")
    
    # 6. 蛋白质特征
    print("\n6. 提取蛋白质特征...")
    
    X_train_prot = np.vstack([load_protein_features(pid) for pid in train_valid_df['pdb_id']])
    X_val_prot = np.vstack([load_protein_features(pid) for pid in val_valid_df['pdb_id']])
    X_test_prot = np.vstack([load_protein_features(pid) for pid in test_valid_df['pdb_id']])
    
    # 7. 合并特征
    print("\n7. 合并配体+蛋白质特征...")
    X_train = np.hstack([X_train_lig, X_train_prot])
    X_val = np.hstack([X_val_lig, X_val_prot])
    X_test = np.hstack([X_test_lig, X_test_prot])
    
    print(f"   特征维度: {X_train.shape[1]}")
    
    # 8. 标签
    y_train = train_valid_df['pK'].values
    y_val = val_valid_df['pK'].values
    y_test = test_valid_df['pK'].values
    
    # 9. 训练模型
    print("\n8. 训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    # 10. 评估
    print("\n9. 评估模型性能:")
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\n   📊 训练集 RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}, R²: {train_r2:.3f}")
    print(f"   📊 验证集 RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}, R²: {val_r2:.3f}")
    print(f"   📊 测试集 RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}, R²: {test_r2:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ 第五阶段完成！使用 OpenBabel + RDKit 提取了真实配体特征")
    print(f"   成功提取比例: 训练集 {len(train_valid_df)}/{len(train_df)}")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    model = main()