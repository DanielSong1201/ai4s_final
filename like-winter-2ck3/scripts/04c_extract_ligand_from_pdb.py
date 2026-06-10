"""
脚本4c: 直接从 PDB 文件中提取配体坐标
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

# 尝试导入 ProDy
try:
    import prody as pr
    PRODY_AVAILABLE = True
    print("✓ ProDy 可用")
except ImportError:
    PRODY_AVAILABLE = False
    print("请安装 ProDy: pip install prody")

# 尝试导入 RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
    print("✓ RDKit 可用")
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit 不可用")

def extract_ligand_from_pdb(pdb_id):
    """
    从 PDB 文件中提取配体（HETATM）
    然后生成 ECFP 指纹
    """
    if not PRODY_AVAILABLE or not RDKIT_AVAILABLE:
        return None
    
    pdb_path = RAW_DATA_DIR / pdb_id / f"{pdb_id}_protein.pdb"
    
    if not pdb_path.exists():
        return None
    
    try:
        # 用 ProDy 读取 PDB
        structure = pr.parsePDB(str(pdb_path))
        
        # 提取 HETATM（非标准残基，通常是配体）
        hetatoms = structure.select('hetatm')
        
        if hetatoms is None or hetatoms.numAtoms() == 0:
            return None
        
        # 获取配体名称（通常是第一个 HETATM 的残基名）
        ligand_name = hetatoms.getResnames()[0] if len(hetatoms.getResnames()) > 0 else 'LIG'
        
        # 提取配体坐标
        coords = hetatoms.getCoords()
        
        # 将原子坐标转换为 RDKit 分子（需要额外处理，这里简化）
        # 由于从坐标构建分子比较复杂，我们先用一个更简单的方法：
        # 直接从 PDB 文件中读取分子 block
        
        # 读取 PDB 文件，提取配体部分的 HETATM 记录
        ligand_pdb_lines = []
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('HETATM'):
                    # 检查是否属于配体（不是水、不是标准辅因子）
                    resname = line[17:20].strip()
                    if resname not in ['HOH', 'WAT', 'SOL']:
                        ligand_pdb_lines.append(line)
        
        if not ligand_pdb_lines:
            return None
        
        # 将配体坐标写入临时 PDB 字符串
        temp_pdb = '\n'.join(ligand_pdb_lines) + '\nEND'
        
        # 用 RDKit 读取 PDB 字符串
        mol = Chem.MolFromPDBBlock(temp_pdb, removeHs=True)
        
        if mol is None:
            return None
        
        # 计算 ECFP 指纹
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        return np.array(fp, dtype=np.float32)
        
    except Exception as e:
        return None

def main():
    print("=" * 60)
    print("步骤4c: 从 PDB 文件直接提取配体")
    print("=" * 60)
    
    if not PRODY_AVAILABLE:
        print("\n❌ ProDy 不可用，请安装: pip install prody")
        return None
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    
    # 只测试小样本
    test_df = train_df.head(50).copy()
    print(f"   测试样本: {len(test_df)}")
    
    # 2. 测试提取特征
    print("\n2. 测试从 PDB 提取配体...")
    
    features = []
    valid_pdbs = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        pdb_id = row['pdb_id']
        fp = extract_ligand_from_pdb(pdb_id)
        if fp is not None:
            features.append(fp)
            valid_pdbs.append(pdb_id)
    
    print(f"\n   成功提取: {len(valid_pdbs)}/{len(test_df)}")
    
    if len(valid_pdbs) == 0:
        print("\n❌ 提取失败，检查 PDB 文件中的 HETATM 记录")
        return None
    
    # 3. 显示示例
    print(f"\n   成功提取的 PDB ID: {valid_pdbs[:5]}")
    
    # 4. 用这些特征训练简单模型（演示）
    print("\n3. 用提取的特征训练模型...")
    
    X = np.vstack(features)
    y = test_df[test_df['pdb_id'].isin(valid_pdbs)]['pK'].values
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # 交叉验证
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    print(f"   交叉验证 R²: {scores.mean():.3f} ± {scores.std():.3f}")
    
    print("\n" + "=" * 60)
    print("✅ 配体提取成功！")
    print("=" * 60)

if __name__ == "__main__":
    main()