"""
脚本1: 解析PDBbind的INDEX文件，生成数据集CSV
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm
from config import INDEX_FILE, PROCESSED_DATA_DIR, RAW_DATA_DIR
from scripts.utils import load_pdbbind_index, check_complex_files

def main():
    print("=" * 60)
    print("步骤1: 解析PDBbind INDEX文件")
    print("=" * 60)
    
    # 1. 加载INDEX数据
    print(f"\n1. 加载INDEX文件: {INDEX_FILE}")
    df = load_pdbbind_index(INDEX_FILE)
    print(f"   共加载 {len(df)} 个复合物")
    
    # 2. 统计有效数据
    valid_affinity = df['pK'].notna()
    print(f"\n2. 亲和力数据统计:")
    print(f"   有效pK值: {valid_affinity.sum()}/{len(df)} ({valid_affinity.sum()/len(df)*100:.1f}%)")
    
    # 3. 检查文件完整性
    print(f"\n3. 检查文件完整性 (这可能需要几分钟)...")
    file_status = []
    for pdb_id in tqdm(df['pdb_id'], desc="检查文件"):
        has_protein, has_ligand = check_complex_files(RAW_DATA_DIR, pdb_id)
        file_status.append({
            'pdb_id': pdb_id,
            'has_protein': has_protein,
            'has_ligand': has_ligand,
            'has_both': has_protein and has_ligand
        })
    
    file_df = pd.DataFrame(file_status)
    df = df.merge(file_df, on='pdb_id')
    
    print(f"\n   完整复合物: {df['has_both'].sum()}/{len(df)}")
    
    # 4. 过滤可用数据
    usable_df = df[df['has_both'] & df['pK'].notna()].copy()
    print(f"\n4. 可用数据: {len(usable_df)} 个复合物")
    
    # 5. 保存
    output_file = PROCESSED_DATA_DIR / "pdbbind_refined_2020.csv"
    usable_df.to_csv(output_file, index=False)
    print(f"\n5. 数据已保存到: {output_file}")
    
    # 6. 统计
    print(f"\n6. 数据统计:")
    print(f"   pK范围: {usable_df['pK'].min():.2f} - {usable_df['pK'].max():.2f}")
    print(f"   pK均值: {usable_df['pK'].mean():.2f} ± {usable_df['pK'].std():.2f}")

if __name__ == "__main__":
    main()