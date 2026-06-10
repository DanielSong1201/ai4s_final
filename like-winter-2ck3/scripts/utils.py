import re
import math
from pathlib import Path
import pandas as pd

def parse_affinity_to_pk(binding_str: str):
    """将 'Kd=49uM' 这样的字符串转换为 pK 值"""
    if not binding_str:
        return None
    
    # 匹配 Kd=数值单位 或 Ki=数值单位
    match = re.search(r'K[dD]?=([\d\.]+)([mun]?M)', binding_str, re.IGNORECASE)
    if not match:
        match = re.search(r'Ki=([\d\.]+)([mun]?M)', binding_str, re.IGNORECASE)
    
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        
        # 单位转换
        if unit in ['UM', 'U']:
            value_m = value * 1e-6
        elif unit in ['NM', 'N']:
            value_m = value * 1e-9
        elif unit in ['PM', 'P']:
            value_m = value * 1e-12
        elif unit == 'M':
            value_m = value
        else:
            value_m = value * 1e-6
        
        if value_m > 0:
            return round(-math.log10(value_m), 2)
    
    return None

def load_pdbbind_index(index_path):
    """加载INDEX文件，返回DataFrame"""
    records = []
    
    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            pdb_id = parts[0].lower()
            resolution = parts[1] if parts[1] != '???' else None
            year = int(parts[2])
            binding_str = parts[3]
            
            pK = parse_affinity_to_pk(binding_str)
            
            records.append({
                'pdb_id': pdb_id,
                'resolution': float(resolution) if resolution else None,
                'year': year,
                'binding_str': binding_str,
                'pK': pK
            })
    
    return pd.DataFrame(records)

def check_complex_files(data_dir, pdb_id):
    """检查蛋白和配体文件是否存在"""
    pdb_file = data_dir / pdb_id / f"{pdb_id}_protein.pdb"
    sdf_file = data_dir / pdb_id / f"{pdb_id}_ligand.sdf"
    return pdb_file.exists(), sdf_file.exists()