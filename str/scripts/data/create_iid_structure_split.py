"""Create IID random train/validation/test splits from raw PDBbind structures."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
for import_root in (PROJECT_ROOT / "str", PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.create_interformer_splits import EVAL_SUBSET_FILES, read_id_file  # noqa: E402
from scripts.data_inspection import load_raw_dataset  # noqa: E402


RAW_INDEX_PATH = Path("data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst")
RAW_COMPLEX_ROOT = Path("data/raw/pdbbind2020/complexes/P-L")
SOURCE_SPLIT_DIR = Path("split")
OUTPUT_SPLIT_DIR = Path("str/split_iid_all_raw")
OUTPUT_DIR = Path("data/processed/iid_split_all_raw")
SPLITS = ("train", "valid", "test")


def read_source_counts(source_split_dir: Path) -> dict[str, int]:
    file_by_split = {
        "train": "timesplit_no_lig_overlap_train",
        "valid": "timesplit_no_lig_overlap_val",
        "test": "timesplit_test",
    }
    counts = {}
    for split, filename in file_by_split.items():
        path = source_split_dir / filename
        counts[split] = len(read_id_file(path)) if path.exists() else 0
    return counts


def target_counts(total: int, source_counts: dict[str, int]) -> dict[str, int]:
    source_total = sum(source_counts.values())
    if source_total <= 0:
        valid = int(round(total * 0.055))
        test = int(round(total * 0.021))
    else:
        valid = int(round(total * source_counts["valid"] / source_total))
        test = int(round(total * source_counts["test"] / source_total))
    train = total - valid - test
    return {"train": train, "valid": valid, "test": test}


def assign_random_splits(df: pd.DataFrame, counts: dict[str, int], seed: int) -> pd.DataFrame:
    pdb_ids = df["pdb_id"].astype(str).str.lower().tolist()
    rng = random.Random(seed)
    rng.shuffle(pdb_ids)

    test_count = counts["test"]
    valid_count = counts["valid"]
    test_ids = set(pdb_ids[:test_count])
    valid_ids = set(pdb_ids[test_count : test_count + valid_count])
    train_ids = set(pdb_ids[test_count + valid_count :])

    split_by_id = {
        **{pdb_id: "train" for pdb_id in train_ids},
        **{pdb_id: "valid" for pdb_id in valid_ids},
        **{pdb_id: "test" for pdb_id in test_ids},
    }
    assigned = df.copy()
    assigned["pdb_id"] = assigned["pdb_id"].astype(str).str.lower()
    assigned.insert(0, "split", assigned["pdb_id"].map(split_by_id))
    assigned.insert(1, "split_source", f"iid_random_structure_split_seed_{seed}")
    return assigned


def write_split_files(assigned: pd.DataFrame, source_split_dir: Path, output_split_dir: Path) -> dict[str, object]:
    output_split_dir.mkdir(parents=True, exist_ok=True)
    primary_file_by_split = {
        "train": "timesplit_no_lig_overlap_train",
        "valid": "timesplit_no_lig_overlap_val",
        "test": "timesplit_test",
    }
    file_counts = {}
    for split, filename in tqdm(primary_file_by_split.items(), desc="Write split files", unit="file"):
        ids = sorted(assigned.loc[assigned["split"].eq(split), "pdb_id"].unique())
        (output_split_dir / filename).write_text("\n".join(ids) + "\n", encoding="utf-8")
        file_counts[filename] = len(ids)

    test_ids = set(assigned.loc[assigned["split"].eq("test"), "pdb_id"])
    for _, filename in tqdm(EVAL_SUBSET_FILES.items(), desc="Write eval subset files", unit="file"):
        source_path = source_split_dir / filename
        if source_path.exists():
            ids = read_id_file(source_path, posebusters=(filename == "posebusters_pdb_ccd_ids.txt"))
            filtered_ids = sorted(set(ids) & test_ids)
        else:
            filtered_ids = []
        (output_split_dir / filename).write_text(
            "\n".join(filtered_ids) + ("\n" if filtered_ids else ""),
            encoding="utf-8",
        )
        file_counts[filename] = len(filtered_ids)
    return {"output_split_dir": str(output_split_dir), "file_counts": file_counts}


def create_iid_structure_split(
    index_path: Path,
    complex_root: Path,
    source_split_dir: Path,
    output_split_dir: Path,
    output_dir: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = load_raw_dataset(index_path=index_path, complex_root=complex_root)
    raw_df["pdb_id"] = raw_df["pdb_id"].astype(str).str.lower()
    raw_df = raw_df[raw_df["pdb_id"].notna()].copy()

    source_counts = read_source_counts(source_split_dir)
    counts = target_counts(len(raw_df), source_counts)
    assigned = assign_random_splits(raw_df, counts, seed)
    write_report = write_split_files(assigned, source_split_dir, output_split_dir)

    output_csv = output_dir / "pdbbind_sequence_cluster_splits.csv"
    report_json = output_dir / "iid_structure_split_report.json"
    assigned.to_csv(output_csv, index=False)

    split_counts = {
        split: int(assigned["split"].eq(split).sum())
        for split in SPLITS
    }
    report = {
        "method": "iid random structure-level split",
        "seed": seed,
        "index_path": str(index_path),
        "complex_root": str(complex_root),
        "source_split_dir": str(source_split_dir),
        "output_split_dir": str(output_split_dir),
        "output_dir": str(output_dir),
        "source_ratio_counts": source_counts,
        "target_counts": counts,
        "assigned_counts": split_counts,
        "write_report": write_report,
        "assigned_csv": str(output_csv),
        "note": "IID mode ignores the 40% sequence-similarity anti-leakage prior and randomly assigns PDB complex IDs.",
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return assigned, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=RAW_INDEX_PATH)
    parser.add_argument("--complex-root", type=Path, default=RAW_COMPLEX_ROOT)
    parser.add_argument("--source-split-dir", type=Path, default=SOURCE_SPLIT_DIR)
    parser.add_argument("--output-split-dir", type=Path, default=OUTPUT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, report = create_iid_structure_split(
        index_path=args.index_path,
        complex_root=args.complex_root,
        source_split_dir=args.source_split_dir,
        output_split_dir=args.output_split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
