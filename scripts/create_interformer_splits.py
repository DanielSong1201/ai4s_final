"""Create train/validation/test splits from raw PDBbind data and Interformer IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.data_inspection import load_raw_dataset


PRIMARY_SPLIT_FILES = {
    "train": "timesplit_no_lig_overlap_train",
    "valid": "timesplit_no_lig_overlap_val",
    "test": "timesplit_test",
}

EVAL_SUBSET_FILES = {
    "is_coreset": "coresetlist",
    "is_diff_test_core": "diff_test+core",
    "is_posebusters_pdb_ccd": "posebusters_pdb_ccd_ids.txt",
    "is_test_no_rec_overlap": "timesplit_test_no_rec_overlap",
    "is_test_sanitizable": "timesplit_test_sanitizable",
}


def read_id_file(path: str | Path, posebusters: bool = False) -> list[str]:
    """Read a split ID file.

    PoseBusters entries are formatted like PDBID_CCD, so only the PDB ID prefix
    is used for matching PDBbind complex IDs.
    """

    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        if posebusters:
            token = token.split("_", 1)[0]
        ids.append(token.lower())
    return ids


def load_interformer_split_ids(split_dir: str | Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    split_dir = Path(split_dir)
    primary = {
        split: set(read_id_file(split_dir / filename))
        for split, filename in PRIMARY_SPLIT_FILES.items()
    }
    subsets = {
        column: set(read_id_file(split_dir / filename, posebusters=(column == "is_posebusters_pdb_ccd")))
        for column, filename in EVAL_SUBSET_FILES.items()
    }
    return primary, subsets


def validate_primary_splits(primary: dict[str, set[str]]) -> dict[str, int]:
    overlaps = {}
    split_names = sorted(primary)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[f"{left}_vs_{right}"] = len(primary[left] & primary[right])
    return overlaps


def create_split_table(
    index_path: str | Path,
    complex_root: str | Path,
    split_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = load_raw_dataset(index_path=index_path, complex_root=complex_root)
    primary, subsets = load_interformer_split_ids(split_dir)
    raw_ids = set(raw_df["pdb_id"])

    split_by_id = {}
    for split_name, ids in primary.items():
        for pdb_id in ids:
            split_by_id[pdb_id] = split_name

    split_df = raw_df.copy()
    split_df.insert(0, "split", split_df["pdb_id"].map(split_by_id).fillna("unassigned"))
    split_df.insert(1, "split_source", "Interformer timesplit/no-lig-overlap files")

    for column, ids in subsets.items():
        split_df[column] = split_df["pdb_id"].isin(ids)

    report = {
        "raw_rows": int(len(raw_df)),
        "primary_split_file_counts": {split: len(ids) for split, ids in primary.items()},
        "matched_primary_split_counts": {
            split: int(split_df["split"].eq(split).sum()) for split in PRIMARY_SPLIT_FILES
        },
        "unassigned_raw_rows": int(split_df["split"].eq("unassigned").sum()),
        "primary_split_overlap_counts": validate_primary_splits(primary),
        "split_ids_missing_from_raw": {
            split: sorted(ids - raw_ids) for split, ids in primary.items()
        },
        "split_ids_missing_from_raw_counts": {
            split: len(ids - raw_ids) for split, ids in primary.items()
        },
        "raw_ids_not_in_primary_interformer_splits_count": int(len(raw_ids - set().union(*primary.values()))),
        "eval_subset_counts_in_raw": {
            column: int(split_df[column].sum()) for column in subsets
        },
        "eval_subset_file_counts": {
            column: len(ids) for column, ids in subsets.items()
        },
    }
    return split_df, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-path",
        default="data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst",
    )
    parser.add_argument(
        "--complex-root",
        default="data/raw/pdbbind2020/complexes/P-L",
    )
    parser.add_argument("--split-dir", default="data/splits/interformer")
    parser.add_argument("--output-csv", default="data/splits/pdbbind_interformer_splits.csv")
    parser.add_argument("--report-json", default="data/splits/pdbbind_interformer_split_report.json")
    args = parser.parse_args()

    split_df, report = create_split_table(
        index_path=args.index_path,
        complex_root=args.complex_root,
        split_dir=args.split_dir,
    )

    output_csv = Path(args.output_csv)
    report_json = Path(args.report_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {output_csv} with {len(split_df)} rows.")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("missing_from_raw")}, indent=2))


if __name__ == "__main__":
    main()
