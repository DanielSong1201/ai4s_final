"""Build an ESM affinity training manifest from validated PDBbind split metadata."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


DEFAULT_SOURCE_CSV = Path(
    "data/processed/sequence_cluster_split_validation/all_raw/pdbbind_sequence_cluster_split_table.csv"
)
DEFAULT_SPLIT_DIR = Path("split_sequence_cluster_all_raw")
DEFAULT_MANIFEST_DIR = Path("str/manifest")
DEFAULT_OUTPUT_CSV = DEFAULT_MANIFEST_DIR / "esm_affinity_manifest.csv"
DEFAULT_REPORT_JSON = DEFAULT_MANIFEST_DIR / "esm_affinity_manifest_report.json"
DEFAULT_INTERFORMER_CSV = DEFAULT_MANIFEST_DIR / "general_PL_2020_sequence_cluster_all_raw.csv"
PRIMARY_SPLIT_FILES = {
    "train": "timesplit_no_lig_overlap_train",
    "valid": "timesplit_no_lig_overlap_val",
    "test": "timesplit_test",
}
REQUIRED_SOURCE_COLUMNS = {
    "pdb_id",
    "protein_path",
    "pocket_path",
    "ligand_sdf_path",
    "ligand_mol2_path",
    "affinity_type",
    "affinity_relation",
    "affinity_value",
    "affinity_unit",
    "affinity_molar",
    "pAffinity",
}
UNIT_TO_MOLAR = {
    "M": 1.0,
    "MM": 1e-3,
    "UM": 1e-6,
    "NM": 1e-9,
    "PM": 1e-12,
    "FM": 1e-15,
}
AFFINITY_PATTERN = re.compile(
    r"(?P<type>Kd|Ki|IC50)\s*(?P<relation>[=<>~])\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fM|pM|nM|uM|µM|mM|M)",
    re.IGNORECASE,
)
AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


def extract_chain_sequences_from_pdb(protein_path: str | Path) -> dict[str, str]:
    """Extract chain-level amino-acid sequences from ATOM records in a PDB file."""

    chains: dict[str, list[str]] = {}
    seen_residues: set[tuple[str, str, str]] = set()
    protein_path = Path(protein_path)

    with protein_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_name = line[17:20].strip().upper()
            aa = AA3_TO_AA1.get(residue_name)
            if aa is None:
                continue
            chain_id = line[21].strip() or "_"
            residue_id = line[22:26].strip()
            insertion_code = line[26].strip()
            key = (chain_id, residue_id, insertion_code)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            chains.setdefault(chain_id, []).append(aa)

    return {chain_id: "".join(sequence) for chain_id, sequence in chains.items() if sequence}


def read_id_file(path: Path) -> list[str]:
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_split_dir(split_dir: Path) -> Path:
    split_dir = project_path(split_dir)
    if split_dir.exists():
        return split_dir

    fallback = PROJECT_ROOT / "str" / split_dir.name
    if fallback.exists():
        return fallback

    return split_dir


def load_split_mapping(split_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    split_dir = resolve_split_dir(split_dir)
    split_to_ids: dict[str, list[str]] = {}
    id_to_split: dict[str, str] = {}
    report: dict[str, object] = {
        "split_dir": str(split_dir),
        "split_file_counts": {},
        "duplicate_ids": {},
        "overlap_counts": {},
    }

    for split, filename in PRIMARY_SPLIT_FILES.items():
        path = split_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required split file is missing: {path}")
        ids = read_id_file(path)
        split_to_ids[split] = ids
        report["split_file_counts"][split] = len(ids)
        duplicates = sorted(pdb_id for pdb_id, count in Counter(ids).items() if count > 1)
        report["duplicate_ids"][split] = duplicates
        if duplicates:
            raise ValueError(f"Duplicate IDs found in {path}: {duplicates[:10]}")

    split_names = list(PRIMARY_SPLIT_FILES)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(set(split_to_ids[left]) & set(split_to_ids[right]))
            report["overlap_counts"][f"{left}_vs_{right}"] = len(overlap)
            if overlap:
                raise ValueError(f"Split overlap found for {left} vs {right}: {overlap[:10]}")

    for split, ids in split_to_ids.items():
        for pdb_id in ids:
            id_to_split[pdb_id] = split

    return id_to_split, report


def require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Source CSV is missing required columns: {missing}")


def path_exists(value: object) -> bool:
    if pd.isna(value):
        return False
    return (PROJECT_ROOT / str(value)).exists()


def extract_sequences(protein_path: str) -> tuple[str, str, int, int]:
    chains = extract_chain_sequences_from_pdb(PROJECT_ROOT / protein_path)
    ordered = {chain_id: chains[chain_id] for chain_id in sorted(chains)}
    sequence = ":".join(ordered.values())
    sequence_length = sum(len(seq) for seq in ordered.values())
    return sequence, json.dumps(ordered, sort_keys=True), len(ordered), sequence_length


def parse_affinity_text(raw_affinity: object) -> dict[str, object] | None:
    if pd.isna(raw_affinity):
        return None
    match = AFFINITY_PATTERN.search(str(raw_affinity))
    if not match:
        return None

    affinity_type = match.group("type")
    relation = match.group("relation")
    value = float(match.group("value"))
    unit = match.group("unit").replace("µ", "u")
    multiplier = UNIT_TO_MOLAR.get(unit.upper())
    if multiplier is None or value <= 0:
        return None

    affinity_molar = value * multiplier
    return {
        "affinity_type": affinity_type,
        "affinity_relation": relation,
        "affinity_value": value,
        "affinity_unit": unit,
        "affinity_molar": affinity_molar,
        "pAffinity": -math.log10(affinity_molar),
    }


def repair_missing_affinity_values(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    repaired_rows = []
    failed_rows = []
    required = ["affinity_type", "affinity_value", "affinity_unit", "affinity_molar", "pAffinity"]

    for index, row in manifest.iterrows():
        has_missing = any(pd.isna(row.get(column)) for column in required)
        if not has_missing:
            continue

        parsed = parse_affinity_text(row.get("raw_affinity"))
        if parsed is None:
            failed_rows.append({"pdb_id": row["pdb_id"], "raw_affinity": row.get("raw_affinity")})
            continue

        for column, value in parsed.items():
            manifest.at[index, column] = value
        repaired_rows.append({"pdb_id": row["pdb_id"], "raw_affinity": row.get("raw_affinity")})

    return manifest, {
        "repaired_count": len(repaired_rows),
        "failed_count": len(failed_rows),
        "repaired_examples": repaired_rows[:20],
        "failed_examples": failed_rows[:20],
    }


def build_manifest(source_csv: Path, split_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    id_to_split, split_report = load_split_mapping(split_dir)
    source_csv = project_path(source_csv)
    source_df = pd.read_csv(source_csv)
    require_columns(source_df, REQUIRED_SOURCE_COLUMNS)

    source_df = source_df.copy()
    source_df["pdb_id"] = source_df["pdb_id"].astype(str).str.lower()
    source_by_id = source_df.drop_duplicates("pdb_id", keep="first").set_index("pdb_id", drop=False)

    split_ids = set(id_to_split)
    source_ids = set(source_by_id.index)
    missing_from_source = sorted(split_ids - source_ids)
    if missing_from_source:
        raise ValueError(f"Split IDs missing from source CSV: {missing_from_source[:20]}")

    manifest = source_by_id.loc[sorted(split_ids)].copy().reset_index(drop=True)
    manifest["split"] = manifest["pdb_id"].map(id_to_split)
    manifest, affinity_repair_report = repair_missing_affinity_values(manifest)

    path_columns = ["protein_path", "pocket_path", "ligand_sdf_path", "ligand_mol2_path"]
    for column in path_columns:
        manifest[f"{column}_exists"] = manifest[column].apply(path_exists)

    sequence_rows = []
    sequence_failures = []
    rows_iter = tqdm(
        manifest.itertuples(index=False),
        total=len(manifest),
        desc="Extract protein sequences",
        unit="complex",
    )
    for row in rows_iter:
        try:
            sequence, chain_json, chain_count, sequence_length = extract_sequences(row.protein_path)
        except Exception as exc:  # pragma: no cover - report exact bad rows instead of hiding them
            sequence = ""
            chain_json = "{}"
            chain_count = 0
            sequence_length = 0
            sequence_failures.append({"pdb_id": row.pdb_id, "error": str(exc)})
        sequence_rows.append(
            {
                "protein_sequence": sequence,
                "protein_chain_sequences_json": chain_json,
                "protein_chain_count": chain_count,
                "protein_sequence_length": sequence_length,
            }
        )

    sequence_df = pd.DataFrame(sequence_rows, index=manifest.index)
    manifest = pd.concat([manifest, sequence_df], axis=1)

    output_columns = [
        "pdb_id",
        "split",
        "protein_sequence",
        "protein_chain_sequences_json",
        "protein_chain_count",
        "protein_sequence_length",
        "protein_path",
        "pocket_path",
        "ligand_sdf_path",
        "ligand_mol2_path",
        "protein_path_exists",
        "pocket_path_exists",
        "ligand_sdf_path_exists",
        "ligand_mol2_path_exists",
        "affinity_type",
        "affinity_relation",
        "affinity_value",
        "affinity_unit",
        "affinity_molar",
        "pAffinity",
        "resolution",
        "release_year",
        "ligand_name",
        "raw_affinity",
        "complex_dir",
    ]
    output_columns = [column for column in output_columns if column in manifest.columns]
    split_order = {"train": 0, "valid": 1, "test": 2}
    manifest["_split_order"] = manifest["split"].map(split_order).fillna(99)
    manifest = manifest[output_columns + ["_split_order"]]
    manifest = manifest.sort_values(["_split_order", "pdb_id"]).drop(columns=["_split_order"]).reset_index(drop=True)

    report = {
        "source_csv": display_path(source_csv),
        "output_rows": int(len(manifest)),
        "split_report": split_report,
        "manifest_split_counts": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "missing_from_source_count": len(missing_from_source),
        "affinity_repair_report": affinity_repair_report,
        "file_existence_counts": {
            column: int(manifest[f"{column}_exists"].sum()) for column in path_columns
        },
        "sequence_failure_count": len(sequence_failures),
        "sequence_failures": sequence_failures[:20],
        "empty_sequence_count": int(manifest["protein_sequence"].eq("").sum()),
        "min_sequence_length": int(manifest["protein_sequence_length"].min()),
        "max_sequence_length": int(manifest["protein_sequence_length"].max()),
    }
    return manifest, report


def write_interformer_csv(manifest: pd.DataFrame, output_csv: Path) -> None:
    interformer_df = manifest.copy()
    interformer_df["Target"] = interformer_df["pdb_id"]
    interformer_df["pIC50"] = interformer_df["pAffinity"]
    leading = ["Target", "pIC50"]
    remaining = [column for column in interformer_df.columns if column not in leading]
    interformer_df = interformer_df[leading + remaining]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    interformer_df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--interformer-csv", type=Path, default=DEFAULT_INTERFORMER_CSV)
    parser.add_argument(
        "--skip-interformer-csv",
        action="store_true",
        help="Do not write the auxiliary Interformer-compatible CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, report = build_manifest(args.source_csv, args.split_dir)

    output_csv = project_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)

    if not args.skip_interformer_csv:
        interformer_csv = project_path(args.interformer_csv)
        write_interformer_csv(manifest, interformer_csv)
        report["interformer_csv"] = display_path(interformer_csv)

    report_json = project_path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote manifest: {display_path(output_csv)} ({len(manifest)} rows)")
    print(f"Wrote report: {display_path(report_json)}")
    if not args.skip_interformer_csv:
        print(f"Wrote Interformer-compatible CSV: {display_path(interformer_csv)}")
    print("Split counts:", report["manifest_split_counts"])


if __name__ == "__main__":
    main()
