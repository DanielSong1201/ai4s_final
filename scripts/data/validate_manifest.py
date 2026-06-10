"""Validate an ESM affinity manifest before implementing dataloaders or training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.create_esm_manifest import (  # noqa: E402
    PRIMARY_SPLIT_FILES,
    display_path,
    project_path,
    read_id_file,
    resolve_split_dir,
)


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_manifest.csv")
DEFAULT_REPORT = Path("str/manifest/esm_affinity_manifest_validation_report.json")
REQUIRED_COLUMNS = {
    "pdb_id",
    "split",
    "protein_sequence",
    "protein_chain_count",
    "protein_sequence_length",
    "protein_path",
    "pocket_path",
    "ligand_sdf_path",
    "ligand_mol2_path",
    "affinity_type",
    "affinity_value",
    "affinity_unit",
    "affinity_molar",
    "pAffinity",
}
PATH_COLUMNS = ["protein_path", "pocket_path", "ligand_sdf_path", "ligand_mol2_path"]
EXPECTED_SPLITS = {"train", "valid", "test"}
VALID_AA_CHARS = set("ACDEFGHIKLMNPQRSTVWYUOX:")  # ':' separates chains in the manifest.
PLAUSIBLE_PAFFINITY_RANGE = (0.0, 15.0)


def add_issue(issues: list[dict[str, object]], severity: str, check: str, message: str, examples=None) -> None:
    issues.append(
        {
            "severity": severity,
            "check": check,
            "message": message,
            "examples": examples or [],
        }
    )


def file_exists(value: object) -> bool:
    if pd.isna(value):
        return False
    return project_path(Path(str(value))).exists()


def file_size(value: object) -> int:
    if pd.isna(value):
        return 0
    path = project_path(Path(str(value)))
    return path.stat().st_size if path.exists() else 0


def validate_required_columns(df: pd.DataFrame, issues: list[dict[str, object]]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        add_issue(issues, "error", "required_columns", "Manifest is missing required columns.", missing)


def validate_ids_and_splits(
    df: pd.DataFrame,
    issues: list[dict[str, object]],
    split_dir: Path | None,
) -> dict[str, object]:
    report: dict[str, object] = {}

    duplicate_mask = df["pdb_id"].duplicated(keep=False)
    duplicates = sorted(df.loc[duplicate_mask, "pdb_id"].astype(str).unique().tolist())
    if duplicates:
        add_issue(issues, "error", "duplicate_pdb_id", "Duplicate pdb_id values found.", duplicates[:20])

    observed_splits = set(df["split"].dropna().astype(str).unique())
    unexpected_splits = sorted(observed_splits - EXPECTED_SPLITS)
    missing_splits = sorted(EXPECTED_SPLITS - observed_splits)
    if unexpected_splits:
        add_issue(issues, "error", "unexpected_split", "Unexpected split values found.", unexpected_splits)
    if missing_splits:
        add_issue(issues, "error", "missing_split", "Expected split values are missing.", missing_splits)

    report["split_counts"] = {key: int(value) for key, value in df["split"].value_counts().to_dict().items()}

    if split_dir is not None:
        split_dir = resolve_split_dir(split_dir)
        report["split_dir"] = display_path(split_dir)
        manifest_by_split = {
            split: set(df.loc[df["split"].eq(split), "pdb_id"].astype(str).str.lower())
            for split in PRIMARY_SPLIT_FILES
        }
        split_file_counts = {}
        for split, filename in PRIMARY_SPLIT_FILES.items():
            path = split_dir / filename
            if not path.exists():
                add_issue(issues, "error", "split_file_missing", f"Split file is missing: {path}")
                continue
            ids = set(read_id_file(path))
            split_file_counts[split] = len(ids)
            missing_from_manifest = sorted(ids - manifest_by_split[split])
            extra_in_manifest = sorted(manifest_by_split[split] - ids)
            if missing_from_manifest:
                add_issue(
                    issues,
                    "error",
                    f"{split}_ids_missing_from_manifest",
                    f"{split} split file contains IDs not present in manifest with the same split.",
                    missing_from_manifest[:20],
                )
            if extra_in_manifest:
                add_issue(
                    issues,
                    "error",
                    f"{split}_manifest_ids_not_in_split_file",
                    f"Manifest contains {split} IDs not present in the split file.",
                    extra_in_manifest[:20],
                )
        report["split_file_counts"] = split_file_counts

    return report


def validate_paths(df: pd.DataFrame, issues: list[dict[str, object]]) -> dict[str, object]:
    report: dict[str, object] = {}
    for column in PATH_COLUMNS:
        exists = df[column].apply(file_exists)
        sizes = df[column].apply(file_size)
        missing = df.loc[~exists, ["pdb_id", column]].head(20).to_dict("records")
        empty = df.loc[exists & sizes.eq(0), ["pdb_id", column]].head(20).to_dict("records")
        report[column] = {
            "exists_count": int(exists.sum()),
            "missing_count": int((~exists).sum()),
            "empty_file_count": int((exists & sizes.eq(0)).sum()),
        }
        if missing:
            add_issue(issues, "error", f"{column}_missing", f"Missing files in {column}.", missing)
        if empty:
            add_issue(issues, "error", f"{column}_empty", f"Empty files in {column}.", empty)
    return report


def validate_labels(df: pd.DataFrame, issues: list[dict[str, object]]) -> dict[str, object]:
    report: dict[str, object] = {}
    p_affinity = pd.to_numeric(df["pAffinity"], errors="coerce")
    affinity_molar = pd.to_numeric(df["affinity_molar"], errors="coerce")
    affinity_value = pd.to_numeric(df["affinity_value"], errors="coerce")

    invalid_label = df[p_affinity.isna() | ~p_affinity.apply(lambda value: math.isfinite(value) if pd.notna(value) else False)]
    if not invalid_label.empty:
        add_issue(
            issues,
            "error",
            "invalid_pAffinity",
            "pAffinity contains non-numeric or non-finite values.",
            invalid_label[["pdb_id", "pAffinity"]].head(20).to_dict("records"),
        )

    invalid_molar = df[affinity_molar.isna() | affinity_molar.le(0)]
    if not invalid_molar.empty:
        add_issue(
            issues,
            "error",
            "invalid_affinity_molar",
            "affinity_molar contains missing or non-positive values.",
            invalid_molar[["pdb_id", "affinity_molar"]].head(20).to_dict("records"),
        )

    invalid_value = df[affinity_value.isna() | affinity_value.le(0)]
    if not invalid_value.empty:
        add_issue(
            issues,
            "error",
            "invalid_affinity_value",
            "affinity_value contains missing or non-positive values.",
            invalid_value[["pdb_id", "affinity_value"]].head(20).to_dict("records"),
        )

    low, high = PLAUSIBLE_PAFFINITY_RANGE
    outside = df[p_affinity.notna() & (~p_affinity.between(low, high))]
    if not outside.empty:
        add_issue(
            issues,
            "warning",
            "pAffinity_outside_plausible_range",
            f"pAffinity values outside the broad plausible range [{low}, {high}] were found.",
            outside[["pdb_id", "pAffinity"]].head(20).to_dict("records"),
        )

    report["pAffinity"] = {
        "min": float(p_affinity.min()),
        "max": float(p_affinity.max()),
        "mean": float(p_affinity.mean()),
        "missing_count": int(p_affinity.isna().sum()),
    }
    report["affinity_type_counts"] = {key: int(value) for key, value in df["affinity_type"].value_counts().to_dict().items()}
    report["affinity_unit_counts"] = {key: int(value) for key, value in df["affinity_unit"].value_counts().to_dict().items()}
    return report


def validate_sequences(df: pd.DataFrame, issues: list[dict[str, object]]) -> dict[str, object]:
    sequence = df["protein_sequence"].fillna("").astype(str)
    lengths = pd.to_numeric(df["protein_sequence_length"], errors="coerce")
    chain_counts = pd.to_numeric(df["protein_chain_count"], errors="coerce")
    computed_lengths = sequence.str.replace(":", "", regex=False).str.len()

    empty = df[sequence.eq("")]
    if not empty.empty:
        add_issue(
            issues,
            "error",
            "empty_protein_sequence",
            "Empty protein sequences found.",
            empty[["pdb_id", "protein_path"]].head(20).to_dict("records"),
        )

    length_mismatch = df[lengths.ne(computed_lengths)]
    if not length_mismatch.empty:
        add_issue(
            issues,
            "error",
            "protein_sequence_length_mismatch",
            "protein_sequence_length does not match the sequence string length.",
            length_mismatch[["pdb_id", "protein_sequence_length"]].head(20).to_dict("records"),
        )

    invalid_chars = []
    for row in df[["pdb_id", "protein_sequence"]].itertuples(index=False):
        bad = sorted(set(str(row.protein_sequence)) - VALID_AA_CHARS)
        if bad:
            invalid_chars.append({"pdb_id": row.pdb_id, "invalid_chars": "".join(bad)})
            if len(invalid_chars) >= 20:
                break
    if invalid_chars:
        add_issue(issues, "error", "invalid_protein_sequence_chars", "Invalid protein sequence characters found.", invalid_chars)

    invalid_chain_counts = df[chain_counts.isna() | chain_counts.le(0)]
    if not invalid_chain_counts.empty:
        add_issue(
            issues,
            "error",
            "invalid_protein_chain_count",
            "protein_chain_count contains missing or non-positive values.",
            invalid_chain_counts[["pdb_id", "protein_chain_count"]].head(20).to_dict("records"),
        )

    return {
        "empty_sequence_count": int(empty.shape[0]),
        "min_sequence_length": int(lengths.min()),
        "max_sequence_length": int(lengths.max()),
        "mean_sequence_length": float(lengths.mean()),
        "invalid_chain_count_rows": int(invalid_chain_counts.shape[0]),
    }


def load_rdkit():
    try:
        from rdkit import RDLogger
        from rdkit import Chem

        RDLogger.DisableLog("rdApp.warning")
        return Chem
    except ImportError:
        return None


def validate_ligand_parsing(df: pd.DataFrame, issues: list[dict[str, object]], parse_limit: int) -> dict[str, object]:
    if parse_limit == 0:
        return {"enabled": False, "reason": "parse limit is 0"}

    Chem = load_rdkit()
    if Chem is None:
        add_issue(
            issues,
            "warning",
            "rdkit_unavailable",
            "RDKit is unavailable, so ligand parse validation was skipped.",
        )
        return {"enabled": False, "reason": "RDKit unavailable"}

    sample = df if parse_limit < 0 else df.head(parse_limit)
    failures = []
    success_count = 0

    for row in sample.itertuples(index=False):
        mol = None
        sdf_path = project_path(Path(row.ligand_sdf_path))
        mol2_path = project_path(Path(row.ligand_mol2_path))
        if sdf_path.exists():
            try:
                supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
                if len(supplier) > 0:
                    mol = supplier[0]
            except Exception:
                mol = None
        if mol is None and mol2_path.exists():
            try:
                mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
            except Exception:
                mol = None

        if mol is None:
            failures.append({"pdb_id": row.pdb_id, "ligand_sdf_path": row.ligand_sdf_path, "ligand_mol2_path": row.ligand_mol2_path})
            if len(failures) >= 20:
                break
        else:
            success_count += 1

    if failures:
        add_issue(
            issues,
            "warning",
            "ligand_parse_failures",
            "Some sampled ligands could not be parsed by RDKit from SDF or MOL2.",
            failures,
        )

    return {
        "enabled": True,
        "checked_rows": int(len(sample)),
        "success_count": int(success_count),
        "failure_count_recorded": int(len(failures)),
        "parse_limit": parse_limit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split-dir", type=Path, default=None, help="Optional Interformer-format split directory to compare against.")
    parser.add_argument(
        "--ligand-parse-limit",
        type=int,
        default=200,
        help="Number of ligand rows to parse with RDKit. Use -1 for all rows, 0 to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    report_path = project_path(args.report_json)
    issues: list[dict[str, object]] = []

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    validate_required_columns(df, issues)
    if any(issue["severity"] == "error" and issue["check"] == "required_columns" for issue in issues):
        report = {"manifest": display_path(manifest_path), "row_count": int(len(df)), "issues": issues}
    else:
        report = {
            "manifest": display_path(manifest_path),
            "row_count": int(len(df)),
            "ids_and_splits": validate_ids_and_splits(df, issues, args.split_dir),
            "paths": validate_paths(df, issues),
            "labels": validate_labels(df, issues),
            "sequences": validate_sequences(df, issues),
            "ligand_parsing": validate_ligand_parsing(df, issues, args.ligand_parse_limit),
            "issues": issues,
        }

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
    report["summary"] = {
        "error_count": error_count,
        "warning_count": warning_count,
        "status": "fail" if error_count else "pass",
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Validated manifest: {display_path(manifest_path)}")
    print(f"Wrote report: {display_path(report_path)}")
    print(f"Rows: {len(df)}")
    print(f"Errors: {error_count}, warnings: {warning_count}")
    if error_count:
        print("Status: FAIL")
        raise SystemExit(1)
    print("Status: PASS")


if __name__ == "__main__":
    main()
