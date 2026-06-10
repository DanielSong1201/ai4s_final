"""Build and validate a minimal batch from the ESM affinity manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_manifest.csv")
DEFAULT_REPORT = Path("str/manifest/esm_affinity_batch_smoke_report.json")
REQUIRED_COLUMNS = {
    "pdb_id",
    "split",
    "protein_sequence",
    "protein_sequence_length",
    "protein_path",
    "pocket_path",
    "ligand_sdf_path",
    "ligand_mol2_path",
    "pAffinity",
}


def file_exists(path_text: str) -> bool:
    return project_path(Path(path_text)).exists()


def load_manifest(manifest_path: Path, split: str, batch_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    df = df[df["split"].eq(split)].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split!r}")
    if batch_size > len(df):
        raise ValueError(f"batch_size={batch_size} exceeds split size {len(df)}")
    return df.sample(n=batch_size, random_state=seed).reset_index(drop=True)


def pad_protein_sequences(sequences: list[str]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    lengths = [len(sequence.replace(":", "")) for sequence in sequences]
    max_len = max(lengths)
    token_array = np.full((len(sequences), max_len), fill_value="", dtype=object)
    mask = np.zeros((len(sequences), max_len), dtype=bool)

    for row_index, sequence in enumerate(sequences):
        flat = sequence.replace(":", "")
        chars = list(flat)
        token_array[row_index, : len(chars)] = chars
        mask[row_index, : len(chars)] = True

    return token_array, mask, lengths


def build_batch(rows: pd.DataFrame) -> dict[str, object]:
    sequences = rows["protein_sequence"].astype(str).tolist()
    protein_tokens, protein_mask, computed_lengths = pad_protein_sequences(sequences)
    labels = rows["pAffinity"].astype(float).to_numpy(dtype=np.float32)

    return {
        "pdb_id": rows["pdb_id"].astype(str).tolist(),
        "split": rows["split"].astype(str).tolist(),
        "protein_tokens": protein_tokens,
        "protein_mask": protein_mask,
        "protein_lengths": np.array(computed_lengths, dtype=np.int32),
        "labels": labels,
        "protein_path": rows["protein_path"].astype(str).tolist(),
        "pocket_path": rows["pocket_path"].astype(str).tolist(),
        "ligand_sdf_path": rows["ligand_sdf_path"].astype(str).tolist(),
        "ligand_mol2_path": rows["ligand_mol2_path"].astype(str).tolist(),
    }


def validate_batch(batch: dict[str, object], source_rows: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    issues: list[dict[str, object]] = []
    labels = batch["labels"]
    protein_mask = batch["protein_mask"]
    protein_lengths = batch["protein_lengths"]

    if np.isnan(labels).any():
        issues.append({"severity": "error", "check": "label_nan", "message": "Batch labels contain NaN."})
    if not np.isfinite(labels).all():
        issues.append({"severity": "error", "check": "label_non_finite", "message": "Batch labels contain non-finite values."})
    if protein_mask.shape[0] != len(source_rows):
        issues.append({"severity": "error", "check": "mask_batch_dim", "message": "Protein mask batch dimension is incorrect."})
    if not np.array_equal(protein_mask.sum(axis=1), protein_lengths):
        issues.append({"severity": "error", "check": "mask_length_mismatch", "message": "Protein mask sums do not match sequence lengths."})

    for column in ["protein_path", "pocket_path", "ligand_sdf_path", "ligand_mol2_path"]:
        missing = [path for path in batch[column] if not file_exists(path)]
        if missing:
            issues.append(
                {
                    "severity": "error",
                    "check": f"{column}_missing",
                    "message": f"Some paths in {column} do not exist.",
                    "examples": missing[:10],
                }
            )

    report = {
        "batch_size": int(len(source_rows)),
        "pdb_ids": batch["pdb_id"],
        "protein_tokens_shape": list(batch["protein_tokens"].shape),
        "protein_mask_shape": list(batch["protein_mask"].shape),
        "protein_length_min": int(protein_lengths.min()),
        "protein_length_max": int(protein_lengths.max()),
        "label_shape": list(labels.shape),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "label_mean": float(labels.mean()),
    }
    return issues, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    report_path = project_path(args.report_json)
    rows = load_manifest(manifest_path, args.split, args.batch_size, args.seed)
    batch = build_batch(rows)
    issues, batch_report = validate_batch(batch, rows)

    error_count = sum(1 for issue in issues if issue["severity"] == "error")
    report = {
        "manifest": display_path(manifest_path),
        "split": args.split,
        "seed": args.seed,
        "batch": batch_report,
        "ligand_graph_status": "not_built",
        "ligand_graph_note": "This smoke test validates paths and batch fields only. Install RDKit/PyTorch before building ligand graph tensors.",
        "issues": issues,
        "summary": {
            "error_count": error_count,
            "status": "fail" if error_count else "pass",
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Built batch from: {display_path(manifest_path)}")
    print(f"Wrote report: {display_path(report_path)}")
    print(f"Split: {args.split}")
    print(f"Batch size: {batch_report['batch_size']}")
    print(f"Protein token shape: {batch_report['protein_tokens_shape']}")
    print(f"Protein mask shape: {batch_report['protein_mask_shape']}")
    print(f"Label shape: {batch_report['label_shape']}")
    print(f"Errors: {error_count}")
    if error_count:
        print("Status: FAIL")
        raise SystemExit(1)
    print("Status: PASS")


if __name__ == "__main__":
    main()
