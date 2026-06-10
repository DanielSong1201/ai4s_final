"""Build a batch that includes cached ligand graph tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402
from scripts.data.smoke_test_manifest_batch import pad_protein_sequences  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_CACHE_DIR = Path("str/manifest/cache/ligand_graphs")
DEFAULT_REPORT = Path("str/manifest/esm_affinity_graph_batch_smoke_report.json")


def load_rows(manifest_path: Path, split: str, batch_size: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    df = df[df["split"].eq(split)].copy()
    if df.empty:
        raise ValueError(f"No rows found for split={split}")
    if batch_size > len(df):
        raise ValueError(f"batch_size={batch_size} exceeds split size {len(df)}")
    return df.sample(n=batch_size, random_state=seed).reset_index(drop=True)


def load_graph(cache_dir: Path, pdb_id: str) -> dict[str, object]:
    path = cache_dir / f"{pdb_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing ligand graph cache: {path}")
    return torch.load(path, weights_only=False)


def collate_graphs(graphs: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    atom_features = []
    atom_coordinates = []
    bond_indices = []
    bond_features = []
    ligand_batch = []
    atom_offset = 0

    for graph_index, graph in enumerate(graphs):
        x = graph["atom_features"].float()
        pos = graph["atom_coordinates"].float()
        edge_index = graph["bond_index"].long()
        edge_attr = graph["bond_features"].float()
        num_atoms = x.shape[0]

        atom_features.append(x)
        atom_coordinates.append(pos)
        if edge_index.numel() > 0:
            bond_indices.append(edge_index + atom_offset)
            bond_features.append(edge_attr)
        ligand_batch.append(torch.full((num_atoms,), graph_index, dtype=torch.long))
        atom_offset += num_atoms

    return {
        "ligand_atom_features": torch.cat(atom_features, dim=0),
        "ligand_atom_coordinates": torch.cat(atom_coordinates, dim=0),
        "ligand_bond_index": torch.cat(bond_indices, dim=1) if bond_indices else torch.empty((2, 0), dtype=torch.long),
        "ligand_bond_features": torch.cat(bond_features, dim=0) if bond_features else torch.empty((0, 4), dtype=torch.float32),
        "ligand_batch": torch.cat(ligand_batch, dim=0),
    }


def build_batch(rows: pd.DataFrame, cache_dir: Path) -> dict[str, object]:
    protein_tokens, protein_mask, protein_lengths = pad_protein_sequences(rows["protein_sequence"].astype(str).tolist())
    graphs = [load_graph(cache_dir, pdb_id) for pdb_id in rows["pdb_id"].astype(str)]
    graph_batch = collate_graphs(graphs)
    labels = torch.tensor(rows["pAffinity"].astype(float).to_numpy(), dtype=torch.float32)

    return {
        "pdb_id": rows["pdb_id"].astype(str).tolist(),
        "protein_tokens_shape": list(protein_tokens.shape),
        "protein_mask": torch.tensor(protein_mask, dtype=torch.bool),
        "protein_lengths": torch.tensor(np.array(protein_lengths), dtype=torch.long),
        "labels": labels,
        **graph_batch,
    }


def validate_batch(batch: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    issues = []
    if torch.isnan(batch["labels"]).any():
        issues.append({"severity": "error", "check": "label_nan", "message": "Labels contain NaN."})
    if batch["ligand_atom_features"].shape[0] != batch["ligand_atom_coordinates"].shape[0]:
        issues.append({"severity": "error", "check": "atom_coord_mismatch", "message": "Atom feature and coordinate counts differ."})
    if batch["ligand_bond_index"].shape[0] != 2:
        issues.append({"severity": "error", "check": "bond_index_shape", "message": "Bond index first dimension must be 2."})
    if batch["ligand_bond_index"].shape[1] != batch["ligand_bond_features"].shape[0]:
        issues.append({"severity": "error", "check": "bond_feature_mismatch", "message": "Bond index and feature counts differ."})

    report = {
        "pdb_ids": batch["pdb_id"],
        "protein_tokens_shape": batch["protein_tokens_shape"],
        "protein_mask_shape": list(batch["protein_mask"].shape),
        "protein_lengths_shape": list(batch["protein_lengths"].shape),
        "labels_shape": list(batch["labels"].shape),
        "ligand_atom_features_shape": list(batch["ligand_atom_features"].shape),
        "ligand_atom_coordinates_shape": list(batch["ligand_atom_coordinates"].shape),
        "ligand_bond_index_shape": list(batch["ligand_bond_index"].shape),
        "ligand_bond_features_shape": list(batch["ligand_bond_features"].shape),
        "ligand_batch_shape": list(batch["ligand_batch"].shape),
        "label_min": float(batch["labels"].min()),
        "label_max": float(batch["labels"].max()),
    }
    return issues, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    cache_dir = project_path(args.cache_dir)
    report_json = project_path(args.report_json)

    rows = load_rows(manifest_path, args.split, args.batch_size, args.seed)
    batch = build_batch(rows, cache_dir)
    issues, batch_report = validate_batch(batch)
    error_count = sum(1 for issue in issues if issue["severity"] == "error")

    report = {
        "manifest": display_path(manifest_path),
        "cache_dir": display_path(cache_dir),
        "split": args.split,
        "batch_size": args.batch_size,
        "batch": batch_report,
        "issues": issues,
        "summary": {"error_count": error_count, "status": "fail" if error_count else "pass"},
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Built graph batch from: {display_path(manifest_path)}")
    print(f"Wrote report: {display_path(report_json)}")
    print(f"Protein token shape: {batch_report['protein_tokens_shape']}")
    print(f"Ligand atom feature shape: {batch_report['ligand_atom_features_shape']}")
    print(f"Ligand bond index shape: {batch_report['ligand_bond_index_shape']}")
    print(f"Label shape: {batch_report['labels_shape']}")
    print(f"Errors: {error_count}")
    if error_count:
        print("Status: FAIL")
        raise SystemExit(1)
    print("Status: PASS")


if __name__ == "__main__":
    main()
