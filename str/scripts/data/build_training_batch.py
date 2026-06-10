"""Build training batches from cached ESM embeddings and ligand graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402
from scripts.data.smoke_test_graph_batch import collate_graphs  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_ESM_CACHE_DIR = Path("str/manifest/cache/esm_embeddings")
DEFAULT_LIGAND_CACHE_DIR = Path("str/manifest/cache/ligand_graphs")
DEFAULT_REPORT = Path("str/manifest/esm_ligand_training_batch_report.json")


class EsmLigandAffinityDataset(Dataset):
    """Dataset backed by cached protein ESM embeddings and ligand graph tensors."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        esm_cache_dir: Path,
        ligand_cache_dir: Path,
        split: str | None = None,
    ) -> None:
        if split is not None:
            manifest = manifest[manifest["split"].eq(split)].copy()
        if manifest.empty:
            raise ValueError(f"No manifest rows found for split={split!r}")

        self.manifest = manifest.reset_index(drop=True)
        self.esm_cache_dir = esm_cache_dir
        self.ligand_cache_dir = ligand_cache_dir

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        pdb_id = str(row["pdb_id"])
        esm_path = self.esm_cache_dir / f"{pdb_id}.pt"
        ligand_path = self.ligand_cache_dir / f"{pdb_id}.pt"

        if not esm_path.exists():
            raise FileNotFoundError(f"Missing ESM embedding cache: {esm_path}")
        if not ligand_path.exists():
            raise FileNotFoundError(f"Missing ligand graph cache: {ligand_path}")

        esm_record = torch.load(esm_path, map_location="cpu", weights_only=False)
        ligand_graph = torch.load(ligand_path, map_location="cpu", weights_only=False)
        protein_embedding = esm_record["embedding"].float()

        return {
            "pdb_id": pdb_id,
            "split": str(row["split"]),
            "protein_embedding": protein_embedding,
            "protein_length": int(protein_embedding.shape[0]),
            "ligand_graph": ligand_graph,
            "label": torch.tensor(float(row["pAffinity"]), dtype=torch.float32),
        }


def pad_protein_embeddings(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_dims = {int(embedding.shape[1]) for embedding in embeddings}
    if len(hidden_dims) != 1:
        raise ValueError(f"Mixed ESM hidden dimensions in one batch: {sorted(hidden_dims)}")

    batch_size = len(embeddings)
    max_length = max(int(embedding.shape[0]) for embedding in embeddings)
    hidden_dim = hidden_dims.pop()
    padded = torch.zeros((batch_size, max_length, hidden_dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    lengths = torch.zeros((batch_size,), dtype=torch.long)

    for index, embedding in enumerate(embeddings):
        length = int(embedding.shape[0])
        padded[index, :length] = embedding
        mask[index, :length] = True
        lengths[index] = length

    return padded, mask, lengths


def collate_esm_ligand_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    protein_embeddings, protein_mask, protein_lengths = pad_protein_embeddings(
        [item["protein_embedding"] for item in items]
    )
    ligand_batch = collate_graphs([item["ligand_graph"] for item in items])
    labels = torch.stack([item["label"] for item in items], dim=0)

    return {
        "pdb_id": [item["pdb_id"] for item in items],
        "split": [item["split"] for item in items],
        "protein_embedding": protein_embeddings,
        "protein_mask": protein_mask,
        "protein_lengths": protein_lengths,
        "labels": labels,
        **ligand_batch,
    }


def select_manifest_rows(manifest_path: Path, split: str, limit: int, seed: int, sample_mode: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"].eq(split)].copy()
    if manifest.empty:
        raise ValueError(f"No rows found for split={split}")
    if limit >= 0:
        sample_size = min(limit, len(manifest))
        if sample_mode == "head":
            manifest = manifest.head(sample_size)
        elif sample_mode == "random":
            manifest = manifest.sample(n=sample_size, random_state=seed)
        else:
            raise ValueError(f"Unsupported sample_mode={sample_mode}")
    return manifest.reset_index(drop=True)


def validate_cache_presence(manifest: pd.DataFrame, esm_cache_dir: Path, ligand_cache_dir: Path) -> dict[str, Any]:
    missing_esm = []
    missing_ligand = []
    for pdb_id in manifest["pdb_id"].astype(str):
        if not (esm_cache_dir / f"{pdb_id}.pt").exists():
            missing_esm.append(pdb_id)
        if not (ligand_cache_dir / f"{pdb_id}.pt").exists():
            missing_ligand.append(pdb_id)
    return {
        "missing_esm_count": len(missing_esm),
        "missing_ligand_count": len(missing_ligand),
        "missing_esm_examples": missing_esm[:20],
        "missing_ligand_examples": missing_ligand[:20],
    }


def shape_report(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "pdb_ids": batch["pdb_id"],
        "protein_embedding_shape": list(batch["protein_embedding"].shape),
        "protein_mask_shape": list(batch["protein_mask"].shape),
        "protein_lengths_shape": list(batch["protein_lengths"].shape),
        "ligand_atom_features_shape": list(batch["ligand_atom_features"].shape),
        "ligand_atom_coordinates_shape": list(batch["ligand_atom_coordinates"].shape),
        "ligand_bond_index_shape": list(batch["ligand_bond_index"].shape),
        "ligand_bond_features_shape": list(batch["ligand_bond_features"].shape),
        "ligand_batch_shape": list(batch["ligand_batch"].shape),
        "labels_shape": list(batch["labels"].shape),
        "label_min": float(batch["labels"].min().item()),
        "label_max": float(batch["labels"].max().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--esm-cache-dir", type=Path, default=DEFAULT_ESM_CACHE_DIR)
    parser.add_argument("--ligand-cache-dir", type=Path, default=DEFAULT_LIGAND_CACHE_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=32, help="Rows to inspect before building a batch. Use -1 for full split.")
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    esm_cache_dir = project_path(args.esm_cache_dir)
    ligand_cache_dir = project_path(args.ligand_cache_dir)
    report_json = project_path(args.report_json)

    rows = select_manifest_rows(manifest_path, args.split, args.limit, args.seed, args.sample_mode)
    cache_report = validate_cache_presence(rows, esm_cache_dir, ligand_cache_dir)
    if cache_report["missing_esm_count"] or cache_report["missing_ligand_count"]:
        report = {
            "manifest": display_path(manifest_path),
            "esm_cache_dir": display_path(esm_cache_dir),
            "ligand_cache_dir": display_path(ligand_cache_dir),
            "split": args.split,
            "inspected_rows": int(len(rows)),
            "sample_mode": args.sample_mode,
            "cache_check": cache_report,
            "summary": {"status": "fail"},
        }
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {display_path(report_json)}")
        print("Status: FAIL")
        raise SystemExit(1)

    dataset = EsmLigandAffinityDataset(rows, esm_cache_dir, ligand_cache_dir, split=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_esm_ligand_batch,
    )
    batch = next(iter(loader))
    batch_report = shape_report(batch)
    report = {
        "manifest": display_path(manifest_path),
        "esm_cache_dir": display_path(esm_cache_dir),
        "ligand_cache_dir": display_path(ligand_cache_dir),
        "split": args.split,
        "inspected_rows": int(len(rows)),
        "sample_mode": args.sample_mode,
        "batch_size": args.batch_size,
        "cache_check": cache_report,
        "batch": batch_report,
        "summary": {"status": "pass"},
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Built ESM + ligand batch from: {display_path(manifest_path)}")
    print(f"Wrote report: {display_path(report_json)}")
    print(f"Protein embedding shape: {batch_report['protein_embedding_shape']}")
    print(f"Ligand atom feature shape: {batch_report['ligand_atom_features_shape']}")
    print(f"Ligand bond index shape: {batch_report['ligand_bond_index_shape']}")
    print(f"Label shape: {batch_report['labels_shape']}")
    print("Status: PASS")


if __name__ == "__main__":
    main()
