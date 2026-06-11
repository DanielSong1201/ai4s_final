"""Train a frozen-ESM + pocket-aware protein pooling + ligand GNN model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.build_training_batch import (  # noqa: E402
    EsmLigandAffinityDataset,
    collate_esm_ligand_batch,
    select_manifest_rows,
    validate_cache_presence,
)
from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402
from scripts.train.train_ligand_gnn_frozen_esm import LigandGNN  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_ESM_CACHE_DIR = Path("str/manifest/cache/esm_embeddings")
DEFAULT_LIGAND_CACHE_DIR = Path("str/manifest/cache/ligand_graphs")
DEFAULT_POCKET_CACHE_DIR = Path("str/manifest/cache/pocket_features")
DEFAULT_OUTPUT_DIR = Path("str/manifest/outputs/pocket_gnn_frozen_esm")


def progress_bar(iterable, desc: str, unit: str):
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
        disable=False,
        miniters=1,
        mininterval=0.0,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def masked_mean_pool(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.unsqueeze(-1).to(embeddings.dtype)
    summed = (embeddings * mask_float).sum(dim=1)
    counts = mask_float.sum(dim=1).clamp_min(1.0)
    return summed / counts


def make_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unsupported loss: {name}")


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    errors = predictions - labels
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    label_var = float(np.sum((labels - labels.mean()) ** 2))
    r2 = 1.0 - float(np.sum(errors**2)) / label_var if label_var > 0 else float("nan")

    def corr(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    label_rank = pd.Series(labels).rank(method="average").to_numpy()
    prediction_rank = pd.Series(predictions).rank(method="average").to_numpy()
    return {
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": r2,
        "pearson": corr(labels, predictions),
        "spearman": corr(label_rank, prediction_rank),
    }


class PocketEsmLigandAffinityDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        esm_cache_dir: Path,
        ligand_cache_dir: Path,
        pocket_cache_dir: Path,
        fallback_to_full_sequence: bool,
    ) -> None:
        self.base = EsmLigandAffinityDataset(manifest, esm_cache_dir, ligand_cache_dir, split=None)
        self.pocket_cache_dir = pocket_cache_dir
        self.fallback_to_full_sequence = fallback_to_full_sequence

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        pdb_id = item["pdb_id"]
        pocket_path = self.pocket_cache_dir / f"{pdb_id}.pt"
        if not pocket_path.exists():
            raise FileNotFoundError(f"Missing pocket feature cache: {pocket_path}")
        pocket = torch.load(pocket_path, map_location="cpu", weights_only=False)

        protein_length = int(item["protein_embedding"].shape[0])
        pocket_mask = pocket["pocket_mask"].bool()
        residue_coordinates = pocket["residue_coordinates"].float()
        if int(pocket_mask.shape[0]) != protein_length:
            if not self.fallback_to_full_sequence:
                raise ValueError(
                    f"Pocket mask length mismatch for {pdb_id}: "
                    f"mask={pocket_mask.shape[0]}, protein_embedding={protein_length}"
                )
            pocket_mask = torch.zeros((protein_length,), dtype=torch.bool)
            residue_coordinates = torch.zeros((protein_length, 3), dtype=torch.float32)

        item["pocket_mask"] = pocket_mask
        item["residue_coordinates"] = residue_coordinates
        item["pocket_residue_count"] = torch.tensor(int(pocket_mask.sum().item()), dtype=torch.long)
        return item


def pad_pocket_features(items: list[dict[str, Any]], max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(items)
    pocket_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    residue_coordinates = torch.zeros((batch_size, max_length, 3), dtype=torch.float32)
    pocket_counts = torch.zeros((batch_size,), dtype=torch.long)

    for index, item in enumerate(items):
        length = min(int(item["pocket_mask"].shape[0]), max_length)
        pocket_mask[index, :length] = item["pocket_mask"][:length]
        residue_coordinates[index, :length] = item["residue_coordinates"][:length]
        pocket_counts[index] = int(item["pocket_residue_count"].item())

    return pocket_mask, residue_coordinates, pocket_counts


def collate_pocket_esm_ligand_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_esm_ligand_batch(items)
    max_length = int(batch["protein_embedding"].shape[1])
    pocket_mask, residue_coordinates, pocket_counts = pad_pocket_features(items, max_length)
    batch["pocket_mask"] = pocket_mask
    batch["residue_coordinates"] = residue_coordinates
    batch["pocket_residue_counts"] = pocket_counts
    return batch


def validate_pocket_cache_presence(manifest: pd.DataFrame, pocket_cache_dir: Path) -> dict[str, Any]:
    missing = []
    pdb_ids = manifest["pdb_id"].astype(str).tolist()
    for pdb_id in progress_bar(pdb_ids, desc="Check pocket cache", unit="sample"):
        if not (pocket_cache_dir / f"{pdb_id}.pt").exists():
            missing.append(pdb_id)
    return {
        "missing_pocket_count": len(missing),
        "missing_pocket_examples": missing[:20],
    }


def build_loader(
    manifest_path: Path,
    esm_cache_dir: Path,
    ligand_cache_dir: Path,
    pocket_cache_dir: Path,
    split: str,
    limit: int,
    sample_mode: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    fallback_to_full_sequence: bool,
) -> tuple[DataLoader, pd.DataFrame]:
    rows = select_manifest_rows(manifest_path, split, limit, seed, sample_mode)
    cache_report = validate_cache_presence(rows, esm_cache_dir, ligand_cache_dir)
    pocket_report = validate_pocket_cache_presence(rows, pocket_cache_dir)
    if cache_report["missing_esm_count"] or cache_report["missing_ligand_count"] or pocket_report["missing_pocket_count"]:
        raise FileNotFoundError(
            f"Cache is incomplete for split={split}: "
            f"missing_esm={cache_report['missing_esm_examples']}, "
            f"missing_ligand={cache_report['missing_ligand_examples']}, "
            f"missing_pocket={pocket_report['missing_pocket_examples']}"
        )

    dataset = PocketEsmLigandAffinityDataset(
        rows,
        esm_cache_dir,
        ligand_cache_dir,
        pocket_cache_dir,
        fallback_to_full_sequence=fallback_to_full_sequence,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pocket_esm_ligand_batch,
    )
    return loader, rows


def infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    batch = next(iter(loader))
    protein_dim = int(batch["protein_embedding"].shape[-1])
    atom_dim = int(batch["ligand_atom_features"].shape[-1])
    edge_dim = int(batch["ligand_bond_features"].shape[-1])
    return protein_dim, atom_dim, edge_dim


class PocketProteinEncoder(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        hidden_dim: int,
        pooling: str,
        dropout: float,
        ligand_dim: int,
        fallback_to_full_sequence: bool,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.fallback_to_full_sequence = fallback_to_full_sequence
        self.residue_projection = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.ligand_context = nn.Linear(ligand_dim, hidden_dim)
        self.conditioned_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def effective_mask(self, protein_mask: torch.Tensor, pocket_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "full_mean":
            return protein_mask
        mask = protein_mask & pocket_mask
        missing = mask.sum(dim=1).eq(0)
        if missing.any() and self.fallback_to_full_sequence:
            mask = mask.clone()
            mask[missing] = protein_mask[missing]
        return mask

    def attention_pool(
        self,
        residue_features: torch.Tensor,
        mask: torch.Tensor,
        ligand_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ligand_vector is None:
            scores = self.attention_score(residue_features).squeeze(-1)
        else:
            context = self.ligand_context(ligand_vector).unsqueeze(1).expand_as(residue_features)
            scores = self.conditioned_score(
                torch.cat([residue_features, context, residue_features * context], dim=-1)
            ).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0).unsqueeze(-1)
        return (residue_features * weights).sum(dim=1)

    def forward(
        self,
        protein_embedding: torch.Tensor,
        protein_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
        ligand_vector: torch.Tensor,
    ) -> torch.Tensor:
        residue_features = self.residue_projection(protein_embedding)
        mask = self.effective_mask(protein_mask, pocket_mask)
        if self.pooling in {"full_mean", "pocket_mean"}:
            return masked_mean_pool(residue_features, mask)
        if self.pooling == "pocket_attention":
            return self.attention_pool(residue_features, mask)
        if self.pooling == "ligand_conditioned_attention":
            return self.attention_pool(residue_features, mask, ligand_vector=ligand_vector)
        raise ValueError(f"Unsupported protein pooling: {self.pooling}")


class FrozenEsmPocketLigandGNN(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        atom_dim: int,
        edge_dim: int,
        protein_hidden_dim: int,
        gnn_hidden_dim: int,
        gnn_layers: int,
        gnn_type: str,
        ligand_pooling: str,
        protein_pooling: str,
        fusion_hidden_dim: int,
        dropout: float,
        fallback_to_full_sequence: bool,
    ) -> None:
        super().__init__()
        self.ligand_gnn = LigandGNN(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            layers=gnn_layers,
            gnn_type=gnn_type,
            pooling=ligand_pooling,
            dropout=dropout,
        )
        self.protein_encoder = PocketProteinEncoder(
            protein_dim=protein_dim,
            hidden_dim=protein_hidden_dim,
            pooling=protein_pooling,
            dropout=dropout,
            ligand_dim=gnn_hidden_dim,
            fallback_to_full_sequence=fallback_to_full_sequence,
        )
        self.ligand_projection = nn.Sequential(
            nn.Linear(gnn_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(protein_hidden_dim + fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        raw_ligand_vector = self.ligand_gnn(batch)
        protein_vector = self.protein_encoder(
            batch["protein_embedding"],
            batch["protein_mask"],
            batch["pocket_mask"],
            raw_ligand_vector,
        )
        ligand_vector = self.ligand_projection(raw_ligand_vector)
        return self.regressor(torch.cat([protein_vector, ligand_vector], dim=-1)).squeeze(-1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    progress = progress_bar(loader, desc="Train pocket GNN batches", unit="batch")
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch)
        loss = criterion(predictions, batch["labels"])
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
        progress.set_postfix(loss=f"{float(loss.item()):.4f}")
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    rows = []
    labels_all = []
    predictions_all = []

    for batch in progress_bar(loader, desc="Evaluate pocket GNN batches", unit="batch"):
        batch = move_batch_to_device(batch, device)
        predictions = model(batch)
        loss = criterion(predictions, batch["labels"])
        labels = batch["labels"].detach().cpu().numpy()
        preds = predictions.detach().cpu().numpy()
        labels_all.append(labels)
        predictions_all.append(preds)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size

        for pdb_id, split, label, pred, pocket_count in zip(
            batch["pdb_id"],
            batch["split"],
            labels,
            preds,
            batch["pocket_residue_counts"].detach().cpu().numpy(),
        ):
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "split": split,
                    "label": float(label),
                    "prediction": float(pred),
                    "error": float(pred - label),
                    "pocket_residue_count": int(pocket_count),
                }
            )

    labels_np = np.concatenate(labels_all) if labels_all else np.array([], dtype=float)
    predictions_np = np.concatenate(predictions_all) if predictions_all else np.array([], dtype=float)
    metrics = compute_metrics(labels_np, predictions_np)
    metrics["loss"] = total_loss / max(total_rows, 1)
    return metrics, pd.DataFrame(rows)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--esm-cache-dir", type=Path, default=DEFAULT_ESM_CACHE_DIR)
    parser.add_argument("--ligand-cache-dir", type=Path, default=DEFAULT_LIGAND_CACHE_DIR)
    parser.add_argument("--pocket-cache-dir", type=Path, default=DEFAULT_POCKET_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--valid-limit", type=int, default=-1)
    parser.add_argument("--test-limit", type=int, default=-1)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--protein-hidden-dim", type=int, default=256)
    parser.add_argument("--protein-pooling", choices=["full_mean", "pocket_mean", "pocket_attention", "ligand_conditioned_attention"], default="pocket_attention")
    parser.add_argument("--gnn-type", choices=["gcn", "sage", "gine"], default="gine")
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--ligand-pooling", choices=["mean", "max", "attention"], default="mean")
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fallback-to-full-sequence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = project_path(args.manifest)
    esm_cache_dir = project_path(args.esm_cache_dir)
    ligand_cache_dir = project_path(args.ligand_cache_dir)
    pocket_cache_dir = project_path(args.pocket_cache_dir)
    output_dir = project_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    train_loader, train_rows = build_loader(
        manifest_path,
        esm_cache_dir,
        ligand_cache_dir,
        pocket_cache_dir,
        args.train_split,
        args.train_limit,
        args.sample_mode,
        args.seed,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        fallback_to_full_sequence=args.fallback_to_full_sequence,
    )
    valid_loader, valid_rows = build_loader(
        manifest_path,
        esm_cache_dir,
        ligand_cache_dir,
        pocket_cache_dir,
        args.valid_split,
        args.valid_limit,
        args.sample_mode,
        args.seed,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        fallback_to_full_sequence=args.fallback_to_full_sequence,
    )
    test_loader, test_rows = build_loader(
        manifest_path,
        esm_cache_dir,
        ligand_cache_dir,
        pocket_cache_dir,
        args.test_split,
        args.test_limit,
        args.sample_mode,
        args.seed,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        fallback_to_full_sequence=args.fallback_to_full_sequence,
    )

    protein_dim, atom_dim, edge_dim = infer_dims(train_loader)
    model = FrozenEsmPocketLigandGNN(
        protein_dim=protein_dim,
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        protein_hidden_dim=args.protein_hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        gnn_type=args.gnn_type,
        ligand_pooling=args.ligand_pooling,
        protein_pooling=args.protein_pooling,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        fallback_to_full_sequence=args.fallback_to_full_sequence,
    ).to(device)
    criterion = make_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_rmse = float("inf")
    best_epoch = -1
    history = []
    start_epoch = 0
    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_path = project_path(args.resume_checkpoint)
        resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        print(f"Resumed from checkpoint: {display_path(resume_path)} at epoch {start_epoch}")

    for epoch in progress_bar(
        range(start_epoch + 1, start_epoch + args.epochs + 1),
        desc="Train pocket GNN epochs",
        unit="epoch",
    ):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)
        valid_metrics, _ = evaluate(model, valid_loader, criterion, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"valid_rmse={valid_metrics['rmse']:.4f} valid_mae={valid_metrics['mae']:.4f} "
            f"valid_pearson={valid_metrics['pearson']:.4f}"
        )

        if valid_metrics["rmse"] < best_valid_rmse:
            best_valid_rmse = valid_metrics["rmse"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "protein_dim": protein_dim,
                    "atom_dim": atom_dim,
                    "edge_dim": edge_dim,
                    "args": vars(args),
                },
                checkpoint_dir / "best.pt",
            )

    checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    valid_metrics, valid_predictions = evaluate(model, valid_loader, criterion, device)
    test_metrics, test_predictions = evaluate(model, test_loader, criterion, device)
    valid_predictions.to_csv(output_dir / "predictions_valid.csv", index=False)
    test_predictions.to_csv(output_dir / "predictions_test.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    metrics = {
        "status": "complete",
        "best_epoch": best_epoch,
        "start_epoch": start_epoch,
        "resumed_from_checkpoint": display_path(project_path(args.resume_checkpoint)) if args.resume_checkpoint else None,
        "resumed_from_epoch": int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint else None,
        "manifest": display_path(manifest_path),
        "esm_cache_dir": display_path(esm_cache_dir),
        "ligand_cache_dir": display_path(ligand_cache_dir),
        "pocket_cache_dir": display_path(pocket_cache_dir),
        "output_dir": display_path(output_dir),
        "device": str(device),
        "row_counts": {
            "train": int(len(train_rows)),
            "valid": int(len(valid_rows)),
            "test": int(len(test_rows)),
        },
        "model": {
            "protein_dim": protein_dim,
            "atom_dim": atom_dim,
            "edge_dim": edge_dim,
            "protein_hidden_dim": args.protein_hidden_dim,
            "protein_pooling": args.protein_pooling,
            "fallback_to_full_sequence": args.fallback_to_full_sequence,
            "gnn_type": args.gnn_type,
            "gnn_layers": args.gnn_layers,
            "gnn_hidden_dim": args.gnn_hidden_dim,
            "ligand_pooling": args.ligand_pooling,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "dropout": args.dropout,
            "loss": args.loss,
        },
        "valid": valid_metrics,
        "test": test_metrics,
        "history": history,
    }
    save_json(output_dir / "metrics.json", metrics)

    print(f"Best epoch: {best_epoch}")
    print(f"Valid metrics: {json.dumps(valid_metrics, indent=2)}")
    print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"Wrote outputs to: {display_path(output_dir)}")


if __name__ == "__main__":
    main()
