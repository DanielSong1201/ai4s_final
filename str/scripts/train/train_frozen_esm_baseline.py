"""Train a frozen-ESM baseline for protein-ligand affinity regression."""

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
from torch.utils.data import DataLoader
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


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_ESM_CACHE_DIR = Path("str/manifest/cache/esm_embeddings")
DEFAULT_LIGAND_CACHE_DIR = Path("str/manifest/cache/ligand_graphs")
DEFAULT_OUTPUT_DIR = Path("str/manifest/outputs/baseline_frozen_esm")


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


def ligand_mean_pool(atom_features: torch.Tensor, ligand_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    feature_dim = atom_features.shape[-1]
    pooled = atom_features.new_zeros((batch_size, feature_dim))
    pooled.index_add_(0, ligand_batch, atom_features)
    counts = torch.bincount(ligand_batch, minlength=batch_size).to(atom_features.device).clamp_min(1)
    return pooled / counts.unsqueeze(-1).to(atom_features.dtype)


class FrozenEsmLigandBaseline(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(protein_dim + ligand_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        protein_vector = masked_mean_pool(batch["protein_embedding"], batch["protein_mask"])
        ligand_vector = ligand_mean_pool(
            batch["ligand_atom_features"],
            batch["ligand_batch"],
            batch_size=batch["labels"].shape[0],
        )
        fused = torch.cat([protein_vector, ligand_vector], dim=-1)
        return self.regressor(fused).squeeze(-1)


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


def build_loader(
    manifest_path: Path,
    esm_cache_dir: Path,
    ligand_cache_dir: Path,
    split: str,
    limit: int,
    sample_mode: str,
    seed: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[DataLoader, pd.DataFrame]:
    rows = select_manifest_rows(manifest_path, split, limit, seed, sample_mode)
    cache_report = validate_cache_presence(rows, esm_cache_dir, ligand_cache_dir)
    if cache_report["missing_esm_count"] or cache_report["missing_ligand_count"]:
        raise FileNotFoundError(
            f"Cache is incomplete for split={split}: "
            f"missing_esm={cache_report['missing_esm_examples']}, "
            f"missing_ligand={cache_report['missing_ligand_examples']}"
        )

    dataset = EsmLigandAffinityDataset(rows, esm_cache_dir, ligand_cache_dir, split=None)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_esm_ligand_batch,
    )
    return loader, rows


def infer_dims(loader: DataLoader) -> tuple[int, int]:
    batch = next(iter(loader))
    protein_dim = int(batch["protein_embedding"].shape[-1])
    ligand_dim = int(batch["ligand_atom_features"].shape[-1])
    return protein_dim, ligand_dim


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
    progress = tqdm(loader, desc="Train batches", unit="batch", leave=False)
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

    for batch in tqdm(loader, desc="Evaluate batches", unit="batch", leave=False):
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

        for pdb_id, split, label, pred in zip(batch["pdb_id"], batch["split"], labels, preds):
            rows.append(
                {
                    "pdb_id": pdb_id,
                    "split": split,
                    "label": float(label),
                    "prediction": float(pred),
                    "error": float(pred - label),
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
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    manifest_path = project_path(args.manifest)
    esm_cache_dir = project_path(args.esm_cache_dir)
    ligand_cache_dir = project_path(args.ligand_cache_dir)
    output_dir = project_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    train_loader, train_rows = build_loader(
        manifest_path,
        esm_cache_dir,
        ligand_cache_dir,
        args.train_split,
        args.train_limit,
        args.sample_mode,
        args.seed,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    valid_loader, valid_rows = build_loader(
        manifest_path,
        esm_cache_dir,
        ligand_cache_dir,
        args.valid_split,
        args.valid_limit,
        args.sample_mode,
        args.seed,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    test_loader, test_rows = build_loader(
        manifest_path,
        esm_cache_dir,
        ligand_cache_dir,
        args.test_split,
        args.test_limit,
        args.sample_mode,
        args.seed,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    protein_dim, ligand_dim = infer_dims(train_loader)
    model = FrozenEsmLigandBaseline(
        protein_dim=protein_dim,
        ligand_dim=ligand_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = make_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_rmse = float("inf")
    best_epoch = -1
    history = []

    for epoch in tqdm(range(1, args.epochs + 1), desc="Train epochs", unit="epoch"):
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
                    "ligand_dim": ligand_dim,
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
        "manifest": display_path(manifest_path),
        "esm_cache_dir": display_path(esm_cache_dir),
        "ligand_cache_dir": display_path(ligand_cache_dir),
        "output_dir": display_path(output_dir),
        "device": str(device),
        "row_counts": {
            "train": int(len(train_rows)),
            "valid": int(len(valid_rows)),
            "test": int(len(test_rows)),
        },
        "model": {
            "protein_dim": protein_dim,
            "ligand_dim": ligand_dim,
            "hidden_dim": args.hidden_dim,
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
