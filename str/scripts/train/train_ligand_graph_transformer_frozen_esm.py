"""Train a frozen-ESM + ligand Graph Transformer model for affinity regression."""

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
DEFAULT_OUTPUT_DIR = Path("str/manifest/outputs/ligand_graph_transformer_frozen_esm")


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


def masked_max_pool(node_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = node_features.masked_fill(~mask.unsqueeze(-1), -torch.inf)
    pooled = masked.max(dim=1).values
    return torch.nan_to_num(pooled, neginf=0.0)


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(node_features).squeeze(-1)
        scores = scores.masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0).unsqueeze(-1)
        return (node_features * weights).sum(dim=1)


class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, ffn_multiplier: int) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("--transformer-hidden-dim must be divisible by --attention-heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, node_features: torch.Tensor, mask: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        batch_size, max_nodes, hidden_dim = node_features.shape
        qkv = self.qkv(self.norm1(node_features))
        qkv = qkv.view(batch_size, max_nodes, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + attn_bias
        invalid_keys = ~mask[:, None, None, :]
        scores = scores.masked_fill(invalid_keys, -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_dropout(attn)

        updated = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, max_nodes, hidden_dim)
        node_features = node_features + self.resid_dropout(self.out(updated))
        node_features = node_features + self.ffn(self.norm2(node_features))
        return node_features.masked_fill(~mask.unsqueeze(-1), 0.0)


class LigandGraphTransformer(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        edge_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        pooling: str,
        rbf_bins: int,
        rbf_max_distance: float,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("--transformer-layers must be positive")
        self.heads = heads
        self.pooling = pooling
        self.rbf_bins = rbf_bins
        self.rbf_max_distance = rbf_max_distance
        self.register_buffer("rbf_centers", torch.linspace(0.0, rbf_max_distance, rbf_bins), persistent=False)

        augmented_atom_dim = atom_dim + 5
        self.atom_encoder = nn.Sequential(
            nn.Linear(augmented_atom_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.distance_bias = nn.Linear(rbf_bins, heads, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, heads, dropout, ffn_multiplier) for _ in range(layers)]
        )
        self.attention_pool = AttentionPool(hidden_dim) if pooling == "attention" else None

    def rbf(self, distances: torch.Tensor) -> torch.Tensor:
        width = self.rbf_max_distance / max(self.rbf_bins - 1, 1)
        width = max(width, 1e-6)
        return torch.exp(-((distances.unsqueeze(-1) - self.rbf_centers) / width) ** 2)

    def pack_graphs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        atom_features = batch["ligand_atom_features"]
        coordinates = batch["ligand_atom_coordinates"].float()
        graph_batch = batch["ligand_batch"].long()
        edge_index = batch["ligand_bond_index"].long()
        edge_features = batch["ligand_bond_features"].float()
        batch_size = int(batch["labels"].shape[0])
        device = atom_features.device

        node_counts = torch.bincount(graph_batch, minlength=batch_size)
        max_nodes = int(node_counts.max().item())
        atom_dim = int(atom_features.shape[-1])
        edge_dim = int(edge_features.shape[-1])

        node_padded = atom_features.new_zeros((batch_size, max_nodes, atom_dim))
        coord_padded = coordinates.new_zeros((batch_size, max_nodes, 3))
        mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=device)
        global_to_local = torch.empty((atom_features.shape[0],), dtype=torch.long, device=device)

        for graph_index in range(batch_size):
            node_ids = torch.where(graph_batch.eq(graph_index))[0]
            count = int(node_ids.numel())
            if count == 0:
                continue
            local = torch.arange(count, device=device)
            global_to_local[node_ids] = local
            node_padded[graph_index, :count] = atom_features[node_ids]
            coord_padded[graph_index, :count] = coordinates[node_ids]
            mask[graph_index, :count] = True

        degrees = atom_features.new_zeros((atom_features.shape[0],))
        if edge_index.numel() > 0:
            dst = edge_index[1]
            degrees.index_add_(0, dst, torch.ones_like(dst, dtype=atom_features.dtype))
        degree_padded = atom_features.new_zeros((batch_size, max_nodes, 1))
        for graph_index in range(batch_size):
            node_ids = torch.where(graph_batch.eq(graph_index))[0]
            count = int(node_ids.numel())
            if count > 0:
                degree_padded[graph_index, :count, 0] = torch.log1p(degrees[node_ids])

        center = (coord_padded * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
        center = center / node_counts.clamp_min(1).view(batch_size, 1, 1).to(coord_padded.dtype)
        centered = coord_padded - center
        radial = torch.linalg.norm(centered, dim=-1, keepdim=True)
        scale = radial.masked_fill(~mask.unsqueeze(-1), 0.0).amax(dim=1, keepdim=True).clamp_min(1.0)
        coord_features = torch.cat([centered / scale, radial / scale, degree_padded], dim=-1)
        node_augmented = torch.cat([node_padded, coord_features], dim=-1)

        distances = torch.cdist(coord_padded, coord_padded).clamp_max(self.rbf_max_distance)
        distance_bias = self.distance_bias(self.rbf(distances)).permute(0, 3, 1, 2)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        attn_bias = distance_bias.masked_fill(~pair_mask, 0.0)

        if edge_index.numel() > 0 and edge_dim > 0:
            src, dst = edge_index[0], edge_index[1]
            edge_graph = graph_batch[src]
            src_local = global_to_local[src]
            dst_local = global_to_local[dst]
            edge_bias = self.edge_bias(edge_features)
            attn_bias[edge_graph, :, src_local, dst_local] += edge_bias.transpose(0, 1).transpose(0, 1)

        return {"node_features": node_augmented, "coordinates": coord_padded, "mask": mask, "attn_bias": attn_bias}

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        packed = self.pack_graphs(batch)
        node_features = self.atom_encoder(packed["node_features"])
        mask = packed["mask"]
        attn_bias = packed["attn_bias"]

        for layer in self.layers:
            node_features = layer(node_features, mask, attn_bias)

        if self.pooling == "mean":
            return masked_mean_pool(node_features, mask)
        if self.pooling == "max":
            return masked_max_pool(node_features, mask)
        if self.pooling == "attention":
            return self.attention_pool(node_features, mask)
        raise ValueError(f"Unsupported pooling: {self.pooling}")


class FrozenEsmLigandGraphTransformer(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        atom_dim: int,
        edge_dim: int,
        protein_hidden_dim: int,
        transformer_hidden_dim: int,
        transformer_layers: int,
        attention_heads: int,
        pooling: str,
        fusion_hidden_dim: int,
        dropout: float,
        rbf_bins: int,
        rbf_max_distance: float,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_dim, protein_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ligand_transformer = LigandGraphTransformer(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            hidden_dim=transformer_hidden_dim,
            layers=transformer_layers,
            heads=attention_heads,
            dropout=dropout,
            pooling=pooling,
            rbf_bins=rbf_bins,
            rbf_max_distance=rbf_max_distance,
            ffn_multiplier=ffn_multiplier,
        )
        self.ligand_projection = nn.Sequential(
            nn.Linear(transformer_hidden_dim, fusion_hidden_dim),
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
        protein_vector = masked_mean_pool(batch["protein_embedding"], batch["protein_mask"])
        protein_vector = self.protein_projection(protein_vector)
        ligand_vector = self.ligand_projection(self.ligand_transformer(batch))
        return self.regressor(torch.cat([protein_vector, ligand_vector], dim=-1)).squeeze(-1)


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


def infer_dims(loader: DataLoader) -> tuple[int, int, int]:
    batch = next(iter(loader))
    protein_dim = int(batch["protein_embedding"].shape[-1])
    atom_dim = int(batch["ligand_atom_features"].shape[-1])
    edge_dim = int(batch["ligand_bond_features"].shape[-1])
    return protein_dim, atom_dim, edge_dim


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
    progress = progress_bar(loader, desc="Train Graph Transformer batches", unit="batch")
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

    for batch in progress_bar(loader, desc="Evaluate Graph Transformer batches", unit="batch"):
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
    parser.add_argument("--protein-hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-hidden-dim", type=int, default=192)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--ffn-multiplier", type=int, default=4)
    parser.add_argument("--pooling", choices=["mean", "max", "attention"], default="attention")
    parser.add_argument("--rbf-bins", type=int, default=32)
    parser.add_argument("--rbf-max-distance", type=float, default=20.0)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-checkpoint", type=Path)
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

    protein_dim, atom_dim, edge_dim = infer_dims(train_loader)
    model = FrozenEsmLigandGraphTransformer(
        protein_dim=protein_dim,
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        protein_hidden_dim=args.protein_hidden_dim,
        transformer_hidden_dim=args.transformer_hidden_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        pooling=args.pooling,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        rbf_bins=args.rbf_bins,
        rbf_max_distance=args.rbf_max_distance,
        ffn_multiplier=args.ffn_multiplier,
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
        desc="Train Graph Transformer epochs",
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
            "transformer_hidden_dim": args.transformer_hidden_dim,
            "transformer_layers": args.transformer_layers,
            "attention_heads": args.attention_heads,
            "ffn_multiplier": args.ffn_multiplier,
            "pooling": args.pooling,
            "rbf_bins": args.rbf_bins,
            "rbf_max_distance": args.rbf_max_distance,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "dropout": args.dropout,
            "loss": args.loss,
            "extra_features": [
                "log node degree",
                "centered ligand xyz coordinates",
                "center-relative radial distance",
                "3D pairwise distance RBF attention bias",
                "bond feature attention bias",
            ],
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
