"""Train a frozen-ESM + spatial pocket-ligand interaction model."""

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

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402
from scripts.train.train_ligand_graph_transformer_frozen_esm import (  # noqa: E402
    AttentionPool,
    GraphTransformerLayer,
    LigandGraphTransformer,
    masked_mean_pool,
    masked_max_pool,
)
from scripts.train.train_pocket_gnn_frozen_esm import (  # noqa: E402
    build_loader,
    compute_metrics,
    infer_dims,
    make_loss,
    move_batch_to_device,
)


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_ESM_CACHE_DIR = Path("str/manifest/cache/esm_embeddings")
DEFAULT_LIGAND_CACHE_DIR = Path("str/manifest/cache/ligand_graphs")
DEFAULT_POCKET_CACHE_DIR = Path("str/manifest/cache/pocket_features")
DEFAULT_OUTPUT_DIR = Path("str/manifest/outputs/spatial_pocket_interaction_frozen_esm")


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


class SpatialPocketEncoder(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        ffn_multiplier: int,
        rbf_bins: int,
        rbf_max_distance: float,
        fallback_to_full_sequence: bool,
    ) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("--pocket-layers must be positive")
        if hidden_dim % heads != 0:
            raise ValueError("--hidden-dim must be divisible by --attention-heads")
        self.heads = heads
        self.rbf_bins = rbf_bins
        self.rbf_max_distance = rbf_max_distance
        self.fallback_to_full_sequence = fallback_to_full_sequence
        self.register_buffer("rbf_centers", torch.linspace(0.0, rbf_max_distance, rbf_bins), persistent=False)
        self.residue_encoder = nn.Sequential(
            nn.Linear(protein_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.distance_bias = nn.Linear(rbf_bins, heads, bias=False)
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, heads, dropout, ffn_multiplier) for _ in range(layers)]
        )

    def rbf(self, distances: torch.Tensor) -> torch.Tensor:
        width = self.rbf_max_distance / max(self.rbf_bins - 1, 1)
        width = max(width, 1e-6)
        return torch.exp(-((distances.unsqueeze(-1) - self.rbf_centers) / width) ** 2)

    def effective_mask(self, protein_mask: torch.Tensor, pocket_mask: torch.Tensor) -> torch.Tensor:
        mask = protein_mask & pocket_mask
        missing = mask.sum(dim=1).eq(0)
        if missing.any() and self.fallback_to_full_sequence:
            mask = mask.clone()
            mask[missing] = protein_mask[missing]
        return mask

    def pack_pocket(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        protein_embedding = batch["protein_embedding"]
        residue_coordinates = batch["residue_coordinates"].float()
        mask = self.effective_mask(batch["protein_mask"], batch["pocket_mask"])
        batch_size, _, protein_dim = protein_embedding.shape
        counts = mask.sum(dim=1).clamp_min(1)
        max_nodes = int(counts.max().item())

        packed_embeddings = protein_embedding.new_zeros((batch_size, max_nodes, protein_dim))
        packed_coords = residue_coordinates.new_zeros((batch_size, max_nodes, 3))
        packed_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=protein_embedding.device)

        for index in range(batch_size):
            residue_ids = torch.where(mask[index])[0]
            if residue_ids.numel() == 0:
                residue_ids = torch.where(batch["protein_mask"][index])[0]
            count = int(residue_ids.numel())
            packed_embeddings[index, :count] = protein_embedding[index, residue_ids]
            packed_coords[index, :count] = residue_coordinates[index, residue_ids]
            packed_mask[index, :count] = True

        return packed_embeddings, packed_coords, packed_mask

    def coordinate_features(self, coordinates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(coordinates.dtype)
        center = (coordinates * mask_float).sum(dim=1, keepdim=True)
        center = center / mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        centered = coordinates - center
        radial = torch.linalg.norm(centered, dim=-1, keepdim=True)
        scale = radial.masked_fill(~mask.unsqueeze(-1), 0.0).amax(dim=1, keepdim=True).clamp_min(1.0)
        return torch.cat([centered / scale, radial / scale], dim=-1)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings, coordinates, mask = self.pack_pocket(batch)
        residue_features = self.residue_encoder(torch.cat([embeddings, self.coordinate_features(coordinates, mask)], dim=-1))
        distances = torch.cdist(coordinates, coordinates).clamp_max(self.rbf_max_distance)
        attn_bias = self.distance_bias(self.rbf(distances)).permute(0, 3, 1, 2)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        attn_bias = attn_bias.masked_fill(~pair_mask, 0.0)

        for layer in self.layers:
            residue_features = layer(residue_features, mask, attn_bias)
        return residue_features, coordinates, mask


class SpatialCrossAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, rbf_bins: int, rbf_max_distance: float) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("--hidden-dim must be divisible by --attention-heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim**-0.5
        self.rbf_bins = rbf_bins
        self.rbf_max_distance = rbf_max_distance
        self.register_buffer("rbf_centers", torch.linspace(0.0, rbf_max_distance, rbf_bins), persistent=False)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.distance_bias = nn.Linear(rbf_bins, heads, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.contact_encoder = nn.Sequential(
            nn.Linear(rbf_bins, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool(hidden_dim)

    def rbf(self, distances: torch.Tensor) -> torch.Tensor:
        width = self.rbf_max_distance / max(self.rbf_bins - 1, 1)
        width = max(width, 1e-6)
        return torch.exp(-((distances.unsqueeze(-1) - self.rbf_centers) / width) ** 2)

    def forward(
        self,
        pocket_nodes: torch.Tensor,
        pocket_coordinates: torch.Tensor,
        pocket_mask: torch.Tensor,
        ligand_nodes: torch.Tensor,
        ligand_coordinates: torch.Tensor,
        ligand_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, pocket_count, hidden_dim = pocket_nodes.shape
        ligand_count = int(ligand_nodes.shape[1])
        distances = torch.cdist(pocket_coordinates, ligand_coordinates).clamp_max(self.rbf_max_distance)
        distance_rbf = self.rbf(distances)

        query = self.query(pocket_nodes).view(batch_size, pocket_count, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(ligand_nodes).view(batch_size, ligand_count, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(ligand_nodes).view(batch_size, ligand_count, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores + self.distance_bias(distance_rbf).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~ligand_mask[:, None, None, :], -torch.inf)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, pocket_count, hidden_dim)

        nearest = distances.masked_fill(~ligand_mask[:, None, :], self.rbf_max_distance).min(dim=-1).values
        contact_features = self.contact_encoder(self.rbf(nearest))
        fused = self.norm(pocket_nodes + self.out(context) + contact_features)
        fused = fused.masked_fill(~pocket_mask.unsqueeze(-1), 0.0)
        return self.pool(fused, pocket_mask)


class SpatialPocketLigandInteractionModel(nn.Module):
    def __init__(
        self,
        protein_dim: int,
        atom_dim: int,
        edge_dim: int,
        hidden_dim: int,
        pocket_layers: int,
        ligand_layers: int,
        attention_heads: int,
        pooling: str,
        fusion_hidden_dim: int,
        dropout: float,
        rbf_bins: int,
        rbf_max_distance: float,
        ffn_multiplier: int,
        fallback_to_full_sequence: bool,
    ) -> None:
        super().__init__()
        self.pocket_encoder = SpatialPocketEncoder(
            protein_dim=protein_dim,
            hidden_dim=hidden_dim,
            layers=pocket_layers,
            heads=attention_heads,
            dropout=dropout,
            ffn_multiplier=ffn_multiplier,
            rbf_bins=rbf_bins,
            rbf_max_distance=rbf_max_distance,
            fallback_to_full_sequence=fallback_to_full_sequence,
        )
        self.ligand_encoder = LigandGraphTransformer(
            atom_dim=atom_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            layers=ligand_layers,
            heads=attention_heads,
            dropout=dropout,
            pooling=pooling,
            rbf_bins=rbf_bins,
            rbf_max_distance=rbf_max_distance,
            ffn_multiplier=ffn_multiplier,
        )
        self.cross_pool = SpatialCrossAttentionPool(
            hidden_dim=hidden_dim,
            heads=attention_heads,
            dropout=dropout,
            rbf_bins=rbf_bins,
            rbf_max_distance=rbf_max_distance,
        )
        self.ligand_projection = nn.Sequential(
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pocket_projection = nn.Sequential(
            nn.Linear(hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def encode_ligand_nodes(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = self.ligand_encoder.pack_graphs(batch)
        node_features = self.ligand_encoder.atom_encoder(packed["node_features"])
        mask = packed["mask"]
        attn_bias = packed["attn_bias"]
        for layer in self.ligand_encoder.layers:
            node_features = layer(node_features, mask, attn_bias)

        if self.ligand_encoder.pooling == "mean":
            ligand_vector = masked_mean_pool(node_features, mask)
        elif self.ligand_encoder.pooling == "max":
            ligand_vector = masked_max_pool(node_features, mask)
        elif self.ligand_encoder.pooling == "attention":
            ligand_vector = self.ligand_encoder.attention_pool(node_features, mask)
        else:
            raise ValueError(f"Unsupported pooling: {self.ligand_encoder.pooling}")

        return node_features, packed["coordinates"], mask, ligand_vector

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        pocket_nodes, pocket_coordinates, pocket_mask = self.pocket_encoder(batch)
        ligand_nodes, ligand_coordinates, ligand_mask, ligand_vector = self.encode_ligand_nodes(batch)
        pocket_vector = self.cross_pool(
            pocket_nodes,
            pocket_coordinates,
            pocket_mask,
            ligand_nodes,
            ligand_coordinates,
            ligand_mask,
        )
        fused = torch.cat([self.pocket_projection(pocket_vector), self.ligand_projection(ligand_vector)], dim=-1)
        return self.regressor(fused).squeeze(-1)


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
    progress = progress_bar(loader, desc="Train spatial pocket batches", unit="batch")
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

    for batch in progress_bar(loader, desc="Evaluate spatial pocket batches", unit="batch"):
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
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--pocket-layers", type=int, default=2)
    parser.add_argument("--ligand-layers", type=int, default=4)
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
    parser.add_argument("--fallback-to-full-sequence", action=argparse.BooleanOptionalAction, default=True)
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
    model = SpatialPocketLigandInteractionModel(
        protein_dim=protein_dim,
        atom_dim=atom_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        pocket_layers=args.pocket_layers,
        ligand_layers=args.ligand_layers,
        attention_heads=args.attention_heads,
        pooling=args.pooling,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        rbf_bins=args.rbf_bins,
        rbf_max_distance=args.rbf_max_distance,
        ffn_multiplier=args.ffn_multiplier,
        fallback_to_full_sequence=args.fallback_to_full_sequence,
    ).to(device)
    criterion = make_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid_rmse = float("inf")
    best_epoch = -1
    history = []

    for epoch in progress_bar(range(1, args.epochs + 1), desc="Train spatial pocket epochs", unit="epoch"):
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
            "hidden_dim": args.hidden_dim,
            "pocket_layers": args.pocket_layers,
            "ligand_layers": args.ligand_layers,
            "attention_heads": args.attention_heads,
            "pooling": args.pooling,
            "rbf_bins": args.rbf_bins,
            "rbf_max_distance": args.rbf_max_distance,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "dropout": args.dropout,
            "loss": args.loss,
            "fallback_to_full_sequence": args.fallback_to_full_sequence,
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
