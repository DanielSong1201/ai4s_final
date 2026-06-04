"""Split-level sequence and pocket similarity analysis for PDBbind splits."""

from __future__ import annotations

import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.create_interformer_splits import create_split_table
from scripts.sequence_leakage_check import (
    AA3_TO_AA1,
    annotate_hits,
    build_sequence_table,
    parse_mmseqs_hits,
    write_fasta,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_INDEX_PATH = PROJECT_ROOT / "data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst"
RAW_COMPLEX_ROOT = PROJECT_ROOT / "data/raw/pdbbind2020/complexes/P-L"
SPLIT_DIR = PROJECT_ROOT / "split"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/split_similarity_analysis"
SPLIT_CSV = OUTPUT_DIR / "pdbbind_interformer_split_from_root_split.csv"
SPLIT_REPORT_JSON = OUTPUT_DIR / "pdbbind_interformer_split_from_root_split_report.json"

SPLITS = ("train", "valid", "test")
SPLIT_PAIRS = (("train", "valid"), ("train", "test"), ("valid", "test"))
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
ELEMENTS = ("C", "N", "O", "S", "P", "METAL", "OTHER")


def ensure_output_dir(output_dir: str | Path = OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_split_table(
    split_dir: str | Path = SPLIT_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    index_path: str | Path = RAW_INDEX_PATH,
    complex_root: str | Path = RAW_COMPLEX_ROOT,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create the auditable PDBbind table from copied Interformer split files."""

    output_dir = ensure_output_dir(output_dir)
    split_df, report = create_split_table(index_path=index_path, complex_root=complex_root, split_dir=split_dir)
    split_csv = output_dir / SPLIT_CSV.name
    report_json = output_dir / SPLIT_REPORT_JSON.name
    split_df.to_csv(split_csv, index=False)
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return split_df, report


def find_mmseqs() -> str | None:
    """Find MMseqs2 even when Homebrew's bin directory is not on PATH."""

    candidate = shutil.which("mmseqs")
    if candidate:
        return candidate
    homebrew_candidate = Path("/opt/homebrew/bin/mmseqs")
    if homebrew_candidate.exists():
        return str(homebrew_candidate)
    return None


def _annotate_pair_hits(hits: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "split_pair",
                "query",
                "target",
                "pident",
                "alnlen",
                "qlen",
                "tlen",
                "evalue",
                "bits",
                "query_split",
                "query_pdb_id",
                "query_chain_id",
                "target_split",
                "target_pdb_id",
                "target_chain_id",
                "query_coverage",
                "target_coverage",
            ]
        )
    annotated = annotate_hits(hits)
    annotated.insert(0, "split_pair", f"{left}_vs_{right}")
    annotated["query_coverage"] = annotated["alnlen"] / annotated["qlen"].replace(0, np.nan)
    annotated["target_coverage"] = annotated["alnlen"] / annotated["tlen"].replace(0, np.nan)
    return annotated


def run_mmseqs_pair(
    mmseqs: str,
    query_fasta: str | Path,
    target_fasta: str | Path,
    output_tsv: str | Path,
    tmp_dir: str | Path,
    min_seq_id: float,
    coverage: float,
) -> None:
    command = [
        mmseqs,
        "easy-search",
        str(query_fasta),
        str(target_fasta),
        str(output_tsv),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--format-output",
        "query,target,pident,alnlen,qlen,tlen,evalue,bits",
        "-v",
        "1",
    ]
    subprocess.run(command, check=True)


def check_cross_split_sequence_similarity(
    split_csv: str | Path = SPLIT_CSV,
    output_dir: str | Path = OUTPUT_DIR,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    min_length: int = 30,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run all train/valid/test cross-split sequence checks with MMseqs2."""

    output_dir = ensure_output_dir(output_dir)
    sequence_dir = output_dir / "sequence_similarity"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    split_csv = Path(split_csv)
    if not split_csv.is_absolute():
        split_csv = output_dir / split_csv.name

    sequence_df = build_sequence_table(split_csv=split_csv, min_length=min_length)
    sequence_table_path = sequence_dir / "chain_sequences.csv"
    sequence_df.to_csv(sequence_table_path, index=False)

    fasta_paths: dict[str, str] = {}
    for split in SPLITS:
        fasta_path = sequence_dir / f"{split}_chains.fasta"
        write_fasta(sequence_df[sequence_df["split"].eq(split)], fasta_path)
        fasta_paths[split] = str(fasta_path)

    mmseqs = find_mmseqs()
    all_hits = []
    pair_reports = {}
    for left, right in SPLIT_PAIRS:
        pair_name = f"{left}_vs_{right}"
        hits_path = sequence_dir / f"{pair_name}_mmseqs_hits.tsv"
        tmp_dir = sequence_dir / f"tmp_{pair_name}"
        if mmseqs:
            run_mmseqs_pair(
                mmseqs=mmseqs,
                query_fasta=fasta_paths[right],
                target_fasta=fasta_paths[left],
                output_tsv=hits_path,
                tmp_dir=tmp_dir,
                min_seq_id=min_seq_id,
                coverage=coverage,
            )
            raw_hits = parse_mmseqs_hits(hits_path)
            method = "mmseqs easy-search"
        else:
            raw_hits = _exact_duplicate_hits_for_pair(sequence_df, query_split=right, target_split=left)
            method = "exact duplicate fallback"

        hits = _annotate_pair_hits(raw_hits, left=left, right=right)
        all_hits.append(hits)
        pair_reports[pair_name] = {
            "hit_count": int(len(hits)),
            "max_pident": None if hits.empty else float(hits["pident"].max()),
            "mean_pident": None if hits.empty else float(hits["pident"].mean()),
        }

    hits_df = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()
    hits_csv = sequence_dir / "cross_split_sequence_hits.csv"
    hits_df.to_csv(hits_csv, index=False)

    summary_df = summarize_hits_by_pair(hits_df, value_column="pident")
    summary_csv = sequence_dir / "cross_split_sequence_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    report = {
        "split_csv": str(split_csv),
        "method": "mmseqs easy-search" if mmseqs else "exact duplicate fallback",
        "mmseqs_path": mmseqs,
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "min_length": min_length,
        "chain_sequence_rows": int(len(sequence_df)),
        "chain_sequence_counts_by_split": {
            split: int(count) for split, count in sequence_df["split"].value_counts().sort_index().items()
        },
        "pair_reports": pair_reports,
        "hit_count": int(len(hits_df)),
        "leakage_detected": bool(len(hits_df) > 0),
        "sequence_table_path": str(sequence_table_path),
        "hits_path": str(hits_csv),
        "summary_path": str(summary_csv),
        "fasta_paths": fasta_paths,
    }
    report_path = sequence_dir / "cross_split_sequence_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return hits_df, report


def _exact_duplicate_hits_for_pair(sequence_df: pd.DataFrame, query_split: str, target_split: str) -> pd.DataFrame:
    target_df = sequence_df[sequence_df["split"].eq(target_split)]
    query_df = sequence_df[sequence_df["split"].eq(query_split)]
    target_by_sequence = target_df.groupby("sequence")["sequence_id"].apply(list).to_dict()
    rows = []
    for row in query_df.itertuples(index=False):
        for target_id in target_by_sequence.get(row.sequence, []):
            rows.append(
                {
                    "query": row.sequence_id,
                    "target": target_id,
                    "pident": 100.0,
                    "alnlen": len(row.sequence),
                    "qlen": len(row.sequence),
                    "tlen": len(row.sequence),
                    "evalue": 0.0,
                    "bits": 0.0,
                }
            )
    return pd.DataFrame(rows)


def parse_pocket_features(pocket_path: str | Path) -> dict[str, object] | None:
    """Parse a PDB pocket and return residue, atom, and geometry descriptors."""

    pocket_path = Path(pocket_path)
    if not pocket_path.exists():
        return None

    coords: list[tuple[float, float, float]] = []
    residue_counts = dict.fromkeys(AMINO_ACIDS, 0)
    element_counts = dict.fromkeys(ELEMENTS, 0)
    residues_seen: set[tuple[str, str, str]] = set()

    with pocket_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append((x, y, z))

            residue_name = line[17:20].strip().upper()
            chain_id = line[21].strip() or "_"
            residue_id = line[22:26].strip()
            insertion_code = line[26].strip()
            residue_key = (chain_id, residue_id, insertion_code)
            aa = AA3_TO_AA1.get(residue_name)
            if aa and residue_key not in residues_seen:
                residue_counts[aa] += 1
                residues_seen.add(residue_key)

            element = line[76:78].strip().upper()
            if not element:
                atom_name = line[12:16].strip().upper()
                element = "".join(ch for ch in atom_name if ch.isalpha())[:1]
            if element in {"FE", "ZN", "MG", "MN", "CA", "CU", "CO", "NI", "NA", "K"}:
                bucket = "METAL"
            elif element in {"C", "N", "O", "S", "P"}:
                bucket = element
            else:
                bucket = "OTHER"
            element_counts[bucket] += 1

    if not coords:
        return None

    coord_array = np.asarray(coords, dtype=float)
    centroid = coord_array.mean(axis=0)
    centered = coord_array - centroid
    distances = np.linalg.norm(centered, axis=1)
    bbox = coord_array.max(axis=0) - coord_array.min(axis=0)
    residue_total = sum(residue_counts.values())
    atom_total = sum(element_counts.values())

    return {
        "n_atoms": int(len(coord_array)),
        "n_residues": int(residue_total),
        "radius_gyration": float(np.sqrt((distances**2).mean())),
        "max_radius": float(distances.max()),
        "bbox_x": float(bbox[0]),
        "bbox_y": float(bbox[1]),
        "bbox_z": float(bbox[2]),
        "bbox_volume": float(np.prod(np.maximum(bbox, 1e-6))),
        "residue_vector": np.array([residue_counts[aa] / residue_total if residue_total else 0.0 for aa in AMINO_ACIDS]),
        "element_vector": np.array([element_counts[element] / atom_total if atom_total else 0.0 for element in ELEMENTS]),
    }


def build_pocket_feature_table(split_df: pd.DataFrame, output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
    output_dir = ensure_output_dir(output_dir)
    rows = []
    for row in split_df[split_df["split"].isin(SPLITS)].itertuples(index=False):
        features = parse_pocket_features(row.pocket_path)
        if features is None:
            continue
        out = {
            "split": row.split,
            "pdb_id": row.pdb_id,
            "pocket_path": row.pocket_path,
            "n_atoms": features["n_atoms"],
            "n_residues": features["n_residues"],
            "radius_gyration": features["radius_gyration"],
            "max_radius": features["max_radius"],
            "bbox_x": features["bbox_x"],
            "bbox_y": features["bbox_y"],
            "bbox_z": features["bbox_z"],
            "bbox_volume": features["bbox_volume"],
        }
        for aa, value in zip(AMINO_ACIDS, features["residue_vector"]):
            out[f"residue_{aa}"] = float(value)
        for element, value in zip(ELEMENTS, features["element_vector"]):
            out[f"element_{element.lower()}"] = float(value)
        rows.append(out)

    feature_df = pd.DataFrame(rows)
    feature_path = output_dir / "pocket_feature_table.csv"
    feature_df.to_csv(feature_path, index=False)
    return feature_df


def _sample_by_split(df: pd.DataFrame, sample_per_split: int, random_state: int) -> pd.DataFrame:
    sampled = []
    for split in SPLITS:
        part = df[df["split"].eq(split)]
        if len(part) > sample_per_split:
            part = part.sample(sample_per_split, random_state=random_state)
        sampled.append(part)
    return pd.concat(sampled, ignore_index=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_cross_split_pocket_similarity(
    pocket_df: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
    sample_per_split: int = 250,
    random_state: int = 42,
    top_k_per_pair: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute sampled pocket similarity across train/valid/test splits."""

    output_dir = ensure_output_dir(output_dir)
    sampled = _sample_by_split(pocket_df, sample_per_split=sample_per_split, random_state=random_state)
    sampled_path = output_dir / "pocket_similarity_sampled_pockets.csv"
    sampled.to_csv(sampled_path, index=False)

    residue_cols = [f"residue_{aa}" for aa in AMINO_ACIDS]
    element_cols = [f"element_{element.lower()}" for element in ELEMENTS]
    geometry_cols = [
        "n_atoms",
        "n_residues",
        "radius_gyration",
        "max_radius",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "bbox_volume",
    ]

    geom = sampled[geometry_cols].astype(float).to_numpy()
    geom[:, 0:2] = np.log1p(geom[:, 0:2])
    geom[:, 7] = np.log1p(geom[:, 7])
    mean = geom.mean(axis=0)
    std = geom.std(axis=0)
    std[std == 0] = 1.0
    geom_z = (geom - mean) / std

    sampled = sampled.reset_index(drop=True)
    pair_rows = []
    for left, right in SPLIT_PAIRS:
        left_idx = sampled.index[sampled["split"].eq(left)].to_list()
        right_idx = sampled.index[sampled["split"].eq(right)].to_list()
        pair_scores = []
        for i, j in itertools.product(left_idx, right_idx):
            residue_similarity = _cosine(
                sampled.loc[i, residue_cols].astype(float).to_numpy(),
                sampled.loc[j, residue_cols].astype(float).to_numpy(),
            )
            element_similarity = _cosine(
                sampled.loc[i, element_cols].astype(float).to_numpy(),
                sampled.loc[j, element_cols].astype(float).to_numpy(),
            )
            geometry_distance = float(np.linalg.norm(geom_z[i] - geom_z[j]))
            geometry_similarity = math.exp(-geometry_distance / math.sqrt(len(geometry_cols)))
            combined_similarity = (
                0.45 * residue_similarity + 0.25 * element_similarity + 0.30 * geometry_similarity
            )
            pair_scores.append(
                {
                    "split_pair": f"{left}_vs_{right}",
                    "left_split": left,
                    "left_pdb_id": sampled.at[i, "pdb_id"],
                    "right_split": right,
                    "right_pdb_id": sampled.at[j, "pdb_id"],
                    "pocket_similarity": float(combined_similarity),
                    "residue_similarity": float(residue_similarity),
                    "element_similarity": float(element_similarity),
                    "geometry_similarity": float(geometry_similarity),
                }
            )
        pair_scores.sort(key=lambda item: item["pocket_similarity"], reverse=True)
        pair_rows.extend(pair_scores[:top_k_per_pair])
        pair_rows.extend(_evenly_sample_rows(pair_scores, max_rows=2000))

    similarity_df = pd.DataFrame(pair_rows).drop_duplicates(
        ["split_pair", "left_pdb_id", "right_pdb_id"], keep="first"
    )
    hits_path = output_dir / "cross_split_pocket_similarity_sampled_pairs.csv"
    similarity_df.to_csv(hits_path, index=False)

    summary_df = summarize_hits_by_pair(similarity_df, value_column="pocket_similarity")
    summary_path = output_dir / "cross_split_pocket_similarity_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    report = {
        "method": "sampled pocket descriptor similarity",
        "sample_per_split": sample_per_split,
        "random_state": random_state,
        "top_k_per_pair": top_k_per_pair,
        "full_pocket_rows": int(len(pocket_df)),
        "sampled_pocket_rows": int(len(sampled)),
        "pair_rows_written": int(len(similarity_df)),
        "feature_table_path": str(output_dir / "pocket_feature_table.csv"),
        "sampled_pockets_path": str(sampled_path),
        "hits_path": str(hits_path),
        "summary_path": str(summary_path),
        "score_definition": "0.45 residue-composition cosine + 0.25 atom-element cosine + 0.30 exp(-standardized-geometry-distance/sqrt(d))",
    }
    report_path = output_dir / "cross_split_pocket_similarity_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return similarity_df, summary_df, report


def _evenly_sample_rows(rows: list[dict[str, object]], max_rows: int) -> list[dict[str, object]]:
    if len(rows) <= max_rows:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_rows, dtype=int)
    return [rows[int(index)] for index in indices]


def summarize_hits_by_pair(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["split_pair", "n_pairs_or_hits", "max", "mean", "median", "p95", "p90", "p75"]
        )
    rows = []
    for pair, group in df.groupby("split_pair"):
        values = group[value_column].astype(float)
        rows.append(
            {
                "split_pair": pair,
                "n_pairs_or_hits": int(len(group)),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "p90": float(values.quantile(0.90)),
                "p75": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows).sort_values("split_pair")


def html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "<p>No rows.</p>"
    return df.head(max_rows).to_html(index=False, escape=False)


def summary_heatmap_html(summary_df: pd.DataFrame, metric: str, title: str) -> str:
    values = {row.split_pair: float(getattr(row, metric)) for row in summary_df.itertuples(index=False)}
    labels = list(SPLITS)
    cells = []
    max_value = max(values.values()) if values else 1.0
    if max_value <= 0:
        max_value = 1.0
    for left in labels:
        row_cells = [f"<th>{left}</th>"]
        for right in labels:
            if left == right:
                row_cells.append("<td class='diag'>same split</td>")
                continue
            pair = f"{left}_vs_{right}" if f"{left}_vs_{right}" in values else f"{right}_vs_{left}"
            value = values.get(pair)
            if value is None:
                row_cells.append("<td class='empty'>-</td>")
            else:
                strength = min(1.0, value / max_value)
                background = f"rgba(214, 76, 76, {0.18 + 0.72 * strength:.3f})"
                row_cells.append(f"<td style='background:{background}'><b>{value:.3f}</b><br><span>{pair}</span></td>")
        cells.append("<tr>" + "".join(row_cells) + "</tr>")
    return f"""
    <style>
    .similarity-heatmap table {{ border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .similarity-heatmap th, .similarity-heatmap td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; }}
    .similarity-heatmap .diag {{ color: #777; background: #f5f5f5; }}
    .similarity-heatmap .empty {{ color: #aaa; }}
    .similarity-heatmap span {{ color: #555; font-size: 12px; }}
    </style>
    <div class="similarity-heatmap">
      <h3>{title}</h3>
      <table>
        <thead><tr><th></th>{''.join(f'<th>{label}</th>' for label in labels)}</tr></thead>
        <tbody>{''.join(cells)}</tbody>
      </table>
    </div>
    """


def svg_histogram(values: Iterable[float], title: str, x_label: str, bins: int = 20) -> str:
    values = np.asarray(list(values), dtype=float)
    width, height = 760, 260
    margin_left, margin_bottom, margin_top, margin_right = 58, 44, 36, 20
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    if len(values) == 0:
        return f"<svg width='{width}' height='{height}'><text x='20' y='40'>{title}: no values</text></svg>"

    counts, edges = np.histogram(values, bins=bins)
    max_count = max(int(counts.max()), 1)
    bars = []
    for idx, count in enumerate(counts):
        x = margin_left + idx * chart_w / bins
        bar_w = chart_w / bins - 2
        bar_h = chart_h * count / max_count
        y = margin_top + chart_h - bar_h
        bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' fill='#4f7fba'/>")

    x_min, x_max = float(edges[0]), float(edges[-1])
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
      <style>text {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; fill: #222; }}</style>
      <text x="{margin_left}" y="22" font-size="16" font-weight="700">{title}</text>
      <line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#222"/>
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#222"/>
      {''.join(bars)}
      <text x="{margin_left}" y="{height - 14}" font-size="12">{x_min:.2f}</text>
      <text x="{margin_left + chart_w - 44}" y="{height - 14}" font-size="12">{x_max:.2f}</text>
      <text x="{margin_left + chart_w / 2 - 40}" y="{height - 14}" font-size="12">{x_label}</text>
      <text x="8" y="{margin_top + 10}" font-size="12">{max_count}</text>
    </svg>
    """
