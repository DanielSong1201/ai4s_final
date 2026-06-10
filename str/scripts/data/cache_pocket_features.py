"""Cache residue-level pocket masks aligned to cached ESM residue embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.create_esm_manifest import AA3_TO_AA1, display_path, project_path  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_CACHE_DIR = Path("str/manifest/cache/pocket_features")
DEFAULT_REPORT = Path("str/manifest/cache/pocket_features_report.json")


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


def residue_key(line: str) -> tuple[str, str, str]:
    chain_id = line[21].strip() or "_"
    residue_id = line[22:26].strip()
    insertion_code = line[26].strip()
    return chain_id, residue_id, insertion_code


def atom_coordinates_from_line(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def parse_protein_residues(path: Path) -> list[dict[str, Any]]:
    residues: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_name = line[17:20].strip().upper()
            aa = AA3_TO_AA1.get(residue_name)
            if aa is None:
                continue
            key = residue_key(line)
            if key not in residues:
                residues[key] = {
                    "key": key,
                    "chain_id": key[0],
                    "residue_id": key[1],
                    "insertion_code": key[2],
                    "residue_name": residue_name,
                    "aa": aa,
                    "atom_coords": [],
                    "ca_coord": None,
                }
                order.append(key)
            coord = atom_coordinates_from_line(line)
            residues[key]["atom_coords"].append(coord)
            if line[12:16].strip() == "CA":
                residues[key]["ca_coord"] = coord

    chain_order = sorted({key[0] for key in order})
    ordered: list[dict[str, Any]] = []
    for chain_id in chain_order:
        ordered.extend(residues[key] for key in order if key[0] == chain_id)
    return ordered


def parse_pocket_residue_keys(path: Path) -> set[tuple[str, str, str]]:
    keys = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name not in AA3_TO_AA1:
                continue
            keys.add(residue_key(line))
    return keys


def residue_coordinate(residue: dict[str, Any]) -> tuple[float, float, float]:
    if residue["ca_coord"] is not None:
        return residue["ca_coord"]
    coords = residue["atom_coords"]
    count = max(len(coords), 1)
    return (
        sum(coord[0] for coord in coords) / count,
        sum(coord[1] for coord in coords) / count,
        sum(coord[2] for coord in coords) / count,
    )


def cache_one_row(row) -> dict[str, Any]:
    protein_path = project_path(Path(row.protein_path))
    pocket_path = project_path(Path(row.pocket_path))
    residues = parse_protein_residues(protein_path)
    pocket_keys = parse_pocket_residue_keys(pocket_path)
    expected_length = int(row.protein_sequence_length)

    mask_values = []
    coords = []
    residue_rows = []
    for index, residue in enumerate(residues):
        key = residue["key"]
        is_pocket = key in pocket_keys
        mask_values.append(is_pocket)
        coords.append(residue_coordinate(residue))
        residue_rows.append(
            {
                "index": index,
                "chain_id": residue["chain_id"],
                "residue_id": residue["residue_id"],
                "insertion_code": residue["insertion_code"],
                "residue_name": residue["residue_name"],
                "aa": residue["aa"],
                "is_pocket": bool(is_pocket),
            }
        )

    sequence = "".join(residue["aa"] for residue in residues)
    manifest_sequence = str(row.protein_sequence).replace(":", "")
    length_matches_manifest = len(residues) == expected_length == len(manifest_sequence)
    sequence_matches_manifest = sequence == manifest_sequence

    pocket_mask = torch.tensor(mask_values, dtype=torch.bool)
    residue_coordinates = torch.tensor(coords, dtype=torch.float32) if coords else torch.empty((0, 3), dtype=torch.float32)
    pocket_indices = torch.where(pocket_mask)[0].long()

    return {
        "pdb_id": row.pdb_id,
        "split": row.split,
        "protein_path": row.protein_path,
        "pocket_path": row.pocket_path,
        "protein_length": int(len(residues)),
        "manifest_sequence_length": expected_length,
        "pocket_residue_count": int(pocket_mask.sum().item()),
        "pocket_mask": pocket_mask,
        "pocket_indices": pocket_indices,
        "residue_coordinates": residue_coordinates,
        "residue_table": residue_rows,
        "length_matches_manifest": bool(length_matches_manifest),
        "sequence_matches_manifest": bool(sequence_matches_manifest),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=-1, help="Limit rows for debugging. Use -1 for all rows.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = project_path(args.manifest)
    cache_dir = project_path(args.cache_dir)
    report_json = project_path(args.report_json)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if args.limit >= 0:
        df = df.head(args.limit).copy()

    written = 0
    skipped_existing = 0
    failures = []
    pocket_counts = []
    length_mismatch = []
    sequence_mismatch = []
    split_counts: Counter[str] = Counter()

    rows_iter = progress_bar(df.itertuples(index=False), desc="Cache pocket features", unit="complex")
    for row in rows_iter:
        output_path = cache_dir / f"{row.pdb_id}.pt"
        try:
            if output_path.exists() and not args.overwrite:
                record = torch.load(output_path, map_location="cpu", weights_only=False)
                skipped_existing += 1
            else:
                record = cache_one_row(row)
                torch.save(record, output_path)
                written += 1

            split_counts[str(record["split"])] += 1
            pocket_counts.append(int(record["pocket_residue_count"]))
            if not bool(record["length_matches_manifest"]):
                length_mismatch.append(str(record["pdb_id"]))
            if not bool(record["sequence_matches_manifest"]):
                sequence_mismatch.append(str(record["pdb_id"]))
        except Exception as exc:
            failures.append({"pdb_id": row.pdb_id, "split": row.split, "error": str(exc)})

    report = {
        "manifest": display_path(manifest_path),
        "cache_dir": display_path(cache_dir),
        "input_rows": int(len(df)),
        "written": written,
        "skipped_existing": skipped_existing,
        "failure_count": len(failures),
        "failures": failures[:20],
        "split_counts": dict(split_counts),
        "length_mismatch_count": len(length_mismatch),
        "length_mismatch_examples": length_mismatch[:20],
        "sequence_mismatch_count": len(sequence_mismatch),
        "sequence_mismatch_examples": sequence_mismatch[:20],
        "pocket_residue_count_min": int(min(pocket_counts)) if pocket_counts else None,
        "pocket_residue_count_max": int(max(pocket_counts)) if pocket_counts else None,
        "pocket_residue_count_mean": float(sum(pocket_counts) / len(pocket_counts)) if pocket_counts else None,
        "empty_pocket_count": int(sum(count == 0 for count in pocket_counts)),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Cached pocket features in: {display_path(cache_dir)}")
    print(f"Wrote report: {display_path(report_json)}")
    print(f"Rows: {len(df)}, written: {written}, skipped_existing: {skipped_existing}, failures: {len(failures)}")
    print(f"Length mismatches: {len(length_mismatch)}, sequence mismatches: {len(sequence_mismatch)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
