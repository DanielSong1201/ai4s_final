"""Create Interformer-compatible splits with <40% cross-split sequence similarity."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.create_interformer_splits import (
    EVAL_SUBSET_FILES,
    create_split_table,
    read_id_file,
)
from scripts.data_inspection import load_raw_dataset
from scripts.sequence_leakage_check import extract_chain_sequences_from_pdb, write_fasta


RAW_INDEX_PATH = Path("data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst")
RAW_COMPLEX_ROOT = Path("data/raw/pdbbind2020/complexes/P-L")
SOURCE_SPLIT_DIR = Path("split")
OUTPUT_SPLIT_DIR = Path("split_sequence_cluster")
OUTPUT_DIR = Path("data/processed/sequence_cluster_split_interformer_compatible")
DEFAULT_ALL_VS_ALL = None
SPLITS = ("train", "valid", "test")


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        if left not in self.parent or right not in self.parent:
            return
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def find_mmseqs() -> str | None:
    candidate = shutil.which("mmseqs")
    if candidate:
        return candidate
    homebrew_candidate = Path("/opt/homebrew/bin/mmseqs")
    if homebrew_candidate.exists():
        return str(homebrew_candidate)
    return None


def split_chain_id(chain_id: str) -> tuple[str, str]:
    if "_" not in chain_id:
        return chain_id.lower(), "_"
    pdb_id, chain = chain_id.split("_", 1)
    return pdb_id.lower(), chain


def load_source_table(
    index_path: str | Path = RAW_INDEX_PATH,
    complex_root: str | Path = RAW_COMPLEX_ROOT,
    source_split_dir: str | Path = SOURCE_SPLIT_DIR,
    universe: str = "interformer_assigned",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if universe == "interformer_assigned":
        split_df, source_report = create_split_table(
            index_path=index_path,
            complex_root=complex_root,
            split_dir=source_split_dir,
        )
        split_df = split_df[split_df["split"].isin(SPLITS)].copy()
    elif universe == "all_raw":
        split_df = load_raw_dataset(index_path=index_path, complex_root=complex_root)
        split_df.insert(0, "split", "unassigned")
        split_df.insert(1, "split_source", "raw PDBbind before sequence-cluster splitting")
        source_report = {
            "raw_rows": int(len(split_df)),
            "universe": "all_raw",
        }
    else:
        raise ValueError("universe must be 'interformer_assigned' or 'all_raw'")

    split_df["pdb_id"] = split_df["pdb_id"].str.lower()
    return split_df, source_report


def build_chain_sequence_table(split_df: pd.DataFrame, min_length: int = 30) -> pd.DataFrame:
    rows = []
    for row in split_df.itertuples(index=False):
        for chain_id, sequence in extract_chain_sequences_from_pdb(row.protein_path).items():
            if len(sequence) < min_length:
                continue
            rows.append(
                {
                    "pdb_id": row.pdb_id,
                    "chain_id": chain_id,
                    "sequence_id": f"{row.pdb_id}_{chain_id}",
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                }
            )
    return pd.DataFrame(rows)


def write_all_vs_all(
    chain_df: pd.DataFrame,
    output_dir: Path,
    min_seq_id: float,
    coverage: float,
) -> Path:
    mmseqs = find_mmseqs()
    if mmseqs is None:
        raise RuntimeError("MMseqs2 was not found. Install mmseqs or pass --all-vs-all-m8.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / "protein_chains.fasta"
    hits_path = output_dir / "pdbbind_seqid_40_all_vs_all.m8"
    tmp_dir = output_dir / "mmseqs_all_vs_all_tmp"
    write_fasta(chain_df.rename(columns={"sequence_id": "sequence_id"}), fasta_path)
    command = [
        mmseqs,
        "easy-search",
        str(fasta_path),
        str(fasta_path),
        str(hits_path),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        "0",
        "--format-output",
        "query,target,pident,alnlen,qcov,tcov,evalue,bits",
        "-v",
        "1",
    ]
    subprocess.run(command, check=True)
    return hits_path


def build_components_from_m8(
    pdb_ids: set[str],
    m8_path: str | Path,
    min_seq_id: float,
    coverage: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    uf = UnionFind(pdb_ids)
    stats = {
        "m8_path": str(m8_path),
        "similar_chain_hits_used": 0,
        "all_rows_seen": 0,
        "rows_skipped_outside_universe": 0,
    }
    threshold_percent = min_seq_id * 100.0

    with Path(m8_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stats["all_rows_seen"] += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            query, target = parts[0], parts[1]
            pident = float(parts[2])
            qcov = float(parts[4])
            tcov = float(parts[5])
            if pident < threshold_percent or qcov < coverage or tcov < coverage:
                continue
            query_pdb, _ = split_chain_id(query)
            target_pdb, _ = split_chain_id(target)
            if query_pdb not in pdb_ids or target_pdb not in pdb_ids:
                stats["rows_skipped_outside_universe"] += 1
                continue
            stats["similar_chain_hits_used"] += 1
            uf.union(query_pdb, target_pdb)

    groups: dict[str, list[str]] = defaultdict(list)
    for pdb_id in sorted(pdb_ids):
        groups[uf.find(pdb_id)].append(pdb_id)

    rows = []
    for index, members in enumerate(sorted(groups.values(), key=lambda values: (-len(values), values[0]))):
        component_id = f"seq_component_{index:05d}"
        for pdb_id in members:
            rows.append({"pdb_id": pdb_id, "component_id": component_id, "component_size": len(members)})

    components_df = pd.DataFrame(rows)
    stats["component_count"] = int(components_df["component_id"].nunique())
    stats["largest_component_size"] = int(components_df["component_size"].max())
    return components_df, stats


def source_target_counts(split_df: pd.DataFrame) -> dict[str, int]:
    counts = split_df["split"].value_counts().to_dict()
    return {split: int(counts.get(split, 0)) for split in SPLITS}


def target_counts_from_interformer_ratio(
    split_df: pd.DataFrame,
    index_path: str | Path,
    complex_root: str | Path,
    source_split_dir: str | Path,
) -> dict[str, int]:
    """Apply Interformer's primary split proportions to a different universe."""

    source_df, _ = load_source_table(
        index_path=index_path,
        complex_root=complex_root,
        source_split_dir=source_split_dir,
        universe="interformer_assigned",
    )
    source_counts = source_target_counts(source_df)
    source_total = sum(source_counts.values())
    total = len(split_df)
    valid = int(round(total * source_counts["valid"] / source_total))
    test = int(round(total * source_counts["test"] / source_total))
    train = total - valid - test
    return {"train": train, "valid": valid, "test": test}


def component_table(split_df: pd.DataFrame, components_df: pd.DataFrame) -> pd.DataFrame:
    merged = split_df.merge(components_df, on="pdb_id", how="left")
    if merged["component_id"].isna().any():
        missing = merged.loc[merged["component_id"].isna(), "pdb_id"].head(10).tolist()
        raise RuntimeError(f"Missing sequence components for PDB IDs: {missing}")

    rows = []
    for component_id, group in merged.groupby("component_id"):
        p_affinity = pd.to_numeric(group["pAffinity"], errors="coerce")
        years = pd.to_numeric(group["release_year"], errors="coerce")
        rows.append(
            {
                "component_id": component_id,
                "component_size": int(len(group)),
                "pdb_ids": sorted(group["pdb_id"].tolist()),
                "pAffinity_mean": float(p_affinity.mean()) if p_affinity.notna().any() else np.nan,
                "release_year_mean": float(years.mean()) if years.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def dp_subset(component_ids: list[str], sizes: dict[str, int], target: int) -> set[str]:
    """Subset-sum DP, preferring sums closest to target without exceeding it."""

    dp: dict[int, tuple[str, int] | None] = {0: None}
    for component_id in component_ids:
        size = sizes[component_id]
        if size > target:
            continue
        for subtotal in sorted(list(dp), reverse=True):
            new_total = subtotal + size
            if new_total <= target and new_total not in dp:
                dp[new_total] = (component_id, subtotal)
        if target in dp:
            break

    best_total = max(dp)
    selected: set[str] = set()
    current = best_total
    while current != 0:
        component_id, previous = dp[current]
        selected.add(component_id)
        current = previous
    return selected


def score_assignment(split_df: pd.DataFrame, target_counts: dict[str, int]) -> float:
    counts = split_df["split"].value_counts().to_dict()
    total = len(split_df)
    score = 0.0
    for split, target in target_counts.items():
        score += abs(counts.get(split, 0) - target) / max(total, 1) * 10.0

    overall_affinity = pd.to_numeric(split_df["pAffinity"], errors="coerce")
    affinity_std = float(overall_affinity.std()) or 1.0
    for split in SPLITS:
        part = pd.to_numeric(split_df.loc[split_df["split"].eq(split), "pAffinity"], errors="coerce")
        if part.notna().any() and overall_affinity.notna().any():
            score += abs(float(part.mean()) - float(overall_affinity.mean())) / affinity_std
    return score


def assign_components_to_splits(
    split_df: pd.DataFrame,
    components_df: pd.DataFrame,
    target_counts: dict[str, int],
    seeds: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    comp_df = component_table(split_df, components_df)
    sizes = dict(zip(comp_df["component_id"], comp_df["component_size"]))
    ids = comp_df["component_id"].tolist()
    best = None

    for seed in range(seeds):
        rng = random.Random(seed)
        ordered = ids[:]
        rng.shuffle(ordered)

        test_components = dp_subset(ordered, sizes, target_counts["test"])
        remaining = [component_id for component_id in ordered if component_id not in test_components]
        valid_components = dp_subset(remaining, sizes, target_counts["valid"])
        train_components = set(ids) - test_components - valid_components

        assignment = {
            **{component_id: "test" for component_id in test_components},
            **{component_id: "valid" for component_id in valid_components},
            **{component_id: "train" for component_id in train_components},
        }
        candidate = split_df.merge(components_df, on="pdb_id", how="left").copy()
        candidate["split"] = candidate["component_id"].map(assignment)
        candidate["split_source"] = "sequence_cluster_min_seq_id_0.4_coverage_0.8"
        score = score_assignment(candidate, target_counts)
        if best is None or score < best[0]:
            best = (score, seed, assignment, candidate)

    _, best_seed, best_assignment, assigned_df = best
    comp_df["assigned_split"] = comp_df["component_id"].map(best_assignment)
    report = {
        "best_seed": int(best_seed),
        "assignment_score": float(best[0]),
        "target_counts": target_counts,
        "assigned_counts": {
            split: int(count)
            for split, count in assigned_df["split"].value_counts().sort_index().items()
        },
        "component_counts_by_split": {
            split: int(count)
            for split, count in comp_df["assigned_split"].value_counts().sort_index().items()
        },
    }
    return assigned_df, comp_df, report


def validate_no_cross_split_hits(
    assigned_df: pd.DataFrame,
    m8_path: str | Path,
    min_seq_id: float,
    coverage: float,
    max_rows: int = 1000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    split_by_pdb = dict(zip(assigned_df["pdb_id"], assigned_df["split"]))
    rows = []
    threshold_percent = min_seq_id * 100.0
    violation_count = 0

    with Path(m8_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            query, target = parts[0], parts[1]
            pident = float(parts[2])
            qcov = float(parts[4])
            tcov = float(parts[5])
            if pident < threshold_percent or qcov < coverage or tcov < coverage:
                continue
            query_pdb, query_chain = split_chain_id(query)
            target_pdb, target_chain = split_chain_id(target)
            query_split = split_by_pdb.get(query_pdb)
            target_split = split_by_pdb.get(target_pdb)
            if query_split is None or target_split is None or query_split == target_split:
                continue
            violation_count += 1
            if len(rows) < max_rows:
                rows.append(
                    {
                        "query": query,
                        "target": target,
                        "pident": pident,
                        "qcov": qcov,
                        "tcov": tcov,
                        "query_split": query_split,
                        "target_split": target_split,
                        "query_pdb_id": query_pdb,
                        "query_chain_id": query_chain,
                        "target_pdb_id": target_pdb,
                        "target_chain_id": target_chain,
                    }
                )
    violations = pd.DataFrame(rows)
    report = {
        "cross_split_violation_count": int(violation_count),
        "leakage_detected": bool(violation_count > 0),
        "violation_rows_saved": int(len(violations)),
    }
    return violations, report


def write_interformer_compatible_files(
    assigned_df: pd.DataFrame,
    source_split_dir: str | Path,
    output_split_dir: str | Path,
) -> dict[str, object]:
    output_split_dir = Path(output_split_dir)
    source_split_dir = Path(source_split_dir)
    output_split_dir.mkdir(parents=True, exist_ok=True)

    file_counts = {}
    primary_file_by_split = {
        "train": "timesplit_no_lig_overlap_train",
        "valid": "timesplit_no_lig_overlap_val",
        "test": "timesplit_test",
    }
    for split, filename in primary_file_by_split.items():
        ids = sorted(assigned_df.loc[assigned_df["split"].eq(split), "pdb_id"].unique())
        (output_split_dir / filename).write_text("\n".join(ids) + "\n", encoding="utf-8")
        file_counts[filename] = len(ids)

    test_ids = set(assigned_df.loc[assigned_df["split"].eq("test"), "pdb_id"])
    for _, filename in EVAL_SUBSET_FILES.items():
        source_path = source_split_dir / filename
        if not source_path.exists():
            ids = []
        else:
            ids = [pdb_id for pdb_id in read_id_file(source_path, posebusters=(filename == "posebusters_pdb_ccd_ids.txt"))]
        filtered_ids = sorted(set(ids) & test_ids)
        (output_split_dir / filename).write_text("\n".join(filtered_ids) + ("\n" if filtered_ids else ""), encoding="utf-8")
        file_counts[filename] = len(filtered_ids)

    return {"output_split_dir": str(output_split_dir), "file_counts": file_counts}


def create_sequence_cluster_split(
    index_path: str | Path = RAW_INDEX_PATH,
    complex_root: str | Path = RAW_COMPLEX_ROOT,
    source_split_dir: str | Path = SOURCE_SPLIT_DIR,
    output_split_dir: str | Path = OUTPUT_SPLIT_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    all_vs_all_m8: str | Path | None = DEFAULT_ALL_VS_ALL,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    min_length: int = 30,
    seeds: int = 128,
    universe: str = "interformer_assigned",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df, source_report = load_source_table(index_path, complex_root, source_split_dir, universe=universe)
    if universe == "interformer_assigned":
        target_counts = source_target_counts(source_df)
    else:
        target_counts = target_counts_from_interformer_ratio(source_df, index_path, complex_root, source_split_dir)
    chain_df = build_chain_sequence_table(source_df, min_length=min_length)
    chain_df.to_csv(output_dir / "chain_sequences.csv", index=False)

    m8_path = Path(all_vs_all_m8) if all_vs_all_m8 else output_dir / "pdbbind_seqid_40_all_vs_all.m8"
    if not m8_path.exists():
        m8_path = write_all_vs_all(chain_df, output_dir, min_seq_id=min_seq_id, coverage=coverage)

    components_df, component_report = build_components_from_m8(
        pdb_ids=set(source_df["pdb_id"]),
        m8_path=m8_path,
        min_seq_id=min_seq_id,
        coverage=coverage,
    )
    assigned_df, component_assignments, assignment_report = assign_components_to_splits(
        source_df,
        components_df,
        target_counts=target_counts,
        seeds=seeds,
    )
    violations, validation_report = validate_no_cross_split_hits(
        assigned_df,
        m8_path=m8_path,
        min_seq_id=min_seq_id,
        coverage=coverage,
    )
    write_report = write_interformer_compatible_files(assigned_df, source_split_dir, output_split_dir)

    assigned_csv = output_dir / "pdbbind_sequence_cluster_splits.csv"
    component_csv = output_dir / "sequence_component_assignments.csv"
    violations_csv = output_dir / "cross_split_sequence_violations.csv"
    assigned_df.to_csv(assigned_csv, index=False)
    component_assignments.to_csv(component_csv, index=False)
    violations.to_csv(violations_csv, index=False)

    report = {
        "method": "sequence connected components + ratio-constrained component split",
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "min_length": min_length,
        "source_split_dir": str(source_split_dir),
        "universe": universe,
        "output_split_dir": str(output_split_dir),
        "output_dir": str(output_dir),
        "source_report": source_report,
        "target_counts": target_counts,
        "chain_rows": int(len(chain_df)),
        "component_report": component_report,
        "assignment_report": assignment_report,
        "validation_report": validation_report,
        "write_report": write_report,
        "assigned_csv": str(assigned_csv),
        "component_csv": str(component_csv),
        "violations_csv": str(violations_csv),
    }
    (output_dir / "sequence_cluster_split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return assigned_df, component_assignments, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=str(RAW_INDEX_PATH))
    parser.add_argument("--complex-root", default=str(RAW_COMPLEX_ROOT))
    parser.add_argument("--source-split-dir", default=str(SOURCE_SPLIT_DIR))
    parser.add_argument("--output-split-dir", default=str(OUTPUT_SPLIT_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--all-vs-all-m8", default=DEFAULT_ALL_VS_ALL)
    parser.add_argument("--min-seq-id", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument(
        "--universe",
        choices=["interformer_assigned", "all_raw"],
        default="interformer_assigned",
    )
    args = parser.parse_args()

    _, _, report = create_sequence_cluster_split(
        index_path=args.index_path,
        complex_root=args.complex_root,
        source_split_dir=args.source_split_dir,
        output_split_dir=args.output_split_dir,
        output_dir=args.output_dir,
        all_vs_all_m8=args.all_vs_all_m8,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        min_length=args.min_length,
        seeds=args.seeds,
        universe=args.universe,
    )
    print(json.dumps(report["assignment_report"], indent=2, ensure_ascii=False))
    print(json.dumps(report["validation_report"], indent=2, ensure_ascii=False))
    print(json.dumps(report["write_report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
