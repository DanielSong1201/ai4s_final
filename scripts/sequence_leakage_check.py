"""Protein sequence leakage checks for PDBbind train/valid/test splits."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd


AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "MSE": "M",
}


def extract_chain_sequences_from_pdb(protein_path: str | Path) -> dict[str, str]:
    """Extract chain-level amino-acid sequences from ATOM records in a PDB file."""

    chains: dict[str, list[str]] = {}
    seen_residues: set[tuple[str, str, str]] = set()
    protein_path = Path(protein_path)

    with protein_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            residue_name = line[17:20].strip().upper()
            aa = AA3_TO_AA1.get(residue_name)
            if aa is None:
                continue
            chain_id = line[21].strip() or "_"
            residue_id = line[22:26].strip()
            insertion_code = line[26].strip()
            key = (chain_id, residue_id, insertion_code)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            chains.setdefault(chain_id, []).append(aa)

    return {chain_id: "".join(sequence) for chain_id, sequence in chains.items() if sequence}


def build_sequence_table(split_csv: str | Path, min_length: int = 30) -> pd.DataFrame:
    """Build a chain-level sequence table from the split CSV."""

    split_df = pd.read_csv(split_csv)
    split_df = split_df[split_df["split"].isin(["train", "valid", "test"])].copy()

    rows = []
    for row in split_df.itertuples(index=False):
        sequences = extract_chain_sequences_from_pdb(row.protein_path)
        for chain_id, sequence in sequences.items():
            if len(sequence) < min_length:
                continue
            rows.append(
                {
                    "split": row.split,
                    "pdb_id": row.pdb_id,
                    "chain_id": chain_id,
                    "sequence_id": f"{row.split}|{row.pdb_id}|{chain_id}",
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                }
            )
    return pd.DataFrame(rows)


def write_fasta(sequence_df: pd.DataFrame, fasta_path: str | Path) -> None:
    """Write a chain sequence table to FASTA."""

    fasta_path = Path(fasta_path)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in sequence_df.itertuples(index=False):
            handle.write(f">{row.sequence_id}\n")
            sequence = row.sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def exact_duplicate_leakage(sequence_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback leakage check for exactly identical chain sequences."""

    train = sequence_df[sequence_df["split"].eq("train")]
    eval_df = sequence_df[sequence_df["split"].isin(["valid", "test"])]
    train_by_sequence = train.groupby("sequence")["sequence_id"].apply(list).to_dict()

    rows = []
    for row in eval_df.itertuples(index=False):
        for train_id in train_by_sequence.get(row.sequence, []):
            rows.append(
                {
                    "query": row.sequence_id,
                    "target": train_id,
                    "pident": 100.0,
                    "alnlen": len(row.sequence),
                    "qlen": len(row.sequence),
                    "tlen": len(row.sequence),
                    "evalue": 0.0,
                    "bits": 0.0,
                }
            )
    return pd.DataFrame(rows)


def run_mmseqs_search(
    query_fasta: str | Path,
    target_fasta: str | Path,
    output_tsv: str | Path,
    tmp_dir: str | Path,
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
) -> bool:
    """Run MMseqs2 easy-search. Returns False when mmseqs is unavailable."""

    if shutil.which("mmseqs") is None:
        return False

    command = [
        "mmseqs",
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
    return True


def parse_mmseqs_hits(hits_path: str | Path) -> pd.DataFrame:
    """Read MMseqs2 hits from the configured tabular output."""

    hits_path = Path(hits_path)
    columns = ["query", "target", "pident", "alnlen", "qlen", "tlen", "evalue", "bits"]
    if not hits_path.exists() or hits_path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(hits_path, sep="\t", names=columns)


def annotate_hits(hits: pd.DataFrame) -> pd.DataFrame:
    """Add split, PDB, and chain fields parsed from hit identifiers."""

    if hits.empty:
        return hits

    query_parts = hits["query"].str.split("|", expand=True)
    target_parts = hits["target"].str.split("|", expand=True)
    annotated = hits.copy()
    annotated["query_split"] = query_parts[0]
    annotated["query_pdb_id"] = query_parts[1]
    annotated["query_chain_id"] = query_parts[2]
    annotated["target_split"] = target_parts[0]
    annotated["target_pdb_id"] = target_parts[1]
    annotated["target_chain_id"] = target_parts[2]
    return annotated


def check_sequence_leakage(
    split_csv: str | Path,
    output_dir: str | Path = "data/splits/sequence_leakage",
    min_seq_id: float = 0.4,
    coverage: float = 0.8,
    min_length: int = 30,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Check whether valid/test protein chains are similar to train chains."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_df = build_sequence_table(split_csv=split_csv, min_length=min_length)
    train_df = sequence_df[sequence_df["split"].eq("train")]
    eval_df = sequence_df[sequence_df["split"].isin(["valid", "test"])]

    sequence_table_path = output_dir / "chain_sequences.csv"
    train_fasta = output_dir / "train_chains.fasta"
    eval_fasta = output_dir / "valid_test_chains.fasta"
    hits_path = output_dir / "mmseqs_train_vs_valid_test_hits.tsv"
    tmp_dir = output_dir / "mmseqs_tmp"

    sequence_df.to_csv(sequence_table_path, index=False)
    write_fasta(train_df, train_fasta)
    write_fasta(eval_df, eval_fasta)

    used_mmseqs = run_mmseqs_search(
        query_fasta=eval_fasta,
        target_fasta=train_fasta,
        output_tsv=hits_path,
        tmp_dir=tmp_dir,
        min_seq_id=min_seq_id,
        coverage=coverage,
    )
    hits = parse_mmseqs_hits(hits_path) if used_mmseqs else exact_duplicate_leakage(sequence_df)
    hits = annotate_hits(hits)

    annotated_hits_path = output_dir / "sequence_leakage_hits.csv"
    hits.to_csv(annotated_hits_path, index=False)

    report = {
        "split_csv": str(split_csv),
        "method": "mmseqs easy-search" if used_mmseqs else "exact duplicate fallback",
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "min_length": min_length,
        "chain_sequence_rows": int(len(sequence_df)),
        "chain_sequence_counts_by_split": {
            split: int(count) for split, count in sequence_df["split"].value_counts().sort_index().items()
        },
        "complex_counts_by_split": {
            split: int(count) for split, count in sequence_df.groupby("split")["pdb_id"].nunique().sort_index().items()
        },
        "hit_count": int(len(hits)),
        "leakage_detected": bool(len(hits) > 0),
        "sequence_table_path": str(sequence_table_path),
        "hits_path": str(annotated_hits_path),
        "fasta_paths": {
            "train": str(train_fasta),
            "valid_test": str(eval_fasta),
        },
    }

    report_path = output_dir / "sequence_leakage_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return hits, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", default="data/splits/pdbbind_interformer_splits.csv")
    parser.add_argument("--output-dir", default="data/splits/sequence_leakage")
    parser.add_argument("--min-seq-id", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--min-length", type=int, default=30)
    args = parser.parse_args()

    _, report = check_sequence_leakage(
        split_csv=args.split_csv,
        output_dir=args.output_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        min_length=args.min_length,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
