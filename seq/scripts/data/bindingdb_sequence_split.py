"""Build BindingDB train/valid/test splits with <40% protein similarity leakage control.

The full BindingDB TSV is too large to load eagerly. The ``clean`` step streams
the raw TSV, extracts a sequence-based affinity table, and writes compact CSV
artifacts. Later steps operate on the cleaned table and unique protein FASTA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback keeps the pipeline runnable.
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []


AFFINITY_COLUMNS = {
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "IC50": "IC50 (nM)",
    "EC50": "EC50 (nM)",
}
DEFAULT_RAW_TSV = Path("seq/data/BindingDB_All.tsv")
DEFAULT_OUTPUT_DIR = Path("seq/processed/bindingdb_sequence_split")
SPLITS = ("train", "valid", "test")
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")
EXACT_NUMERIC_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\s*$")


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        if left not in self.parent or right not in self.parent:
            return
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class CleanRow:
    reactant_set_id: str
    ligand_smiles: str
    monomer_id: str
    target_name: str
    protein_id: str
    protein_sequence: str
    protein_length: int
    uniprot_id: str
    affinity_type: str
    affinity_nM: float
    p_affinity: float
    num_chains: int
    pdb_ids: str
    pubchem_cid: str
    chembl_ligand_id: str


def find_mmseqs() -> str | None:
    candidate = shutil.which("mmseqs")
    if candidate:
        return candidate
    homebrew_candidate = Path("/opt/homebrew/bin/mmseqs")
    if homebrew_candidate.exists():
        return str(homebrew_candidate)
    return None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def protein_id_from_sequence(sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]
    return f"prot_{digest}"


def parse_exact_nm(value: str) -> float | None:
    match = EXACT_NUMERIC_RE.match(value or "")
    if not match:
        return None
    parsed = float(match.group(1))
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def clean_sequence(sequence: str) -> str:
    return re.sub(r"\s+", "", sequence or "").upper()


def first_nonempty(*values: str) -> str:
    for value in values:
        value = (value or "").strip()
        if value:
            return value
    return ""


def progress(iterable, description: str, total: int | None = None, unit: str = "it"):
    return tqdm(iterable, desc=description, total=total, unit=unit)


def row_affinity(row: dict[str, str], preferred_types: list[str]) -> tuple[str, float] | None:
    found = []
    for affinity_type in preferred_types:
        value = parse_exact_nm(row.get(AFFINITY_COLUMNS[affinity_type], ""))
        if value is not None:
            found.append((affinity_type, value))
    if len(found) != 1:
        return None
    return found[0]


def stream_clean_rows(
    raw_tsv: str | Path,
    preferred_types: list[str],
    single_chain_only: bool,
    min_protein_length: int,
    max_protein_length: int | None,
    limit_rows: int | None,
) -> tuple[list[CleanRow], dict[str, object]]:
    required = [
        "BindingDB Reactant_set_id",
        "Ligand SMILES",
        "BindingDB MonomerID",
        "Target Name",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "BindingDB Target Chain Sequence 1",
        "UniProt (SwissProt) Primary ID of Target Chain 1",
        "UniProt (TrEMBL) Primary ID of Target Chain 1",
        "PDB ID(s) for Ligand-Target Complex",
        "PubChem CID",
        "ChEMBL ID of Ligand",
    ] + [AFFINITY_COLUMNS[name] for name in preferred_types]

    stats: dict[str, object] = {
        "raw_tsv": str(raw_tsv),
        "preferred_affinity_types": preferred_types,
        "single_chain_only": single_chain_only,
        "min_protein_length": min_protein_length,
        "max_protein_length": max_protein_length,
        "raw_rows_seen": 0,
        "missing_required_columns": [],
        "kept_rows_before_aggregation": 0,
        "skipped_missing_smiles": 0,
        "skipped_bad_chain_count": 0,
        "skipped_missing_sequence": 0,
        "skipped_bad_sequence": 0,
        "skipped_length": 0,
        "skipped_affinity": 0,
    }
    rows: list[CleanRow] = []

    with Path(raw_tsv).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        stats["missing_required_columns"] = missing
        if missing:
            raise ValueError(f"Missing required BindingDB columns: {missing}")

        total = limit_rows
        for row in progress(reader, "stream BindingDB TSV", total=total, unit="rows"):
            stats["raw_rows_seen"] += 1
            if limit_rows is not None and stats["raw_rows_seen"] > limit_rows:
                break

            smiles = (row.get("Ligand SMILES") or "").strip()
            if not smiles:
                stats["skipped_missing_smiles"] += 1
                continue

            chain_text = (row.get("Number of Protein Chains in Target (>1 implies a multichain complex)") or "").strip()
            try:
                num_chains = int(float(chain_text)) if chain_text else 0
            except ValueError:
                num_chains = 0
            if single_chain_only and num_chains != 1:
                stats["skipped_bad_chain_count"] += 1
                continue

            sequence = clean_sequence(row.get("BindingDB Target Chain Sequence 1", ""))
            if not sequence:
                stats["skipped_missing_sequence"] += 1
                continue
            if not VALID_AA_RE.match(sequence):
                stats["skipped_bad_sequence"] += 1
                continue
            if len(sequence) < min_protein_length or (max_protein_length is not None and len(sequence) > max_protein_length):
                stats["skipped_length"] += 1
                continue

            affinity = row_affinity(row, preferred_types)
            if affinity is None:
                stats["skipped_affinity"] += 1
                continue
            affinity_type, affinity_nM = affinity
            p_affinity = 9.0 - math.log10(affinity_nM)

            rows.append(
                CleanRow(
                    reactant_set_id=(row.get("BindingDB Reactant_set_id") or "").strip(),
                    ligand_smiles=smiles,
                    monomer_id=(row.get("BindingDB MonomerID") or "").strip(),
                    target_name=(row.get("Target Name") or "").strip(),
                    protein_id=protein_id_from_sequence(sequence),
                    protein_sequence=sequence,
                    protein_length=len(sequence),
                    uniprot_id=first_nonempty(
                        row.get("UniProt (SwissProt) Primary ID of Target Chain 1", ""),
                        row.get("UniProt (TrEMBL) Primary ID of Target Chain 1", ""),
                    ),
                    affinity_type=affinity_type,
                    affinity_nM=affinity_nM,
                    p_affinity=p_affinity,
                    num_chains=num_chains,
                    pdb_ids=(row.get("PDB ID(s) for Ligand-Target Complex") or "").strip(),
                    pubchem_cid=(row.get("PubChem CID") or "").strip(),
                    chembl_ligand_id=(row.get("ChEMBL ID of Ligand") or "").strip(),
                )
            )
            stats["kept_rows_before_aggregation"] += 1

    return rows, stats


def aggregate_rows(rows: list[CleanRow]) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[CleanRow]] = defaultdict(list)
    for row in progress(rows, "group duplicate measurements", unit="rows"):
        grouped[(row.protein_id, row.ligand_smiles, row.affinity_type)].append(row)

    aggregated = []
    replicate_counts = []
    for (_, _, _), group in progress(grouped.items(), "aggregate duplicate groups", total=len(grouped), unit="groups"):
        values = [item.affinity_nM for item in group]
        p_values = [item.p_affinity for item in group]
        first = group[0]
        replicate_counts.append(len(group))
        aggregated.append(
            {
                "sample_id": f"bd_{len(aggregated):08d}",
                "bindingdb_reactant_set_ids": ";".join(item.reactant_set_id for item in group if item.reactant_set_id),
                "ligand_smiles": first.ligand_smiles,
                "monomer_id": first.monomer_id,
                "target_name": first.target_name,
                "protein_id": first.protein_id,
                "protein_sequence": first.protein_sequence,
                "protein_length": first.protein_length,
                "uniprot_id": first.uniprot_id,
                "affinity_type": first.affinity_type,
                "affinity_nM_median": median(values),
                "p_affinity_median": median(p_values),
                "replicate_count": len(group),
                "num_chains": first.num_chains,
                "pdb_ids": first.pdb_ids,
                "pubchem_cid": first.pubchem_cid,
                "chembl_ligand_id": first.chembl_ligand_id,
            }
        )

    stats = {
        "aggregated_rows": len(aggregated),
        "duplicate_groups": sum(1 for count in replicate_counts if count > 1),
        "max_replicates_per_group": max(replicate_counts) if replicate_counts else 0,
    }
    return aggregated, stats


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_fasta(path: str | Path, rows: list[dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['protein_id']}\n")
            sequence = row["protein_sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def command_clean(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    clean_csv = output_dir / "bindingdb_clean.csv"
    proteins_csv = output_dir / "unique_proteins.csv"
    proteins_fasta = output_dir / "unique_proteins.fasta"
    report_path = output_dir / "clean_report.json"

    rows, clean_stats = stream_clean_rows(
        raw_tsv=args.raw_tsv,
        preferred_types=args.affinity_types,
        single_chain_only=not args.allow_multichain,
        min_protein_length=args.min_protein_length,
        max_protein_length=args.max_protein_length,
        limit_rows=args.limit_rows,
    )
    aggregated, aggregate_stats = aggregate_rows(rows)
    if not aggregated:
        raise RuntimeError("No rows survived cleaning. Relax filters or inspect the raw TSV.")

    clean_fields = list(aggregated[0].keys())
    write_csv(clean_csv, aggregated, clean_fields)

    protein_map = {}
    for row in aggregated:
        protein_map[row["protein_id"]] = {
            "protein_id": row["protein_id"],
            "protein_sequence": row["protein_sequence"],
            "protein_length": row["protein_length"],
            "target_name": row["target_name"],
            "uniprot_id": row["uniprot_id"],
        }
    proteins = sorted(protein_map.values(), key=lambda item: item["protein_id"])
    write_csv(proteins_csv, proteins, ["protein_id", "protein_sequence", "protein_length", "target_name", "uniprot_id"])
    write_fasta(proteins_fasta, proteins)

    report = {
        **clean_stats,
        **aggregate_stats,
        "clean_csv": str(clean_csv),
        "unique_proteins_csv": str(proteins_csv),
        "unique_proteins_fasta": str(proteins_fasta),
        "unique_proteins": len(proteins),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_mmseqs_all_vs_all(fasta: Path, output_dir: Path, min_seq_id: float, coverage: float) -> Path:
    mmseqs = find_mmseqs()
    if mmseqs is None:
        raise RuntimeError("MMseqs2 not found. Install mmseqs or ensure /opt/homebrew/bin/mmseqs exists.")
    hits_path = output_dir / "mmseqs_all_vs_all.tsv"
    tmp_dir = output_dir / "mmseqs_all_vs_all_tmp"
    command = [
        mmseqs,
        "easy-search",
        str(fasta),
        str(fasta),
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


def components_from_hits(protein_ids: list[str], hits_path: Path, min_seq_id: float, coverage: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    uf = UnionFind(protein_ids)
    protein_set = set(protein_ids)
    min_percent = min_seq_id * 100.0
    stats = {
        "hits_path": str(hits_path),
        "mmseqs_rows_seen": 0,
        "similar_edges_used": 0,
        "self_hits_seen": 0,
        "rows_skipped_outside_protein_set": 0,
    }
    if hits_path.exists():
        with hits_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in progress(handle, "parse MMseqs all-vs-all hits", unit="hits"):
                stats["mmseqs_rows_seen"] += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                query, target = parts[0], parts[1]
                if query not in protein_set or target not in protein_set:
                    stats["rows_skipped_outside_protein_set"] += 1
                    continue
                if query == target:
                    stats["self_hits_seen"] += 1
                    continue
                pident = float(parts[2])
                qcov = float(parts[4])
                tcov = float(parts[5])
                if pident >= min_percent and qcov >= coverage and tcov >= coverage:
                    uf.union(query, target)
                    stats["similar_edges_used"] += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for protein_id in progress(protein_ids, "build protein components", total=len(protein_ids), unit="proteins"):
        groups[uf.find(protein_id)].append(protein_id)

    components = []
    for index, members in enumerate(sorted(groups.values(), key=lambda values: (-len(values), values[0]))):
        component_id = f"component_{index:06d}"
        for member in members:
            components.append({"protein_id": member, "component_id": component_id, "component_size": len(members)})

    stats["component_count"] = len(groups)
    stats["largest_component_size"] = max((len(values) for values in groups.values()), default=0)
    return components, stats


def command_cluster(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    proteins_csv = Path(args.proteins_csv or output_dir / "unique_proteins.csv")
    fasta = Path(args.proteins_fasta or output_dir / "unique_proteins.fasta")
    hits_path = Path(args.hits_path) if args.hits_path else run_mmseqs_all_vs_all(fasta, output_dir, args.min_seq_id, args.coverage)
    proteins = read_csv_rows(proteins_csv)
    components, stats = components_from_hits([row["protein_id"] for row in proteins], hits_path, args.min_seq_id, args.coverage)

    components_csv = output_dir / "protein_components.csv"
    write_csv(components_csv, components, ["protein_id", "component_id", "component_size"])
    report = {
        **stats,
        "min_seq_id": args.min_seq_id,
        "coverage": args.coverage,
        "proteins_csv": str(proteins_csv),
        "proteins_fasta": str(fasta),
        "protein_components_csv": str(components_csv),
    }
    write_json(output_dir / "cluster_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def split_components(
    clean_rows: list[dict[str, str]],
    component_rows: list[dict[str, str]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, object]], dict[str, object]]:
    ratio_sum = train_ratio + valid_ratio + test_ratio
    ratios = {"train": train_ratio / ratio_sum, "valid": valid_ratio / ratio_sum, "test": test_ratio / ratio_sum}
    component_by_protein = {row["protein_id"]: row["component_id"] for row in component_rows}

    component_stats: dict[str, dict[str, object]] = {}
    for row in progress(clean_rows, "summarize samples by component", total=len(clean_rows), unit="samples"):
        component_id = component_by_protein.get(row["protein_id"])
        if component_id is None:
            raise RuntimeError(f"Missing component for protein_id={row['protein_id']}")
        item = component_stats.setdefault(component_id, {"component_id": component_id, "sample_count": 0, "protein_ids": set()})
        item["sample_count"] += 1
        item["protein_ids"].add(row["protein_id"])

    components = list(component_stats.values())
    rng = random.Random(seed)
    rng.shuffle(components)
    components.sort(key=lambda item: (-int(item["sample_count"]), item["component_id"]))

    total_samples = len(clean_rows)
    target = {split: total_samples * ratio for split, ratio in ratios.items()}
    counts = {split: 0 for split in SPLITS}
    component_split = {}
    for component in progress(components, "assign components to splits", total=len(components), unit="components"):
        chosen = min(SPLITS, key=lambda split: counts[split] / target[split] if target[split] else float("inf"))
        component_split[component["component_id"]] = chosen
        counts[chosen] += int(component["sample_count"])

    split_rows = []
    for row in progress(clean_rows, "write sample split labels", total=len(clean_rows), unit="samples"):
        output = dict(row)
        output["component_id"] = component_by_protein[row["protein_id"]]
        output["split"] = component_split[output["component_id"]]
        split_rows.append(output)

    component_output = []
    for component in progress(components, "write component split labels", total=len(components), unit="components"):
        component_id = component["component_id"]
        protein_ids = sorted(component["protein_ids"])
        component_output.append(
            {
                "component_id": component_id,
                "split": component_split[component_id],
                "sample_count": component["sample_count"],
                "protein_count": len(protein_ids),
                "protein_ids": ";".join(protein_ids),
            }
        )

    stats = {
        "seed": seed,
        "ratios": ratios,
        "sample_counts": counts,
        "component_counts": {split: sum(1 for row in component_output if row["split"] == split) for split in SPLITS},
        "protein_counts": {
            split: len({row["protein_id"] for row in split_rows if row["split"] == split}) for split in SPLITS
        },
    }
    return split_rows, component_output, stats


def command_split(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    clean_csv = Path(args.clean_csv or output_dir / "bindingdb_clean.csv")
    components_csv = Path(args.components_csv or output_dir / "protein_components.csv")
    clean_rows = read_csv_rows(clean_csv)
    component_rows = read_csv_rows(components_csv)
    split_rows, component_output, stats = split_components(
        clean_rows=clean_rows,
        component_rows=component_rows,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_csv = output_dir / "bindingdb_clean_with_split.csv"
    write_csv(split_csv, split_rows, list(split_rows[0].keys()))
    component_split_csv = output_dir / "component_splits.csv"
    write_csv(component_split_csv, component_output, ["component_id", "split", "sample_count", "protein_count", "protein_ids"])
    for split in progress(SPLITS, "write split FASTA files", total=len(SPLITS), unit="splits"):
        rows = [row for row in split_rows if row["split"] == split]
        write_csv(output_dir / f"{split}.csv", rows, list(split_rows[0].keys()))

    report = {
        **stats,
        "clean_csv": str(clean_csv),
        "components_csv": str(components_csv),
        "split_csv": str(split_csv),
        "component_split_csv": str(component_split_csv),
    }
    write_json(output_dir / "split_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def run_cross_split_mmseqs(split_csv: Path, output_dir: Path, min_seq_id: float, coverage: float) -> Path:
    mmseqs = find_mmseqs()
    if mmseqs is None:
        raise RuntimeError("MMseqs2 not found. Install mmseqs or ensure /opt/homebrew/bin/mmseqs exists.")

    rows = read_csv_rows(split_csv)
    fasta_by_split = {}
    for split in SPLITS:
        unique = {}
        for row in rows:
            if row["split"] == split:
                unique[row["protein_id"]] = {
                    "protein_id": f"{split}|{row['protein_id']}",
                    "protein_sequence": row["protein_sequence"],
                }
        fasta = output_dir / f"{split}_proteins.fasta"
        write_fasta(fasta, list(unique.values()))
        fasta_by_split[split] = fasta

    all_hits = output_dir / "cross_split_mmseqs_hits.tsv"
    with all_hits.open("w", encoding="utf-8") as combined:
        pairs = (("train", "valid"), ("train", "test"), ("valid", "test"))
        for left, right in progress(pairs, "run cross-split MMseqs", total=len(pairs), unit="pairs"):
            hits = output_dir / f"mmseqs_{left}_vs_{right}.tsv"
            tmp = output_dir / f"mmseqs_{left}_vs_{right}_tmp"
            command = [
                mmseqs,
                "easy-search",
                str(fasta_by_split[left]),
                str(fasta_by_split[right]),
                str(hits),
                str(tmp),
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
            if hits.exists():
                with hits.open("r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        combined.write(line)
    return all_hits


def parse_cross_split_hits(hits_path: Path, min_seq_id: float, coverage: float) -> tuple[list[dict[str, object]], dict[str, object]]:
    min_percent = min_seq_id * 100.0
    rows = []
    if hits_path.exists():
        with hits_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in progress(handle, "parse cross-split hits", unit="hits"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                query, target = parts[0], parts[1]
                query_split, query_protein = query.split("|", 1)
                target_split, target_protein = target.split("|", 1)
                pident = float(parts[2])
                qcov = float(parts[4])
                tcov = float(parts[5])
                if query_split == target_split:
                    continue
                if pident >= min_percent and qcov >= coverage and tcov >= coverage:
                    rows.append(
                        {
                            "query": query,
                            "target": target,
                            "query_split": query_split,
                            "target_split": target_split,
                            "query_protein_id": query_protein,
                            "target_protein_id": target_protein,
                            "pident": pident,
                            "qcov": qcov,
                            "tcov": tcov,
                        }
                    )
    stats = {
        "hit_count": len(rows),
        "leakage_detected": bool(rows),
        "min_seq_id": min_seq_id,
        "coverage": coverage,
    }
    return rows, stats


def command_validate(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    split_csv_arg = getattr(args, "split_csv", None)
    split_csv = Path(split_csv_arg or output_dir / "bindingdb_clean_with_split.csv")
    hits_path = Path(args.hits_path) if args.hits_path else run_cross_split_mmseqs(
        split_csv=split_csv,
        output_dir=output_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
    )
    hits, stats = parse_cross_split_hits(hits_path, args.min_seq_id, args.coverage)
    hits_csv = output_dir / "cross_split_leakage_hits.csv"
    write_csv(
        hits_csv,
        hits,
        ["query", "target", "query_split", "target_split", "query_protein_id", "target_protein_id", "pident", "qcov", "tcov"],
    )
    report = {
        **stats,
        "split_csv": str(split_csv),
        "hits_path": str(hits_path),
        "hits_csv": str(hits_csv),
    }
    write_json(output_dir / "validation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def command_run_all(args: argparse.Namespace) -> None:
    command_clean(args)
    command_cluster(args)
    command_split(args)
    command_validate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    clean = subparsers.add_parser("clean", help="Stream BindingDB TSV and write compact sequence-affinity files.")
    add_common(clean)
    clean.add_argument("--raw-tsv", type=Path, default=DEFAULT_RAW_TSV)
    clean.add_argument("--affinity-types", nargs="+", choices=sorted(AFFINITY_COLUMNS), default=["Ki"])
    clean.add_argument("--allow-multichain", action="store_true")
    clean.add_argument("--min-protein-length", type=int, default=30)
    clean.add_argument("--max-protein-length", type=int)
    clean.add_argument("--limit-rows", type=int, help="Debug option: stop after this many raw TSV rows.")
    clean.set_defaults(func=command_clean)

    cluster = subparsers.add_parser("cluster", help="Run MMseqs all-vs-all and build protein similarity components.")
    add_common(cluster)
    cluster.add_argument("--proteins-csv", type=Path)
    cluster.add_argument("--proteins-fasta", type=Path)
    cluster.add_argument("--hits-path", type=Path, help="Reuse an existing MMseqs tabular all-vs-all output.")
    cluster.add_argument("--min-seq-id", type=float, default=0.4)
    cluster.add_argument("--coverage", type=float, default=0.8)
    cluster.set_defaults(func=command_cluster)

    split = subparsers.add_parser("split", help="Assign whole protein components to train/valid/test.")
    add_common(split)
    split.add_argument("--clean-csv", type=Path)
    split.add_argument("--components-csv", type=Path)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--valid-ratio", type=float, default=0.1)
    split.add_argument("--test-ratio", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=2026)
    split.set_defaults(func=command_split)

    validate = subparsers.add_parser("validate", help="Validate that cross-split protein similarity hits are absent.")
    add_common(validate)
    validate.add_argument("--split-csv", type=Path)
    validate.add_argument("--hits-path", type=Path, help="Reuse an existing cross-split MMseqs output.")
    validate.add_argument("--min-seq-id", type=float, default=0.4)
    validate.add_argument("--coverage", type=float, default=0.8)
    validate.set_defaults(func=command_validate)

    run_all = subparsers.add_parser("run-all", help="Run clean, cluster, split, and validate in sequence.")
    add_common(run_all)
    run_all.add_argument("--raw-tsv", type=Path, default=DEFAULT_RAW_TSV)
    run_all.add_argument("--affinity-types", nargs="+", choices=sorted(AFFINITY_COLUMNS), default=["Ki"])
    run_all.add_argument("--allow-multichain", action="store_true")
    run_all.add_argument("--min-protein-length", type=int, default=30)
    run_all.add_argument("--max-protein-length", type=int)
    run_all.add_argument("--limit-rows", type=int)
    run_all.add_argument("--proteins-csv", type=Path)
    run_all.add_argument("--proteins-fasta", type=Path)
    run_all.add_argument("--hits-path", type=Path)
    run_all.add_argument("--min-seq-id", type=float, default=0.4)
    run_all.add_argument("--coverage", type=float, default=0.8)
    run_all.add_argument("--clean-csv", type=Path)
    run_all.add_argument("--components-csv", type=Path)
    run_all.add_argument("--train-ratio", type=float, default=0.8)
    run_all.add_argument("--valid-ratio", type=float, default=0.1)
    run_all.add_argument("--test-ratio", type=float, default=0.1)
    run_all.add_argument("--seed", type=int, default=2026)
    run_all.set_defaults(func=command_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
