"""Cache RDKit ligand graphs as PyTorch tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_CACHE_DIR = Path("str/manifest/cache/ligand_graphs")
DEFAULT_REPORT = Path("str/manifest/cache/ligand_graphs_report.json")


def load_rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    return Chem


def parse_ligand(row, Chem):
    sdf_path = project_path(Path(row.ligand_sdf_path))
    mol2_path = project_path(Path(row.ligand_mol2_path))

    if sdf_path.exists():
        try:
            supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
            if len(supplier) > 0 and supplier[0] is not None:
                return supplier[0], "sdf"
        except Exception:
            pass

    if mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
        if mol is not None:
            return mol, "mol2"

    return None, "failed"


def hybridization_id(atom) -> int:
    name = str(atom.GetHybridization())
    values = {
        "UNSPECIFIED": 0,
        "S": 1,
        "SP": 2,
        "SP2": 3,
        "SP3": 4,
        "SP3D": 5,
        "SP3D2": 6,
        "OTHER": 7,
    }
    return values.get(name, 0)


def chirality_id(atom) -> int:
    tag = str(atom.GetChiralTag())
    values = {
        "CHI_UNSPECIFIED": 0,
        "CHI_TETRAHEDRAL_CW": 1,
        "CHI_TETRAHEDRAL_CCW": 2,
        "CHI_OTHER": 3,
    }
    return values.get(tag, 0)


def bond_type_id(bond) -> int:
    name = str(bond.GetBondType())
    values = {
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "AROMATIC": 4,
        "DATIVE": 5,
    }
    return values.get(name, 0)


def stereo_id(bond) -> int:
    name = str(bond.GetStereo())
    values = {
        "STEREONONE": 0,
        "STEREOANY": 1,
        "STEREOZ": 2,
        "STEREOE": 3,
        "STEREOCIS": 4,
        "STEREOTRANS": 5,
    }
    return values.get(name, 0)


def atom_features(mol) -> torch.Tensor:
    rows = []
    for atom in mol.GetAtoms():
        rows.append(
            [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                hybridization_id(atom),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                atom.GetTotalNumHs(),
                chirality_id(atom),
                atom.GetMass() * 0.01,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def atom_coordinates(mol) -> torch.Tensor:
    num_atoms = mol.GetNumAtoms()
    if mol.GetNumConformers() == 0:
        return torch.zeros((num_atoms, 3), dtype=torch.float32)
    conf = mol.GetConformer()
    coords = []
    for idx in range(num_atoms):
        pos = conf.GetAtomPosition(idx)
        coords.append([pos.x, pos.y, pos.z])
    return torch.tensor(coords, dtype=torch.float32)


def bond_graph(mol) -> tuple[torch.Tensor, torch.Tensor]:
    edges = []
    features = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = [
            bond_type_id(bond),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            stereo_id(bond),
        ]
        edges.append([begin, end])
        features.append(feat)
        edges.append([end, begin])
        features.append(feat)

    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4), dtype=torch.float32)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(features, dtype=torch.float32)
    return edge_index, edge_attr


def graph_from_mol(row, mol, source: str) -> dict[str, object]:
    x = atom_features(mol)
    pos = atom_coordinates(mol)
    edge_index, edge_attr = bond_graph(mol)
    return {
        "pdb_id": row.pdb_id,
        "split": row.split,
        "source": source,
        "atom_features": x,
        "atom_coordinates": pos,
        "bond_index": edge_index,
        "bond_features": edge_attr,
        "num_atoms": int(mol.GetNumAtoms()),
        "num_bonds": int(mol.GetNumBonds()),
        "pAffinity": float(row.pAffinity),
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
    Chem = load_rdkit()
    manifest_path = project_path(args.manifest)
    cache_dir = project_path(args.cache_dir)
    report_json = project_path(args.report_json)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if args.limit >= 0:
        df = df.head(args.limit).copy()

    failures = []
    written = 0
    skipped_existing = 0
    atom_counts = []
    edge_counts = []
    source_counts: dict[str, int] = {}

    rows_iter = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Cache ligand graphs",
        unit="ligand",
    )
    for row in rows_iter:
        output_path = cache_dir / f"{row.pdb_id}.pt"
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            graph = torch.load(output_path, weights_only=False)
            atom_counts.append(int(graph["num_atoms"]))
            edge_counts.append(int(graph["bond_index"].shape[1]))
            source = str(graph.get("source", "cached"))
            source_counts[source] = source_counts.get(source, 0) + 1
            continue

        mol, source = parse_ligand(row, Chem)
        source_counts[source] = source_counts.get(source, 0) + 1
        if mol is None:
            failures.append({"pdb_id": row.pdb_id, "split": row.split})
            continue

        graph = graph_from_mol(row, mol, source)
        torch.save(graph, output_path)
        written += 1
        atom_counts.append(int(graph["num_atoms"]))
        edge_counts.append(int(graph["bond_index"].shape[1]))

    report = {
        "manifest": display_path(manifest_path),
        "cache_dir": display_path(cache_dir),
        "input_rows": int(len(df)),
        "written": written,
        "skipped_existing": skipped_existing,
        "failure_count": len(failures),
        "failures": failures[:20],
        "source_counts": source_counts,
        "num_atoms_min": int(min(atom_counts)) if atom_counts else None,
        "num_atoms_max": int(max(atom_counts)) if atom_counts else None,
        "num_atoms_mean": float(sum(atom_counts) / len(atom_counts)) if atom_counts else None,
        "num_directed_edges_min": int(min(edge_counts)) if edge_counts else None,
        "num_directed_edges_max": int(max(edge_counts)) if edge_counts else None,
        "num_directed_edges_mean": float(sum(edge_counts) / len(edge_counts)) if edge_counts else None,
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Cached ligand graphs in: {display_path(cache_dir)}")
    print(f"Wrote report: {display_path(report_json)}")
    print(f"Rows: {len(df)}, written: {written}, skipped_existing: {skipped_existing}, failures: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
