"""Create a trainable manifest by filtering rows that cannot produce ligand graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STR_ROOT = PROJECT_ROOT / "str"
for import_root in (STR_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.data.create_esm_manifest import display_path, project_path  # noqa: E402


DEFAULT_MANIFEST = Path("str/manifest/esm_affinity_manifest.csv")
DEFAULT_OUTPUT = Path("str/manifest/esm_affinity_trainable_manifest.csv")
DEFAULT_FAILURES = Path("str/manifest/ligand_parse_failures.csv")
DEFAULT_REPORT = Path("str/manifest/esm_affinity_trainable_manifest_report.json")


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
                return supplier[0], "sdf", ""
        except Exception as exc:
            sdf_error = str(exc)
        else:
            sdf_error = "SDF supplier returned no molecule"
    else:
        sdf_error = "SDF path does not exist"

    if mol2_path.exists():
        try:
            mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
            if mol is not None:
                return mol, "mol2", ""
        except Exception as exc:
            return None, "failed", f"SDF failed: {sdf_error}; MOL2 failed: {exc}"
        return None, "failed", f"SDF failed: {sdf_error}; MOL2 returned no molecule"

    return None, "failed", f"SDF failed: {sdf_error}; MOL2 path does not exist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures-csv", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Chem = load_rdkit()
    manifest_path = project_path(args.manifest)
    df = pd.read_csv(manifest_path)

    keep_rows = []
    failure_rows = []
    parse_source_counts: dict[str, int] = {}

    for row in df.itertuples(index=False):
        mol, source, error = parse_ligand(row, Chem)
        parse_source_counts[source] = parse_source_counts.get(source, 0) + 1
        if mol is None:
            failure_rows.append(
                {
                    "pdb_id": row.pdb_id,
                    "split": row.split,
                    "ligand_sdf_path": row.ligand_sdf_path,
                    "ligand_mol2_path": row.ligand_mol2_path,
                    "error": error,
                }
            )
            continue
        keep_rows.append(row.pdb_id)

    trainable = df[df["pdb_id"].isin(keep_rows)].copy()
    failures = pd.DataFrame(failure_rows)

    output_csv = project_path(args.output_csv)
    failures_csv = project_path(args.failures_csv)
    report_json = project_path(args.report_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    trainable.to_csv(output_csv, index=False)
    failures.to_csv(failures_csv, index=False)

    report = {
        "input_manifest": display_path(manifest_path),
        "output_manifest": display_path(output_csv),
        "failures_csv": display_path(failures_csv),
        "input_rows": int(len(df)),
        "trainable_rows": int(len(trainable)),
        "filtered_rows": int(len(failures)),
        "input_split_counts": {key: int(value) for key, value in df["split"].value_counts().to_dict().items()},
        "trainable_split_counts": {key: int(value) for key, value in trainable["split"].value_counts().to_dict().items()},
        "filtered_split_counts": {key: int(value) for key, value in failures["split"].value_counts().to_dict().items()} if not failures.empty else {},
        "parse_source_counts": parse_source_counts,
        "filtered_pdb_ids": failures["pdb_id"].tolist() if not failures.empty else [],
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote trainable manifest: {display_path(output_csv)} ({len(trainable)} rows)")
    print(f"Wrote ligand parse failures: {display_path(failures_csv)} ({len(failures)} rows)")
    print(f"Wrote report: {display_path(report_json)}")
    print("Trainable split counts:", report["trainable_split_counts"])


if __name__ == "__main__":
    main()
