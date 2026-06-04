"""Reusable raw PDBbind data inspection utilities.

The functions in this module intentionally read from the original PDBbind index
and structure tree, not from files under data/processed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


AFFINITY_PATTERN = re.compile(
    r"^(?P<kind>Kd|Ki|IC50)(?P<relation><=|>=|<|>|=)"
    r"(?P<value>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(?P<unit>[A-Za-z]+)$"
)

UNIT_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

EXPECTED_STRUCTURE_SUFFIXES = {
    "protein_path": "_protein.pdb",
    "pocket_path": "_pocket.pdb",
    "ligand_sdf_path": "_ligand.sdf",
    "ligand_mol2_path": "_ligand.mol2",
}


def parse_affinity(raw_affinity: str) -> dict[str, object]:
    """Parse a PDBbind affinity token such as Kd=49uM into numeric fields."""

    match = AFFINITY_PATTERN.match(str(raw_affinity).strip())
    if not match:
        return {
            "affinity_type": pd.NA,
            "affinity_relation": pd.NA,
            "affinity_value": math.nan,
            "affinity_unit": pd.NA,
            "affinity_molar": math.nan,
            "pAffinity": math.nan,
            "affinity_parse_ok": False,
        }

    value = float(match.group("value"))
    unit = match.group("unit")
    molar = value * UNIT_TO_MOLAR[unit] if unit in UNIT_TO_MOLAR else math.nan
    p_affinity = -math.log10(molar) if molar > 0 else math.nan

    return {
        "affinity_type": match.group("kind"),
        "affinity_relation": match.group("relation"),
        "affinity_value": value,
        "affinity_unit": unit,
        "affinity_molar": molar,
        "pAffinity": p_affinity,
        "affinity_parse_ok": unit in UNIT_TO_MOLAR,
    }


def read_raw_pdbbind_index(index_path: str | Path) -> pd.DataFrame:
    """Read the original INDEX_general_PL file into a tabular dataframe."""

    records = []
    index_path = Path(index_path)
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            left, _, comment = line.partition("//")
            fields = left.split()
            if len(fields) < 4:
                records.append(
                    {
                        "line_number": line_number,
                        "pdb_id": pd.NA,
                        "resolution": pd.NA,
                        "release_year": pd.NA,
                        "raw_affinity": pd.NA,
                        "reference": pd.NA,
                        "ligand_name": pd.NA,
                        "comment": comment.strip(),
                        "raw_line": raw_line.rstrip("\n"),
                        "index_parse_ok": False,
                    }
                )
                continue

            comment = comment.strip()
            reference = pd.NA
            ligand_name = pd.NA
            extra_comment = comment
            comment_match = re.match(r"(?P<ref>\S+)\s+\((?P<ligand>[^)]*)\)\s*(?P<extra>.*)$", comment)
            if comment_match:
                reference = comment_match.group("ref")
                ligand_name = comment_match.group("ligand")
                extra_comment = comment_match.group("extra").strip()

            record = {
                "line_number": line_number,
                "pdb_id": fields[0].lower(),
                "resolution": fields[1],
                "release_year": int(fields[2]) if fields[2].isdigit() else pd.NA,
                "raw_affinity": fields[3],
                "reference": reference,
                "ligand_name": ligand_name,
                "comment": extra_comment,
                "raw_line": raw_line.rstrip("\n"),
                "index_parse_ok": True,
            }
            record.update(parse_affinity(fields[3]))
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    frame["resolution_numeric"] = pd.to_numeric(frame["resolution"], errors="coerce")
    return frame


def iter_complex_dirs(complex_root: str | Path) -> Iterable[Path]:
    """Yield leaf complex directories under the original P-L structure tree."""

    complex_root = Path(complex_root)
    for bucket_dir in sorted(path for path in complex_root.iterdir() if path.is_dir()):
        for complex_dir in sorted(path for path in bucket_dir.iterdir() if path.is_dir()):
            yield complex_dir


def build_complex_directory_table(complex_root: str | Path) -> pd.DataFrame:
    """Create a table of raw complex directories and expected structure files."""

    rows = []
    for complex_dir in iter_complex_dirs(complex_root):
        pdb_id = complex_dir.name.lower()
        row = {
            "pdb_id": pdb_id,
            "complex_dir": str(complex_dir),
            "year_bucket": complex_dir.parent.name,
        }
        for column, suffix in EXPECTED_STRUCTURE_SUFFIXES.items():
            path = complex_dir / f"{pdb_id}{suffix}"
            row[column] = str(path)
            row[f"has_{column.replace('_path', '')}"] = path.exists()
            row[f"{column.replace('_path', '')}_bytes"] = path.stat().st_size if path.exists() else math.nan
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def load_raw_dataset(index_path: str | Path, complex_root: str | Path) -> pd.DataFrame:
    """Join raw index records with raw structure-file inventory."""

    index_df = read_raw_pdbbind_index(index_path)
    complex_df = build_complex_directory_table(complex_root)
    return index_df.merge(complex_df, on="pdb_id", how="left", validate="one_to_one")


def table_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataset-level row/column and duplicate statistics."""

    return pd.DataFrame(
        [
            ("rows", len(df)),
            ("columns", df.shape[1]),
            ("unique_pdb_id", df["pdb_id"].nunique(dropna=True)),
            ("duplicate_pdb_id_rows", int(df["pdb_id"].duplicated(keep=False).sum())),
            ("index_parse_failures", int((~df["index_parse_ok"].fillna(False)).sum())),
            ("affinity_parse_failures", int((~df["affinity_parse_ok"].fillna(False)).sum())),
            ("missing_complex_dir", int(df["complex_dir"].isna().sum())),
        ],
        columns=["metric", "value"],
    )


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return column dtype, missing count, and distinct-count information."""

    rows = []
    for column in df.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "missing_pct": round(float(df[column].isna().mean() * 100), 3),
                "unique": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def numeric_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric columns in the raw inspection dataframe."""

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().transpose().reset_index().rename(columns={"index": "column"})


def categorical_feature_summary(df: pd.DataFrame, columns: list[str] | None = None, top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Return top value counts for selected categorical columns."""

    if columns is None:
        columns = ["affinity_type", "affinity_relation", "affinity_unit", "year_bucket", "resolution"]
    summaries = {}
    for column in columns:
        if column in df.columns:
            summaries[column] = (
                df[column]
                .value_counts(dropna=False)
                .head(top_n)
                .rename_axis(column)
                .reset_index(name="count")
            )
    return summaries


def missing_structure_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing expected raw structure files."""

    rows = []
    for column in EXPECTED_STRUCTURE_SUFFIXES:
        flag_column = f"has_{column.replace('_path', '')}"
        if flag_column in df.columns:
            present = int(df[flag_column].fillna(False).sum())
            rows.append(
                {
                    "file_type": column.replace("_path", ""),
                    "present": present,
                    "missing": int(len(df) - present),
                    "present_pct": round(present / len(df) * 100, 3) if len(df) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def sample_missing_rows(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return examples with any missing expected structure file or complex dir."""

    missing_flags = ["complex_dir"] + [
        f"has_{column.replace('_path', '')}" for column in EXPECTED_STRUCTURE_SUFFIXES
    ]
    mask = df["complex_dir"].isna()
    for flag in missing_flags[1:]:
        if flag in df.columns:
            mask = mask | (~df[flag].fillna(False))
    return df.loc[mask, ["pdb_id", "raw_affinity", "complex_dir", *missing_flags[1:]]].head(n)


def raw_data_quality_report(index_path: str | Path, complex_root: str | Path) -> dict[str, object]:
    """Build all first-pass raw data quality tables used by the notebook."""

    df = load_raw_dataset(index_path=index_path, complex_root=complex_root)
    return {
        "raw_df": df,
        "table_overview": table_overview(df),
        "column_overview": column_overview(df),
        "numeric_summary": numeric_feature_summary(df),
        "missing_structure_summary": missing_structure_summary(df),
        "categorical_summaries": categorical_feature_summary(df),
        "missing_examples": sample_missing_rows(df),
    }
