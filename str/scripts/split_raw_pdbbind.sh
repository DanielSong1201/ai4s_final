#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
IID="${IID:-false}"
SPLIT_NAME="${SPLIT_NAME:-split}"

INDEX_PATH="${INDEX_PATH:-data/raw/pdbbind2020/index/index/INDEX_general_PL.2020R1.lst}"
COMPLEX_ROOT="${COMPLEX_ROOT:-data/raw/pdbbind2020/complexes/P-L}"
SOURCE_SPLIT_DIR="${SOURCE_SPLIT_DIR:-split}"

SEED="${SEED:-42}"
MIN_SEQ_ID="${MIN_SEQ_ID:-0.4}"
COVERAGE="${COVERAGE:-0.8}"
MIN_LENGTH="${MIN_LENGTH:-30}"
SEEDS="${SEEDS:-128}"
UNIVERSE="${UNIVERSE:-all_raw}"
ALL_VS_ALL_M8="${ALL_VS_ALL_M8:-}"

usage() {
  cat <<'EOF'
Usage: bash str/scripts/split_raw_pdbbind.sh [options]

Options:
  --iid true|false            true: random PDB-complex split; false: sequence-similarity-constrained PDB-complex split.
                              Overrides the IID environment variable.
  --split-name NAME           Name of the processed split output under data/processed/NAME.
                              Must be a single directory name, not a path. Overrides SPLIT_NAME.
  --index-path PATH           PDBbind INDEX_general_PL file.
  --complex-root PATH         PDBbind P-L complex root.
  --source-split-dir PATH     Source Interformer split directory for ratios/subset files.
  --output-split-dir PATH     Output Interformer-format split directory.
  --output-dir PATH           Output processed split report/table directory.
  --seed N                    Random seed for IID split.
  --min-seq-id FLOAT          Sequence identity threshold for constrained split.
  --coverage FLOAT            Coverage threshold for constrained split and validation.
  --min-length N              Minimum protein-chain length used for sequence checks.
  --seeds N                   Number of component-assignment seeds for constrained split.
  --universe NAME             interformer_assigned or all_raw.
  --all-vs-all-m8 PATH        Optional precomputed MMseqs all-vs-all result.
  -h, --help                  Show this help message.

Defaults can also be edited in this script or provided as environment variables.
Argument priority: command-line option > environment variable > bash default.
EOF
}

require_arg_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "Missing value for ${option}" >&2
    usage >&2
    exit 2
  fi
}

OUTPUT_SPLIT_DIR_ARG=""
OUTPUT_DIR_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iid)
      require_arg_value "$1" "${2:-}"
      IID="$2"
      shift 2
      ;;
    --index-path)
      require_arg_value "$1" "${2:-}"
      INDEX_PATH="$2"
      shift 2
      ;;
    --split-name)
      require_arg_value "$1" "${2:-}"
      SPLIT_NAME="$2"
      shift 2
      ;;
    --complex-root)
      require_arg_value "$1" "${2:-}"
      COMPLEX_ROOT="$2"
      shift 2
      ;;
    --source-split-dir)
      require_arg_value "$1" "${2:-}"
      SOURCE_SPLIT_DIR="$2"
      shift 2
      ;;
    --output-split-dir)
      require_arg_value "$1" "${2:-}"
      OUTPUT_SPLIT_DIR_ARG="$2"
      shift 2
      ;;
    --output-dir)
      require_arg_value "$1" "${2:-}"
      OUTPUT_DIR_ARG="$2"
      shift 2
      ;;
    --seed)
      require_arg_value "$1" "${2:-}"
      SEED="$2"
      shift 2
      ;;
    --min-seq-id)
      require_arg_value "$1" "${2:-}"
      MIN_SEQ_ID="$2"
      shift 2
      ;;
    --coverage)
      require_arg_value "$1" "${2:-}"
      COVERAGE="$2"
      shift 2
      ;;
    --min-length)
      require_arg_value "$1" "${2:-}"
      MIN_LENGTH="$2"
      shift 2
      ;;
    --seeds)
      require_arg_value "$1" "${2:-}"
      SEEDS="$2"
      shift 2
      ;;
    --universe)
      require_arg_value "$1" "${2:-}"
      UNIVERSE="$2"
      shift 2
      ;;
    --all-vs-all-m8)
      require_arg_value "$1" "${2:-}"
      ALL_VS_ALL_M8="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IID_LOWER="$(printf '%s' "${IID}" | tr '[:upper:]' '[:lower:]')"
if [[ "${IID_LOWER}" == "true" || "${IID_LOWER}" == "1" || "${IID_LOWER}" == "yes" ]]; then
  IID_MODE="true"
elif [[ "${IID_LOWER}" == "false" || "${IID_LOWER}" == "0" || "${IID_LOWER}" == "no" ]]; then
  IID_MODE="false"
else
  echo "Invalid IID value: ${IID}. Use true or false." >&2
  exit 2
fi

if [[ "${IID_MODE}" == "true" ]]; then
  DEFAULT_OUTPUT_SPLIT_DIR="str/split_iid_all_raw"
else
  DEFAULT_OUTPUT_SPLIT_DIR="str/split_sequence_cluster_all_raw"
fi

if [[ ! "${SPLIT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid SPLIT_NAME: ${SPLIT_NAME}. Use a single directory name with letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi

DEFAULT_OUTPUT_DIR="data/processed/${SPLIT_NAME}"
OUTPUT_SPLIT_DIR="${OUTPUT_SPLIT_DIR_ARG:-${OUTPUT_SPLIT_DIR:-${DEFAULT_OUTPUT_SPLIT_DIR}}}"
OUTPUT_DIR="${OUTPUT_DIR_ARG:-${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}}"
ASSIGNED_CSV="${ASSIGNED_CSV:-${OUTPUT_DIR}/pdbbind_sequence_cluster_splits.csv}"
LEAKAGE_OUTPUT_DIR="${LEAKAGE_OUTPUT_DIR:-${OUTPUT_DIR}/sequence_leakage_check}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Raw PDBbind split pipeline"
echo "ROOT_DIR=${ROOT_DIR}"
echo "IID=${IID_MODE}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "INDEX_PATH=${INDEX_PATH}"
echo "COMPLEX_ROOT=${COMPLEX_ROOT}"
echo "SOURCE_SPLIT_DIR=${SOURCE_SPLIT_DIR}"
echo "OUTPUT_SPLIT_DIR=${OUTPUT_SPLIT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ASSIGNED_CSV=${ASSIGNED_CSV}"
echo "SEED=${SEED}"
echo "MIN_SEQ_ID=${MIN_SEQ_ID}"
echo "COVERAGE=${COVERAGE}"
echo "MIN_LENGTH=${MIN_LENGTH}"
echo "UNIVERSE=${UNIVERSE}"
echo

CURRENT_STAGE="initialization"
trap 'echo "[FAIL] ${CURRENT_STAGE}" >&2' ERR

require_path() {
  local path="$1"
  local message="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${message}: ${path}" >&2
    exit 1
  fi
}

run_stage() {
  local stage="$1"
  shift
  CURRENT_STAGE="${stage}"
  local start_time
  start_time="$(date +%s)"
  echo "[START] ${stage}"
  "$@"
  local end_time
  end_time="$(date +%s)"
  echo "[DONE] ${stage} ($((end_time - start_time))s)"
  echo
}

require_path "${INDEX_PATH}" "PDBbind index file"
require_path "${COMPLEX_ROOT}" "PDBbind complex root"
require_path "${SOURCE_SPLIT_DIR}" "source Interformer split directory for ratios/subset files"

if [[ "${IID_MODE}" == "true" ]]; then
  run_stage "[1/3] Create IID random structure-level split" \
    python str/scripts/data/create_iid_structure_split.py \
      --index-path "${INDEX_PATH}" \
      --complex-root "${COMPLEX_ROOT}" \
      --source-split-dir "${SOURCE_SPLIT_DIR}" \
      --output-split-dir "${OUTPUT_SPLIT_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --seed "${SEED}"
else
  sequence_args=(
    --index-path "${INDEX_PATH}"
    --complex-root "${COMPLEX_ROOT}"
    --source-split-dir "${SOURCE_SPLIT_DIR}"
    --output-split-dir "${OUTPUT_SPLIT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --min-seq-id "${MIN_SEQ_ID}"
    --coverage "${COVERAGE}"
    --min-length "${MIN_LENGTH}"
    --seeds "${SEEDS}"
    --universe "${UNIVERSE}"
  )
  if [[ -n "${ALL_VS_ALL_M8}" ]]; then
    sequence_args+=(--all-vs-all-m8 "${ALL_VS_ALL_M8}")
  fi
  run_stage "[1/3] Create sequence-cluster split with 40% similarity prior" \
    python scripts/create_sequence_cluster_split.py "${sequence_args[@]}"
fi

run_stage "[2/3] Verify Interformer-style split files" \
  python - "${OUTPUT_SPLIT_DIR}" "${ASSIGNED_CSV}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

split_dir = Path(sys.argv[1])
assigned_csv = Path(sys.argv[2])
required = {
    "train": "timesplit_no_lig_overlap_train",
    "valid": "timesplit_no_lig_overlap_val",
    "test": "timesplit_test",
}

report = {"split_dir": str(split_dir), "assigned_csv": str(assigned_csv), "files": {}}
all_ids = {}
for split, filename in required.items():
    path = split_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    ids = [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]
    all_ids[split] = set(ids)
    report["files"][filename] = {"rows": len(ids), "unique": len(set(ids))}

overlaps = {}
splits = list(required)
for i, left in enumerate(splits):
    for right in splits[i + 1:]:
        overlaps[f"{left}_vs_{right}"] = len(all_ids[left] & all_ids[right])
report["overlaps"] = overlaps

if any(count > 0 for count in overlaps.values()):
    raise ValueError(f"Primary split files overlap: {overlaps}")
if not assigned_csv.exists():
    raise FileNotFoundError(f"Missing assigned CSV: {assigned_csv}")
df = pd.read_csv(assigned_csv)
report["assigned_csv_rows"] = int(len(df))
report["assigned_split_counts"] = {
    split: int(df["split"].eq(split).sum())
    for split in required
}
print(json.dumps(report, indent=2, ensure_ascii=False))
PY

run_stage "[3/3] Check train-vs-valid/test sequence leakage" \
  python scripts/sequence_leakage_check.py \
    --split-csv "${ASSIGNED_CSV}" \
    --output-dir "${LEAKAGE_OUTPUT_DIR}" \
    --min-seq-id "${MIN_SEQ_ID}" \
    --coverage "${COVERAGE}" \
    --min-length "${MIN_LENGTH}"

echo "Split completed."
echo "Use this for manifest generation:"
echo "SOURCE_CSV=${ASSIGNED_CSV} SPLIT_DIR=${OUTPUT_SPLIT_DIR} PYTHONPATH=\$(pwd) bash str/scripts/build_manifest_from_split.sh"
