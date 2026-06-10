#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
SPLIT_NAME="${SPLIT_NAME:-split}"
IID="${IID:-false}"

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

MANIFEST_DIR="${MANIFEST_DIR:-str/manifest}"
LIGAND_PARSE_LIMIT="${LIGAND_PARSE_LIMIT:--1}"

LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR:-${MANIFEST_DIR}/cache/ligand_graphs}"
LIGAND_CACHE_LIMIT="${LIGAND_CACHE_LIMIT:--1}"
LIGAND_OVERWRITE="${LIGAND_OVERWRITE:-0}"

ESM_CACHE_DIR="${ESM_CACHE_DIR:-${MANIFEST_DIR}/cache/esm_embeddings}"
ESM_MODEL_NAME="${ESM_MODEL_NAME:-facebook/esm2_t6_8M_UR50D}"
ESM_DEVICE="${ESM_DEVICE:-auto}"
ESM_LIMIT="${ESM_LIMIT:--1}"
ESM_CHUNK_BATCH_SIZE="${ESM_CHUNK_BATCH_SIZE:-4}"
ESM_MAX_RESIDUES_PER_CHUNK="${ESM_MAX_RESIDUES_PER_CHUNK:-1022}"
ESM_FLOAT16_OUTPUT="${ESM_FLOAT16_OUTPUT:-1}"
ESM_OVERWRITE="${ESM_OVERWRITE:-0}"
ESM_LOCAL_FILES_ONLY="${ESM_LOCAL_FILES_ONLY:-0}"

BATCH_SPLIT="${BATCH_SPLIT:-train}"
BATCH_LIMIT="${BATCH_LIMIT:-128}"
BATCH_SIZE="${BATCH_SIZE:-8}"

usage() {
  cat <<'EOF'
Usage: bash str/scripts/prepare_until_baseline.sh [options]

Runs the full pre-training pipeline:
  1. Split raw PDBbind structures
  2. Build manifest
  3. Validate manifest
  4. Build ligand graph cache
  5. Build ESM embedding cache
  6. Validate ESM + ligand training batch

It stops before baseline training.

Options:
  --split-name NAME           Name under data/processed/NAME. Also used to infer the split folder when not provided.
  --iid true|false            true: IID/random PDB-complex split; false: sequence-constrained PDB-complex split.
  --output-split-dir PATH     Output Interformer-format split directory.
  --manifest-dir PATH         Output manifest directory.
  --esm-model-name NAME       Hugging Face ESM model name.
  --esm-device DEVICE         auto, cpu, cuda, cuda:0, or mps.
  --esm-limit N               Limit ESM cache rows. Use -1 for all.
  --ligand-cache-limit N      Limit ligand graph cache rows. Use -1 for all.
  --batch-limit N             Rows to check when building final training batch.
  --batch-size N              Batch size for final training batch check.
  -h, --help                  Show this help message.

All options can also be edited in this script or passed as environment variables.
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
while [[ $# -gt 0 ]]; do
  case "$1" in
    --split-name)
      require_arg_value "$1" "${2:-}"
      SPLIT_NAME="$2"
      shift 2
      ;;
    --iid)
      require_arg_value "$1" "${2:-}"
      IID="$2"
      shift 2
      ;;
    --output-split-dir)
      require_arg_value "$1" "${2:-}"
      OUTPUT_SPLIT_DIR_ARG="$2"
      shift 2
      ;;
    --manifest-dir)
      require_arg_value "$1" "${2:-}"
      MANIFEST_DIR="$2"
      shift 2
      ;;
    --esm-model-name)
      require_arg_value "$1" "${2:-}"
      ESM_MODEL_NAME="$2"
      shift 2
      ;;
    --esm-device)
      require_arg_value "$1" "${2:-}"
      ESM_DEVICE="$2"
      shift 2
      ;;
    --esm-limit)
      require_arg_value "$1" "${2:-}"
      ESM_LIMIT="$2"
      shift 2
      ;;
    --ligand-cache-limit)
      require_arg_value "$1" "${2:-}"
      LIGAND_CACHE_LIMIT="$2"
      shift 2
      ;;
    --batch-limit)
      require_arg_value "$1" "${2:-}"
      BATCH_LIMIT="$2"
      shift 2
      ;;
    --batch-size)
      require_arg_value "$1" "${2:-}"
      BATCH_SIZE="$2"
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

if [[ ! "${SPLIT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid SPLIT_NAME: ${SPLIT_NAME}. Use a single directory name with letters, numbers, dot, underscore, or hyphen." >&2
  exit 2
fi

IID_LOWER="$(printf '%s' "${IID}" | tr '[:upper:]' '[:lower:]')"
if [[ "${IID_LOWER}" == "true" || "${IID_LOWER}" == "1" || "${IID_LOWER}" == "yes" ]]; then
  IID_MODE="true"
elif [[ "${IID_LOWER}" == "false" || "${IID_LOWER}" == "0" || "${IID_LOWER}" == "no" ]]; then
  IID_MODE="false"
else
  echo "Invalid IID value: ${IID}. Use true or false." >&2
  exit 2
fi

if [[ -n "${OUTPUT_SPLIT_DIR_ARG}" ]]; then
  OUTPUT_SPLIT_DIR="${OUTPUT_SPLIT_DIR_ARG}"
else
  OUTPUT_SPLIT_DIR="${OUTPUT_SPLIT_DIR:-str/splits/${SPLIT_NAME}}"
fi

SOURCE_CSV="data/processed/${SPLIT_NAME}/pdbbind_sequence_cluster_splits.csv"
MANIFEST_CSV="${MANIFEST_DIR}/esm_affinity_manifest.csv"
TRAINABLE_MANIFEST="${MANIFEST_DIR}/esm_affinity_trainable_manifest.csv"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

echo "Prepare until baseline pipeline"
echo "ROOT_DIR=${ROOT_DIR}"
echo "SPLIT_NAME=${SPLIT_NAME}"
echo "IID=${IID_MODE}"
echo "OUTPUT_SPLIT_DIR=${OUTPUT_SPLIT_DIR}"
echo "SOURCE_CSV=${SOURCE_CSV}"
echo "MANIFEST_DIR=${MANIFEST_DIR}"
echo "LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR}"
echo "ESM_CACHE_DIR=${ESM_CACHE_DIR}"
echo "ESM_MODEL_NAME=${ESM_MODEL_NAME}"
echo "ESM_DEVICE=${ESM_DEVICE}"
echo "ESM_LIMIT=${ESM_LIMIT}"
echo "BATCH_LIMIT=${BATCH_LIMIT}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo

CURRENT_STAGE="initialization"
trap 'echo "[FAIL] ${CURRENT_STAGE}" >&2' ERR

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

split_args=(
  --iid "${IID_MODE}"
  --split-name "${SPLIT_NAME}"
  --output-split-dir "${OUTPUT_SPLIT_DIR}"
  --index-path "${INDEX_PATH}"
  --complex-root "${COMPLEX_ROOT}"
  --source-split-dir "${SOURCE_SPLIT_DIR}"
  --seed "${SEED}"
  --min-seq-id "${MIN_SEQ_ID}"
  --coverage "${COVERAGE}"
  --min-length "${MIN_LENGTH}"
  --seeds "${SEEDS}"
  --universe "${UNIVERSE}"
)
if [[ -n "${ALL_VS_ALL_M8}" ]]; then
  split_args+=(--all-vs-all-m8 "${ALL_VS_ALL_M8}")
fi

run_stage "[1/3] Split raw PDBbind structures" \
  bash str/scripts/split_raw_pdbbind.sh "${split_args[@]}"

run_stage "[2/3] Build manifest from split" \
  bash str/scripts/build_manifest_from_split.sh \
    --split-name "${SPLIT_NAME}" \
    --split-dir "${OUTPUT_SPLIT_DIR}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --ligand-parse-limit "${LIGAND_PARSE_LIMIT}"

run_stage "[3/3] Validate manifest, cache ligand/ESM features, and validate training batch" \
  env \
    MANIFEST="${MANIFEST_CSV}" \
    TRAINABLE_MANIFEST="${TRAINABLE_MANIFEST}" \
    SPLIT_DIR="${OUTPUT_SPLIT_DIR}" \
    LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
    LIGAND_CACHE_LIMIT="${LIGAND_CACHE_LIMIT}" \
    LIGAND_OVERWRITE="${LIGAND_OVERWRITE}" \
    ESM_CACHE_DIR="${ESM_CACHE_DIR}" \
    ESM_MODEL_NAME="${ESM_MODEL_NAME}" \
    ESM_DEVICE="${ESM_DEVICE}" \
    ESM_LIMIT="${ESM_LIMIT}" \
    ESM_CHUNK_BATCH_SIZE="${ESM_CHUNK_BATCH_SIZE}" \
    ESM_MAX_RESIDUES_PER_CHUNK="${ESM_MAX_RESIDUES_PER_CHUNK}" \
    ESM_FLOAT16_OUTPUT="${ESM_FLOAT16_OUTPUT}" \
    ESM_OVERWRITE="${ESM_OVERWRITE}" \
    ESM_LOCAL_FILES_ONLY="${ESM_LOCAL_FILES_ONLY}" \
    BATCH_SPLIT="${BATCH_SPLIT}" \
    BATCH_LIMIT="${BATCH_LIMIT}" \
    BATCH_SIZE="${BATCH_SIZE}" \
  bash str/scripts/validate_after_manifest.sh

echo "Preparation completed. Baseline training can start now."
echo "Frozen ESM baseline:"
echo "MANIFEST=${TRAINABLE_MANIFEST} ESM_CACHE_DIR=${ESM_CACHE_DIR} LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR} bash str/scripts/run_frozen_esm_baseline.sh"
echo "Ligand GNN baseline:"
echo "MANIFEST=${TRAINABLE_MANIFEST} ESM_CACHE_DIR=${ESM_CACHE_DIR} LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR} bash str/scripts/run_ligand_gnn_baseline.sh"
