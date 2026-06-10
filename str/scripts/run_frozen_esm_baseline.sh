#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
MANIFEST="${MANIFEST:-str/manifest/esm_affinity_trainable_manifest.csv}"
ESM_CACHE_DIR="${ESM_CACHE_DIR:-str/manifest/cache/esm_embeddings}"
LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR:-str/manifest/cache/ligand_graphs}"
OUTPUT_DIR="${OUTPUT_DIR:-str/manifest/outputs/baseline_frozen_esm}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALID_SPLIT="${VALID_SPLIT:-valid}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TRAIN_LIMIT="${TRAIN_LIMIT:--1}"
VALID_LIMIT="${VALID_LIMIT:--1}"
TEST_LIMIT="${TEST_LIMIT:--1}"
SAMPLE_MODE="${SAMPLE_MODE:-head}"
SEED="${SEED:-42}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS="${LOSS:-mse}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DEVICE="${DEVICE:-auto}"
PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT:-128}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Frozen ESM baseline training"
echo "ROOT_DIR=${ROOT_DIR}"
echo "MANIFEST=${MANIFEST}"
echo "ESM_CACHE_DIR=${ESM_CACHE_DIR}"
echo "LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "VALID_LIMIT=${VALID_LIMIT}"
echo "TEST_LIMIT=${TEST_LIMIT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "HIDDEN_DIM=${HIDDEN_DIM}"
echo "DROPOUT=${DROPOUT}"
echo "LR=${LR}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
echo "LOSS=${LOSS}"
echo "DEVICE=${DEVICE}"
echo "PRETRAIN_CHECK_LIMIT=${PRETRAIN_CHECK_LIMIT}"
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

require_path "${MANIFEST}" "trainable manifest"
require_path "${ESM_CACHE_DIR}" "ESM cache directory"
require_path "${LIGAND_CACHE_DIR}" "ligand graph cache directory"

run_stage "[1/2] Validate training batch before baseline" \
  python str/scripts/data/build_training_batch.py \
    --manifest "${MANIFEST}" \
    --esm-cache-dir "${ESM_CACHE_DIR}" \
    --ligand-cache-dir "${LIGAND_CACHE_DIR}" \
    --report-json "${OUTPUT_DIR}/pretrain_batch_check.json" \
    --split "${TRAIN_SPLIT}" \
    --limit "${PRETRAIN_CHECK_LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --sample-mode "${SAMPLE_MODE}" \
    --seed "${SEED}"

run_stage "[2/2] Train frozen ESM baseline" \
  python str/scripts/train/train_frozen_esm_baseline.py \
    --manifest "${MANIFEST}" \
    --esm-cache-dir "${ESM_CACHE_DIR}" \
    --ligand-cache-dir "${LIGAND_CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --train-split "${TRAIN_SPLIT}" \
    --valid-split "${VALID_SPLIT}" \
    --test-split "${TEST_SPLIT}" \
    --train-limit "${TRAIN_LIMIT}" \
    --valid-limit "${VALID_LIMIT}" \
    --test-limit "${TEST_LIMIT}" \
    --sample-mode "${SAMPLE_MODE}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --loss "${LOSS}" \
    --grad-clip "${GRAD_CLIP}" \
    --device "${DEVICE}"

echo "Baseline completed. Outputs are in: ${OUTPUT_DIR}"
