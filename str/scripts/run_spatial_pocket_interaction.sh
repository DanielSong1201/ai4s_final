#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-str/manifest/esm_affinity_trainable_manifest.csv}"
ESM_CACHE_DIR="${ESM_CACHE_DIR:-str/manifest/cache/esm_embeddings}"
LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR:-str/manifest/cache/ligand_graphs}"
POCKET_CACHE_DIR="${POCKET_CACHE_DIR:-str/manifest/cache/pocket_features}"
OUTPUT_DIR="${OUTPUT_DIR:-str/manifest/outputs/spatial_pocket_interaction_frozen_esm}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VALID_SPLIT="${VALID_SPLIT:-valid}"
TEST_SPLIT="${TEST_SPLIT:-test}"
TRAIN_LIMIT="${TRAIN_LIMIT:--1}"
VALID_LIMIT="${VALID_LIMIT:--1}"
TEST_LIMIT="${TEST_LIMIT:--1}"
SAMPLE_MODE="${SAMPLE_MODE:-head}"
SEED="${SEED:-42}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
POCKET_LAYERS="${POCKET_LAYERS:-2}"
LIGAND_LAYERS="${LIGAND_LAYERS:-4}"
ATTENTION_HEADS="${ATTENTION_HEADS:-6}"
FFN_MULTIPLIER="${FFN_MULTIPLIER:-4}"
POOLING="${POOLING:-attention}"
RBF_BINS="${RBF_BINS:-32}"
RBF_MAX_DISTANCE="${RBF_MAX_DISTANCE:-20.0}"
FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS="${LOSS:-mse}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
DEVICE="${DEVICE:-auto}"
FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE:-1}"
PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT:-128}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

echo "Frozen ESM + spatial pocket-ligand interaction training"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MANIFEST=${MANIFEST}"
echo "ESM_CACHE_DIR=${ESM_CACHE_DIR}"
echo "LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR}"
echo "POCKET_CACHE_DIR=${POCKET_CACHE_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "VALID_LIMIT=${VALID_LIMIT}"
echo "TEST_LIMIT=${TEST_LIMIT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "HIDDEN_DIM=${HIDDEN_DIM}"
echo "POCKET_LAYERS=${POCKET_LAYERS}"
echo "LIGAND_LAYERS=${LIGAND_LAYERS}"
echo "ATTENTION_HEADS=${ATTENTION_HEADS}"
echo "POOLING=${POOLING}"
echo "RBF_BINS=${RBF_BINS}"
echo "RBF_MAX_DISTANCE=${RBF_MAX_DISTANCE}"
echo "LR=${LR}"
echo "LOSS=${LOSS}"
echo "DEVICE=${DEVICE}"
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
require_path "${POCKET_CACHE_DIR}" "pocket feature cache directory"

run_stage "[1/2] Validate ESM + ligand batch before spatial pocket interaction" \
  "${PYTHON_BIN}" str/scripts/data/build_training_batch.py \
    --manifest "${MANIFEST}" \
    --esm-cache-dir "${ESM_CACHE_DIR}" \
    --ligand-cache-dir "${LIGAND_CACHE_DIR}" \
    --report-json "${OUTPUT_DIR}/pretrain_batch_check.json" \
    --split "${TRAIN_SPLIT}" \
    --limit "${PRETRAIN_CHECK_LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --sample-mode "${SAMPLE_MODE}" \
    --seed "${SEED}"

fallback_arg="--no-fallback-to-full-sequence"
if [[ "${FALLBACK_TO_FULL_SEQUENCE}" == "1" || "${FALLBACK_TO_FULL_SEQUENCE}" == "true" || "${FALLBACK_TO_FULL_SEQUENCE}" == "TRUE" ]]; then
  fallback_arg="--fallback-to-full-sequence"
fi

run_stage "[2/2] Train frozen ESM + spatial pocket-ligand interaction" \
  "${PYTHON_BIN}" str/scripts/train/train_spatial_pocket_interaction_frozen_esm.py \
    --manifest "${MANIFEST}" \
    --esm-cache-dir "${ESM_CACHE_DIR}" \
    --ligand-cache-dir "${LIGAND_CACHE_DIR}" \
    --pocket-cache-dir "${POCKET_CACHE_DIR}" \
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
    --pocket-layers "${POCKET_LAYERS}" \
    --ligand-layers "${LIGAND_LAYERS}" \
    --attention-heads "${ATTENTION_HEADS}" \
    --ffn-multiplier "${FFN_MULTIPLIER}" \
    --pooling "${POOLING}" \
    --rbf-bins "${RBF_BINS}" \
    --rbf-max-distance "${RBF_MAX_DISTANCE}" \
    --fusion-hidden-dim "${FUSION_HIDDEN_DIM}" \
    --dropout "${DROPOUT}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --loss "${LOSS}" \
    --grad-clip "${GRAD_CLIP}" \
    --device "${DEVICE}" \
    "${fallback_arg}"

echo "Spatial pocket-ligand interaction model completed. Outputs are in: ${OUTPUT_DIR}"
