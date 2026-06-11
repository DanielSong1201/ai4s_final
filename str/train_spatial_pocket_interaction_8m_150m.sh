#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Unable to locate python or python3. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

MANIFEST="${MANIFEST:-str/manifest/esm_affinity_trainable_manifest.csv}"
LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR:-str/manifest/cache/ligand_graphs}"
POCKET_CACHE_DIR="${POCKET_CACHE_DIR:-str/manifest/cache/pocket_features}"
OUTPUT_ROOT="${OUTPUT_ROOT:-str/output}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"
MODEL_NAME="${MODEL_NAME:-spatial_pocket_interaction_frozen_esm}"
RESET_OUTPUT="${RESET_OUTPUT:-1}"

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
DEVICE="${DEVICE:-auto}"
PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT:-128}"

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
FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE:-1}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

echo "Train 4.5 spatial pocket-ligand interaction for 8M and 150M"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MANIFEST=${MANIFEST}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "RESET_OUTPUT=${RESET_OUTPUT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
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

plot_history() {
  local scale_name="$1"
  local output_dir="$2"
  local history_csv="${output_dir}/history.csv"
  local plot_png="${PLOT_DIR}/${MODEL_NAME}_${scale_name}_history.png"
  local summary_json="${PLOT_DIR}/${MODEL_NAME}_${scale_name}_history_plot.json"
  require_path "${history_csv}" "history CSV for ${MODEL_NAME}_${scale_name}"
  run_stage "[plot] ${MODEL_NAME}_${scale_name}" \
    "${PYTHON_BIN}" str/scripts/plot/plot_history.py \
      --history-csv "${history_csv}" \
      --output-png "${plot_png}" \
      --title "${MODEL_NAME} (${scale_name})" \
      --summary-json "${summary_json}"
}

train_scale() {
  local scale_name="$1"
  local esm_cache_dir="str/manifest/cache/esm_embeddings_${scale_name}"
  local output_dir="${OUTPUT_ROOT}/${scale_name}/${MODEL_NAME}"

  require_path "${esm_cache_dir}" "${scale_name} ESM cache directory"
  if [[ "${RESET_OUTPUT}" == "1" || "${RESET_OUTPUT}" == "true" || "${RESET_OUTPUT}" == "TRUE" ]]; then
    rm -rf "${output_dir}"
  fi

  run_stage "[${scale_name}] Train ${MODEL_NAME}" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      MANIFEST="${MANIFEST}" \
      ESM_CACHE_DIR="${esm_cache_dir}" \
      LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
      POCKET_CACHE_DIR="${POCKET_CACHE_DIR}" \
      OUTPUT_DIR="${output_dir}" \
      TRAIN_SPLIT="${TRAIN_SPLIT}" VALID_SPLIT="${VALID_SPLIT}" TEST_SPLIT="${TEST_SPLIT}" \
      TRAIN_LIMIT="${TRAIN_LIMIT}" VALID_LIMIT="${VALID_LIMIT}" TEST_LIMIT="${TEST_LIMIT}" \
      SAMPLE_MODE="${SAMPLE_MODE}" SEED="${SEED}" \
      EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      HIDDEN_DIM="${HIDDEN_DIM}" POCKET_LAYERS="${POCKET_LAYERS}" LIGAND_LAYERS="${LIGAND_LAYERS}" \
      ATTENTION_HEADS="${ATTENTION_HEADS}" FFN_MULTIPLIER="${FFN_MULTIPLIER}" POOLING="${POOLING}" \
      RBF_BINS="${RBF_BINS}" RBF_MAX_DISTANCE="${RBF_MAX_DISTANCE}" \
      FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM}" DROPOUT="${DROPOUT}" LR="${LR}" \
      WEIGHT_DECAY="${WEIGHT_DECAY}" LOSS="${LOSS}" GRAD_CLIP="${GRAD_CLIP}" \
      DEVICE="${DEVICE}" FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE}" \
      PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT}" \
    bash str/scripts/run_spatial_pocket_interaction.sh

  plot_history "${scale_name}" "${output_dir}"
}

require_path "${MANIFEST}" "trainable manifest"
require_path "${LIGAND_CACHE_DIR}" "ligand graph cache directory"
require_path "${POCKET_CACHE_DIR}" "pocket feature cache directory"
mkdir -p "${OUTPUT_ROOT}" "${PLOT_DIR}"

train_scale "8m"
train_scale "150m"

echo "4.5 training completed."
echo "Outputs: ${OUTPUT_ROOT}/8m/${MODEL_NAME} and ${OUTPUT_ROOT}/150m/${MODEL_NAME}"
echo "Plots: ${PLOT_DIR}"
