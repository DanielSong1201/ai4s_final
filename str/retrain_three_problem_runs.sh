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
OUTPUT_ROOT="${OUTPUT_ROOT:-str/output/retrain_three_problem_runs}"
PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"

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

PROTEIN_HIDDEN_DIM="${PROTEIN_HIDDEN_DIM:-256}"
FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS="${LOSS:-mse}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"

TRANSFORMER_LR="${TRANSFORMER_LR:-5e-4}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-4}"
TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-192}"
ATTENTION_HEADS="${ATTENTION_HEADS:-6}"
FFN_MULTIPLIER="${FFN_MULTIPLIER:-4}"
TRANSFORMER_POOLING="${TRANSFORMER_POOLING:-attention}"
RBF_BINS="${RBF_BINS:-32}"
RBF_MAX_DISTANCE="${RBF_MAX_DISTANCE:-20.0}"

POCKET_LR="${POCKET_LR:-1e-3}"
PROTEIN_POOLING="${PROTEIN_POOLING:-pocket_attention}"
GNN_TYPE="${GNN_TYPE:-gine}"
GNN_LAYERS="${GNN_LAYERS:-3}"
GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM:-128}"
LIGAND_POOLING="${LIGAND_POOLING:-mean}"
POCKET_CACHE_LIMIT="${POCKET_CACHE_LIMIT:--1}"
POCKET_OVERWRITE="${POCKET_OVERWRITE:-0}"
FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE:-1}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

echo "Retrain selected runs"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MANIFEST=${MANIFEST}"
echo "LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR}"
echo "POCKET_CACHE_DIR=${POCKET_CACHE_DIR}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
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

plot_history() {
  local model_name="$1"
  local scale_name="$2"
  local output_dir="$3"
  local history_csv="${output_dir}/history.csv"
  local plot_png="${PLOT_DIR}/${model_name}_${scale_name}_history.png"
  local summary_json="${PLOT_DIR}/${model_name}_${scale_name}_history_plot.json"

  require_path "${history_csv}" "history CSV for ${model_name}_${scale_name}"
  run_stage "[plot] ${model_name}_${scale_name}" \
    "${PYTHON_BIN}" str/scripts/plot/plot_history.py \
      --history-csv "${history_csv}" \
      --output-png "${plot_png}" \
      --title "${model_name} (${scale_name}, retrain)" \
      --summary-json "${summary_json}"
}

run_transformer() {
  local scale_name="$1"
  local esm_cache_dir="$2"
  local output_dir="${OUTPUT_ROOT}/${scale_name}/ligand_graph_transformer_frozen_esm"

  require_path "${esm_cache_dir}" "${scale_name} ESM cache directory"
  run_stage "[${scale_name}] Retrain ligand Graph Transformer" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      MANIFEST="${MANIFEST}" \
      ESM_CACHE_DIR="${esm_cache_dir}" \
      LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
      OUTPUT_DIR="${output_dir}" \
      TRAIN_SPLIT="${TRAIN_SPLIT}" VALID_SPLIT="${VALID_SPLIT}" TEST_SPLIT="${TEST_SPLIT}" \
      TRAIN_LIMIT="${TRAIN_LIMIT}" VALID_LIMIT="${VALID_LIMIT}" TEST_LIMIT="${TEST_LIMIT}" \
      SAMPLE_MODE="${SAMPLE_MODE}" SEED="${SEED}" \
      EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      PROTEIN_HIDDEN_DIM="${PROTEIN_HIDDEN_DIM}" TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS}" \
      TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM}" ATTENTION_HEADS="${ATTENTION_HEADS}" \
      FFN_MULTIPLIER="${FFN_MULTIPLIER}" POOLING="${TRANSFORMER_POOLING}" \
      RBF_BINS="${RBF_BINS}" RBF_MAX_DISTANCE="${RBF_MAX_DISTANCE}" \
      FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM}" DROPOUT="${DROPOUT}" LR="${TRANSFORMER_LR}" \
      WEIGHT_DECAY="${WEIGHT_DECAY}" LOSS="${LOSS}" GRAD_CLIP="${GRAD_CLIP}" \
      DEVICE="${DEVICE}" PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT}" \
    bash str/scripts/run_ligand_graph_transformer_baseline.sh

  plot_history "ligand_graph_transformer_frozen_esm" "${scale_name}" "${output_dir}"
}

run_pocket() {
  local scale_name="$1"
  local esm_cache_dir="$2"
  local output_dir="${OUTPUT_ROOT}/${scale_name}/pocket_gnn_frozen_esm"

  require_path "${esm_cache_dir}" "${scale_name} ESM cache directory"
  run_stage "[${scale_name}] Retrain pocket-aware ligand GNN" \
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
      POCKET_CACHE_LIMIT="${POCKET_CACHE_LIMIT}" POCKET_OVERWRITE="${POCKET_OVERWRITE}" \
      FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE}" \
      EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      PROTEIN_HIDDEN_DIM="${PROTEIN_HIDDEN_DIM}" PROTEIN_POOLING="${PROTEIN_POOLING}" \
      GNN_TYPE="${GNN_TYPE}" GNN_LAYERS="${GNN_LAYERS}" GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM}" \
      LIGAND_POOLING="${LIGAND_POOLING}" FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM}" \
      DROPOUT="${DROPOUT}" LR="${POCKET_LR}" WEIGHT_DECAY="${WEIGHT_DECAY}" LOSS="${LOSS}" \
      GRAD_CLIP="${GRAD_CLIP}" DEVICE="${DEVICE}" PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT}" \
    bash str/scripts/run_pocket_gnn_baseline.sh

  plot_history "pocket_gnn_frozen_esm" "${scale_name}" "${output_dir}"
}

require_path "${MANIFEST}" "trainable manifest"
require_path "${LIGAND_CACHE_DIR}" "ligand graph cache directory"
require_path "${POCKET_CACHE_DIR}" "pocket cache directory"
mkdir -p "${OUTPUT_ROOT}" "${PLOT_DIR}"

run_transformer "8m" "str/manifest/cache/esm_embeddings_8m"
run_transformer "150m" "str/manifest/cache/esm_embeddings_150m"
run_pocket "8m" "str/manifest/cache/esm_embeddings_8m"

echo "Selected retraining completed."
echo "Outputs: ${OUTPUT_ROOT}"
echo "Plots: ${PLOT_DIR}"
