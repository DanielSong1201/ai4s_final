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
OUTPUT_ROOT="${OUTPUT_ROOT:-str/output}"

LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR:-str/manifest/cache/ligand_graphs}"
LIGAND_CACHE_REPORT="${LIGAND_CACHE_REPORT:-str/manifest/cache/ligand_graphs_report.json}"
LIGAND_CACHE_LIMIT="${LIGAND_CACHE_LIMIT:--1}"
LIGAND_OVERWRITE="${LIGAND_OVERWRITE:-0}"

POCKET_CACHE_DIR="${POCKET_CACHE_DIR:-str/manifest/cache/pocket_features}"
POCKET_CACHE_LIMIT="${POCKET_CACHE_LIMIT:--1}"
POCKET_OVERWRITE="${POCKET_OVERWRITE:-0}"

ESM_DEVICE="${ESM_DEVICE:-auto}"
ESM_LIMIT="${ESM_LIMIT:--1}"
ESM_CHUNK_BATCH_SIZE="${ESM_CHUNK_BATCH_SIZE:-4}"
ESM_MAX_RESIDUES_PER_CHUNK="${ESM_MAX_RESIDUES_PER_CHUNK:-1022}"
ESM_FLOAT16_OUTPUT="${ESM_FLOAT16_OUTPUT:-1}"
ESM_OVERWRITE="${ESM_OVERWRITE:-0}"
ESM_LOCAL_FILES_ONLY="${ESM_LOCAL_FILES_ONLY:-0}"

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

BASELINE_HIDDEN_DIM="${BASELINE_HIDDEN_DIM:-256}"
PROTEIN_HIDDEN_DIM="${PROTEIN_HIDDEN_DIM:-256}"
FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM:-256}"
DROPOUT="${DROPOUT:-0.1}"
LR="${LR:-1e-3}"
TRANSFORMER_LR="${TRANSFORMER_LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS="${LOSS:-mse}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"

GNN_TYPE="${GNN_TYPE:-gine}"
GNN_LAYERS="${GNN_LAYERS:-3}"
GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM:-128}"
GNN_POOLING="${GNN_POOLING:-mean}"

TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-4}"
TRANSFORMER_HIDDEN_DIM="${TRANSFORMER_HIDDEN_DIM:-192}"
ATTENTION_HEADS="${ATTENTION_HEADS:-6}"
FFN_MULTIPLIER="${FFN_MULTIPLIER:-4}"
TRANSFORMER_POOLING="${TRANSFORMER_POOLING:-attention}"
RBF_BINS="${RBF_BINS:-32}"
RBF_MAX_DISTANCE="${RBF_MAX_DISTANCE:-20.0}"

PROTEIN_POOLING="${PROTEIN_POOLING:-pocket_attention}"
FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE:-1}"

PLOT_DIR="${PLOT_DIR:-${OUTPUT_ROOT}/plots}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_ROOT}/tensorboard}"
START_TENSORBOARD="${START_TENSORBOARD:-1}"
TENSORBOARD_HOST="${TENSORBOARD_HOST:-0.0.0.0}"
TENSORBOARD_PORT="${TENSORBOARD_PORT:-6006}"
TENSORBOARD_LOG="${TENSORBOARD_LOG:-${OUTPUT_ROOT}/tensorboard.log}"
TENSORBOARD_PID_FILE="${TENSORBOARD_PID_FILE:-${OUTPUT_ROOT}/tensorboard.pid}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-0}"

echo "Full ESM-scale training pipeline"
echo "ROOT_DIR=${ROOT_DIR}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "MANIFEST=${MANIFEST}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LIGAND_CACHE_DIR=${LIGAND_CACHE_DIR}"
echo "POCKET_CACHE_DIR=${POCKET_CACHE_DIR}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "DEVICE=${DEVICE}"
echo "ESM_DEVICE=${ESM_DEVICE}"
echo "TENSORBOARD_DIR=${TENSORBOARD_DIR}"
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

check_tensorboard_environment() {
  "${PYTHON_BIN}" - <<'PY'
try:
    import tensorboard  # noqa: F401
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "TensorBoard is required for the full training pipeline.\n"
        f"Import error: {exc}\n"
        "Install it with: python -m pip install tensorboard"
    )
PY
}

plot_history() {
  local model_name="$1"
  local scale_name="$2"
  local output_dir="$3"
  local history_csv="${output_dir}/history.csv"
  local plot_png="${PLOT_DIR}/${model_name}_${scale_name}_history.png"
  local summary_json="${PLOT_DIR}/${model_name}_${scale_name}_history_plot.json"
  local run_name="${scale_name}/${model_name}"

  require_path "${history_csv}" "history CSV for ${model_name}_${scale_name}"
  run_stage "[plot] ${model_name}_${scale_name}" \
    "${PYTHON_BIN}" str/scripts/plot/plot_history.py \
      --history-csv "${history_csv}" \
      --output-png "${plot_png}" \
      --title "${model_name} (${scale_name})" \
      --summary-json "${summary_json}" \
      --tensorboard-dir "${TENSORBOARD_DIR}" \
      --tensorboard-run-name "${run_name}"
}

start_tensorboard() {
  if [[ "${START_TENSORBOARD}" != "1" && "${START_TENSORBOARD}" != "true" && "${START_TENSORBOARD}" != "TRUE" ]]; then
    echo "TensorBoard startup disabled: START_TENSORBOARD=${START_TENSORBOARD}"
    return 0
  fi

  mkdir -p "${OUTPUT_ROOT}"
  if [[ -f "${TENSORBOARD_PID_FILE}" ]]; then
    local old_pid
    old_pid="$(cat "${TENSORBOARD_PID_FILE}")"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "TensorBoard is already running: pid=${old_pid}, logdir=${TENSORBOARD_DIR}"
      echo "Open: http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"
      return 0
    fi
  fi

  if command -v tensorboard >/dev/null 2>&1; then
    nohup tensorboard --logdir "${TENSORBOARD_DIR}" --host "${TENSORBOARD_HOST}" --port "${TENSORBOARD_PORT}" \
      > "${TENSORBOARD_LOG}" 2>&1 &
  elif "${PYTHON_BIN}" -c "import tensorboard" >/dev/null 2>&1; then
    nohup "${PYTHON_BIN}" -m tensorboard.main --logdir "${TENSORBOARD_DIR}" --host "${TENSORBOARD_HOST}" --port "${TENSORBOARD_PORT}" \
      > "${TENSORBOARD_LOG}" 2>&1 &
  else
    echo "TensorBoard is required but unavailable." >&2
    echo "Install it with: ${PYTHON_BIN} -m pip install tensorboard" >&2
    return 1
  fi
  echo "$!" > "${TENSORBOARD_PID_FILE}"
  echo "TensorBoard started: pid=$(cat "${TENSORBOARD_PID_FILE}")"
  echo "TensorBoard log: ${TENSORBOARD_LOG}"
  echo "Open: http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"
}

require_path "${MANIFEST}" "trainable manifest"
mkdir -p "${OUTPUT_ROOT}" "${PLOT_DIR}" "${TENSORBOARD_DIR}"

run_stage "[preflight] Check TensorBoard environment" \
  check_tensorboard_environment

ligand_args=(
  --manifest "${MANIFEST}"
  --cache-dir "${LIGAND_CACHE_DIR}"
  --report-json "${LIGAND_CACHE_REPORT}"
  --limit "${LIGAND_CACHE_LIMIT}"
)
if [[ "${LIGAND_OVERWRITE}" == "1" ]]; then
  ligand_args+=(--overwrite)
fi
run_stage "[setup] Cache ligand graph tensors" \
  "${PYTHON_BIN}" str/scripts/data/cache_ligand_graphs.py "${ligand_args[@]}"

scale_names=("8m" "150m")
scale_models=("facebook/esm2_t6_8M_UR50D" "facebook/esm2_t30_150M_UR50D")

for index in "${!scale_names[@]}"; do
  scale_name="${scale_names[$index]}"
  esm_model_name="${scale_models[$index]}"
  esm_cache_dir="str/manifest/cache/esm_embeddings_${scale_name}"
  esm_cache_report="str/manifest/cache/esm_embeddings_${scale_name}_report.json"

  esm_args=(
    --manifest "${MANIFEST}"
    --cache-dir "${esm_cache_dir}"
    --report-json "${esm_cache_report}"
    --model-name "${esm_model_name}"
    --device "${ESM_DEVICE}"
    --limit "${ESM_LIMIT}"
    --chunk-batch-size "${ESM_CHUNK_BATCH_SIZE}"
    --max-residues-per-chunk "${ESM_MAX_RESIDUES_PER_CHUNK}"
  )
  if [[ "${ESM_FLOAT16_OUTPUT}" == "1" ]]; then
    esm_args+=(--float16-output)
  fi
  if [[ "${ESM_OVERWRITE}" == "1" ]]; then
    esm_args+=(--overwrite)
  fi
  if [[ "${ESM_LOCAL_FILES_ONLY}" == "1" ]]; then
    esm_args+=(--local-files-only)
  fi

  run_stage "[${scale_name}] Precompute ${esm_model_name} ESM cache" \
    "${PYTHON_BIN}" str/scripts/data/cache_esm_embeddings.py "${esm_args[@]}"
done

for index in "${!scale_names[@]}"; do
  scale_name="${scale_names[$index]}"
  esm_model_name="${scale_models[$index]}"
  esm_cache_dir="str/manifest/cache/esm_embeddings_${scale_name}"

  baseline_output="${OUTPUT_ROOT}/${scale_name}/baseline_frozen_esm"
  run_stage "[${scale_name}] Train 5.5 frozen ESM baseline" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      MANIFEST="${MANIFEST}" \
      ESM_CACHE_DIR="${esm_cache_dir}" \
      LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
      OUTPUT_DIR="${baseline_output}" \
      TRAIN_SPLIT="${TRAIN_SPLIT}" VALID_SPLIT="${VALID_SPLIT}" TEST_SPLIT="${TEST_SPLIT}" \
      TRAIN_LIMIT="${TRAIN_LIMIT}" VALID_LIMIT="${VALID_LIMIT}" TEST_LIMIT="${TEST_LIMIT}" \
      SAMPLE_MODE="${SAMPLE_MODE}" SEED="${SEED}" \
      EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      HIDDEN_DIM="${BASELINE_HIDDEN_DIM}" DROPOUT="${DROPOUT}" LR="${LR}" \
      WEIGHT_DECAY="${WEIGHT_DECAY}" LOSS="${LOSS}" GRAD_CLIP="${GRAD_CLIP}" \
      DEVICE="${DEVICE}" PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT}" \
    bash str/scripts/run_frozen_esm_baseline.sh
  plot_history "baseline_frozen_esm" "${scale_name}" "${baseline_output}"

  gnn_output="${OUTPUT_ROOT}/${scale_name}/ligand_gnn_frozen_esm"
  run_stage "[${scale_name}] Train 5.6 ligand GNN" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      MANIFEST="${MANIFEST}" \
      ESM_CACHE_DIR="${esm_cache_dir}" \
      LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
      OUTPUT_DIR="${gnn_output}" \
      TRAIN_SPLIT="${TRAIN_SPLIT}" VALID_SPLIT="${VALID_SPLIT}" TEST_SPLIT="${TEST_SPLIT}" \
      TRAIN_LIMIT="${TRAIN_LIMIT}" VALID_LIMIT="${VALID_LIMIT}" TEST_LIMIT="${TEST_LIMIT}" \
      SAMPLE_MODE="${SAMPLE_MODE}" SEED="${SEED}" \
      EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      PROTEIN_HIDDEN_DIM="${PROTEIN_HIDDEN_DIM}" GNN_TYPE="${GNN_TYPE}" \
      GNN_LAYERS="${GNN_LAYERS}" GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM}" \
      POOLING="${GNN_POOLING}" FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM}" \
      DROPOUT="${DROPOUT}" LR="${LR}" WEIGHT_DECAY="${WEIGHT_DECAY}" LOSS="${LOSS}" \
      GRAD_CLIP="${GRAD_CLIP}" DEVICE="${DEVICE}" PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT}" \
    bash str/scripts/run_ligand_gnn_baseline.sh
  plot_history "ligand_gnn_frozen_esm" "${scale_name}" "${gnn_output}"

  transformer_output="${OUTPUT_ROOT}/${scale_name}/ligand_graph_transformer_frozen_esm"
  run_stage "[${scale_name}] Train 5.7 ligand Graph Transformer" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      MANIFEST="${MANIFEST}" \
      ESM_CACHE_DIR="${esm_cache_dir}" \
      LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
      OUTPUT_DIR="${transformer_output}" \
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
  plot_history "ligand_graph_transformer_frozen_esm" "${scale_name}" "${transformer_output}"

  pocket_output="${OUTPUT_ROOT}/${scale_name}/pocket_gnn_frozen_esm"
  run_stage "[${scale_name}] Train 5.8 pocket-aware ligand GNN" \
    env \
      PYTHON_BIN="${PYTHON_BIN}" \
      MANIFEST="${MANIFEST}" \
      ESM_CACHE_DIR="${esm_cache_dir}" \
      LIGAND_CACHE_DIR="${LIGAND_CACHE_DIR}" \
      POCKET_CACHE_DIR="${POCKET_CACHE_DIR}" \
      OUTPUT_DIR="${pocket_output}" \
      TRAIN_SPLIT="${TRAIN_SPLIT}" VALID_SPLIT="${VALID_SPLIT}" TEST_SPLIT="${TEST_SPLIT}" \
      TRAIN_LIMIT="${TRAIN_LIMIT}" VALID_LIMIT="${VALID_LIMIT}" TEST_LIMIT="${TEST_LIMIT}" \
      SAMPLE_MODE="${SAMPLE_MODE}" SEED="${SEED}" \
      POCKET_CACHE_LIMIT="${POCKET_CACHE_LIMIT}" POCKET_OVERWRITE="${POCKET_OVERWRITE}" \
      FALLBACK_TO_FULL_SEQUENCE="${FALLBACK_TO_FULL_SEQUENCE}" \
      EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
      PROTEIN_HIDDEN_DIM="${PROTEIN_HIDDEN_DIM}" PROTEIN_POOLING="${PROTEIN_POOLING}" \
      GNN_TYPE="${GNN_TYPE}" GNN_LAYERS="${GNN_LAYERS}" GNN_HIDDEN_DIM="${GNN_HIDDEN_DIM}" \
      LIGAND_POOLING="${GNN_POOLING}" FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM}" \
      DROPOUT="${DROPOUT}" LR="${LR}" WEIGHT_DECAY="${WEIGHT_DECAY}" LOSS="${LOSS}" \
      GRAD_CLIP="${GRAD_CLIP}" DEVICE="${DEVICE}" PRETRAIN_CHECK_LIMIT="${PRETRAIN_CHECK_LIMIT}" \
    bash str/scripts/run_pocket_gnn_baseline.sh
  plot_history "pocket_gnn_frozen_esm" "${scale_name}" "${pocket_output}"
done

start_tensorboard

echo "Full training pipeline completed."
echo "Outputs: ${OUTPUT_ROOT}"
echo "Plots: ${PLOT_DIR}"
echo "TensorBoard logdir: ${TENSORBOARD_DIR}"
