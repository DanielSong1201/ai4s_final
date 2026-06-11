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

usage() {
  cat <<'EOF'
Usage: bash str/scripts/plot_history_separate.sh EXPERIMENT_OUTPUT_DIR [OUTPUT_DIR]

Example:
  bash str/scripts/plot_history_separate.sh str/output/8m/baseline_frozen_esm
  bash str/scripts/plot_history_separate.sh str/output/8m/baseline_frozen_esm str/output/plots/baseline_8m_metrics

This script reads EXPERIMENT_OUTPUT_DIR/history.csv and creates one PNG per
numeric metric column except epoch.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

EXPERIMENT_DIR="$1"
OUTPUT_DIR="${2:-}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --experiment-dir "${EXPERIMENT_DIR}"
  --mode separate
  --summary-json "${EXPERIMENT_DIR}/history_separate_plot.json"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi

"${PYTHON_BIN}" str/scripts/plot/plot_experiment_history.py "${args[@]}"
