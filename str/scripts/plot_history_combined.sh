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
Usage: bash str/scripts/plot_history_combined.sh EXPERIMENT_OUTPUT_DIR [OUTPUT_PNG]

Example:
  bash str/scripts/plot_history_combined.sh str/output/8m/baseline_frozen_esm
  bash str/scripts/plot_history_combined.sh str/output/8m/baseline_frozen_esm str/output/plots/baseline_8m_combined.png

This script reads EXPERIMENT_OUTPUT_DIR/history.csv and plots all numeric metrics
except epoch into one line chart.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

EXPERIMENT_DIR="$1"
OUTPUT_PNG="${2:-}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --experiment-dir "${EXPERIMENT_DIR}"
  --mode combined
  --summary-json "${EXPERIMENT_DIR}/history_combined_plot.json"
)
if [[ -n "${OUTPUT_PNG}" ]]; then
  args+=(--output-png "${OUTPUT_PNG}")
fi

"${PYTHON_BIN}" str/scripts/plot/plot_experiment_history.py "${args[@]}"
