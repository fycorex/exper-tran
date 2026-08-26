#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON=".venv-primary-ml-cka/bin/python"
BINARY_OUTPUT="${BINARY_SCALE50_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3_scale50_binary}"
MULTI_OUTPUT="${MULTI_SCALE50_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3_scale50_multiclass}"

run_one() {
  local mode="$1"
  local output="$2"
  local pairs="${3:-P20 P19 P22 P14 P16 P21 P02 P06 P11}"
  V3_SCALE50_OUTPUT_DIR="$output" \
  V3_SCALE50_SEMANTIC_MODE="$mode" \
  V3_SCALE50_PAIR_IDS="$pairs" \
    bash experiments/2026-08-semantic-contrastive-v3/run_scale50_multiclass.sh
}

# Both modes use the same 50-image cohort and the same attack configuration.
# They differ only in the representation objective:
#   prototype             = binary target-vs-source contrastive baseline
#   multiclass_prototype  = target-vs-all-nine-non-target prototype loss
run_one prototype "$BINARY_OUTPUT" "P20 P19 P22 P14 P16 P21 P02 P06 P11"
run_one multiclass_prototype "$MULTI_OUTPUT" "P20 P21 P22"

env PYTHONPATH=src "$PYTHON" \
  experiments/2026-08-semantic-contrastive-v3/src/compare_scale50.py \
  --old-output "$BINARY_OUTPUT" \
  --new-output "$MULTI_OUTPUT" \
  --output "$MULTI_OUTPUT/summaries/scale_50_binary_vs_multiclass.csv"
