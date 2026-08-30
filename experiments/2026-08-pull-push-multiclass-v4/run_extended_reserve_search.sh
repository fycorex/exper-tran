#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/2026-08-pull-push-multiclass-v4"
OUTPUT="${V4_RESERVE_OUTPUT:-$ROOT/outputs/pull_push_multiclass_v4_reserve8_diverse10}"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"
CONFIG="$EXPERIMENT/config/extended_reserve_search.yaml"

run_search() {
  local pair="$1"
  local transitions="$2"
  shift 2
  # shellcheck disable=SC2086
  PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_compare.py" \
    --config "$EXPERIMENT/config/primary.yaml" --output-dir "$OUTPUT" \
    --pairs "$pair" --transitions $transitions --arm-config "$CONFIG" \
    --arms "$@" --resume --fail-on-error
}

cd "$ROOT"
PYTHONPATH="$ROOT/src:$EXPERIMENT/src" "$PYTHON" \
  "$EXPERIMENT/src/prepare_reserve_tuning.py" --output-dir "$OUTPUT"

run_search P16 "T01 T02 T03 T04 T05 T06 T07 T08 T09 T10" \
  p16_ext_push010 p16_ext_push020 p16_ext_push035 \
  p16_ext_rho075_push025 p16_ext_rho125_push025 \
  p16_ext_tau005_push025 p16_ext_tau020_push025

run_search P14 "T01 T02 T03 T04 T05 T06 T07 T08 T09 T10" \
  p14_ext_baseline p14_ext_push025 p14_ext_push050 \
  p14_ext_rho035_push050 p14_ext_rho075_push050 p14_ext_tau020_push050

run_search P19 "T04 T10" \
  p19_ext_baseline p19_ext_push050 p19_ext_push075 p19_ext_push125 \
  p19_ext_push150 p19_ext_rho035 p19_ext_rho065

for pair in P14 P16; do
  run_search "$pair" "T01 T02 T03 T04 T05 T06 T07 T08 T09 T10" \
    multi_ext_rho025 multi_ext_rho050 multi_ext_rho100 multi_ext_tau020_rho050
done
run_search P19 "T04 T10" \
  multi_ext_rho025 multi_ext_rho050 multi_ext_rho100 multi_ext_tau020_rho050

PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/summarize.py" --output-dir "$OUTPUT"
