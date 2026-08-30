#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/2026-08-pull-push-multiclass-v4"
OUTPUT="${V4_RESERVE_OUTPUT:-$ROOT/outputs/pull_push_multiclass_v4_reserve8_diverse10}"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"
ARMS=(
  p16_reserve_baseline
  p16_reserve_push025
  p16_reserve_push050
  p16_reserve_rho075_push050
  p16_reserve_rho125_push050
)

cd "$ROOT"
PYTHONPATH="$ROOT/src:$EXPERIMENT/src" "$PYTHON" \
  "$EXPERIMENT/src/prepare_reserve_tuning.py" --output-dir "$OUTPUT"
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_compare.py" \
  --config "$EXPERIMENT/config/primary.yaml" \
  --output-dir "$OUTPUT" \
  --pairs P16 \
  --arm-config "$EXPERIMENT/config/p16_reserve_search.yaml" \
  --arms "${ARMS[@]}" \
  --resume --fail-on-error
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/summarize.py" \
  --output-dir "$OUTPUT"
