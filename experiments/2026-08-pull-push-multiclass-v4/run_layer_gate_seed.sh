#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/2026-08-pull-push-multiclass-v4"
OUTPUT="${V4_RESERVE_OUTPUT:-$ROOT/outputs/pull_push_multiclass_v4_reserve8_diverse10}"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"
CONFIG="$EXPERIMENT/config/layer_gate_seed.yaml"

cd "$ROOT"
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_compare.py" \
  --config "$EXPERIMENT/config/primary.yaml" --output-dir "$OUTPUT" \
  --pairs P16 --transitions T02 T10 --arm-config "$CONFIG" \
  --arms p16_layer11_standard p16_layer23_standard \
  p16_layer17_gate_standard p16_layer17_gate_small \
  --resume --fail-on-error
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_compare.py" \
  --config "$EXPERIMENT/config/primary.yaml" --output-dir "$OUTPUT" \
  --pairs P19 --transitions T10 --arm-config "$CONFIG" \
  --arms p19_layer7_standard p19_layer11_standard p19_layer15_gate_small \
  p19_layer15_small_seed43 p19_layer15_small_seed44 \
  --resume --fail-on-error
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/summarize.py" --output-dir "$OUTPUT"
