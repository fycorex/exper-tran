#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/2026-08-pull-push-multiclass-v4"
OUTPUT="${V4_RESERVE_OUTPUT:-$ROOT/outputs/pull_push_multiclass_v4_reserve8_diverse10}"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"
CONFIG="$EXPERIMENT/config/optimization_diagnostic.yaml"

cd "$ROOT"
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_compare.py" \
  --config "$EXPERIMENT/config/primary.yaml" --output-dir "$OUTPUT" \
  --pairs P16 --transitions T02 T10 --arm-config "$CONFIG" \
  --arms p16_diag_best_standard p16_diag_best_small_steps p16_diag_lower_aux \
  --resume --fail-on-error

PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_compare.py" \
  --config "$EXPERIMENT/config/primary.yaml" --output-dir "$OUTPUT" \
  --pairs P19 --transitions T10 --arm-config "$CONFIG" \
  --arms p19_diag_best_standard p19_diag_best_small_steps p19_diag_weaker_push \
  --resume --fail-on-error

PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/evaluate_optimization_diagnostics.py" \
  --output-dir "$OUTPUT" --pair P16 --model-id OpenGVLab/InternVL3_5-4B-HF \
  --arms p16_diag_best_standard p16_diag_best_small_steps p16_diag_lower_aux
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/evaluate_optimization_diagnostics.py" \
  --output-dir "$OUTPUT" --pair P19 --model-id google/gemma-4-E4B-it \
  --arms p19_diag_best_standard p19_diag_best_small_steps p19_diag_weaker_push

PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/summarize.py" --output-dir "$OUTPUT"
