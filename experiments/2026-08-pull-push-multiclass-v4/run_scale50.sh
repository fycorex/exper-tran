#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/2026-08-pull-push-multiclass-v4"
CONFIG="$EXPERIMENT/config/scale50.yaml"
OUTPUT="${V4_SCALE50_OUTPUT:-$ROOT/outputs/pull_push_multiclass_v4_scale50_diverse10}"
DATA_ROOT="${IMAGENET_ROOT:-$ROOT/data/imagenet_diverse10_minimal}"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"

cd "$ROOT"
"$PYTHON" -m unittest discover -s "$EXPERIMENT/tests"
"$PYTHON" "$EXPERIMENT/src/download_required_imagenet.py" \
  --config "$CONFIG" --data-root "$DATA_ROOT"
IMAGENET_ROOT="$DATA_ROOT" PYTHONPATH="$ROOT/src" \
  "$PYTHON" "$EXPERIMENT/src/prepare_data.py" \
  --config "$CONFIG" --output-dir "$OUTPUT"
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/screen_transitions.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" --resume
PYTHONPATH="$ROOT/src" "$PYTHON" "$EXPERIMENT/src/run_scale50.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" --resume
