#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V4_OUTPUT_DIR:-outputs/pull_push_multiclass_v4}"
CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/primary.yaml"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Data preparation and screening are resumable. The actual attack smoke covers
# both loss definitions with only two optimization steps on P14/T08.
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR"
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/screen_transitions.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR" --resume
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR" --fail-on-error \
  --pairs P14 --transitions T08 --steps 2 \
  --arms pull_push_small_steps multiclass_small_steps
