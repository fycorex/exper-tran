#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V4_OUTPUT_DIR:-outputs/pull_push_multiclass_v4_diverse10}"
CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/primary.yaml"
read -r -a PAIRS <<< "${V4_PAIRS:-P14 P16 P19}"
read -r -a TUNING_TRANSITIONS <<< "${V4_TUNING_TRANSITIONS:-T02 T04 T08}"
read -r -a ALL_TRANSITIONS <<< "${V4_TRANSITIONS:-T01 T02 T03 T04 T05 T06 T07 T08 T09 T10}"

export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR"

"$PYTHON" -m unittest discover \
  -s experiments/2026-08-pull-push-multiclass-v4/tests

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/screen_transitions.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR" --resume

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/audit_distances.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR"

# Phase 1: isolate the effect of loss definition and optimizer schedule on
# three fixed transitions for each small-to-large pair.
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs "${PAIRS[@]}" \
  --transitions "${TUNING_TRANSITIONS[@]}" \
  --arms pull_push_standard multiclass_standard \
    pull_push_small_steps multiclass_small_steps

# Phase 2: use the requested small-step schedule over all ten balanced
# transitions. Completed tuning cells are automatically resumed.
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs "${PAIRS[@]}" \
  --transitions "${ALL_TRANSITIONS[@]}" \
  --arms pull_push_small_steps multiclass_small_steps

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/summarize.py \
  --output-dir "$OUTPUT_DIR"
