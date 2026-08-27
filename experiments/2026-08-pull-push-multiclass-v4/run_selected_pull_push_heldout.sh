#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V4_OUTPUT_DIR:-outputs/pull_push_multiclass_v4_diverse10}"
PRIMARY_CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/primary.yaml"
SEARCH_CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/pull_push_search.yaml"
HELDOUT=(T01 T03 T05 T06 T07 T09 T10)
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hyperparameters are selected only from T02/T04/T08. These invocations apply
# one frozen choice per family to the seven held-out semantic transitions.
"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$PRIMARY_CONFIG" --arm-config "$SEARCH_CONFIG" \
  --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs P14 --transitions "${HELDOUT[@]}" \
  --arms pull_push_rho050_balanced

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$PRIMARY_CONFIG" --arm-config "$SEARCH_CONFIG" \
  --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs P16 --transitions "${HELDOUT[@]}" \
  --arms pull_push_rho100_balanced

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$PRIMARY_CONFIG" --arm-config "$SEARCH_CONFIG" \
  --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs P19 --transitions "${HELDOUT[@]}" \
  --arms pull_push_rho050_balanced

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/summarize.py \
  --output-dir "$OUTPUT_DIR"
