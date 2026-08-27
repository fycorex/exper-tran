#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V4_OUTPUT_DIR:-outputs/pull_push_multiclass_v4_diverse10}"
PRIMARY_CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/primary.yaml"
REFINE_CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/pull_push_refine.yaml"
TUNING=(T02 T04 T08)
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$PRIMARY_CONFIG" --arm-config "$REFINE_CONFIG" \
  --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs P14 --transitions "${TUNING[@]}" \
  --arms p14_rho065_balanced p14_rho050_push075 p14_rho050_push125

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$PRIMARY_CONFIG" --arm-config "$REFINE_CONFIG" \
  --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs P16 --transitions "${TUNING[@]}" \
  --arms p16_rho075_balanced p16_rho125_balanced p16_rho100_push075

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/run_compare.py \
  --config "$PRIMARY_CONFIG" --arm-config "$REFINE_CONFIG" \
  --output-dir "$OUTPUT_DIR" --resume --fail-on-error \
  --pairs P19 --transitions "${TUNING[@]}" \
  --arms p19_rho035_balanced p19_rho065_balanced p19_rho050_push075

"$PYTHON" experiments/2026-08-pull-push-multiclass-v4/src/summarize.py \
  --output-dir "$OUTPUT_DIR"
