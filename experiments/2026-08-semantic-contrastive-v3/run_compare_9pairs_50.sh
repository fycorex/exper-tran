#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OLD_OUTPUT="${OLD_SCALE50_OUTPUT_DIR:-outputs/proxy_selector_cka_v2_scale50}"
NEW_OUTPUT="${NEW_SCALE50_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3_scale50}"
PYTHON=".venv-primary-ml-cka/bin/python"

# The old controlled baseline is resumable and normally already archived.
bash scripts/run_all9_semantic_scale50.sh "$OLD_OUTPUT"

# The new ten-class prototype loss uses the isolated v3 output directory.
V3_SCALE50_OUTPUT_DIR="$NEW_OUTPUT" \
  bash experiments/2026-08-semantic-contrastive-v3/run_scale50_multiclass.sh

env PYTHONPATH=src "$PYTHON" \
  experiments/2026-08-semantic-contrastive-v3/src/compare_scale50.py \
  --old-output "$OLD_OUTPUT" \
  --new-output "$NEW_OUTPUT" \
  --output "$NEW_OUTPUT/summaries/scale_50_old_vs_new.csv"
