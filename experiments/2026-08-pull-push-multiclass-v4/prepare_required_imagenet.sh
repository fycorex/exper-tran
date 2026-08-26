#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON=".venv-primary-ml-cka/bin/python"
CONFIG="experiments/2026-08-pull-push-multiclass-v4/config/primary.yaml"
DATA_ROOT="${V4_IMAGENET_ROOT:-data/imagenet_diverse10_minimal}"
OUTPUT_DIR="${V4_OUTPUT_DIR:-outputs/pull_push_multiclass_v4_diverse10}"
COOKIE_ARGS=()
if [[ -n "${IMAGENET_COOKIE_FILE:-}" ]]; then
  COOKIE_ARGS=(--cookie-file "$IMAGENET_COOKIE_FILE")
fi

export PYTHONPATH=src

"$PYTHON" \
  experiments/2026-08-pull-push-multiclass-v4/src/download_required_imagenet.py \
  --config "$CONFIG" \
  --data-root "$DATA_ROOT" \
  "${COOKIE_ARGS[@]}"

IMAGENET_ROOT="$DATA_ROOT" "$PYTHON" \
  experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR"

echo "Prepared minimal diverse-10 data at: $DATA_ROOT"
echo "Prepared canonical manifests at: $OUTPUT_DIR"
