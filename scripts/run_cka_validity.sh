#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2}"
WAIT_SECONDS="${PRIMARY_ML_CKA_GPU_WAIT_SECONDS:-60}"

cd "$ROOT"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while ! .venv-primary-ml-cka/bin/python -c \
  'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; do
  printf '%s CUDA unavailable; waiting %ss\n' "$(date -u +%FT%TZ)" "$WAIT_SECONDS"
  sleep "$WAIT_SECONDS"
done

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_cka_validity.py \
  --output-dir "$OUTPUT_DIR" --resume
