#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config experiments/2026-08-qwen-transfer-diagnostics/config/intra_large_to_small_recovery.yaml \
  --output-dir outputs/proxy_selector_cka_v2 \
  --resume
