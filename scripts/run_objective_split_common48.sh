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

.venv-primary-ml-cka/bin/pytest -q

# Use the same all-NF4 proxy policy for all three families. Target evaluation
# remains BF16. This removes the prior P21-only precision advantage.
unset PRIMARY_ML_CKA_KEEP_VISION_BF16

for pair_id in P20 P21 P22; do
  bash scripts/run_experiment.sh models inspect-taps \
    --pair-id "$pair_id" --output-dir "$OUTPUT_DIR"
done

# Exercise the full-reference cache, gradient-ratio calibration, attack save,
# and unconditional target evaluation before committing to the long sweep.
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config \
    experiments/2026-08-qwen-transfer-diagnostics/config/objective_split_validation.yaml \
  --output-dir "$OUTPUT_DIR" \
  --resume

for family in p20 p21 p22; do
  PYTHONPATH=src .venv-primary-ml-cka/bin/python \
    experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
    --config \
      "experiments/2026-08-qwen-transfer-diagnostics/config/objective_split_${family}.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --resume
done
