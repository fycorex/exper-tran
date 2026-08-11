#!/usr/bin/env bash
set -u -o pipefail

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

.venv-primary-ml-cka/bin/pytest -q \
  tests/unit \
  tests/integration/test_main_confirmation_disjoint.py \
  tests/integration/test_proxy_visual_preprocessing.py \
  tests/integration/test_qwen_visual_preprocessing.py || exit 1

# Qwen's all-NF4 loader is the validated A4000 path. Keeping only its vision
# module in BF16 currently leaves a language-layer 4-bit state uninitialized.
unset PRIMARY_ML_CKA_KEEP_VISION_BF16
bash scripts/run_experiment.sh models inspect-taps \
  --pair-id P20 --output-dir "$OUTPUT_DIR" || exit 1
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config \
    experiments/2026-08-qwen-transfer-diagnostics/config/projected_tap_qwen_baseline.yaml \
  --output-dir "$OUTPUT_DIR" \
  --resume

export PRIMARY_ML_CKA_KEEP_VISION_BF16=1
bash scripts/run_experiment.sh models inspect-taps \
  --pair-id P21 --output-dir "$OUTPUT_DIR" || exit 1
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config \
    experiments/2026-08-qwen-transfer-diagnostics/config/projected_tap_internvl_baseline.yaml \
  --output-dir "$OUTPUT_DIR" \
  --resume

unset PRIMARY_ML_CKA_KEEP_VISION_BF16
bash scripts/run_experiment.sh models inspect-taps \
  --pair-id P22 --output-dir "$OUTPUT_DIR" || exit 1
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config \
    experiments/2026-08-qwen-transfer-diagnostics/config/projected_tap_gemma_baseline.yaml \
  --output-dir "$OUTPUT_DIR" \
  --resume
