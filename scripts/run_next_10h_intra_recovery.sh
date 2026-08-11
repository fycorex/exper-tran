#!/usr/bin/env bash
set -u -o pipefail

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while systemctl --user is-active --quiet proxy-selector-intra-recovery.service; do
  sleep 60
done

.venv-primary-ml-cka/bin/pytest \
  tests/unit/attack/test_loss_sign.py \
  tests/unit/models/test_proxy_precision.py \
  tests/integration/test_lambda_selection.py

export PRIMARY_ML_CKA_KEEP_VISION_BF16=1
if ! PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config experiments/2026-08-qwen-transfer-diagnostics/config/p21_mixed_precision_semantic.yaml \
  --output-dir outputs/proxy_selector_cka_v2 \
  --resume; then
  echo "P21 mixed-precision phase failed; continuing with resumable P22 phases" >&2
fi
unset PRIMARY_ML_CKA_KEEP_VISION_BF16

for config in \
  p22_semantic_seed42.yaml \
  p22_semantic_seed43.yaml; do
  if ! PYTHONPATH=src .venv-primary-ml-cka/bin/python \
    experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
    --config "experiments/2026-08-qwen-transfer-diagnostics/config/$config" \
    --output-dir outputs/proxy_selector_cka_v2 \
    --resume; then
    echo "$config failed; keeping completed trials and continuing" >&2
  fi
done
