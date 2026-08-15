#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2}"
SCAN_CONFIG="$ROOT/experiments/2026-08-qwen-transfer-diagnostics/config/contrastive_cka_rho_alpha_smoke.yaml"
SCAN_DIAGNOSTICS="$OUTPUT_DIR/diagnostics/contrastive_cka_rho_alpha_smoke"
FINAL_DIAGNOSTICS="$OUTPUT_DIR/diagnostics/contrastive_cka_three_pair_full_smoke"
SELECTED_CONFIG="$FINAL_DIAGNOSTICS/selected_config.yaml"

cd "$ROOT"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv-primary-ml-cka/bin/pytest -q \
  tests/unit/attack/test_linear_cka.py \
  tests/unit/attack/test_loss_sign.py \
  tests/unit/evaluation/test_representation_metrics.py \
  tests/unit/experiment/test_scaled_resume.py \
  tests/integration/test_finite_difference.py

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config "$SCAN_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/select_contrastive_cka_smoke.py \
  --scan-summary "$SCAN_DIAGNOSTICS/summary.csv" \
  --output-config "$SELECTED_CONFIG" \
  --steps 20

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config "$SELECTED_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/summarize_contrastive_cka_smoke.py \
  --summary "$FINAL_DIAGNOSTICS/summary.csv" \
  --output-dir "$FINAL_DIAGNOSTICS/final_summary"
