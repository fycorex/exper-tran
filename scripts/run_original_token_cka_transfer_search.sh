#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2}"
EXPERIMENT="$ROOT/experiments/2026-08-qwen-transfer-diagnostics"
GENERATED="$OUTPUT_DIR/diagnostics/original_token_cka_generated_configs"

SCAN_CONFIG="$EXPERIMENT/config/original_token_cka_wide_scan.yaml"
MID_CONFIG="$GENERATED/mid30.yaml"
FULL_CONFIG="$GENERATED/full100.yaml"
REF48_CONFIG="$GENERATED/reference48.yaml"

cd "$ROOT"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv-primary-ml-cka/bin/pytest -q \
  tests/unit/attack/test_linear_cka.py \
  tests/unit/attack/test_loss_sign.py \
  tests/unit/evaluation/test_representation_metrics.py \
  tests/unit/experiment/test_scaled_resume.py \
  tests/integration/test_finite_difference.py

# Stage 1: broad original token-CKA mechanism/proxy scan (60 trials).
.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/run_sweep.py" \
  --config "$SCAN_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/select_original_token_cka.py" \
  --stage scan-to-mid \
  --summary "$OUTPUT_DIR/diagnostics/original_token_cka_wide_scan/summary.csv" \
  --output-config "$MID_CONFIG"

# Stage 2: three complementary candidates per family plus CLS-only, 30 steps.
.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/run_sweep.py" \
  --config "$MID_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/select_original_token_cka.py" \
  --stage mid-to-full \
  --summary "$OUTPUT_DIR/diagnostics/original_token_cka_mid_transfer/summary.csv" \
  --output-config "$FULL_CONFIG"

# Stage 3: two best CKA candidates per family plus CLS-only, full 100 steps.
.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/run_sweep.py" \
  --config "$FULL_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/select_original_token_cka.py" \
  --stage full-to-48 \
  --summary "$OUTPUT_DIR/diagnostics/original_token_cka_full100/summary.csv" \
  --output-config "$REF48_CONFIG"

# Stage 4: best full-budget candidate with the larger 48-reference bank.
.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/run_sweep.py" \
  --config "$REF48_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

.venv-primary-ml-cka/bin/python \
  "$EXPERIMENT/src/summarize_original_token_cka.py" \
  --summary "$OUTPUT_DIR/diagnostics/original_token_cka_mid_transfer/summary.csv" \
  --summary "$OUTPUT_DIR/diagnostics/original_token_cka_full100/summary.csv" \
  --summary "$OUTPUT_DIR/diagnostics/original_token_cka_reference48/summary.csv" \
  --output-dir "$OUTPUT_DIR/diagnostics/original_token_cka_transfer_search_summary"
