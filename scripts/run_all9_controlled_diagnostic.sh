#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2}"
WAIT_SECONDS="${PRIMARY_ML_CKA_GPU_WAIT_SECONDS:-60}"
DIAGNOSTICS_NAME="objective_split_all9v2_common48_rho03"
RESULT_SUFFIX="all9v2_common48_rho03"

cd "$ROOT"
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PRIMARY_ML_CKA_KEEP_VISION_BF16

while ! .venv-primary-ml-cka/bin/python -c \
  'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; do
  printf '%s CUDA unavailable; waiting %ss\n' "$(date -u +%FT%TZ)" "$WAIT_SECONDS"
  sleep "$WAIT_SECONDS"
done

bash scripts/run_experiment.sh tests run --output-dir "$OUTPUT_DIR"

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/preflight_contrastive_clean.py \
  --output-dir "$OUTPUT_DIR"

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config \
    experiments/2026-08-qwen-transfer-diagnostics/config/objective_split_all9_smoke.yaml \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --fail-on-error

# Priority 1: reverse-direction intra-family pairs.
for config in \
  objective_split_all9_intra_small_to_large \
  objective_split_all9_cross \
  objective_split_all9_intra_large_to_small; do
  .venv-primary-ml-cka/bin/python \
    experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
    --config \
      "experiments/2026-08-qwen-transfer-diagnostics/config/${config}.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --resume
done

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/materialize_disjoint_calibration.py \
  --output-dir "$OUTPUT_DIR" \
  --common-clean-manifest \
    "diagnostics/${DIAGNOSTICS_NAME}/common_clean.jsonl" \
  --output-manifest \
    "evaluation/manifests/calibration_disjoint_all9v2.jsonl"

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_cka_validity.py \
  --output-dir "$OUTPUT_DIR" \
  --diagnostics-name "$DIAGNOSTICS_NAME" \
  --result-name "cka_validity_${RESULT_SUFFIX}" \
  --calibration-manifest \
    "evaluation/manifests/calibration_disjoint_all9v2.jsonl" \
  --resume

for pair_id in P02 P06 P11 P14 P16 P19 P20 P21 P22; do
  .venv-primary-ml-cka/bin/python \
    experiments/2026-08-qwen-transfer-diagnostics/src/run_decision_geometry.py \
    --output-dir "$OUTPUT_DIR" \
    --diagnostics-name "$DIAGNOSTICS_NAME" \
    --result-name "decision_geometry_${RESULT_SUFFIX}" \
    --pair-id "$pair_id"
done

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/summarize_all9_selector.py \
  --output-dir "$OUTPUT_DIR" \
  --diagnostics-name "$DIAGNOSTICS_NAME" \
  --cka-result-name "cka_validity_${RESULT_SUFFIX}" \
  --geometry-result-name "decision_geometry_${RESULT_SUFFIX}" \
  --result-name "selector_analysis_${RESULT_SUFFIX}" \
  --objective cls_only

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/summarize_all9_selector.py \
  --output-dir "$OUTPUT_DIR" \
  --diagnostics-name "$DIAGNOSTICS_NAME" \
  --cka-result-name "cka_validity_${RESULT_SUFFIX}" \
  --geometry-result-name "decision_geometry_${RESULT_SUFFIX}" \
  --result-name "selector_analysis_${RESULT_SUFFIX}" \
  --objective semantic_only
