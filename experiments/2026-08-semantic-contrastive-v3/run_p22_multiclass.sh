#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

EXPERIMENT="experiments/2026-08-semantic-contrastive-v3"
PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3}"

env PYTHONPATH=src "$PYTHON" "$EXPERIMENT/src/prepare_v3_data.py" \
    --output-dir "$OUTPUT_DIR" \
    --reference-count 48

env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH=src \
    "$PYTHON" "$EXPERIMENT/src/run_ablation.py" \
    --config "$EXPERIMENT/config/multiclass_p22_smoke.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --resume \
    --fail-on-error

env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH=src \
    "$PYTHON" "$EXPERIMENT/src/run_ablation.py" \
    --config "$EXPERIMENT/config/multiclass_p22.yaml" \
    --output-dir "$OUTPUT_DIR" \
    --arms \
        multiclass_margin_r015 \
        multiclass_margin_r025 \
        multiclass_margin_r035 \
        multiclass_ce_margin_r025 \
    --checkpoint-steps {15..50} \
    --resume \
    --fail-on-error

for arm in \
    multiclass_margin_r015 \
    multiclass_margin_r025 \
    multiclass_margin_r035 \
    multiclass_ce_margin_r025
do
    env \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TRANSFORMERS_VERBOSITY=error \
        PYTHONPATH=src \
        "$PYTHON" "$EXPERIMENT/src/evaluate_checkpoints.py" \
        --config "$EXPERIMENT/config/multiclass_p22.yaml" \
        --arm "$arm" \
        --steps 50 \
        --output-dir "$OUTPUT_DIR"
done
