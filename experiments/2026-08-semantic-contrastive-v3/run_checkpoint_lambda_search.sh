#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

EXPERIMENT="experiments/2026-08-semantic-contrastive-v3"
PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3}"
CHECKPOINTS=(15 20 25 30 35 40 45)
ARMS=(lambda_r02 lambda_r03 lambda_r04)

for pair in p21 p22; do
    config="$EXPERIMENT/config/checkpoint_lambda_${pair}.yaml"
    env \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TRANSFORMERS_VERBOSITY=error \
        PYTHONPATH=src \
        "$PYTHON" "$EXPERIMENT/src/run_ablation.py" \
        --config "$config" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint-steps "${CHECKPOINTS[@]}" \
        --resume \
        --fail-on-error

    for arm in "${ARMS[@]}"; do
        env \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            TRANSFORMERS_VERBOSITY=error \
            PYTHONPATH=src \
            "$PYTHON" "$EXPERIMENT/src/evaluate_checkpoints.py" \
            --config "$config" \
            --arm "$arm" \
            --steps 45 \
            --output-dir "$OUTPUT_DIR"
    done
done
