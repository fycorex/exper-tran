#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

EXPERIMENT="experiments/2026-08-semantic-contrastive-v3"
PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3}"
CHECKPOINTS=(20 22 24 26 28 30 32 34 36 38 40)

for pair in p21 p22; do
    config="$EXPERIMENT/config/focused_ratio_${pair}.yaml"
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

    while IFS= read -r arm; do
        env \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            TRANSFORMERS_VERBOSITY=error \
            PYTHONPATH=src \
            "$PYTHON" "$EXPERIMENT/src/evaluate_checkpoints.py" \
            --config "$config" \
            --arm "$arm" \
            --steps 40 \
            --output-dir "$OUTPUT_DIR"
    done < <(
        "$PYTHON" -c \
            'import sys,yaml; print(*[x["name"] for x in yaml.safe_load(open(sys.argv[1]))["arms"]], sep="\n")' \
            "$config"
    )
done
