#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2_scale50}"
WAIT_SECONDS="${PRIMARY_ML_CKA_GPU_WAIT_SECONDS:-60}"
DATA_CONFIG="$ROOT/configs/data/scale_50_common.yaml"
ATTACK_CONFIG="$ROOT/configs/attacks/all9_semantic_scale50.yaml"

cd "$ROOT"
export PYTHONPATH=src
export PRIMARY_ML_CKA_DATA_CONFIG="$DATA_CONFIG"
export PRIMARY_ML_CKA_EVALUATE_ALL_FROZEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while ! .venv-primary-ml-cka/bin/python -c \
  'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; do
  printf '%s CUDA unavailable; waiting %ss\n' "$(date -u +%FT%TZ)" "$WAIT_SECONDS"
  sleep "$WAIT_SECONDS"
done

if [[ ! -f "$OUTPUT_DIR/evaluation/manifests/source_validation_candidates.jsonl" ]]; then
  bash scripts/run_experiment.sh data prepare --output-dir "$OUTPUT_DIR"
fi

for screen_spec in \
  "P20:Qwen__Qwen3.5-2B" \
  "P14:Qwen__Qwen3.5-4B" \
  "P21:OpenGVLab__InternVL3_5-2B-HF" \
  "P16:OpenGVLab__InternVL3_5-4B-HF" \
  "P22:google__gemma-4-E2B-it" \
  "P19:google__gemma-4-E4B-it"; do
  pair_id="${screen_spec%%:*}"
  safe_model="${screen_spec#*:}"
  if [[ ! -f "$OUTPUT_DIR/evaluation/${safe_model}__clean_screen.jsonl" ]]; then
    bash scripts/run_experiment.sh target screen \
      --pair-id "$pair_id" --output-dir "$OUTPUT_DIR"
  fi
done

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/preflight_contrastive_clean.py \
  --output-dir "$OUTPUT_DIR"

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/materialize_common_clean.py \
  --output-dir "$OUTPUT_DIR" --count 50

for tap_spec in \
  "P02:Qwen__Qwen3.5-4B" \
  "P06:openai__clip-vit-large-patch14" \
  "P11:google__siglip2-so400m-patch14-384" \
  "P14:Qwen__Qwen3.5-2B" \
  "P16:OpenGVLab__InternVL3_5-2B-HF" \
  "P21:OpenGVLab__InternVL3_5-4B-HF" \
  "P19:google__gemma-4-E2B-it" \
  "P22:google__gemma-4-E4B-it"; do
  pair_id="${tap_spec%%:*}"
  safe_model="${tap_spec#*:}"
  if [[ ! -f "$OUTPUT_DIR/taps/${safe_model}.json" ]]; then
    bash scripts/run_experiment.sh models inspect-taps \
      --pair-id "$pair_id" --output-dir "$OUTPUT_DIR"
  fi
done
bash scripts/run_experiment.sh tests run --output-dir "$OUTPUT_DIR"

for pair_id in P20 P19 P22 P14 P16 P21 P02 P06 P11; do
  bash scripts/run_experiment.sh attack scaled \
    --pair-id "$pair_id" --resume --image-count 50 \
    --config "$ATTACK_CONFIG" --output-dir "$OUTPUT_DIR"
done

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/summarize_scale50_semantic.py \
  --output-dir "$OUTPUT_DIR"
