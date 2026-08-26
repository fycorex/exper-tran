#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V3_SCALE50_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3_scale50}"

env PYTHONPATH=src "$PYTHON" \
  experiments/2026-08-semantic-contrastive-v3/src/prepare_scale50_data.py \
  --output-dir "$OUTPUT_DIR"

export PYTHONPATH=src
export PRIMARY_ML_CKA_DATA_CONFIG="$ROOT/configs/data/scale_50_common.yaml"
export PRIMARY_ML_CKA_SOURCE_MANIFEST=attack_images_scale50.jsonl
export PRIMARY_ML_CKA_EVALUATE_ALL_FROZEN=1
export PRIMARY_ML_CKA_FIXED_REFERENCE_BANK=1
export PRIMARY_ML_CKA_SEMANTIC_MODE="${V3_SCALE50_SEMANTIC_MODE:-multiclass_prototype}"
export PRIMARY_ML_CKA_CLS_LOSS_MODE="${V3_SCALE50_CLS_LOSS_MODE:-ce_margin}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash scripts/run_experiment.sh tests run --output-dir "$OUTPUT_DIR"
for tap_spec in \
  "P02:Qwen__Qwen3.5-4B" \
  "P06:openai__clip-vit-large-patch14" \
  "P11:google__siglip2-so400m-patch14-384" \
  "P14:Qwen__Qwen3.5-2B" \
  "P16:OpenGVLab__InternVL3_5-2B-HF" \
  "P19:google__gemma-4-E2B-it" \
  "P20:Qwen__Qwen3.5-4B" \
  "P21:OpenGVLab__InternVL3_5-4B-HF" \
  "P22:google__gemma-4-E4B-it"; do
  pair_id="${tap_spec%%:*}"
  bash scripts/run_experiment.sh models inspect-taps --pair-id "$pair_id" --output-dir "$OUTPUT_DIR"
done

for pair_id in P20 P19 P22 P14 P16 P21 P02 P06 P11; do
  mode="${V3_SCALE50_SEMANTIC_MODE:-multiclass_prototype}"
  case "$pair_id" in
    P20|P14) recipe_pair=P20 ;;
    P21|P16) recipe_pair=P21 ;;
    P22|P19) recipe_pair=P22 ;;
    *) recipe_pair=generic ;;
  esac
  if [[ "$mode" == "prototype" && "$recipe_pair" == "generic" ]]; then
    config="$ROOT/configs/attacks/binary_prototype_scale50_generic.yaml"
  elif [[ "$mode" == "prototype" ]]; then
    config="$ROOT/configs/attacks/binary_prototype_scale50_${recipe_pair,,}.yaml"
  else
    config="$ROOT/configs/attacks/${mode}_scale50_${recipe_pair,,}.yaml"
  fi
  if [[ "$mode" == "prototype" ]]; then
    case "$recipe_pair" in
      P20) export PRIMARY_ML_CKA_CLS_LOSS_MODE=ce_margin; export PRIMARY_ML_CKA_SEMANTIC_TARGET_WEIGHT=1; export PRIMARY_ML_CKA_SEMANTIC_SOURCE_WEIGHT=1 ;;
      P21) export PRIMARY_ML_CKA_CLS_LOSS_MODE=ce_margin; export PRIMARY_ML_CKA_SEMANTIC_TARGET_WEIGHT=1; export PRIMARY_ML_CKA_SEMANTIC_SOURCE_WEIGHT=0.5 ;;
      P22) export PRIMARY_ML_CKA_CLS_LOSS_MODE=margin_only; export PRIMARY_ML_CKA_SEMANTIC_TARGET_WEIGHT=1; export PRIMARY_ML_CKA_SEMANTIC_SOURCE_WEIGHT=0.25 ;;
      *) export PRIMARY_ML_CKA_CLS_LOSS_MODE=ce_margin; export PRIMARY_ML_CKA_SEMANTIC_TARGET_WEIGHT=1; export PRIMARY_ML_CKA_SEMANTIC_SOURCE_WEIGHT=1 ;;
    esac
  else
    export PRIMARY_ML_CKA_SEMANTIC_TARGET_WEIGHT=1
    export PRIMARY_ML_CKA_SEMANTIC_SOURCE_WEIGHT=1
    export PRIMARY_ML_CKA_CLS_LOSS_MODE=margin_only
  fi
  bash scripts/run_experiment.sh attack scaled \
    --pair-id "$pair_id" --resume --image-count 50 \
    --config "$config" --output-dir "$OUTPUT_DIR"
done

env PYTHONPATH=src "$PYTHON" \
  experiments/2026-08-qwen-transfer-diagnostics/src/summarize_scale50_semantic.py \
  --output-dir "$OUTPUT_DIR"
