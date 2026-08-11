#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIZE="${1:?usage: run_scaled_experiment.sh 8|50|500 [PAIR_ID|all] [OUTPUT_DIR] [--resume] [PROMPT_ID] [ATTACK_CONFIG]}"
PAIR_ID="${2:-all}"
OUTPUT_DIR="${3:-outputs/proxy_selector_scale_${SIZE}}"
RESUME_FLAG="${4:-}"
PROMPT_ID="${5:-original}"
ATTACK_CONFIG="${6:-configs/attacks/qwen_scale_selected.yaml}"

case "$SIZE" in
  8|50|500) ;;
  *) echo "SIZE must be 8, 50, or 500" >&2; exit 2 ;;
esac

if [[ -n "$RESUME_FLAG" && "$RESUME_FLAG" != "--resume" ]]; then
  echo "Fourth argument must be --resume when provided" >&2
  exit 2
fi

export PRIMARY_ML_CKA_DATA_CONFIG="$ROOT/configs/data/scale_${SIZE}.yaml"
export PRIMARY_ML_CKA_PROMPT_ID="$PROMPT_ID"

PAIR_ARGS=()
if [[ "$PAIR_ID" != "all" ]]; then
  PAIR_ARGS=(--pair-id "$PAIR_ID")
fi

RESUME_ARGS=()
if [[ "$RESUME_FLAG" == "--resume" ]]; then
  RESUME_ARGS=(--resume)
fi

cd "$ROOT"
NEEDS_SETUP=true
if [[ "$RESUME_FLAG" == "--resume" \
  && -f "$OUTPUT_DIR/evaluation/manifests/source_validation_candidates.jsonl" \
  && -f "$OUTPUT_DIR/summaries/test_report.txt" \
  && -d "$OUTPUT_DIR/taps" ]] \
  && compgen -G "$OUTPUT_DIR/evaluation/manifests/*__main.jsonl" > /dev/null \
  && compgen -G "$OUTPUT_DIR/taps/*.json" > /dev/null; then
  NEEDS_SETUP=false
fi

if [[ "$NEEDS_SETUP" == true ]]; then
  bash scripts/run_experiment.sh data prepare \
    "${PAIR_ARGS[@]}" --output-dir "$OUTPUT_DIR"
  bash scripts/run_experiment.sh target screen \
    "${PAIR_ARGS[@]}" --output-dir "$OUTPUT_DIR"
  bash scripts/run_experiment.sh models inspect-taps \
    "${PAIR_ARGS[@]}" --output-dir "$OUTPUT_DIR"
  bash scripts/run_experiment.sh tests run \
    "${PAIR_ARGS[@]}" --output-dir "$OUTPUT_DIR"
fi
bash scripts/run_experiment.sh attack scaled \
  "${PAIR_ARGS[@]}" "${RESUME_ARGS[@]}" \
  --image-count "$SIZE" \
  --config "$ATTACK_CONFIG" \
  --output-dir "$OUTPUT_DIR"
