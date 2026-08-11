#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNING_OUTPUT="${1:-outputs/proxy_selector_cka_v2}"
SCALE_ROOT="${2:-outputs/proxy_selector_selected_scales}"
RESUME_FLAG="${3:---resume}"

if [[ "$RESUME_FLAG" != "--resume" && "$RESUME_FLAG" != "--fresh" ]]; then
  echo "Third argument must be --resume or --fresh" >&2
  exit 2
fi

cd "$ROOT"

TUNING_NEEDS_SETUP=true
if [[ "$RESUME_FLAG" == "--resume" \
  && -f "$TUNING_OUTPUT/evaluation/manifests/source_validation_candidates.jsonl" \
  && -f "$TUNING_OUTPUT/evaluation/manifests/target_training_references.jsonl" \
  && -f "$TUNING_OUTPUT/summaries/test_report.txt" \
  && -d "$TUNING_OUTPUT/taps" ]]; then
  TUNING_NEEDS_SETUP=false
fi

if [[ "$TUNING_NEEDS_SETUP" == true ]]; then
  bash scripts/run_experiment.sh data prepare --output-dir "$TUNING_OUTPUT"
  bash scripts/run_experiment.sh target screen --output-dir "$TUNING_OUTPUT"
  bash scripts/run_experiment.sh models inspect-taps --output-dir "$TUNING_OUTPUT"
  bash scripts/run_experiment.sh tests run --output-dir "$TUNING_OUTPUT"
fi

PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --output-dir "$TUNING_OUTPUT" --resume

SELECTION_DIR="$TUNING_OUTPUT/diagnostics/pair_prompt_sweep_v2"
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/materialize_selected_configs.py \
  --selected "$SELECTION_DIR/selected.json" \
  --base-config configs/attacks/primary_ml_cka.yaml \
  --output-dir "$SELECTION_DIR/selected_configs"

SCALE_RESUME=""
if [[ "$RESUME_FLAG" == "--resume" ]]; then
  SCALE_RESUME="--resume"
fi

while IFS=$'\t' read -r pair_id prompt_id attack_config; do
  if [[ "$pair_id" == "pair_id" ]]; then
    continue
  fi
  for size in 8 50 500; do
    output_dir="$SCALE_ROOT/$pair_id/scale_$size"
    bash scripts/run_scaled_experiment.sh \
      "$size" "$pair_id" "$output_dir" "$SCALE_RESUME" \
      "$prompt_id" "$attack_config"
  done
done < "$SELECTION_DIR/selected_configs/selected_runs.tsv"
