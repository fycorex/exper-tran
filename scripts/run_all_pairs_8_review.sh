#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNING_OUTPUT="${1:-outputs/proxy_selector_cka_v2}"
SCALE_ROOT="${2:-outputs/proxy_selector_all_pairs_8}"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/run_experiment.sh models inspect-taps \
  --output-dir "$TUNING_OUTPUT"
bash scripts/run_experiment.sh tests run \
  --output-dir "$TUNING_OUTPUT"

PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --config experiments/2026-08-qwen-transfer-diagnostics/config/all_pairs_8_review.yaml \
  --output-dir "$TUNING_OUTPUT" --resume

SELECTION_DIR="$TUNING_OUTPUT/diagnostics/pair_prompt_sweep_v2"
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/materialize_selected_configs.py \
  --selected "$SELECTION_DIR/selected.json" \
  --base-config configs/attacks/primary_ml_cka.yaml \
  --output-dir "$SELECTION_DIR/selected_configs"

while IFS=$'\t' read -r pair_id prompt_id attack_config; do
  if [[ "$pair_id" == "pair_id" ]]; then
    continue
  fi
  output_dir="$SCALE_ROOT/$pair_id/scale_8"
  bash scripts/run_scaled_experiment.sh \
    8 "$pair_id" "$output_dir" --resume "$prompt_id" "$attack_config"
done < "$SELECTION_DIR/selected_configs/selected_runs.tsv"

PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/summarize_scale8_review.py \
  --scale-root "$SCALE_ROOT" \
  --output "$SCALE_ROOT/scale_8_review.csv"

echo "8-image review complete. The 50-image stage is intentionally gated." 
