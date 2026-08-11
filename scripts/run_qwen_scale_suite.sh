#!/usr/bin/env bash
set -euo pipefail

PAIR_ID="${1:-P20}"
PROMPT_ID="${2:-original}"
RESUME_FLAG="${3:-}"

if [[ -n "$RESUME_FLAG" && "$RESUME_FLAG" != "--resume" ]]; then
  echo "Third argument must be --resume when provided" >&2
  exit 2
fi

for size in 8 50 500; do
  output_dir="outputs/proxy_selector_${PAIR_ID}_scale_${size}"
  bash scripts/run_scaled_experiment.sh \
    "$size" "$PAIR_ID" "$output_dir" "$RESUME_FLAG" "$PROMPT_ID"
done
