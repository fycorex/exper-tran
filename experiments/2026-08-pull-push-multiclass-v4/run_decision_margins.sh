#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiments/2026-08-pull-push-multiclass-v4"
OUTPUT="${V4_SCALE50_OUTPUT:-$ROOT/outputs/pull_push_multiclass_v4_scale50_diverse10}"
PYTHON="$ROOT/.venv-primary-ml-cka/bin/python"

cd "$ROOT"
"$PYTHON" -m unittest discover -s "$EXPERIMENT/tests"
PYTHONPATH="$ROOT/src" "$PYTHON" \
  "$EXPERIMENT/src/analyze_decision_margins.py" \
  --config "$EXPERIMENT/config/scale50_tuned.yaml" \
  --output-dir "$OUTPUT" --resume
