#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-primary-ml-cka"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Missing environment: $VENV. Create it with python3.11 -m venv $VENV" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export IMAGENET_ROOT="${IMAGENET_ROOT:-$ROOT/data/imagenet_vehicle_official}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" -m primary_ml_cka.cli.main "$@"
