#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

while systemctl --user is-active --quiet proxy-selector-all-pairs-8.service; do
  sleep 60
done

exec bash scripts/run_intra_large_to_small_recovery.sh
