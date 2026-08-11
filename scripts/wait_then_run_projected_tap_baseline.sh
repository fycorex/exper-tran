#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2}"

while systemctl --user is-active --quiet \
  proxy-selector-projected-tap-baseline.service; do
  sleep 60
done

exec /bin/bash "$ROOT/scripts/run_projected_tap_intra_baseline.sh" "$OUTPUT_DIR"
