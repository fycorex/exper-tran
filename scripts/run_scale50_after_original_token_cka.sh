#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-outputs/proxy_selector_cka_v2_scale50}"
CKA_UNIT="proxy-selector-original-token-cka-search.service"

while true; do
  state="$(systemctl --user is-active "$CKA_UNIT" 2>/dev/null || true)"
  case "$state" in
    active|activating|deactivating)
      sleep 30
      ;;
    *)
      break
      ;;
  esac
done

# Give the previous process a short window to release CUDA allocations before
# the resumable scale-50 runner loads the next model.
sleep 30
cd "$ROOT"
exec bash scripts/run_all9_semantic_scale50.sh "$OUTPUT_DIR"
