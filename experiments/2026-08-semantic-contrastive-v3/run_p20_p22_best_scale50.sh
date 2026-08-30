#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Run sequentially on the single A4000. Each child uses an independent output
# directory and is resumable, so interruption cannot mix corrected results
# with the earlier legacy-projected scale-50 artifacts.
bash experiments/2026-08-semantic-contrastive-v3/run_p20_multiclass_scale50_vision.sh
bash experiments/2026-08-semantic-contrastive-v3/run_p22_multiclass_scale50_vision.sh
