#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export V3_SCALE50_OUTPUT_DIR="${V3_P20_SCALE50_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3_scale50_multiclass_p20_vision17}"
export V3_SCALE50_SEMANTIC_MODE=multiclass_prototype
export V3_SCALE50_PAIR_IDS=P20
export PRIMARY_ML_CKA_REPRESENTATION_TYPE=vision_encoder
export PRIMARY_ML_CKA_REPRESENTATION_LAYER=17
export PRIMARY_ML_CKA_REPRESENTATION_POOLING=mean

bash experiments/2026-08-semantic-contrastive-v3/run_scale50_multiclass.sh
