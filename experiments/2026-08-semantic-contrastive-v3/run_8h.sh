#!/usr/bin/env bash
set -Eeuo pipefail

# Resumable eight-hour launcher for the V3 eight-image experiment only.
# This script intentionally never invokes the 50- or 500-image runners.

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPOSITORY_ROOT"

EXPERIMENT="experiments/2026-08-semantic-contrastive-v3"
PYTHON=".venv-primary-ml-cka/bin/python"
OUTPUT_DIR="${V3_OUTPUT_DIR:-outputs/proxy_selector_semantic_contrastive_v3}"
TIME_BUDGET_SECONDS="${V3_TIME_BUDGET_SECONDS:-28800}"
POLL_SECONDS="${V3_EXISTING_JOB_POLL_SECONDS:-30}"
START_EPOCH="$(date +%s)"
DEADLINE_EPOCH="$((START_EPOCH + TIME_BUDGET_SECONDS))"
DIAGNOSTICS="$OUTPUT_DIR/diagnostics"
mkdir -p "$DIAGNOSTICS"
LOG_PATH="$DIAGNOSTICS/run_8h_$(date -u +%Y%m%dT%H%M%SZ).log"

exec > >(tee -a "$LOG_PATH") 2>&1

exec 9>"$OUTPUT_DIR/run_8h.lock"
if ! flock -n 9; then
    echo "Another V3 eight-hour runner already owns $OUTPUT_DIR/run_8h.lock" >&2
    exit 2
fi

remaining_seconds() {
    local now
    now="$(date +%s)"
    echo "$((DEADLINE_EPOCH - now))"
}

refresh_summary() {
    echo "Refreshing completed-trial summary..."
    env PYTHONPATH=src "$PYTHON" \
        "$EXPERIMENT/src/summarize_results.py" \
        --output-dir "$OUTPUT_DIR" || true
    echo "Completed state files:"
    find "$OUTPUT_DIR/states" -type f -name '*.json' -print 2>/dev/null \
        | sort || true
}

finish() {
    local exit_code=$?
    trap - EXIT INT TERM
    refresh_summary
    echo "V3 runner stopped with exit_code=$exit_code at $(date -u +%FT%TZ)"
    echo "Log: $LOG_PATH"
    exit "$exit_code"
}
trap finish EXIT INT TERM

run_with_budget() {
    local label="$1"
    shift
    local remaining
    remaining="$(remaining_seconds)"
    if ((remaining <= 60)); then
        echo "Time budget exhausted before $label"
        return 124
    fi
    echo "Starting $label with ${remaining}s remaining"
    set +e
    timeout --signal=TERM --kill-after=60 "${remaining}s" "$@"
    local exit_code=$?
    set -e
    if ((exit_code == 124 || exit_code == 137)); then
        echo "Time budget reached during $label; completed states remain resumable"
        return 124
    fi
    if ((exit_code != 0)); then
        echo "$label failed with exit_code=$exit_code" >&2
        return "$exit_code"
    fi
}

echo "V3 eight-hour runner"
echo "Start: $(date -u +%FT%TZ)"
echo "Deadline: $(date -u -d "@$DEADLINE_EPOCH" +%FT%TZ)"
echo "Output: $OUTPUT_DIR"
echo "Scope: P20/P21/P22, eight images only"

# A manually started V3 ablation may already be using the A4000. Count the wait
# against the eight-hour budget and never launch a duplicate GPU process.
while pgrep -f '[s]emantic-contrastive-v3/src/run_ablation.py' >/dev/null; do
    if (( $(remaining_seconds) <= 60 )); then
        echo "Time budget exhausted while waiting for the existing V3 job"
        exit 124
    fi
    echo "Existing V3 ablation detected; waiting ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
done

run_with_budget "unit tests" \
    env PYTHONPATH=src .venv-primary-ml-cka/bin/pytest -q

CORE_ARMS=(
    cls_only
    semantic_only
    cls_plus_semantic
    contrastive_only
    cls_plus_contrastive
)

# P20 is included to verify completeness, but --resume makes its finished
# states essentially free. P21/P22 then fill the same controlled A-E matrix.
for pair in p20 p21 p22; do
    run_with_budget "${pair^^} core A-E ablation" \
        env \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TRANSFORMERS_VERBOSITY=error \
        PYTHONPATH=src \
        "$PYTHON" "$EXPERIMENT/src/run_ablation.py" \
        --config "$EXPERIMENT/config/ablation_${pair}.yaml" \
        --arms "${CORE_ARMS[@]}" \
        --resume \
        --fail-on-error || exit $?
    refresh_summary
done

# Secondary advisor-requested comparison. It runs only after the core matrix.
for pair in p21 p22; do
    run_with_budget "${pair^^} mean-reference comparison" \
        env \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        TRANSFORMERS_VERBOSITY=error \
        PYTHONPATH=src \
        "$PYTHON" "$EXPERIMENT/src/run_ablation.py" \
        --config "$EXPERIMENT/config/ablation_${pair}.yaml" \
        --arms mean_reference_only \
        --resume \
        --fail-on-error || exit $?
    refresh_summary
done

# Complete the required component/layer/pooling audit only after attack arms.
run_with_budget "eight-model embedding audit" \
    env \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRANSFORMERS_VERBOSITY=error \
    PYTHONPATH=src \
    "$PYTHON" "$EXPERIMENT/src/audit_embeddings.py" \
    --config "$EXPERIMENT/config/embedding_audit.yaml" \
    --output-dir "$OUTPUT_DIR" || exit $?

echo "All scheduled V3 eight-image work completed within the time budget."
