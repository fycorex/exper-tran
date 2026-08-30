# Tracked Results Manifest

This repository snapshot tracks all compact, auditable result products needed
to reproduce the reported conclusions. Historical result files already
tracked under `outputs/` remain in Git.

In addition to the headline files below, compact CSV/JSON/JSONL/TXT artifacts
under the four V4 output trees are tracked, including per-trial states,
clean-screen manifests, checkpoint evaluations, and raw parsed target outputs.

## V4 primary and tuning results

- `outputs/pull_push_multiclass_v4_diverse10/summaries/results.csv`
- `outputs/pull_push_multiclass_v4_diverse10/diagnostics/prototype_distances.csv`
- `outputs/pull_push_multiclass_v4_diverse10/diagnostics/prototype_distances.json`
- `outputs/pull_push_multiclass_v4_diverse10/diagnostics/clean_screen_summary.csv`
- `outputs/pull_push_multiclass_v4_reserve8_diverse10/summaries/results.csv`
- `outputs/pull_push_multiclass_v4_reserve8_diverse10/diagnostics/optimization/P16_checkpoints.csv`
- `outputs/pull_push_multiclass_v4_reserve8_diverse10/diagnostics/optimization/P16_gradients.csv`
- `outputs/pull_push_multiclass_v4_reserve8_diverse10/diagnostics/optimization/P19_checkpoints.csv`
- `outputs/pull_push_multiclass_v4_reserve8_diverse10/diagnostics/optimization/P19_gradients.csv`

## V4 50-image confirmation

- `outputs/pull_push_multiclass_v4_scale50_diverse10/summaries/scale50_results.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/summaries/scale50_tuned_results.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/clean_screen_summary.csv`

## V4 post-hoc CKA, distance, and perturbation analysis

- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/pair_metrics.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/transition_metrics.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/correlations.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/summary.json`

## V4 target decision audit

- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/P14_per_image.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/P16_per_image.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/P19_per_image.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/transition_decision_metrics.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/decision_correlations.csv`

## Intentionally excluded generated artifacts

The following stay local and ignored:

- ImageNet source/reference images and canonical copies (licensed dataset).
- Frozen clean/adversarial PNG collections (large, reproducible binaries).
- `.pt` representation caches (large, machine-generated intermediates).
- model snapshots, tokenizer caches, Python environments, CUDA caches.

Their omission does not remove aggregate metrics or auditability: image IDs,
configuration, compact summaries, per-image target decision metrics, and all
code needed to regenerate them are tracked.
