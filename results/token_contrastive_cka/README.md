# Token-level contrastive CKA results

This bundle isolates experiments that used the full token-level contrastive
objective:

```text
L = L_cls + lambda * [CKA(adv, clean) - alpha * CKA(adv, target)]
```

Minimizing this objective pushes the adversarial representation away from its
clean source and toward the target-reference representation. CKA is computed
per image over spatial visual tokens.

## Corrected projected-tap intra-family experiment

The most directly interpretable evidence is in
`projected_tap_intra_trials.csv`. It uses the corrected classifier-facing
projected visual taps for the three large-to-small intra-family pairs.

| Pair | Classification only | Best contrastive token CKA |
| --- | ---: | ---: |
| P20 Qwen 4B to 2B | 2/8 | 3/8 |
| P21 InternVL 4B to 2B | 0/8 | 1/8 |
| P22 Gemma E4B to E2B | 0/8 | 0/8 |

The CKA mechanism did change the intended quantities. For example, with
`lambda=1, alpha=2`, reference-CKA gain was approximately `+0.228`, `+0.226`,
and `+0.202` for P20, P21, and P22 respectively. The corresponding source-CKA
drops were approximately `0.764`, `0.747`, and `0.642`.

## Historical sweep

`historical_sweep_by_pair.csv` aggregates historical token-contrastive trials
that met the strict 8/8 proxy gate and were evaluated on the target:

- 72 evaluated trials;
- 50/72 produced 0/8 target hits;
- 61/72 produced at most 1/8 target hits;
- the best observed result was 4/8, from P20.

These historical trials cover more prompts and hyperparameters, but are not a
single fully controlled matrix. Some were produced before every tap and common
source-cohort correction. They support the statement that token-level
contrastive CKA was weak and inconsistent, not that it was universally zero.

## Important exclusion

The latest all-nine target-only CKA arm used `source_weight=0`. It is therefore
not included here as evidence about the full source-away plus target-toward
objective. A fully controlled all-nine contrastive arm with common images, 48
references, and gradient-ratio calibration has not yet been run.

## Provenance

The exported tables are derived from:

- `outputs/proxy_selector_cka_v2/diagnostics/projected_tap_intra_baseline/summary.csv`
- `outputs/proxy_selector_cka_v2/diagnostics/pair_prompt_sweep_v2/summary.csv`

Reproduce the bundle from the repository root:

```bash
python3 scripts/export_token_contrastive_cka_results.py
```
