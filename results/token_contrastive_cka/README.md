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
projected visual taps for the three large-to-small intra-family pairs. The CSV
contains both TASR and ASR, along with the proxy gate, free-generation gate,
loss weights, attack parameters, CKA diagnostics, gradient diagnostics, and
runtime.

| Pair | Classification TASR / ASR | Best contrastive CKA TASR / ASR |
| --- | ---: | ---: |
| P20 Qwen 4B to 2B | 2/8 / 5/8 | 3/8 / 5/8 |
| P21 InternVL 4B to 2B | 0/8 / 1/8 | 1/8 / 2/8 |
| P22 Gemma E4B to E2B | 0/8 / 3/8 | 0/8 / 3/8 |

“Best” is selected first by TASR and then by ASR. The selected P20, P21, and
P22 contrastive settings are respectively `(lambda, alpha) = (1, 2)`,
`(3, 1)`, and `(1, 0.5)`; every trial and parameter combination remains in
the detailed CSV.

The CKA mechanism did change the intended quantities. For example, with
`lambda=1, alpha=2`, reference-CKA gain was approximately `+0.228`, `+0.226`,
and `+0.202` for P20, P21, and P22 respectively. The corresponding source-CKA
drops were approximately `0.764`, `0.747`, and `0.642`.

## Historical sweep

`historical_strict_trials.csv` contains every historical token-contrastive
trial that met the strict 8/8 proxy gate and was evaluated on the target.
`historical_sweep_by_pair.csv` aggregates those rows and reports both TASR and
ASR:

- 52 trials with both proxy 8/8 and target clean-valid 8/8;
- 30/52 produced 0/8 target hits;
- 41/52 produced at most 1/8 target hits;
- the best observed result was 4/8, from P20.

These historical trials cover more prompts and hyperparameters, but are not a
single fully controlled matrix. Some were produced before every tap and common
source-cohort correction. They support the statement that token-level
contrastive CKA was weak and inconsistent, not that it was universally zero.
Trials with a zero or partial target clean-valid denominator are excluded from
this aggregate so TASR and ASR always use the same eight-image denominator.

## Metric definitions

- `TASR = target_hits / target_denominator`: adversarial output is class 7.
- `ASR = untargeted_hits / target_denominator`: adversarial output differs
  from clean source class 8. TASR hits are therefore included in ASR.
- `proxy_hits`: images satisfying the strict closed-set proxy target gate.
- `free_generation_hits`: proxy free-generation outputs equal to class 7.

The exported protocol columns include `lambda_cka`, source weight, target
weight alpha, semantic weight, seed, steps, batch size, reference count,
epsilon, step size, and momentum. `summary.json` also records random start,
canvas size, margin, probability threshold, and gradient-ratio status.

## Important exclusion

The latest all-nine target-only CKA arm used `source_weight=0`. It is therefore
not included here as evidence about the full source-away plus target-toward
objective. A fully controlled all-nine contrastive arm with common images, 48
references, and gradient-ratio calibration has not yet been run.

The archived trials use the original same-spatial-index target-token
comparison. Unrelated images do not guarantee semantic patch correspondence,
but changing that correspondence changes the hypothesis. The optional
`clean_anchor_soft` variant is therefore archived separately; historical CSV
values are never silently reinterpreted under another definition.

## Provenance

The exported tables are derived from:

- `outputs/proxy_selector_cka_v2/diagnostics/projected_tap_intra_baseline/summary.csv`
- `outputs/proxy_selector_cka_v2/diagnostics/pair_prompt_sweep_v2/summary.csv`

Reproduce the bundle from the repository root:

```bash
python3 scripts/export_token_contrastive_cka_results.py
```
