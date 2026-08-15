# Original token-level contrastive CKA transfer search

This bundle records the completed controlled search for the attack objective

```text
L = L_cls + lambda_eff * [CKA(adv, clean) - alpha * CKA(adv, target)]
```

It uses no semantic auxiliary (`beta=0`) and preserves the original
same-spatial-index, per-image token-CKA definition. All three intra-family
large-to-small pairs use the same eight canonical clean images.

The staged search contains 60 ten-step rho/alpha trials, 12 thirty-step
transfer trials, nine full 100-step trials, and three 100-step confirmations
using 48 target references. Every state, target generation output, selected
configuration, and compact combined table is included here.

## Full-budget result

All reported full-budget attacks reached the strict 8/8 proxy gate and 8/8
proxy free-generation target check.

| Pair | CLS TASR / ASR | Best 8-ref CKA TASR / ASR | 48-ref CKA TASR / ASR |
| --- | ---: | ---: | ---: |
| P20 Qwen 4B to 2B | 3/8 / 7/8 | 1/8 / 8/8 | 1/8 / 3/8 |
| P21 InternVL 4B to 2B | 0/8 / 5/8 | 0/8 / 4/8 | 0/8 / 3/8 |
| P22 Gemma E4B to E2B | 0/8 / 2/8 | 0/8 / 2/8 | 0/8 / 1/8 |

The CKA mechanism itself succeeded: selected CKA trials reduced source CKA
and increased target-reference CKA. The result therefore separates failure to
optimize the auxiliary objective from failure of that optimized geometry to
increase targeted transfer. It supports a bounded conclusion for this attack,
transition, budget, and parameter search; it is not a proof about every
possible CKA formulation.

## Files

- `combined/all_trials.csv`: compact mid/full/48-reference comparison.
- `*/summary.csv`: full protocol and metrics for each stage.
- `*/trials/`: complete resumable trial states.
- `*/target_outputs/`: frozen-PNG target generations and parsed labels.
- `selected_configs/`: generated promotion configs and selection provenance.

Reproduce from the repository root with:

```bash
bash scripts/run_original_token_cka_transfer_search.sh outputs/proxy_selector_cka_v2
python3 scripts/archive_completed_experiment_results.py
```
