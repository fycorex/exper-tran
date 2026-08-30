# Semantic Pull+Push Multiclass V4 — Technical Results Report

## Technical summary

The tuned binary prototype pull+push attack reached **802/1500 targeted
successes (53.5% TASR)** across three small-to-large intra-family model pairs,
ten semantically diverse ImageNet transitions, and 50 clean-valid images per
cell. The corresponding untargeted ASR was **60.7% (910/1500)**. Relative to
the first frozen 50-image settings, tuning improved aggregate TASR from 45.0%
to 53.5% (+8.5 percentage points).

The gain was present in all three families: Qwen P14 improved from 53.4% to
58.6%, InternVL P16 from 18.6% to 26.2%, and Gemma P19 from 63.0% to 75.6%.
InternVL remains materially harder than the other two pairs.

Post-hoc analysis supports a more precise explanation than “farther classes
transfer less.” Within model pair, target-side projected prototype distance
was negatively associated with TASR (stratified Spearman rho=-0.571,
permutation p=0.0016), while the clean target decision margin was positively
associated with TASR (rho=0.541, p=0.0037). The finite target margin movement
caused by the frozen perturbation was much more predictive (rho=0.888,
p<1e-5), and normalized gap closure was stronger still (rho=0.950, p<1e-5).
Representation CKA is high for all three same-family pairs, but transition CKA
does not behave as a simple “closer means easier” selector in this cohort.

## Tuned 50-image results

| Pair | Direction | Original TASR | Tuned TASR | Tuned ASR | Absolute gain |
|---|---|---:|---:|---:|---:|
| P14 | Qwen 2B -> 4B | 267/500 (53.4%) | 293/500 (58.6%) | 336/500 (67.2%) | +5.2 pp |
| P16 | InternVL 2B -> 4B | 93/500 (18.6%) | 131/500 (26.2%) | 159/500 (31.8%) | +7.6 pp |
| P19 | Gemma E2B -> E4B | 315/500 (63.0%) | 378/500 (75.6%) | 415/500 (83.0%) | +12.6 pp |
| **Total** | three pairs | **675/1500 (45.0%)** | **802/1500 (53.5%)** | **910/1500 (60.7%)** | **+8.5 pp** |

Each pair aggregates ten source-to-target transitions with 50 images per
transition. TASR is unconditional targeted attack success on the frozen PNGs;
ASR counts any target prediction different from the clean source class.

The frozen tuned settings were:

| Pair | rho0 | Temperature | Target pull | Source push | Vision layer | Steps |
|---|---:|---:|---:|---:|---:|---:|
| P14 | 0.5 | 0.2 | 1.0 | 0.5 | 17 | 50 |
| P16 | 1.0 | 0.1 | 1.0 | 0.25 | 17 | 50 |
| P19 | 0.5 | 0.1 | 1.0 | 0.5 | 15 | 50 |

Here `rho0` is the initial semantic/classification input-gradient ratio used
to calibrate the effective semantic coefficient. Target pull and source push
are independent logits inside the binary prototype contrastive term.

## Representation similarity is real but not sufficient

The proxy and target projected representations were extracted post hoc for
the same 480 disjoint class-reference images (48 per class). All three
same-family pairs have high matched-image CKA well above their 1,000-shuffle
null distributions:

| Pair | Global CKA | Normalized CKA | Aggregate TASR |
|---|---:|---:|---:|
| P16 | 0.9328 | 0.9297 | 26.2% |
| P14 | 0.9515 | 0.9491 | 58.6% |
| P19 | 0.9912 | 0.9905 | 75.6% |

These three pair-level points are descriptive, not enough for a reliable
cross-pair correlation claim. Within pair, however, the 30 transition cells
show that projected source-target prototype distance is associated with lower
TASR: proxy distance rho=-0.539 (p=0.0032), target distance rho=-0.571
(p=0.0016). Class-conditioned transition CKA is weaker and negatively related
to TASR in this fixed catalog, so CKA should not be interpreted as a monotonic
transition-difficulty score without additional controls.

## Target decision difficulty and achieved margin movement explain the gap

The post-hoc teacher-forced closed-set audit evaluates each already-frozen
clean/adversarial pair on the target model. It does not participate in attack
generation. Across the 30 transition cells:

| Diagnostic | Stratified Spearman with TASR | Permutation p |
|---|---:|---:|
| Clean target robust margin | +0.541 | 0.0037 |
| Target robust margin change | +0.888 | <1e-5 |
| Normalized target gap closure | +0.950 | <1e-5 |
| Closed-set boundary crossings | +0.997 | <1e-5 |

The result separates two effects. First, some target classes start much
farther from the target decision boundary. Second, successful perturbations
must move the target margin far enough to close that initial gap. For example,
P16 T01 closed about 107% of its mean initial gap and achieved 27/50 TASR,
whereas P16 T02 closed about 19% and achieved 0/50. This is consistent with
the user hypothesis that semantically distant transitions are harder, but it
also shows that distance alone is incomplete: actual target-side margin
movement is the dominant immediate correlate.

## Experimental scope and metric definitions

- P14: Qwen 2B proxy -> Qwen 4B target.
- P16: InternVL 2B proxy -> InternVL 4B target.
- P19: Gemma E2B proxy -> Gemma E4B target.
- Ten ImageNet classes cover animals, insects, food, instruments,
  electronics, beverages, natural scenes, furniture, sports equipment, and
  transport. The transitions form a balanced cycle so each class appears once
  as source and once as target.
- Every 50-image cell uses images clean-correct for all six participating
  models. Source and target reference banks contain 48 disjoint training
  images per class.
- Attack budget: L-infinity 16/255, step 1/255, momentum 1, random start,
  50 steps, seed 42.
- Target models remain black-box during attack generation. Target outputs and
  representations are used only after adversarial PNGs are frozen.
- TASR: fraction of clean-valid frozen adversarial images generated as the
  specified target class.
- ASR: fraction generated as any class other than the clean source class.
- Gap closure: finite target robust-margin change divided by the magnitude of
  the negative clean margin; values near one mean the initial decision gap was
  closed on average.

## Robustness checks and limitations

1. Hyperparameters were selected on independent eight-image reserve cohorts;
   the 50-image common-clean cohort was used for confirmation rather than
   retuning.
2. The 30 transition cells are clustered in only three model pairs. Reported
   within-pair correlations use pair-stratified ranks and 100,000 within-pair
   permutations, but do not establish population-level causality.
3. The decision audit uses teacher-forced ten-way scores, whereas final TASR
   uses greedy generation. Boundary crossings closely match TASR here, but the
   two evaluation mechanisms are not mathematically identical.
4. CKA absolute values are calibrated against shuffled-image nulls. The
   minimum empirical p with 1,000 CKA permutations is approximately 0.001.
5. The constant L-infinity budget has no correlation statistic; its rho and p
   are intentionally recorded as `NaN` rather than interpreted.
6. Raw ImageNet files, adversarial PNG collections, and cached model feature
   tensors are not stored in Git because of licensing and repository-size
   constraints. Manifests, configurations, aggregate results, per-image
   decision diagnostics, and reproduction scripts are included.

## Recommended next steps

1. Keep P14/P19 settings frozen and focus additional optimization diagnostics
   on P16 transitions where the target margin moves too little.
2. Test whether initial target margin and normalized prototype distance can
   prospectively select easier transitions before attack generation.
3. Add more model pairs, especially cross-family pairs, before making a
   general CKA-versus-transfer claim.
4. Repeat a subset with multiple seeds to quantify attack stochasticity.

## Reproducibility and result files

The exact run commands are in [README.md](README.md). The primary tracked
artifacts are listed in [RESULTS_MANIFEST.md](RESULTS_MANIFEST.md). The central
tables are:

- `outputs/pull_push_multiclass_v4_scale50_diverse10/summaries/scale50_tuned_results.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/transition_metrics.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/correlations.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/transition_decision_metrics.csv`
- `outputs/pull_push_multiclass_v4_scale50_diverse10/diagnostics/tuned_transfer_correlations/decision/decision_correlations.csv`
