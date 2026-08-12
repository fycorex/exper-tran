# Post-hoc corrections for the 8→7 objective split

This note records the corrected interpretation of the P20/P21/P22 eight-image
diagnostic. Final TASR is still greedy generation on frozen adversarial PNGs.
Margins, gradients, and gap closure below use the separate teacher-forced
closed-set ten-class diagnostic.

## Targeted and untargeted transfer

Zero targeted hits does not mean that a perturbation had no cross-model effect.
For the semantic objective, P20/P21/P22 targeted hits were 4/8, 0/8, and 0/8,
while untargeted hits were 6/8, 5/8, and 8/8. The supported statement is that
targeted-direction transfer is weak for P21/P22, not that their perturbations do
not transfer.

## Target-side difficulty and gap closure

For the robust class-7 margin, `s_7 - max(s_not_7)`, the semantic objective gave:

| Pair | Clean margin | Margin change | Mean per-image gap closure | TASR | Untargeted ASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| P20 | -8.094 | +7.989 | 0.975 | 4/8 | 6/8 |
| P21 | -11.828 | +6.625 | 0.553 | 0/8 | 5/8 |
| P22 | -15.051 | +8.395 | 0.473 | 0/8 | 8/8 |

P22 therefore had the largest mean class-7 margin movement but started farthest
from the boundary. All 12 robust-margin boundary-crossing counts exactly matched
the corresponding greedy-generation targeted-hit counts. Raw TASR across these
models is confounded by initial target-class difficulty and should be reported
with margin movement and gap closure.

## Clean-query local CKA

Local proxy-target CKA uses the canonical clean source image to choose its
proxy-space calibration neighbors. It produces one observation per pair and
source image rather than duplicating the same selector value across four attack
objectives. Three of eight source queries also occur in the original calibration
manifest; their matching bank row is now excluded before top-k selection. Each
N=8 neighborhood is also calibrated against 1,000 shuffled correspondences.
The leave-query-out values below supersede the earlier overlapping results.

| Pair | Raw local CKA | Null mean | CKA excess | Normalized CKA | Semantic TASR |
| --- | ---: | ---: | ---: | ---: | ---: |
| P20 | 0.9792 | 0.7239 | 0.2553 | 0.9131 | 4/8 |
| P21 | 0.9623 | 0.7271 | 0.2352 | 0.8519 | 0/8 |
| P22 | 0.9979 | 0.8176 | 0.1804 | 0.9891 | 0/8 |

These eight-image results no longer use an overlapping self-neighbor or test a
post-attack query under the name of a pre-attack selector. Raw local CKA should
be interpreted alongside its null mean, excess, normalized value, z-score, and
Monte Carlo empirical p resolution.

## CKA permutation calibration

Every layer/subset CKA is now compared with a shuffled-correspondence null. For
five-image single-class subsets, all 119 non-identity permutations are used; for
larger subsets, 1,000 seeded permutations are used by default.

For projected-to-projected CKA:

| Pair | Subset | True CKA | Null mean | z-score | Empirical p |
| --- | --- | ---: | ---: | ---: | ---: |
| P20 | global, N=50 | 0.9518 | 0.3601 | 32.52 | 0.0010 |
| P21 | global, N=50 | 0.9172 | 0.3377 | 30.25 | 0.0010 |
| P22 | global, N=50 | 0.9935 | 0.3956 | 33.71 | 0.0010 |
| P20 | class 7, N=5 | 0.9899 | 0.6445 | 2.11 | 0.0083 |
| P21 | class 7, N=5 | 0.9831 | 0.7243 | 2.15 | 0.0083 |
| P22 | class 7, N=5 | 0.9990 | 0.9179 | 2.90 | 0.0083 |

The N=5 null means confirm strong small-sample saturation. Absolute values such
as 0.999 must not be interpreted like ordinary correlations. The matched-image
signal nevertheless remains above the permutation null, so the calibrated
result is not that the representation evidence disappears.

For N=10/N=50 and local N=8, an empirical p-value of `1/1001` means that zero of
the 1,000 sampled permutations reached the observed CKA. It is not presented as
an exact tail probability. All 24 local neighborhoods reached this Monte Carlo
resolution limit and are therefore summarized as empirical `p <= 0.001`.

## Scope

This correction does not rerun or alter the adversarial attacks. It fixes
post-hoc selector/calibration design and adds a target-difficulty-normalized
diagnostic. Testing more source→target class transitions remains a new attack
experiment and is required before making a family-wide transferability claim.
