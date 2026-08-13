# Pair transfer diagnostics

## Question

Why did same-family Qwen targeted attacks initially report zero transfer, and
whether the same failure modes affect the other proxy adapters, and which CKA
lambda and target-reference alpha retain the strict proxy gate while improving
conditional target success across all configured pairs?

## Corrections established before the sweep

- The differentiable Qwen proxy path now matches the native processor's
  256-patch (16x16) visual input instead of the incorrect 196-patch path.
- Generative scoring includes only semantic answer tokens. Chat-template
  boundary newlines are excluded.
- Proxy teacher forcing, proxy free generation, and target generation use the
  same non-thinking chat-template prefix.
- CLIP, SigLIP, InternVL, and Gemma preprocessing and token taps are checked
  independently against their native processors.
- Qwen-4B to Qwen-2B is included as reverse pair P20.
- The existing-method audit found that Qwen and InternVL CKA were tapped before
  their multimodal projector while classification acted on projected language-
  model input tokens. Both now use `get_image_features().pooler_output`, matching
  Gemma's attack layer and the actual model input semantics.
- Smoke and scaled attacks consume the full configured step budget; the strict
  8/8 proxy criterion remains an evaluation gate rather than an early-stop rule.
- Confirmation now evaluates the frozen PNGs on the target and records held-out
  TASR instead of writing attack-only rows.

## Environment and inputs

Run from the repository root with the existing `.venv-primary-ml-cka`, local
`.hf-cache`, prepared canonical ImageNet images, clean-screen manifests, and
validated proxy taps. One NVIDIA RTX A4000 is sufficient; models are
loaded sequentially and no 59 GB checkpoint is used.

## Reproduce the diagnostic sweep

```bash
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-qwen-transfer-diagnostics/src/run_sweep.py \
  --output-dir outputs/proxy_selector_cka_v2 --resume
```

The default matrix contains 221 one-batch trials across all seven configured
pairs, five lambdas, and three alphas (lambda zero is deduplicated). The three
prompt variants apply only to generative proxies; CLIP/SigLIP use one fixed
closed-set text-classifier definition and therefore use only the original
target prompt. Expected runtime on an A4000 is well above ten hours.
Incremental state is written to
`outputs/proxy_selector_cka_v2/diagnostics/pair_prompt_sweep_v2`.

For the focused large-to-small recovery sweep, run:

```bash
bash scripts/run_intra_large_to_small_recovery.sh
```

This uses a separate `recovery_full100` phase, never overwrites the original
review, disables proxy-gate early stopping, and probes family-calibrated CKA
weights. Resumable state is written under
`outputs/proxy_selector_cka_v2/diagnostics/intra_large_to_small_recovery/`.

The follow-up single-GPU queue covers roughly the next ten hours without
starting the 50-image stage:

```bash
bash scripts/run_next_10h_intra_recovery.sh
```

It waits for the first recovery, tests a BF16 vision tower with an NF4 language
model for P21, and then runs proxy-only semantic-centroid ablations for P22 at
two seeds. Every phase is independently resumable, and a failed/OOM phase does
not discard completed trials or prevent the next phase from running.

The corrected existing-method baseline (no new transfer method) is:

```bash
bash scripts/run_projected_tap_intra_baseline.sh \
  outputs/proxy_selector_cka_v2
```

It revalidates the three large intra-family proxy taps and runs 15 full-budget
eight-image trials across P20, P21, and P22. Each family includes a lambda-zero
classification control and four lambda/alpha combinations. Incremental results
are written to `diagnostics/projected_tap_intra_baseline`. The runner waits when
CUDA is unavailable and never falls back to CPU.

The clean objective-split diagnostic is:

```bash
bash scripts/run_objective_split_common48.sh \
  outputs/proxy_selector_cka_v2
```

It first runs a one-step end-to-end GPU validation and then compares four
objectives for each of P20/P21/P22: classification only, target-CKA only,
semantic-centroid only, and target-CKA plus semantic centroid. All trials use
the same eight source images, all 48 target references, seed 42, 100 attack
steps, no source-repulsion term, and an initial auxiliary/classification
input-gradient ratio of 0.3. The calibrated auxiliary weight is then fixed for
all attack steps. All proxies use the same NF4 policy and targets use BF16. Frozen
adversarial images are evaluated on the target even when the proxy result is
below 8/8; the strict 8/8 rule is retained only as a promotion gate. Results
are resumable under `diagnostics/objective_split_common48_rho03`. This script
does not run the 50- or 500-image stages.

The complete controlled nine-pair matrix uses a separate phase and diagnostics
directory so it cannot overwrite the corrected three-pair run. It first runs
the three small-to-large intra-family pairs, then the three cross-family pairs,
then reruns the three large-to-small pairs on the same eight-model intersection:
six generative models plus CLIP and SigLIP. There are 36 clean-valid source
candidates in that intersection;
the first eight are frozen in
`diagnostics/objective_split_all9v2_common48_rho03/common_clean.jsonl`. Each pair
uses the same four objectives, 48 references, seed 42, 100 steps, and initial
gradient ratio 0.3:

```bash
bash scripts/run_all9_controlled_diagnostic.sh \
  outputs/proxy_selector_cka_v2
```

The script is resumable. Its six-pair one-step smoke is fail-fast and verifies
that every planned smoke state is complete before any 100-step trial begins.
It continues through all nine-pair CKA validity,
leave-query-out local CKA, decision geometry, gap closure, and pair-level
Spearman summaries after the 36 attacks finish. Generative proxy gates are
reported as `generative_strict`; CLIP/SigLIP gates are reported as
`contrastive_closed_set`, and their repeated argmax is not labeled as free
generation. Contrastive semantic loss uses the same classifier-facing projected
image representation as its closed-set logits; spatial tokens remain separate
for token CKA. Selector reporting treats `cls_only` as primary and semantic as
secondary; their artifacts are written to separate `cls_only/` and
`semantic_only/` result directories. Before CKA extraction, the runner creates
`evaluation/manifests/calibration_disjoint_all9v2.jsonl`, preserving five
images per class while excluding every frozen attack image. Thus the selector
analysis is out-of-sample with respect to the attacked source images without
requiring any attack to be rerun. The existing P20/P21/P22 results remain the corrected historical
three-pair diagnostic; only the new all-nine phase is used for strict common-
image pair comparisons.

On the single A4000, the P02 and P19 8-bit Gemma E4B targets do not leave enough
memory for an input-gradient graph even at one image per microbatch. These two
pairs therefore retain the exact teacher-forced clean/adversarial margins and
gap closure under `torch.no_grad`, while their gradient-alignment fields are
marked unavailable. The other seven pairs retain the full gradient diagnostics;
the runner does not substitute a lower-precision E4B target merely to
manufacture those fields. Completed per-pair geometry files are resumable.

Post-hoc CKA validity uses the eight canonical clean source images as local
queries, not their adversarial variants. It writes one local-similarity row per
pair and source image, avoiding repeated similarity observations across attack
objectives. If a query also occurs in the calibration manifest, every matching
calibration row is excluded before selecting its neighbors. Each resulting
eight-neighbor local CKA is calibrated against a correspondence-shuffled null
and reports raw, excess, and null-headroom-normalized values. Layer/subset CKA
uses the same calibration: all 119 non-identity permutations are used for
five-image class subsets, while larger subsets default to 1,000 seeded
permutations. A Monte Carlo value at the minimum resolution means no sampled
permutation reached the observation; it is reported as an empirical bound, not
an exact tail probability. Run it with:

```bash
bash scripts/run_cka_validity.sh outputs/proxy_selector_cka_v2
```

The decision-geometry output is explicitly a post-hoc, teacher-forced,
closed-set diagnostic; final TASR remains greedy generation on frozen PNGs.
Alongside raw margins it reports target gap closure,
`(M_adv - M_clean) / (-M_clean)`, for clean margins below zero. This separates
target-direction movement from the initial distance to the target boundary:

```bash
bash scripts/run_decision_geometry.sh outputs/proxy_selector_cka_v2
```

The running artifact labels `target_cka_only`, `semantic_only`, and
`target_cka_semantic` are historical shorthand. Classification loss remains in
all three; their precise meanings are `cls_plus_target_cka`,
`cls_plus_semantic`, and `cls_plus_target_cka_semantic`. Future diagnostic
configs should use the explicit names after this active run has completed.

## Scale runs

Run and stop at the all-pair eight-image review gate:

```bash
bash scripts/run_all_pairs_8_review.sh \
  outputs/proxy_selector_cka_v2 \
  outputs/proxy_selector_all_pairs_8
```

This covers the three cross-family pairs plus both directions for Qwen,
InternVL, and Gemma. It writes `scale_8_review.csv` and intentionally does not
start the 50-image stage.

The full resumable entry point performs the 8-image sweep, materializes one
selected prompt/lambda/alpha configuration per eligible pair, and then runs
the 8/50/500-image experiments:

```bash
bash scripts/run_full_long_experiment.sh \
  outputs/proxy_selector_cka_v2 \
  outputs/proxy_selector_selected_scales \
  --resume
```

Pairs that do not satisfy the strict 8/8 proxy gate and eight-image clean
target denominator are not silently promoted to the scale stage. Individual
selected configurations can also be run with the lower-level entry point:

```bash
bash scripts/run_scaled_experiment.sh 8 P20 outputs/proxy_selector_scale_8
bash scripts/run_scaled_experiment.sh 50 P20 outputs/proxy_selector_scale_50
bash scripts/run_scaled_experiment.sh 500 P20 outputs/proxy_selector_scale_500
```

Pass `--resume` as the fourth argument to skip setup and resume completed
batches. The scaled screening path retains the final partial batch, so the
requested counts are exactly 8, 50, and 500 rather than 8, 48, and 496. The
50/500 configurations use source-class ImageNet training images; validation
has only 50 images per class and cannot supply 500 attacks.

The controlled 50-image confirmation keeps the selected recipe from the
all-nine eight-image controlled diagnostic (`classification + semantic
centroid`, initial gradient ratio 0.3). It prepares 100 candidates, screens all
six generative models plus CLIP and SigLIP, freezes the first 50 images that
every model calls class 8, and attacks them in resumable batches of at most
eight. This is an all-model-consensus clean cohort, so its TASR is conditional
on every participating model classifying the clean image correctly; it is not
an unfiltered random ImageNet sample:

```bash
bash scripts/run_all9_semantic_scale50.sh \
  outputs/proxy_selector_cka_v2_scale50
```

Every frozen adversarial PNG is evaluated on the target, even when its batch
does not satisfy the strict all-proxy-hit gate. The final
`summaries/scale_50_semantic_all9.csv` therefore reports both unconditional
50-image TASR/ASR and targeted hits conditional on the per-image proxy mask.
All seven source batches use the same frozen first 48 target references as the
eight-image controlled diagnostic; the final two-image partial batch does not
change or rotate that semantic prototype. This output directory is dedicated
to scale-50 and must not be reused by the older confirmation pipeline, whose
main/confirmation manifests have different semantics.

## Status

The projected-token sweep is resumable and is the only baseline that should be
used for the next decision. Pre-correction artifacts remain available for the
audit but must not be mixed with corrected trials. The durable audit input is
`audit/artifact.json`; HTML packaging was unavailable on this host because the
Node runtime is not installed.
