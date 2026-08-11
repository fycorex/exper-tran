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

## Status

The projected-token sweep is resumable and is the only baseline that should be
used for the next decision. Pre-correction artifacts remain available for the
audit but must not be mixed with corrected trials. The durable audit input is
`audit/artifact.json`; HTML packaging was unavailable on this host because the
Node runtime is not installed.
