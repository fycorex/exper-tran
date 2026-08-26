# Semantic Contrastive V3

## Question

This experiment tests whether source-vs-target semantic contrast in an explicit
Vision Encoder representation improves targeted adversarial transfer over the
v2 target-only semantic attraction loss, and whether the classification loss
helps or conflicts with that representation objective.

Semantic ImageNet labels remain 1–10. Model-facing answers are zero-based
strings 0–9: pickup truck (semantic class 8) is output `7`, and garbage truck
(semantic class 7) is output `6`.

## Controlled scope

- P20 Qwen 4B → Qwen 2B first, then P21 and P22.
- The same eight v2 all-model-clean source images are reused.
- Source and target banks contain 48 deterministic ImageNet training images and
  are disjoint from attacked validation images.
- Attack: L-inf 16/255, step 1/255, momentum 1, random start, seed 42, 100 steps.
- Target models are loaded only after frozen adversarial PNGs are written.
- No 50- or 500-image run is started by this experiment.

## Setup and reproduction

The repository environment and local pinned Hugging Face snapshots from v2 are
required. Set `IMAGENET_ROOT` to an ImageNet root containing `train/` and `val/`.

```bash
export IMAGENET_ROOT=/path/to/imagenet
export PYTHONPATH=src

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-semantic-contrastive-v3/src/prepare_v3_data.py

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-semantic-contrastive-v3/src/audit_tokenization.py

.venv-primary-ml-cka/bin/python \
  experiments/2026-08-semantic-contrastive-v3/src/audit_embeddings.py \
  --model Qwen/Qwen3.5-4B

# Cheapest P20 end-to-end validation.
.venv-primary-ml-cka/bin/python \
  experiments/2026-08-semantic-contrastive-v3/src/run_ablation.py \
  --config experiments/2026-08-semantic-contrastive-v3/config/ablation_p20.yaml \
  --steps 1 --arms cls_plus_contrastive --resume

# Full P20 controlled ablation (run only after smoke passes).
.venv-primary-ml-cka/bin/python \
  experiments/2026-08-semantic-contrastive-v3/src/run_ablation.py \
  --config experiments/2026-08-semantic-contrastive-v3/config/ablation_p20.yaml \
  --resume

# Resumable wall-clock-bounded runner for the complete P20/P21/P22 eight-image
# work. It waits for an already-running V3 process, never starts 50/500 images,
# and refreshes the summary on success, error, or timeout.
bash experiments/2026-08-semantic-contrastive-v3/run_8h.sh
```

The runner defaults to 28,800 seconds. For a shorter validation of its timeout
and resume behavior, set `V3_TIME_BUDGET_SECONDS`, for example:

```bash
V3_TIME_BUDGET_SECONDS=600 \
  bash experiments/2026-08-semantic-contrastive-v3/run_8h.sh
```

Results are written under `outputs/proxy_selector_semantic_contrastive_v3/`.
Every trial has an independent state JSON and is resumable.

The P22 third-class diagnosis is followed by an optional ten-class prototype
objective. It builds one fixed reference centroid for each of the ten vehicle
classes and minimizes a ten-way proxy-only prototype CE, so the target
prototype must beat every non-target prototype rather than only the pickup
truck prototype:

```bash
bash experiments/2026-08-semantic-contrastive-v3/run_p22_multiclass.sh
```

The script prepares disjoint 48-image class banks, runs an 8-reference
one-step smoke, then runs the `rho_0=0.15/0.25/0.35` margin trajectories and
the `rho_0=0.25` CE-plus-margin trajectory. It evaluates every saved checkpoint
from step 15 through step 50 and is resumable at both the trial and checkpoint
levels.

The matching Qwen 4B-to-2B diagnostic keeps the previously audited P20 Vision
Encoder layer 17, runs 100 steps, and evaluates fixed checkpoints without using
the target for attack generation or checkpoint selection:

```bash
bash experiments/2026-08-semantic-contrastive-v3/run_p20_multiclass.sh
```

## Loss arms

1. classification only (`ce_margin`)
2. v2 target-only semantic attraction only
3. classification + v2 target-only semantic attraction
4. prototype source-vs-target contrastive only
5. classification + prototype contrastive
6. mean-reference source-vs-target contrastive only

The combined arms use initial auxiliary/classification pixel-gradient ratio
`rho_0=0.3`. Pure representation arms use coefficient 1 without CLS-relative
calibration.

The follow-up weight search keeps the original unit-weight InfoNCE result as a
baseline and varies the initial auxiliary/weighted-CLS gradient ratio together
with target/source logit weights. It is intentionally limited to P21/P22 and
30-step screening before any new 100-step confirmation:

```bash
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-semantic-contrastive-v3/src/run_ablation.py \
  --config experiments/2026-08-semantic-contrastive-v3/config/weight_search_p21.yaml \
  --resume --fail-on-error
```

### Current eight-image diagnostic results

The original 100-step ablation and the follow-up 30-step search are complete.
The strongest observed targeted-transfer results are:

| Pair | Recipe | Steps | Proxy | TASR | ASR |
|---|---|---:|---:|---:|---:|
| P20 Qwen 4B -> 2B | CLS + unit prototype contrastive | 100 | 8/8 | 4/8 | 7/8 |
| P21 InternVL 4B -> 2B | CLS + prototype, rho=.3, target/source=1/.5 | 30 | 8/8 | 5/8 | 8/8 |
| P22 Gemma E4B -> E2B | CLS + prototype, rho=.3, target/source=1/.25 | 20--40 | 8/8 | 2/8 | 7--8/8 |

P21 is strongly non-monotonic in attack length: TASR is 3/8, 5/8, 1/8,
and 1/8 at 20, 30, 40, and 50 steps respectively, although the proxy gate is
8/8 at every stop. P22 is 2/8 from 20 through 40 steps and falls to 0/8 at 50.
Consequently, a 100-step surrogate optimum is not used as an automatic
confirmation criterion.

For P22 at 30 steps and target/source weights 1/.25, splitting the
classification term produced 0/8 for CE+margin, 1/8 for closed-set CE, 2/8 for
margin-only, and 1/8 for target-token NLL in the follow-up rerun. None exceeded
the existing 2/8 best.

The subsequent ten-class prototype diagnostic directly contrasted the target
prototype against all nine non-target class prototypes. This removed the
binary objective's assumption that pickup truck was the only relevant
negative, but it did not improve P22 TASR:

| Classification | Initial rho | Best strict step | Proxy | TASR | ASR |
|---|---:|---:|---:|---:|---:|
| margin-only | .15 | 32 | 8/8 | 1/8 | 8/8 |
| margin-only | .25 | 40 | 8/8 | 2/8 | 8/8 |
| margin-only | .35 | 45 | 8/8 | 2/8 | 7/8 |
| CE + margin | .25 | 26 | 8/8 | 1/8 | 8/8 |

Here "strict" requires proxy closed-set and free-generation 8/8, minimum
target margin at least 8, and minimum closed-set target probability at least
0.999. At the best checkpoints, target class 9 remains the dominant wrong
output. Thus the P22 limitation is not explained solely by binary source-class
repulsion, insufficient proxy convergence, or a narrow nearby rho choice.

The passed attack seed now initializes Python, NumPy, CPU Torch, and CUDA Torch
before proxy materialization. Exact bitwise reproducibility is nevertheless not
available for the present Gemma path: strict PyTorch deterministic mode rejects
the differentiable antialiased bicubic resize backward because CUDA provides no
deterministic implementation. Two nominally identical NF4/sign-PGD runs can
therefore diverge. Result tables must retain the seed and should use repeat or
multi-seed confirmation rather than treating one best-of-grid trial as a stable
estimate.

The P20 proxy-only layer audit found source/target prototype cosine 0.952 at
layer 12, 0.929 at layer 17, and 0.99996 at layer 23. P20 therefore fixes layer
17 for its controlled loss ablation; this selection uses no target-model data.

## Expected outputs

```text
outputs/proxy_selector_semantic_contrastive_v3/
  evaluation/manifests/{attack,source_reference,target_reference}.jsonl
  diagnostics/tokenization_audit.{csv,json}
  diagnostics/embedding_audit.{json,md}
  diagnostics/gradient_trace.{csv,json}
  diagnostics/ablation_8.csv
  diagnostics/summary.md
  states/<pair>/<arm>.json
```

Eight-image findings are diagnostics rather than final statistical claims.
