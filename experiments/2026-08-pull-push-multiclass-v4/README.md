# Pull+Push versus Multiclass V4

The consolidated technical findings are in [REPORT.md](REPORT.md), and the
tracked output inventory is in [RESULTS_MANIFEST.md](RESULTS_MANIFEST.md).

## Question

Does binary source-target prototype pull+push or 10-class prototype
classification yield higher targeted transfer, and does using smaller PGD
steps for more iterations improve it? The primary optimization pairs are the
three small-to-large intra-family directions: P14, P16, and P19.

The experiment replaces the single pickup-truck to garbage-truck transition
with a balanced cycle over ten semantically diverse ImageNet classes. Every
class appears once as source and once as target. It also records proxy-space
class prototype cosine distance before choosing or interpreting attacks.

The fixed catalog deliberately spans different visual and semantic domains:

| Local label | WNID | Class | Domain |
|---:|---|---|---|
| 1 | `n01443537` | goldfish | animal |
| 2 | `n02279972` | monarch butterfly | insect |
| 3 | `n07753275` | pineapple | food |
| 4 | `n02676566` | acoustic guitar | instrument |
| 5 | `n03642806` | laptop | electronics |
| 6 | `n07920052` | espresso | beverage |
| 7 | `n09472597` | volcano | natural scene |
| 8 | `n04099969` | rocking chair | furniture |
| 9 | `n04254680` | soccer ball | sports equipment |
| 10 | `n04146614` | school bus | transport |

Local semantic labels 1--10 map to model-facing output codes 0--9. Prompts are
generated from this catalog rather than importing the historical vehicle list.

## Controlled comparison

- `pull_push`: binary InfoNCE between the target and source prototypes.
- `multiclass`: cross-entropy over all ten class prototypes.
- Standard schedule: 50 steps at 1/255.
- Small-step schedule: 100 steps at 0.5/255.
- Common constants: L-inf 16/255, momentum 1, random start, rho0=0.25,
  margin-only classification loss, 48 references per class, seed 42.

The target remains black-box during attack generation. Target generation is
performed only after adversarial PNGs have been frozen.

## Inputs and environment

- Authorized ImageNet access. The included preparation script streams only
  the first 98 usable training images for each selected WNID (48 references +
  50 disjoint attack candidates), rather than downloading or extracting the
  full ImageNet-1K dataset. The local vehicle subset is not used.
- Pinned local model snapshots under `.hf-cache`.
- Existing `.venv-primary-ml-cka` environment.
- One CUDA GPU; the scripts refuse CPU attack execution.

## Reproduction

Prepare the minimal raw subset and canonical manifests:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/prepare_required_imagenet.sh
```

The official per-synset endpoint currently works without authentication. If
an authenticated session is required later, export the path to a Netscape
cookie jar before running the same command; the cookie file remains local and
must never be committed:

```bash
export IMAGENET_COOKIE_FILE=/secure/path/imagenet-cookies.txt
bash experiments/2026-08-pull-push-multiclass-v4/prepare_required_imagenet.sh
```

The prepared layout is `data/imagenet_diverse10_minimal/train/<WNID>`. Source
references use the first 48 deterministic files and candidates use the next
50, with a hard overlap check. To use an already prepared tree instead:

```bash
export IMAGENET_ROOT=/path/to/minimal-or-full/imagenet

PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py
```

Then screen pair-specific clean-valid eight-image cohorts:

```bash
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-pull-push-multiclass-v4/src/screen_transitions.py --resume
```

Audit prototype distances:

```bash
PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-pull-push-multiclass-v4/src/audit_distances.py
```

Run the resumable primary comparison:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_primary.sh
```

After the primary matrix, run the focused pull+push hyperparameter search:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_pull_push_search.sh
```

This reuses the three tuning transitions and tests two independent quantities:
the initial auxiliary/classification gradient ratio (`rho=0.1,0.5,1.0`) and
the source-push/target-pull logit ratio (`0.5,1,2`, with target pull fixed at
one). The existing `rho=0.25`, balanced 50-step trial is the baseline and is
not rerun. Because effective lambda is calibrated from `rho`, multiplying both
semantic logits by the same constant is not treated as another independent
hyperparameter.

Freeze the per-family choices selected on T02/T04/T08 and evaluate them on the
seven held-out transitions:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_selected_pull_push_heldout.sh
```

The frozen choices are P14 `rho=0.5`, P16 `rho=1.0`, and P19 `rho=0.5`, all
with balanced target pull and source push. This stage must not retune on the
held-out results.

For the independent 50-image confirmation cohort, a final narrow refinement
around those settings is available:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_pull_push_refine.sh
```

It runs only 27 tuning trials: three nearby configurations per family on
T02/T04/T08. The subsequent 50-image cohort is treated as confirmation data,
not as another source of hyperparameter updates.

Run the independent 50-image confirmation after refinement:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_scale50.sh
```

For each of the ten transitions, this selects the same 50 clean-correct images
across all six participating models and therefore across P14/P16/P19. It runs
seven batches (`8+8+8+8+8+8+2`) while fixing the same 48 source and target
references for every batch. The selected settings are P14 `rho=0.5` with
target:source logits `1:0.75`, P16 `rho=1.0` with `1:1`, and P19 `rho=0.5`
with `1:1`. Every frozen PNG is evaluated, including images that miss the
strict proxy gate. The summary reports unconditional TASR/ASR and TASR among
proxy-hit images separately. Results are written to
`outputs/pull_push_multiclass_v4_scale50_diverse10/`.

After the independent reserve-8 parameter search, run the frozen tuned
confirmation without overwriting the original 50-image states:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_scale50_tuned.sh
```

This reuses the exact same common-clean 50-image cohort and fixed references.
It writes separate `states_scale50_tuned` and `tuned_pull_push` artifacts. The
frozen settings are P14 `rho=.5, tau=.2, pull:push=1:.5`, P16
`rho=1, tau=.1, 1:.25`, and P19 `rho=.5, tau=.1, 1:.5`. Its summary is
`summaries/scale50_tuned_results.csv`. The original layers 17/17/15 remain
fixed because the later layer search covered only T02/T10, not all transitions.

After the tuned 50-image confirmation finishes, run the post-hoc correlation
analysis:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_cka_correlations.sh
```

It extracts projected representations for the same 480 disjoint reference
images in each proxy and target. For every transition it reports proxy- and
target-side prototype distance plus class-conditioned CKA on the combined 48
source and 48 target images. CKA is calibrated against 1,000 shuffled-image
permutations. Correlations use the 30 tuned 50-image cells and include a
within-pair stratified permutation analysis. Target representations are used
only after all adversarial PNGs and target evaluations are frozen.

For target-side teacher-forced closed-set difficulty and finite attack effects:

```bash
bash experiments/2026-08-pull-push-multiclass-v4/run_decision_margins.sh
```

This adds clean/adversarial target-vs-source and robust margins, target
probability/rank, entropy, margin change, gap closure, and closed-set boundary
crossings. These are post-hoc diagnostics and are never exposed during attack
generation.

The first phase compares both losses and both schedules on three fixed
transitions. The second phase applies both small-step objectives to all ten
transitions. Use `V4_PAIRS`, `V4_TRANSITIONS`, or `V4_ARMS` to narrow a run.

Focused tests:

```bash
python -m unittest discover \
  -s experiments/2026-08-pull-push-multiclass-v4/tests
```

## Outputs

Results are written to `outputs/pull_push_multiclass_v4_diverse10/`:

- `diagnostics/prototype_distances.csv`
- `diagnostics/prototype_distances.json`
- `states/<pair>/<transition>/<arm>.json`
- `summaries/results.csv`
- scale50: `summaries/scale50_results.csv`

All conclusions from eight-image transition cells are diagnostic. The ten
semantically diverse transitions reduce the single-transition and
vehicle-only confounds but do not by themselves establish population-level
ImageNet transfer rates.
