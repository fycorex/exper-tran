# Pull+Push versus Multiclass V4

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

- A full ImageNet-1K-style tree at `IMAGENET_ROOT`, with both `train/<WNID>`
  and `val/<WNID>` directories for every class listed above. The repository's
  local `data/imagenet_vehicle_official` subset is insufficient for this
  diverse-class experiment.
- Pinned local model snapshots under `.hf-cache`.
- Existing `.venv-primary-ml-cka` environment.
- One CUDA GPU; the scripts refuse CPU attack execution.

## Reproduction

Prepare canonical data and pair-specific clean-valid eight-image cohorts:

```bash
export IMAGENET_ROOT=/path/to/ILSVRC2012

PYTHONPATH=src .venv-primary-ml-cka/bin/python \
  experiments/2026-08-pull-push-multiclass-v4/src/prepare_data.py

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

All conclusions from eight-image transition cells are diagnostic. The ten
semantically diverse transitions reduce the single-transition and
vehicle-only confounds but do not by themselves establish population-level
ImageNet transfer rates.
