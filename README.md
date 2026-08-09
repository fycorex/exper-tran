# Per-image Proxy CKA + Proxy-Selector Transfer

Reproducible targeted image-attack experiment using a proxy-only classification
objective and per-image token-level proxy CKA. Attack generation is proxy-only;
target representations are used solely by a separate post-attack global/local
CKA analysis that tests whether representation similarity predicts conditional
TASR. The six fixed model pairs and all attack constants live under `configs/`.

The attack source and target classes are not embedded in the implementation.
Set `source_human_label` and `target_human_label` in
`configs/data/imagenet_vehicle10.yaml`; every data, loss, and evaluation stage
uses those values.

## Local assets

- ImageNet vehicle subset: `data/imagenet_vehicle_official`
- Hugging Face cache: `.hf-cache/hub`
- Python environment: `.venv-primary-ml-cka`
- Generated artifacts: `outputs/primary_ml_cka_v1`

Both data and model weights are intentionally excluded from Git.

## Setup and checks

```bash
python3.11 -m venv .venv-primary-ml-cka
source .venv-primary-ml-cka/bin/activate
pip install -e '.[test]'
export HF_HOME="$PWD/.hf-cache"
export IMAGENET_ROOT="$PWD/data/imagenet_vehicle_official"
bash scripts/run_experiment.sh tests run
bash scripts/run_experiment.sh run all --dry-run
```

Run `python -m primary_ml_cka.cli.main --help` for all stages. Real attack
generation is gated on passing correctness tests and validated proxy-tap
records. See `docs/reproduction.md` before starting GPU work.
