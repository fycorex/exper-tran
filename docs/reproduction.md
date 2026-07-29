# Reproduction

## Environment

```bash
source .venv-primary-ml-cka/bin/activate
export HF_HOME="$PWD/.hf-cache"
export IMAGENET_ROOT="$PWD/data/imagenet_vehicle_official"
```

All tensor computation requires CUDA. Commands fail instead of falling back to
CPU.

## Validation and Execution

```bash
bash scripts/run_experiment.sh data prepare
bash scripts/run_experiment.sh target screen
bash scripts/run_experiment.sh tests run
bash scripts/run_experiment.sh models inspect-taps
bash scripts/run_experiment.sh attack smoke
bash scripts/run_experiment.sh attack main
bash scripts/run_experiment.sh selection lambda
bash scripts/run_experiment.sh evaluation main
bash scripts/run_experiment.sh attack confirm
bash scripts/run_experiment.sh report summarize
```

Use `--pair-id P06` to restrict a stage and `--dry-run` to validate orchestration
without loading a model. `--config` selects an alternate attack YAML; batch size
and canvas size are read at runtime.

Main attacks require a passing CUDA test report, a validated proxy tap, a
clean-screened manifest, and a successful pair smoke result. Blocked pairs are
serialized with exact errors. Never interpret missing or partial rows as
successful results.
