import argparse
import json
from pathlib import Path

import yaml

from primary_ml_cka.config.loader import load_config
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    selected = json.loads(args.selected.read_text(encoding="utf-8"))
    base = load_config(args.base_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_id, trial in sorted(selected.items()):
        config = dict(base)
        config["lambdas"] = [float(trial["lambda_cka"])]
        config["cka_target_weight"] = float(trial["alpha"])
        config_path = args.output_dir / f"{pair_id}.yaml"
        atomic_text_write(
            config_path,
            yaml.safe_dump(config, sort_keys=False),
        )
        rows.append(
            "\t".join(
                (
                    pair_id,
                    str(trial["prompt_id"]),
                    config_path.resolve().as_posix(),
                )
            )
        )
    if not rows:
        raise RuntimeError(
            "No pair passed the strict proxy and target-denominator selection gate"
        )
    atomic_text_write(
        args.output_dir / "selected_runs.tsv",
        "pair_id\tprompt_id\tconfig_path\n" + "\n".join(rows) + "\n",
    )
    print(f"materialized_pairs={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
