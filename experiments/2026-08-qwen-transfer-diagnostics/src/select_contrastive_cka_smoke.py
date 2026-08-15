#!/usr/bin/env python3
"""Select one mechanism-valid rho/alpha setting per intra-family pair."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


PAIR_IDS = ("P20", "P21", "P22")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-summary", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def score(row: dict[str, str]) -> tuple[float, ...]:
    return (
        float(row["proxy_hits"]),
        float(row["free_generation_hits"]),
        float(row["reference_cka_gain"]),
        float(row["source_cka_drop"]),
        -float(row["rho"]),
        -float(row["alpha"]),
    )


def main() -> None:
    args = arguments()
    with args.scan_summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    selected: dict[str, dict[str, str]] = {}
    missing = []
    for pair_id in PAIR_IDS:
        candidates = [
            row
            for row in rows
            if row["pair_id"] == pair_id
            and row["status"] == "complete"
            and is_true(row["source_repulsion_achieved"])
            and is_true(row["target_attraction_achieved"])
        ]
        if not candidates:
            missing.append(pair_id)
            continue
        selected[pair_id] = max(candidates, key=score)
    if missing:
        raise RuntimeError(
            "No source-away/target-toward setting for: " + ", ".join(missing)
        )

    trials = []
    for pair_id in PAIR_IDS:
        row = selected[pair_id]
        alpha = float(row["alpha"])
        rho = float(row["rho"])
        trials.append(
            {
                "pair": pair_id,
                "prompt": "original",
                "objective": f"selected_clean_anchor_a{alpha:g}",
                "lambda": 1,
                "rho": rho,
                "source_weight": 1,
                "alpha": alpha,
                "beta": 0,
                "target_cka_mode": "clean_anchor_soft",
                "target_alignment_temperature": 0.07,
            }
        )

    config = {
        "diagnostics_name": "contrastive_cka_three_pair_full_smoke",
        "phase": "contrastive_cka_three_pair_full_smoke",
        "seed": 42,
        "steps": args.steps,
        "early_stop_proxy_gate": False,
        "materialize_selection": False,
        "require_auxiliary_progress": True,
        "evaluate_target": True,
        "common_clean": True,
        "common_clean_pairs": list(PAIR_IDS),
        "image_count": 8,
        "reference_count": 8,
        "trials": trials,
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    selection_path = args.output_config.with_suffix(".selection.json")
    selection_path.write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    for pair_id in PAIR_IDS:
        row = selected[pair_id]
        print(
            f"{pair_id}: rho={row['rho']} alpha={row['alpha']} "
            f"proxy={row['proxy_hits']}/8 source_drop={row['source_cka_drop']} "
            f"target_gain={row['reference_cka_gain']}",
            flush=True,
        )
    print(args.output_config, flush=True)


if __name__ == "__main__":
    main()
