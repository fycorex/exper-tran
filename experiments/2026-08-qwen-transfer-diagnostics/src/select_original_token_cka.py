#!/usr/bin/env python3
"""Promote promising original token-CKA settings through a staged search."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


PAIR_IDS = ("P20", "P21", "P22")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--stage", choices=("scan-to-mid", "mid-to-full", "full-to-48"), required=True)
    return parser.parse_args()


def number(row: dict[str, str], key: str, default: float = float("-inf")) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def truth(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes"}


def signature(row: dict[str, str]) -> tuple[float, float]:
    return (number(row, "rho"), number(row, "alpha"))


def unique_top(
    rows: list[dict[str, str]],
    scorers: list,
    count: int,
) -> list[dict[str, str]]:
    chosen: list[dict[str, str]] = []
    seen: set[tuple[float, float]] = set()
    for scorer in scorers:
        for row in sorted(rows, key=scorer, reverse=True):
            key = signature(row)
            if key not in seen:
                chosen.append(row)
                seen.add(key)
                break
        if len(chosen) >= count:
            return chosen
    for row in sorted(rows, key=scorers[0], reverse=True):
        key = signature(row)
        if key not in seen:
            chosen.append(row)
            seen.add(key)
        if len(chosen) >= count:
            break
    return chosen


def scan_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    mechanism = [
        row
        for row in rows
        if truth(row, "source_repulsion_achieved")
        and truth(row, "target_attraction_achieved")
    ]
    pool = mechanism or rows
    scorers = [
        lambda r: (
            number(r, "proxy_hits", -1),
            number(r, "free_generation_hits", -1),
            number(r, "reference_cka_gain"),
            number(r, "source_cka_drop"),
        ),
        lambda r: (
            number(r, "reference_cka_gain"),
            number(r, "proxy_hits", -1),
            number(r, "source_cka_drop"),
        ),
        lambda r: (
            number(r, "proxy_min_probability"),
            number(r, "proxy_min_margin"),
            number(r, "reference_cka_gain"),
        ),
    ]
    return unique_top(pool, scorers, 3)


def transfer_candidates(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    cka_rows = [row for row in rows if row.get("objective", "") != "classification_only"]
    mechanism = [
        row
        for row in cka_rows
        if truth(row, "source_repulsion_achieved")
        and truth(row, "target_attraction_achieved")
    ]
    pool = mechanism or cka_rows
    scorers = [
        lambda r: (
            number(r, "proxy_hits", -1) == 8,
            number(r, "free_generation_hits", -1) == 8,
            number(r, "target_hits", -1),
            number(r, "target_hits_among_proxy_hits", -1),
            number(r, "proxy_hits", -1),
            number(r, "reference_cka_gain"),
        ),
        lambda r: (
            number(r, "target_hits", -1),
            number(r, "asr_percent", -1),
            number(r, "proxy_hits", -1),
            number(r, "reference_cka_gain"),
        ),
        lambda r: (
            number(r, "proxy_min_probability"),
            number(r, "proxy_min_margin"),
            number(r, "reference_cka_gain"),
        ),
    ]
    return unique_top(pool, scorers, count)


def trial(pair_id: str, row: dict[str, str], tag: str, reference_count: int) -> dict[str, object]:
    return {
        "pair": pair_id,
        "prompt": "original",
        "objective": tag,
        "lambda": 1,
        "rho": number(row, "rho"),
        "source_weight": 1,
        "alpha": number(row, "alpha"),
        "beta": 0,
        "reference_count": reference_count,
        "target_cka_mode": "spatial_index_legacy",
    }


def baseline(pair_id: str, tag: str, reference_count: int) -> dict[str, object]:
    return {
        "pair": pair_id,
        "prompt": "original",
        "objective": tag,
        "lambda": 0,
        "source_weight": 0,
        "alpha": 0,
        "beta": 0,
        "reference_count": reference_count,
        "target_cka_mode": "spatial_index_legacy",
    }


def main() -> None:
    args = arguments()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        source = [row for row in csv.DictReader(stream) if row.get("status") == "complete"]

    chosen: dict[str, list[dict[str, str]]] = {}
    trials: list[dict[str, object]] = []
    if args.stage == "scan-to-mid":
        diagnostics_name = "original_token_cka_mid_transfer"
        steps = 30
        reference_count = 8
        for pair_id in PAIR_IDS:
            rows = [row for row in source if row["pair_id"] == pair_id]
            if len(rows) != 20:
                raise RuntimeError(f"Expected 20 scan rows for {pair_id}; got {len(rows)}")
            chosen[pair_id] = scan_candidates(rows)
            trials.append(baseline(pair_id, "classification_only", reference_count))
            for index, row in enumerate(chosen[pair_id], start=1):
                trials.append(trial(pair_id, row, f"original_cka_mid_{index}", reference_count))
    elif args.stage == "mid-to-full":
        diagnostics_name = "original_token_cka_full100"
        steps = 100
        reference_count = 8
        for pair_id in PAIR_IDS:
            rows = [row for row in source if row["pair_id"] == pair_id]
            chosen[pair_id] = transfer_candidates(rows, 2)
            if len(chosen[pair_id]) != 2:
                raise RuntimeError(f"Could not select two full candidates for {pair_id}")
            trials.append(baseline(pair_id, "classification_only", reference_count))
            for index, row in enumerate(chosen[pair_id], start=1):
                trials.append(trial(pair_id, row, f"original_cka_full_{index}", reference_count))
    else:
        diagnostics_name = "original_token_cka_reference48"
        steps = 100
        reference_count = 48
        for pair_id in PAIR_IDS:
            rows = [row for row in source if row["pair_id"] == pair_id]
            chosen[pair_id] = transfer_candidates(rows, 1)
            if len(chosen[pair_id]) != 1:
                raise RuntimeError(f"Could not select a 48-reference candidate for {pair_id}")
            trials.append(trial(pair_id, chosen[pair_id][0], "original_cka_reference48", reference_count))

    config = {
        "diagnostics_name": diagnostics_name,
        "phase": diagnostics_name,
        "seed": 42,
        "steps": steps,
        "early_stop_proxy_gate": False,
        "materialize_selection": False,
        "evaluate_target": True,
        "require_auxiliary_progress": False,
        "common_clean": True,
        "common_clean_pairs": list(PAIR_IDS),
        "image_count": 8,
        "reference_count": reference_count,
        "trials": trials,
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    args.output_config.with_suffix(".selection.json").write_text(
        json.dumps(chosen, indent=2) + "\n", encoding="utf-8"
    )
    for pair_id, rows in chosen.items():
        settings = ", ".join(
            f"rho={row['rho']} alpha={row['alpha']} target_gain={row['reference_cka_gain']}"
            for row in rows
        )
        print(f"{pair_id}: {settings}", flush=True)
    print(args.output_config, flush=True)


if __name__ == "__main__":
    main()
