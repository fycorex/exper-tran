#!/usr/bin/env python3
"""Summarize all completed V4 pair/transition/arm states."""

import argparse
import csv
import json
from pathlib import Path

from common import DEFAULT_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.output_dir / "states").glob("*/*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        attack = payload["attack"]
        rows.append(
            {
                "pair_id": payload["pair_id"],
                "transition_id": payload["transition_id"],
                "source_human_label": payload["source_human_label"],
                "target_human_label": payload["target_human_label"],
                "arm": payload["arm"],
                "semantic_mode": payload["semantic_mode"],
                "steps": payload["steps"],
                "step_size": payload["step_size"],
                "rho": payload["rho"],
                "proxy_hits": attack["proxy_target_hit_count"],
                "proxy_denominator": attack["proxy_target_hit_denominator"],
                "tasr_hits": payload["tasr_hits"],
                "tasr_denominator": payload["clean_valid_count"],
                "tasr_percent": payload["tasr_percent"],
                "asr_hits": payload["asr_hits"],
                "asr_percent": payload["asr_percent"],
                "semantic_gap_clean": attack["semantic_gap_clean"],
                "semantic_gap_adversarial": attack["semantic_gap_adversarial"],
                "semantic_gap_gain": attack["semantic_gap_gain"],
                "effective_lambda": attack["effective_lambda_cka"],
                "elapsed_seconds": attack["elapsed_seconds"],
            }
        )
    summary = args.output_dir / "summaries" / "results.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No completed V4 states were found")
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(summary.resolve(), flush=True)


if __name__ == "__main__":
    main()
