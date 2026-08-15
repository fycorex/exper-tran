#!/usr/bin/env python3
"""Validate and summarize the three-pair full CKA smoke."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAIR_IDS = ("P20", "P21", "P22")
FIELDS = (
    "pair_id",
    "rho",
    "alpha",
    "effective_lambda_cka",
    "proxy_hits",
    "proxy_denominator",
    "free_generation_hits",
    "target_hits",
    "target_denominator",
    "tasr_percent",
    "untargeted_hits",
    "asr_percent",
    "cka_adv_source",
    "cka_adv_reference",
    "source_cka_drop",
    "reference_cka_gain",
    "source_repulsion_achieved",
    "target_attraction_achieved",
    "elapsed_seconds",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> None:
    args = arguments()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows = []
    for pair_id in PAIR_IDS:
        candidates = [
            row for row in source if row["pair_id"] == pair_id and row["status"] == "complete"
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one complete row for {pair_id}; got {len(candidates)}")
        row = candidates[0]
        if int(row["target_denominator"]) != 8:
            raise RuntimeError(f"{pair_id} target denominator is not 8")
        if not (
            is_true(row["source_repulsion_achieved"])
            and is_true(row["target_attraction_achieved"])
        ):
            raise RuntimeError(f"{pair_id} did not achieve both CKA components")
        rows.append({field: row[field] for field in FIELDS})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['pair_id']}: proxy={row['proxy_hits']}/8 "
            f"TASR={row['target_hits']}/8 ASR={row['untargeted_hits']}/8 "
            f"source_drop={float(row['source_cka_drop']):+.4f} "
            f"target_gain={float(row['reference_cka_gain']):+.4f}",
            flush=True,
        )
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
