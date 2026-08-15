#!/usr/bin/env python3
"""Merge controlled original token-CKA stages into one audit table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "stage",
    "pair_id",
    "objective",
    "rho",
    "alpha",
    "reference_count",
    "effective_lambda_cka",
    "proxy_hits",
    "proxy_denominator",
    "free_generation_hits",
    "proxy_min_margin",
    "proxy_min_probability",
    "target_hits",
    "target_denominator",
    "target_hits_among_proxy_hits",
    "proxy_hit_target_denominator",
    "tasr_percent",
    "untargeted_hits",
    "asr_percent",
    "cka_adv_source",
    "cka_adv_reference",
    "source_cka_drop",
    "reference_cka_gain",
    "source_repulsion_achieved",
    "target_attraction_achieved",
    "grad_ml_l1",
    "grad_cka_weighted_l1",
    "grad_component_cosine",
    "elapsed_seconds",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows: list[dict[str, object]] = []
    for path in args.summary:
        with path.open(newline="", encoding="utf-8") as stream:
            source = list(csv.DictReader(stream))
        for row in source:
            if row.get("status") != "complete":
                continue
            if float(row.get("beta") or 0) != 0:
                raise RuntimeError(f"Semantic loss leaked into {path}")
            if row.get("target_cka_mode") != "spatial_index_legacy":
                raise RuntimeError(f"Non-original target CKA mode in {path}")
            if row.get("target_denominator") not in {"", "8"}:
                raise RuntimeError(f"Unexpected target denominator in {path}")
            merged = {"stage": path.parent.name}
            merged.update({field: row.get(field, "") for field in FIELDS if field != "stage"})
            rows.append(merged)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "all_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "all_trials.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    for stage in sorted({str(row["stage"]) for row in rows}):
        print(stage, flush=True)
        for pair_id in ("P20", "P21", "P22"):
            candidates = [
                row for row in rows if row["stage"] == stage and row["pair_id"] == pair_id
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda row: (
                    float(row["target_hits"] or -1),
                    float(row["proxy_hits"] or -1),
                    float(row["reference_cka_gain"] or -1e9),
                ),
            )
            print(
                f"  {pair_id}: {best['objective']} proxy={best['proxy_hits']}/8 "
                f"TASR={best['target_hits']}/8 ASR={best['untargeted_hits']}/8 "
                f"target_gain={best['reference_cka_gain']}",
                flush=True,
            )
    print(csv_path, flush=True)


if __name__ == "__main__":
    main()
