#!/usr/bin/env python3
"""Export the historical token-level contrastive CKA evidence bundle."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "proxy_selector_cka_v2" / "diagnostics"
DESTINATION = ROOT / "results" / "token_contrastive_cka"

PROJECTED_COLUMNS = (
    "pair_id",
    "prompt_id",
    "lambda_cka",
    "alpha",
    "proxy_hits",
    "proxy_denominator",
    "target_hits",
    "target_denominator",
    "tasr_percent",
    "untargeted_hits",
    "cka_adv_source",
    "cka_adv_reference",
    "reference_cka_gain",
    "source_cka_drop",
    "status",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_projected_baseline() -> dict[str, dict[str, int]]:
    source = OUTPUT / "projected_tap_intra_baseline" / "summary.csv"
    rows = read_rows(source)
    selected = [{column: row[column] for column in PROJECTED_COLUMNS} for row in rows]
    write_rows(DESTINATION / "projected_tap_intra_trials.csv", PROJECTED_COLUMNS, selected)

    summary: dict[str, dict[str, int]] = {}
    for pair_id in sorted({row["pair_id"] for row in rows}):
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        baseline = next(row for row in pair_rows if float(row["lambda_cka"]) == 0)
        contrastive = [row for row in pair_rows if float(row["lambda_cka"]) > 0]
        summary[pair_id] = {
            "classification_only_target_hits": int(baseline["target_hits"]),
            "best_contrastive_target_hits": max(int(row["target_hits"]) for row in contrastive),
            "denominator": int(baseline["target_denominator"]),
            "contrastive_trial_count": len(contrastive),
        }
    return summary


def export_historical_sweep() -> dict[str, object]:
    source = OUTPUT / "pair_prompt_sweep_v2" / "summary.csv"
    rows = [
        row
        for row in read_rows(source)
        if float(row["lambda_cka"]) > 0
        and row["target_denominator"]
        and int(row["proxy_hits"]) == int(row["proxy_denominator"]) == 8
    ]
    fieldnames = (
        "pair_id",
        "strict_evaluated_trials",
        "zero_hit_trials",
        "at_most_one_hit_trials",
        "at_least_two_hit_trials",
        "best_target_hits",
        "denominator",
    )
    aggregate: list[dict[str, object]] = []
    for pair_id in sorted({row["pair_id"] for row in rows}):
        hits = [int(row["target_hits"]) for row in rows if row["pair_id"] == pair_id]
        aggregate.append(
            {
                "pair_id": pair_id,
                "strict_evaluated_trials": len(hits),
                "zero_hit_trials": sum(value == 0 for value in hits),
                "at_most_one_hit_trials": sum(value <= 1 for value in hits),
                "at_least_two_hit_trials": sum(value >= 2 for value in hits),
                "best_target_hits": max(hits),
                "denominator": 8,
            }
        )
    write_rows(DESTINATION / "historical_sweep_by_pair.csv", fieldnames, aggregate)
    counts = Counter(int(row["target_hits"]) for row in rows)
    return {
        "strict_evaluated_trials": len(rows),
        "zero_hit_trials": sum(value == 0 for value in map(int, (r["target_hits"] for r in rows))),
        "at_most_one_hit_trials": sum(
            value <= 1 for value in map(int, (r["target_hits"] for r in rows))
        ),
        "at_least_two_hit_trials": sum(
            value >= 2 for value in map(int, (r["target_hits"] for r in rows))
        ),
        "best_target_hits": max(counts),
        "hit_distribution": {str(key): counts[key] for key in sorted(counts)},
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    summary = {
        "loss_definition": "L_cls + lambda * (CKA(adv, clean) - alpha * CKA(adv, target))",
        "observation_level": "per-image spatial tokens",
        "projected_tap_intra_baseline": export_projected_baseline(),
        "historical_pair_prompt_sweep": export_historical_sweep(),
        "excluded": [
            "target-only token CKA trials with source_weight=0",
            "semantic-centroid trials",
            "batch-level pooled CKA trials",
        ],
    }
    (DESTINATION / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in sorted(DESTINATION.glob("*")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
