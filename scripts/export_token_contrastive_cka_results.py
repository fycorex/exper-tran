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

TRIAL_COLUMNS = (
    "pair_id",
    "prompt_id",
    "objective",
    "lambda_cka",
    "source_cka_weight",
    "target_cka_weight_alpha",
    "semantic_target_weight",
    "seed",
    "steps",
    "batch_size",
    "reference_bank_size",
    "epsilon_linf",
    "step_size",
    "momentum",
    "proxy_hits",
    "proxy_denominator",
    "free_generation_hits",
    "target_hits",
    "target_denominator",
    "tasr_percent",
    "untargeted_hits",
    "asr_percent",
    "proxy_min_margin",
    "proxy_min_probability",
    "cka_adv_source",
    "cka_adv_reference",
    "reference_cka_gain",
    "source_cka_drop",
    "grad_ml_l1",
    "grad_cka_weighted_l1",
    "grad_component_cosine",
    "elapsed_seconds",
    "status",
)

PROTOCOL = {
    "seed": 42,
    "steps": 100,
    "batch_size": 8,
    "reference_bank_size": 8,
    "epsilon_linf": 16 / 255,
    "step_size": 1 / 255,
    "momentum": 1.0,
    "random_start": True,
    "canvas_size": 224,
    "class_margin": 2.0,
    "proxy_probability_threshold": 0.9,
    "require_proxy_free_generation": True,
    "gradient_ratio": None,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def enriched_trial(row: dict[str, str]) -> dict[str, object]:
    denominator = int(row["target_denominator"]) if row["target_denominator"] else 0
    untargeted_hits = int(row["untargeted_hits"]) if row["untargeted_hits"] else 0
    lambda_cka = float(row["lambda_cka"])
    return {
        "pair_id": row["pair_id"],
        "prompt_id": row["prompt_id"],
        "objective": "classification_only" if lambda_cka == 0 else "token_contrastive_cka",
        "lambda_cka": row["lambda_cka"],
        "source_cka_weight": 1.0,
        "target_cka_weight_alpha": row["alpha"],
        "semantic_target_weight": 0.0,
        "seed": PROTOCOL["seed"],
        "steps": PROTOCOL["steps"],
        "batch_size": PROTOCOL["batch_size"],
        "reference_bank_size": PROTOCOL["reference_bank_size"],
        "epsilon_linf": PROTOCOL["epsilon_linf"],
        "step_size": PROTOCOL["step_size"],
        "momentum": PROTOCOL["momentum"],
        "proxy_hits": row["proxy_hits"],
        "proxy_denominator": row["proxy_denominator"],
        "free_generation_hits": row["free_generation_hits"],
        "target_hits": row["target_hits"],
        "target_denominator": row["target_denominator"],
        "tasr_percent": row["tasr_percent"],
        "untargeted_hits": row["untargeted_hits"],
        "asr_percent": 100.0 * untargeted_hits / denominator if denominator else "",
        "proxy_min_margin": row["proxy_min_margin"],
        "proxy_min_probability": row["proxy_min_probability"],
        "cka_adv_source": row["cka_adv_source"],
        "cka_adv_reference": row["cka_adv_reference"],
        "reference_cka_gain": row["reference_cka_gain"],
        "source_cka_drop": row["source_cka_drop"],
        "grad_ml_l1": row["grad_ml_l1"],
        "grad_cka_weighted_l1": row["grad_cka_weighted_l1"],
        "grad_component_cosine": row["grad_component_cosine"],
        "elapsed_seconds": row["elapsed_seconds"],
        "status": row["status"],
    }


def export_projected_baseline() -> dict[str, dict[str, object]]:
    source = OUTPUT / "projected_tap_intra_baseline" / "summary.csv"
    rows = read_rows(source)
    selected = [enriched_trial(row) for row in rows]
    write_rows(DESTINATION / "projected_tap_intra_trials.csv", TRIAL_COLUMNS, selected)

    summary: dict[str, dict[str, object]] = {}
    for pair_id in sorted({row["pair_id"] for row in rows}):
        pair_rows = [row for row in rows if row["pair_id"] == pair_id]
        baseline = next(row for row in pair_rows if float(row["lambda_cka"]) == 0)
        contrastive = [row for row in pair_rows if float(row["lambda_cka"]) > 0]
        best = max(
            contrastive,
            key=lambda row: (int(row["target_hits"]), int(row["untargeted_hits"])),
        )
        summary[pair_id] = {
            "classification_only_target_hits": int(baseline["target_hits"]),
            "classification_only_untargeted_hits": int(baseline["untargeted_hits"]),
            "classification_only_tasr_percent": float(baseline["tasr_percent"]),
            "classification_only_asr_percent": float(
                enriched_trial(baseline)["asr_percent"]
            ),
            "best_contrastive_target_hits": int(best["target_hits"]),
            "best_contrastive_untargeted_hits": int(best["untargeted_hits"]),
            "best_contrastive_tasr_percent": float(best["tasr_percent"]),
            "best_contrastive_asr_percent": float(enriched_trial(best)["asr_percent"]),
            "best_contrastive_lambda_cka": float(best["lambda_cka"]),
            "best_contrastive_target_weight_alpha": float(best["alpha"]),
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
        and int(row["target_denominator"] or 0) == 8
        and int(row["proxy_hits"]) == int(row["proxy_denominator"]) == 8
    ]
    write_rows(
        DESTINATION / "historical_strict_trials.csv",
        TRIAL_COLUMNS,
        [enriched_trial(row) for row in rows],
    )
    fieldnames = (
        "pair_id",
        "strict_evaluated_trials",
        "zero_hit_trials",
        "at_most_one_hit_trials",
        "at_least_two_hit_trials",
        "best_target_hits",
        "mean_tasr_percent",
        "mean_asr_percent",
        "maximum_asr_percent",
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
                "mean_tasr_percent": sum(12.5 * value for value in hits) / len(hits),
                "mean_asr_percent": sum(
                    float(enriched_trial(row)["asr_percent"])
                    for row in rows
                    if row["pair_id"] == pair_id
                )
                / len(hits),
                "maximum_asr_percent": max(
                    float(enriched_trial(row)["asr_percent"])
                    for row in rows
                    if row["pair_id"] == pair_id
                ),
                "denominator": 8,
            }
        )
    write_rows(DESTINATION / "historical_sweep_by_pair.csv", fieldnames, aggregate)
    counts = Counter(int(row["target_hits"]) for row in rows)
    asr_values = [float(enriched_trial(row)["asr_percent"]) for row in rows]
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
        "mean_tasr_percent": sum(float(row["tasr_percent"]) for row in rows) / len(rows),
        "mean_asr_percent": sum(asr_values) / len(asr_values),
        "maximum_asr_percent": max(asr_values),
        "hit_distribution": {str(key): counts[key] for key in sorted(counts)},
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    summary = {
        "loss_definition": "L_cls + lambda * (CKA(adv, clean) - alpha * CKA(adv, target))",
        "observation_level": "per-image spatial tokens",
        "protocol": PROTOCOL,
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
