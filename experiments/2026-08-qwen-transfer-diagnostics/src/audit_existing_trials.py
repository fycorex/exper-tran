"""Summarize completed, proxy-eligible eight-image diagnostic trials.

This is descriptive only. Trials reuse images and vary prompts/objectives, so
the correlations are audit signals rather than independent estimates.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "reference_cka_gain",
    "source_cka_drop",
    "proxy_min_margin",
    "proxy_min_probability",
)


def _rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    index = 0
    while index < len(array):
        stop = index + 1
        while stop < len(array) and array[order[stop]] == array[order[index]]:
            stop += 1
        ranks[order[index:stop]] = (index + stop - 1) / 2 + 1
        index = stop
    return ranks


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    project_root = Path.cwd()
    diagnostics = project_root / "outputs/proxy_selector_cka_v2/diagnostics"
    eligible = []
    for path in sorted(diagnostics.glob("*/summary.csv")):
        # The new projected-token baseline is excluded until it finishes; this
        # report audits only the historical implementation being replaced.
        if path.parent.name == "projected_tap_intra_baseline":
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("status") != "complete"
                    or row.get("proxy_hits") != "8"
                    or row.get("free_generation_hits") != "8"
                    or row.get("target_denominator") != "8"
                ):
                    continue
                numeric = dict(row)
                numeric["source"] = path.parent.name
                numeric["target_hits"] = float(row["target_hits"])
                for metric in METRICS:
                    try:
                        numeric[metric] = float(row[metric])
                    except (KeyError, TypeError, ValueError):
                        numeric[metric] = math.nan
                eligible.append(numeric)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row["pair_id"])].append(row)
    per_pair = []
    for pair_id, rows in sorted(grouped.items()):
        hits = [float(row["target_hits"]) for row in rows]
        reference = [float(row["reference_cka_gain"]) for row in rows]
        per_pair.append(
            {
                "pair_id": pair_id,
                "eligible_trials": len(rows),
                "mean_target_hits_of_8": round(sum(hits) / len(hits), 3),
                "best_target_hits_of_8": int(max(hits)),
                "reference_gain_spearman": (
                    round(value, 3)
                    if (value := _correlation(_rankdata(reference), _rankdata(hits)))
                    is not None
                    else None
                ),
            }
        )

    correlations = {}
    target_hits = [float(row["target_hits"]) for row in eligible]
    for metric in METRICS:
        pairs = [
            (float(row[metric]), float(row["target_hits"]))
            for row in eligible
            if math.isfinite(float(row[metric]))
        ]
        x, y = (list(values) for values in zip(*pairs, strict=True))
        correlations[metric] = {
            "n": len(pairs),
            "pearson": round(_correlation(x, y), 3),
            "spearman": round(_correlation(_rankdata(x), _rankdata(y)), 3),
        }

    payload = {
        "scope": "historical pre-projector/mixed-tap diagnostic trials only",
        "eligibility": "status complete; proxy closed-set 8/8; proxy free-generation 8/8; target denominator 8",
        "eligible_trials": len(eligible),
        "target_hits": int(sum(target_hits)),
        "target_opportunities": 8 * len(eligible),
        "pooled_correlations": correlations,
        "per_pair": per_pair,
        "limitations": (
            "Repeated images, heterogeneous prompts/objectives, and pair confounding; "
            "correlations are descriptive diagnostics, not inferential evidence."
        ),
    }
    output = diagnostics / "existing_method_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
