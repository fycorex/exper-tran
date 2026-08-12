import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write

PAIR_IDS = ("P02", "P06", "P11", "P14", "P16", "P19", "P20", "P21", "P22")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-name", default="objective_split_all9_common48_rho03")
    parser.add_argument("--cka-result-name", default="cka_validity_all9_common48_rho03")
    parser.add_argument("--geometry-result-name", default="decision_geometry_all9_common48_rho03")
    parser.add_argument("--objective", default="semantic_only")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(path, buffer.getvalue())


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_norm = sum((x - left_mean) ** 2 for x in left) ** 0.5
    right_norm = sum((y - right_mean) ** 2 for y in right) ** 0.5
    return numerator / (left_norm * right_norm) if left_norm and right_norm else float("nan")


def _spearman(left: list[float], right: list[float]) -> float:
    return _correlation(_average_ranks(left), _average_ranks(right))


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    diagnostics = output_dir / "diagnostics" / args.diagnostics_name
    cka_dir = output_dir / "diagnostics" / args.cka_result_name
    geometry_dir = output_dir / "diagnostics" / args.geometry_result_name
    attack_rows = _read_csv(diagnostics / "summary.csv")
    null_rows = _read_csv(cka_dir / "cka_permutation_null.csv")
    local_rows = _read_csv(cka_dir / "local_cka.csv")
    geometry_rows = []
    for pair_id in PAIR_IDS:
        geometry_rows.extend(_read_csv(geometry_dir / f"{pair_id}_summary.csv"))

    projected_global = {
        row["pair_id"]: row
        for row in null_rows
        if row["proxy_layer"] == row["target_layer"] == "projected" and row["subset"] == "global"
    }
    local_by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in local_rows:
        local_by_pair[row["pair_id"]].append(row)
    attack_by_pair = {
        row["pair_id"]: row
        for row in attack_rows
        if row["objective"] == args.objective and row["status"] == "complete"
    }
    geometry_by_pair = {
        row["pair_id"]: row
        for row in geometry_rows
        if row["objective"] == args.objective and row["margin_kind"] == "robust"
    }
    missing = [
        pair_id
        for pair_id in PAIR_IDS
        if pair_id not in projected_global
        or pair_id not in local_by_pair
        or pair_id not in attack_by_pair
        or pair_id not in geometry_by_pair
    ]
    if missing:
        raise RuntimeError(f"Incomplete all-pair selector inputs: {missing}")

    rows = []
    for pair_id in PAIR_IDS:
        pair = get_pair(pair_id)
        attack = attack_by_pair[pair_id]
        geometry = geometry_by_pair[pair_id]
        local = local_by_pair[pair_id]
        rows.append(
            {
                "pair_id": pair_id,
                "pair_type": pair.exp_type.value,
                "proxy_model": pair.proxy_model,
                "target_model": pair.target_model,
                "objective": args.objective,
                "global_cka": projected_global[pair_id]["true_cka"],
                "global_cka_null_mean": projected_global[pair_id]["null_mean"],
                "global_cka_z_score": projected_global[pair_id]["z_score"],
                "mean_local_cka": sum(float(row["local_cka"]) for row in local) / len(local),
                "mean_local_cka_normalized": sum(
                    float(row["local_cka_normalized"]) for row in local
                )
                / len(local),
                "proxy_gate_type": attack["proxy_gate_type"],
                "proxy_hits": attack["proxy_hits"],
                "target_hits": attack["target_hits"],
                "tasr_percent": attack["tasr_percent"],
                "untargeted_hits": attack["untargeted_hits"],
                "mean_clean_margin": geometry["mean_clean_margin"],
                "mean_margin_change": geometry["mean_margin_change"],
                "mean_gap_closure": geometry["mean_gap_closure"],
            }
        )

    result_dir = output_dir / "diagnostics" / "selector_analysis_all9_common48_rho03"
    _write_csv(result_dir / "pair_summary.csv", rows)
    metrics = ("global_cka", "global_cka_z_score", "mean_local_cka_normalized")
    outcomes = ("tasr_percent", "mean_gap_closure")
    correlations = []
    for metric in metrics:
        for outcome in outcomes:
            correlations.append(
                {
                    "metric": metric,
                    "outcome": outcome,
                    "pair_count": len(rows),
                    "spearman": _spearman(
                        [float(row[metric]) for row in rows],
                        [float(row[outcome]) for row in rows],
                    ),
                }
            )
    _write_csv(result_dir / "spearman.csv", correlations)
    atomic_text_write(
        result_dir / "summary.json",
        json.dumps(
            {"objective": args.objective, "pairs": rows, "correlations": correlations},
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
