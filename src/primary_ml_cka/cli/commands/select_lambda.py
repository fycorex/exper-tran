import csv
from collections import defaultdict
from statistics import mean

from primary_ml_cka.experiment.orchestration import CommandContext
from primary_ml_cka.reporting.summaries import (
    LambdaCandidate,
    select_positive_lambda,
    write_selected,
)


def run(context: CommandContext) -> str:
    path = context.output_dir / "summaries" / "all_results.csv"
    if context.dry_run:
        return f"dry-run: select positive lambda from representation-only columns in {path}"
    grouped_rows: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] == "main":
                grouped_rows[(row["pair_id"], float(row["lambda"]))].append(row)
    candidates_by_pair: dict[str, list[LambdaCandidate]] = defaultdict(list)
    for (pair_id, lambda_cka), rows in grouped_rows.items():
        candidates_by_pair[pair_id].append(
            LambdaCandidate(
                pair_id,
                lambda_cka,
                mean(float(row["proxy_representation_shift"]) for row in rows),
                mean(float(row["reference_cka_gain"]) for row in rows),
                mean(float(row["source_cka_drop"]) for row in rows),
                mean(float(row["proxy_target_nll"]) for row in rows),
            )
        )
    selected = tuple(select_positive_lambda(values) for values in candidates_by_pair.values())
    write_selected(context.output_dir / "summaries" / "selected_lambda.json", selected)
    return f"selected positive lambdas for {len(selected)} pairs"
