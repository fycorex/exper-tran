import argparse
import csv
import io
import json
from pathlib import Path

from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    rows = []
    for pair in MODEL_PAIRS:
        path = (
            args.scale_root
            / pair.pair_id
            / "scale_8"
            / "summaries"
            / "scale_8_results.csv"
        )
        candidates = []
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                candidates = list(csv.DictReader(handle))
        valid = [row for row in candidates if row.get("status") == "ok"]
        row = valid[-1] if valid else (candidates[-1] if candidates else {})
        target_hits = row.get("targeted_hit_count", "")
        proxy_hits = row.get("proxy_target_hit_count", "")
        denominator = row.get("clean_valid_count", "")
        review = {
            "pair_id": pair.pair_id,
            "type": pair.exp_type.value,
            "proxy_model": pair.proxy_model,
            "target_model": pair.target_model,
            "status": row.get("status", "missing"),
            "proxy_hits": proxy_hits,
            "target_hits": target_hits,
            "denominator": denominator,
            "tasr_percent": row.get("tasr_percent", ""),
            "lambda": row.get("lambda", ""),
            "alpha": row.get("cka_target_weight", ""),
            "reference_cka_gain": row.get("reference_cka_gain", ""),
            "passes_proxy_gate": str(proxy_hits) == "8",
            "passes_3_of_8_review": (
                str(denominator) == "8"
                and str(target_hits).isdigit()
                and int(target_hits) >= 3
            ),
            "failure_reason": row.get("failure_reason", "no selected scale-8 run"),
        }
        rows.append(review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(args.output, stream.getvalue())
    atomic_text_write(
        args.output.with_suffix(".json"),
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )
    passed = sum(row["passes_3_of_8_review"] for row in rows)
    print(f"scale8_review_passed={passed}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
