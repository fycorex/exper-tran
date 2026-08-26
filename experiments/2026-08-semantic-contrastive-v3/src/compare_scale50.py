#!/usr/bin/env python3
"""Join old and new 50-image aggregate tables without mixing artifacts."""

import argparse
import csv
from pathlib import Path


def read_rows(path: Path, prefix: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["pair_id"]: row for row in csv.DictReader(handle)}
    return {pair: {f"{prefix}_{key}": value for key, value in row.items()} for pair, row in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--new-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = read_rows(args.old_output / "summaries/scale_50_semantic_all9.csv", "old")
    new = read_rows(args.new_output / "summaries/scale_50_semantic_all9.csv", "new")
    pairs = sorted(set(old) | set(new))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id"]
    fields.extend(sorted(next(iter(old.values()), {}).keys()))
    fields.extend(sorted(next(iter(new.values()), {}).keys()))
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"pair_id": pair, **old.get(pair, {}), **new.get(pair, {})})
    print(f"wrote {args.output} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
