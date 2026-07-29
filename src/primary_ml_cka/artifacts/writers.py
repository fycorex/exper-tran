import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path

from primary_ml_cka.artifacts.schemas import ALL_RESULTS_COLUMNS, ResultRow
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write


def write_json(path: Path, value: object) -> None:
    payload = asdict(value) if is_dataclass(value) else value
    atomic_text_write(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_results_csv(path: Path, rows: Iterable[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ALL_RESULTS_COLUMNS)
        for row in rows:
            writer.writerow(row.values)
    temporary.replace(path)
