import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from primary_ml_cka.infrastructure.atomic_io import atomic_text_write


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    relative_path: Path
    human_label: int
    class_name: str
    synset: str


def write_manifest(path: Path, records: Iterable[ImageRecord]) -> None:
    lines = []
    for record in records:
        payload = asdict(record)
        payload["relative_path"] = record.relative_path.as_posix()
        lines.append(json.dumps(payload, sort_keys=True))
    atomic_text_write(path, "".join(f"{line}\n" for line in lines))


def read_manifest(path: Path) -> tuple[ImageRecord, ...]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                payload["relative_path"] = Path(payload["relative_path"])
                records.append(ImageRecord(**payload))
    return tuple(records)
