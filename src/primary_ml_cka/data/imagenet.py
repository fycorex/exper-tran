from pathlib import Path

from primary_ml_cka.artifacts.hashes import path_order_key
from primary_ml_cka.config.schema import DataConfig
from primary_ml_cka.data.manifests import ImageRecord
from primary_ml_cka.domain.labels import CLASS_NAMES, CLASS_SYNSETS


def _images(root: Path, split: str, human_label: int) -> tuple[ImageRecord, ...]:
    synset = CLASS_SYNSETS[human_label - 1]
    class_name = CLASS_NAMES[human_label - 1]
    directory = root / split / synset
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing ImageNet synset directory: {directory}")
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png"}
    ]
    records = [
        ImageRecord(
            image_id=path.relative_to(root).as_posix(),
            relative_path=path.relative_to(root),
            human_label=human_label,
            class_name=class_name,
            synset=synset,
        )
        for path in paths
    ]
    return tuple(sorted(records, key=lambda record: path_order_key(record.relative_path)))


def discover_vehicle_pools(
    root: Path, config: DataConfig
) -> tuple[tuple[ImageRecord, ...], tuple[ImageRecord, ...]]:
    candidates = _images(root, "val", config.source_human_label)
    references = _images(root, "train", config.target_human_label)
    if len(candidates) < config.candidate_count:
        raise ValueError(
            f"Expected at least {config.candidate_count} validation images for "
            f"source human label {config.source_human_label}, found {len(candidates)}"
        )
    if len(references) < config.target_reference_count:
        raise ValueError(
            f"Expected at least {config.target_reference_count} training images for "
            f"target human label {config.target_human_label}, found {len(references)}"
        )
    return (
        candidates[: config.candidate_count],
        references[: config.target_reference_count],
    )
