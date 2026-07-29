from collections.abc import Sequence

from primary_ml_cka.data.manifests import ImageRecord
from primary_ml_cka.domain.constants import BATCH_SIZE


def clean_valid_split(
    candidates: Sequence[ImageRecord],
    parsed_labels: Sequence[int | None],
    source_human_label: int,
    main_max_count: int,
    confirmation_max_count: int,
    batch_size: int = BATCH_SIZE,
) -> tuple[tuple[ImageRecord, ...], tuple[ImageRecord, ...]]:
    if len(candidates) != len(parsed_labels):
        raise ValueError("Candidates and labels must have equal length")
    valid = [
        item
        for item, label in zip(candidates, parsed_labels, strict=True)
        if label == source_human_label
    ]
    main_count = min(main_max_count, len(valid))
    main_count -= main_count % batch_size
    main = tuple(valid[:main_count])
    remaining = valid[main_count:]
    confirmation_count = min(confirmation_max_count, len(remaining))
    confirmation_count -= confirmation_count % batch_size
    return main, tuple(remaining[:confirmation_count])


def require_minimum_sets(
    main: Sequence[ImageRecord],
    confirmation: Sequence[ImageRecord],
    batch_size: int = BATCH_SIZE,
) -> tuple[bool, bool]:
    return len(main) >= batch_size, len(confirmation) >= batch_size
