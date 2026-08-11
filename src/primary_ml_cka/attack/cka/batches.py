from collections.abc import Sequence

from primary_ml_cka.data.manifests import ImageRecord


def fixed_reference_batch(
    pool: Sequence[ImageRecord],
    source_batch_index: int,
    batch_size: int,
    reference_count: int | None = None,
) -> tuple[ImageRecord, ...]:
    count = batch_size if reference_count is None else reference_count
    if count < batch_size:
        raise ValueError("Reference count must be at least one attack batch")
    if len(pool) < count:
        raise ValueError(f"Reference pool has {len(pool)} images; requested {count}")
    start = source_batch_index * batch_size
    return tuple(pool[(start + offset) % len(pool)] for offset in range(count))
