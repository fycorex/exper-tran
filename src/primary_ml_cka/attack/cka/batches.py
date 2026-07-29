from collections.abc import Sequence

from primary_ml_cka.data.manifests import ImageRecord


def fixed_reference_batch(
    pool: Sequence[ImageRecord], source_batch_index: int, batch_size: int
) -> tuple[ImageRecord, ...]:
    if len(pool) < batch_size:
        raise ValueError("Reference pool is smaller than one batch")
    start = source_batch_index * batch_size
    return tuple(pool[(start + offset) % len(pool)] for offset in range(batch_size))
