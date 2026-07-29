from pathlib import Path

from primary_ml_cka.attack.cka.batches import fixed_reference_batch
from primary_ml_cka.data.manifests import ImageRecord


def test_reference_order_fixed_across_calls() -> None:
    pool = tuple(ImageRecord(str(i), Path(str(i)), 9, "tow truck", "n04461696") for i in range(64))
    first = fixed_reference_batch(pool, 3, batch_size=5)
    assert first == fixed_reference_batch(pool, 3, batch_size=5)
    assert tuple(item.image_id for item in first) == tuple(str(i) for i in range(15, 20))
