from pathlib import Path

from primary_ml_cka.attack.cka.batches import fixed_reference_batch
from primary_ml_cka.data.manifests import ImageRecord


def test_reference_order_fixed_across_calls() -> None:
    pool = tuple(ImageRecord(str(i), Path(str(i)), 9, "tow truck", "n04461696") for i in range(64))
    first = fixed_reference_batch(pool, 3, batch_size=5)
    assert first == fixed_reference_batch(pool, 3, batch_size=5)
    assert tuple(item.image_id for item in first) == tuple(str(i) for i in range(15, 20))


def test_main_and_confirmation_reference_ranges_are_disjoint() -> None:
    pool = tuple(ImageRecord(str(i), Path(str(i)), 9, "tow truck", "n04461696") for i in range(48))
    main = tuple(
        item.image_id
        for batch_index in range(4)
        for item in fixed_reference_batch(pool, batch_index, batch_size=8)
    )
    confirmation = tuple(
        item.image_id
        for batch_index in range(4, 6)
        for item in fixed_reference_batch(pool, batch_index, batch_size=8)
    )
    assert main == tuple(str(index) for index in range(32))
    assert confirmation == tuple(str(index) for index in range(32, 48))
    assert set(main).isdisjoint(confirmation)
