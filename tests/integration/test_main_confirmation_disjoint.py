from pathlib import Path

from primary_ml_cka.data.manifests import ImageRecord
from primary_ml_cka.data.selection import clean_valid_split


def test_main_and_confirmation_are_disjoint_and_batched() -> None:
    candidates = tuple(
        ImageRecord(str(i), Path(str(i)), 4, "minivan", "n03770679") for i in range(50)
    )
    main, confirmation = clean_valid_split(
        candidates,
        (4,) * 50,
        source_human_label=4,
        main_max_count=32,
        confirmation_max_count=16,
    )
    assert len(main) == 32
    assert len(confirmation) == 16
    assert set(main).isdisjoint(confirmation)


def test_scaled_main_keeps_valid_partial_batch() -> None:
    candidates = tuple(
        ImageRecord(str(i), Path(str(i)), 8, "pickup truck", "n03930630")
        for i in range(58)
    )
    main, confirmation = clean_valid_split(
        candidates,
        (8,) * 58,
        source_human_label=8,
        main_max_count=50,
        confirmation_max_count=8,
        allow_partial_main_batch=True,
    )
    assert len(main) == 50
    assert len(confirmation) == 8
    assert set(main).isdisjoint(confirmation)
