from pathlib import Path

from primary_ml_cka.data.manifests import ImageRecord
from primary_ml_cka.data.selection import clean_valid_split


def test_main_and_confirmation_are_disjoint_and_batched() -> None:
    candidates = tuple(
        ImageRecord(str(i), Path(str(i)), 4, "minivan", "n03770679") for i in range(50)
    )
    main, confirmation = clean_valid_split(candidates, (4,) * 50, source_human_label=4)
    assert len(main) == 32
    assert len(confirmation) == 16
    assert set(main).isdisjoint(confirmation)
