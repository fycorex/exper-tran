from pathlib import Path

from primary_ml_cka.attack.cka.batches import fixed_reference_batch
from primary_ml_cka.data.manifests import ImageRecord


def _record(index: int) -> ImageRecord:
    return ImageRecord(str(index), Path(f"{index}.png"), 7, "garbage truck", "n03417042")


def test_full_reference_bank_is_selected_deterministically() -> None:
    records = tuple(_record(index) for index in range(48))
    selected = fixed_reference_batch(records, 0, 8, reference_count=48)
    assert selected == records


def test_reference_bank_rejects_unavailable_count() -> None:
    records = tuple(_record(index) for index in range(8))
    try:
        fixed_reference_batch(records, 0, 8, reference_count=48)
    except ValueError as exc:
        assert "requested 48" in str(exc)
    else:
        raise AssertionError("Expected unavailable reference bank to fail")
