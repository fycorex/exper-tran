import importlib.util
from pathlib import Path

from primary_ml_cka.data.manifests import ImageRecord

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "2026-08-qwen-transfer-diagnostics"
    / "src"
    / "materialize_disjoint_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_disjoint_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(image_id: str, label: int) -> ImageRecord:
    return ImageRecord(image_id, Path(f"{image_id}.png"), label, f"class-{label}", "synset")


def test_disjoint_calibration_replaces_attack_overlap_and_preserves_class_counts() -> None:
    calibration = (
        _record("class-1-a", 1),
        _record("attack", 8),
        _record("class-8-b", 8),
    )
    candidates = (
        _record("attack", 8),
        _record("replacement", 8),
    )

    result = MODULE.disjoint_calibration(
        calibration,
        candidates,
        (_record("attack", 8),),
    )

    assert [record.human_label for record in result] == [1, 8, 8]
    assert {record.image_id for record in result} == {
        "class-1-a",
        "class-8-b",
        "replacement",
    }
