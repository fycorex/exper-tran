import pytest

from primary_ml_cka.config.schema import DataConfig
from primary_ml_cka.config.validation import validate_data_config
from primary_ml_cka.domain.labels import human_label_to_index


def test_human_label_conversions() -> None:
    assert human_label_to_index(7) == 6
    assert human_label_to_index(8) == 7
    assert human_label_to_index(3) == 2


def test_source_and_target_labels_are_configurable() -> None:
    config = DataConfig(
        source_human_label=4,
        target_human_label=9,
        candidate_count=50,
        target_reference_count=48,
        main_max_count=32,
        confirmation_max_count=16,
    )
    validate_data_config(config)
    assert human_label_to_index(config.target_human_label) == 8


@pytest.mark.parametrize("label", [0, 11, -1])
def test_invalid_human_label(label: int) -> None:
    with pytest.raises(ValueError):
        human_label_to_index(label)
