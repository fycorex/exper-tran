import pytest

from primary_ml_cka.domain.output_codes import (
    human_label_to_output_code,
    output_code_to_human_label,
)


def test_semantic_labels_map_to_zero_based_output_codes() -> None:
    assert [human_label_to_output_code(label) for label in range(1, 11)] == [
        str(code) for code in range(10)
    ]
    assert [output_code_to_human_label(str(code)) for code in range(10)] == list(range(1, 11))


@pytest.mark.parametrize("invalid", [0, 11])
def test_rejects_invalid_semantic_label(invalid: int) -> None:
    with pytest.raises(ValueError):
        human_label_to_output_code(invalid)
