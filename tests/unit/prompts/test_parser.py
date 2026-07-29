import pytest

from primary_ml_cka.prompts.parser import parse_exact_label


@pytest.mark.parametrize("raw,label", [("8", 8), ("\n \n7\nextra", 7), ("10\n", 10)])
def test_exact_first_non_empty_line(raw: str, label: int) -> None:
    parsed = parse_exact_label(raw)
    assert parsed.label == label
    assert parsed.status == "ok"
    assert parsed.raw_output == raw


@pytest.mark.parametrize("raw", ["7.", "label 7", "11", "", "\n \n"])
def test_rejects_non_exact_output(raw: str) -> None:
    assert parse_exact_label(raw).label is None
