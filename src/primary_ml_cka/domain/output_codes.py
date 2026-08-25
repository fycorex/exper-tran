"""Mapping between semantic ImageNet labels and model-facing answer codes."""

OUTPUT_CODES = tuple(str(index) for index in range(10))


def human_label_to_output_code(human_label: int) -> str:
    """Map semantic labels 1..10 to the model-facing strings ``"0"``..``"9"``."""
    if not 1 <= human_label <= 10:
        raise ValueError(human_label)
    return str(human_label - 1)


def output_code_to_human_label(output_code: str) -> int:
    """Map an exact model-facing output code back to a semantic label."""
    if output_code not in OUTPUT_CODES:
        raise ValueError(output_code)
    return int(output_code) + 1
