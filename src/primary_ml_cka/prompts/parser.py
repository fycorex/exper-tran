from dataclasses import dataclass

from primary_ml_cka.domain.output_codes import OUTPUT_CODES, output_code_to_human_label


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    raw_output: str
    first_non_empty_line: str | None
    label: int | None
    status: str


def parse_exact_label(raw_output: str) -> ParsedLabel:
    line = next((line.strip() for line in raw_output.splitlines() if line.strip()), None)
    if line is None:
        return ParsedLabel(raw_output, None, None, "empty")
    if line not in OUTPUT_CODES:
        return ParsedLabel(raw_output, line, None, "invalid")
    return ParsedLabel(raw_output, line, output_code_to_human_label(line), "ok")
