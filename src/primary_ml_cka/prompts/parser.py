from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedLabel:
    raw_output: str
    first_non_empty_line: str | None
    label: int | None
    status: str


def parse_exact_label(raw_output: str) -> ParsedLabel:
    line = next((line.strip() for line in raw_output.splitlines() if line.strip()), None)
    accepted = {str(index) for index in range(1, 11)}
    if line is None:
        return ParsedLabel(raw_output, None, None, "empty")
    if line not in accepted:
        return ParsedLabel(raw_output, line, None, "invalid")
    return ParsedLabel(raw_output, line, int(line), "ok")
