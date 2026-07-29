from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    raw_output: str
    parsed_label: int | None
    parser_status: str
    backend: str
