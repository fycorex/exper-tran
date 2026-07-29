from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.prompts.parser import parse_exact_label


@dataclass(frozen=True, slots=True)
class DecodingConfig:
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    max_new_tokens: int = 4
    disable_thinking: bool = True


@dataclass(slots=True)
class BlackBoxTargetAPI:
    """Adapter for a closed target that returns decoded text only."""

    request: Callable[[Path, str, DecodingConfig], str]
    decoding: DecodingConfig = DecodingConfig()

    def generate_label(self, image_path: Path, prompt: str) -> GenerationOutput:
        raw = self.request(image_path, prompt, self.decoding)
        parsed = parse_exact_label(raw)
        return GenerationOutput(raw, parsed.label, parsed.status, "black_box_api")
