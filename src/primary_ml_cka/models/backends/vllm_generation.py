"""Optional no-gradient generation backend; never import from attack code."""

from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.prompts.parser import parse_exact_label


@dataclass(slots=True)
class VLLMTargetGenerator:
    engine: object
    sampling_params: object

    def generate_label(self, image_path: Path, prompt: str) -> GenerationOutput:
        request = {"prompt": prompt, "multi_modal_data": {"image": str(image_path)}}
        result = self.engine.generate(request, self.sampling_params)
        raw = result[0].outputs[0].text
        parsed = parse_exact_label(raw)
        return GenerationOutput(raw, parsed.label, parsed.status, "vllm")
