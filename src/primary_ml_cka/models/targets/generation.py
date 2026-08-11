from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.prompts.chat_templates import (
    classification_messages,
    render_chat_template,
)
from primary_ml_cka.prompts.parser import parse_exact_label


@dataclass(slots=True)
class TransformersTargetGenerator:
    model: object
    processor: object

    def generate_label(self, image_path: Path, prompt: str) -> GenerationOutput:
        messages = classification_messages(prompt).prompt_only
        rendered = render_chat_template(
            self.processor, messages, add_generation_prompt=True
        )
        with Image.open(image_path) as image:
            inputs = self.processor(text=rendered, images=image.convert("RGB"), return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = self.model.generate(
            **inputs,
            temperature=0,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=4,
            use_cache=True,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        parsed = parse_exact_label(raw)
        return GenerationOutput(raw, parsed.label, parsed.status, "transformers")
