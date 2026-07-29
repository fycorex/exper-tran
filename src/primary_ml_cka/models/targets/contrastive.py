from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoModel, AutoTokenizer

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.preprocessing import ensure_canvas
from primary_ml_cka.models.common.loading import freeze_module, local_snapshot
from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.models.proxies.clip import CLIP_PREPROCESS
from primary_ml_cka.models.proxies.contrastive import ContrastiveProxy


@dataclass(slots=True)
class ContrastiveTargetGenerator:
    classifier: ContrastiveProxy
    canvas_size: int

    def generate_label(self, image_path: Path, prompt: str) -> GenerationOutput:
        with Image.open(image_path) as image:
            pixels = (
                pil_to_tensor(image.convert("RGB"))
                .cuda(non_blocking=False)
                .float()
                .div(255.0)
                .unsqueeze(0)
            )
        pixels = ensure_canvas(pixels, self.canvas_size)
        with torch.no_grad():
            output = self.classifier.target_loss(
                pixels,
                human_target_label=1,
                prompt=prompt,
            )
        if output.class_logits is None:
            raise RuntimeError("Contrastive target did not return ten-class logits")
        label = int(output.class_logits.argmax(dim=1).item()) + 1
        raw = str(label)
        return GenerationOutput(raw, label, "ok", "transformers-clip-contrastive")


def load_clip_target_generator(
    model_id: str,
    hf_home: Path,
    attack_config: AttackConfig,
) -> ContrastiveTargetGenerator:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU target evaluation is forbidden")
    snapshot = local_snapshot(hf_home, model_id)
    model = freeze_module(AutoModel.from_pretrained(snapshot, local_files_only=True).cuda())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    classifier = ContrastiveProxy(
        model,
        tokenizer,
        CLIP_PREPROCESS,
        model_id,
        class_margin=attack_config.class_margin,
        rank_weight=attack_config.rank_weight,
        suppression_weight=attack_config.other_suppression_weight,
    )
    return ContrastiveTargetGenerator(classifier, attack_config.canvas_size)
