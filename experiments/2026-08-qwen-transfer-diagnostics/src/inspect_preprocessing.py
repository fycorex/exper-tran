from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.proxies.clip import CLIP_PREPROCESS
from primary_ml_cka.models.proxies.siglip2 import SIGLIP2_PREPROCESS
from primary_ml_cka.models.proxies.visual import (
    gemma_visual_inputs,
    internvl_visual_inputs,
    qwen_visual_inputs,
)

MODELS = (
    ("openai/clip-vit-large-patch14", CLIP_PREPROCESS),
    ("google/siglip2-so400m-patch14-384", SIGLIP2_PREPROCESS),
    ("Qwen/Qwen3.5-2B", lambda images: qwen_visual_inputs(images)["pixel_values"]),
    (
        "OpenGVLab/InternVL3_5-2B-HF",
        lambda images: internvl_visual_inputs(images)["pixel_values"],
    ),
)


def report_difference(model_id: str, manual_fn, image: Image.Image) -> None:
    snapshot = local_snapshot(Path(".hf-cache"), model_id, MODEL_REVISIONS[model_id])
    processor = load_processor(snapshot)
    native = processor.image_processor(images=image, return_tensors="pt")
    tensor = pil_to_tensor(image).float().div(255).unsqueeze(0)
    manual = manual_fn(tensor).float().cpu()
    native_pixels = native["pixel_values"].float().cpu()
    payload = {
        "model_id": model_id,
        "native_shape": tuple(native_pixels.shape),
        "manual_shape": tuple(manual.shape),
    }
    if native_pixels.numel() == manual.numel():
        difference = native_pixels.reshape_as(manual).sub(manual).abs()
        payload.update(
            mean_absolute_difference=float(difference.mean()),
            maximum_absolute_difference=float(difference.max()),
        )
    if "image_grid_thw" in native:
        payload["native_grid"] = native["image_grid_thw"].tolist()
    print(payload)


def report_gemma(image: Image.Image) -> None:
    model_id = "google/gemma-4-E2B-it"
    snapshot = local_snapshot(Path(".hf-cache"), model_id, MODEL_REVISIONS[model_id])
    processor = load_processor(snapshot)
    native = processor.image_processor(images=image, return_tensors="pt")
    tensor = pil_to_tensor(image).float().div(255).unsqueeze(0)
    manual = gemma_visual_inputs(processor, tensor)
    difference = native["pixel_values"].float().sub(manual["pixel_values"].float()).abs()
    print(
        {
            "model_id": model_id,
            "native_shape": tuple(native["pixel_values"].shape),
            "manual_shape": tuple(manual["pixel_values"].shape),
            "mean_absolute_difference": float(difference.mean()),
            "maximum_absolute_difference": float(difference.max()),
            "position_ids_equal": torch.equal(
                native["image_position_ids"], manual["image_position_ids"].cpu()
            ),
        }
    )


def main() -> None:
    path = Path(
        "outputs/proxy_selector_cka_v2/canonical_images/"
        "source_candidates/0000_ILSVRC2012_val_00025131.png"
    )
    with Image.open(path) as source:
        image = source.convert("RGB")
        for model_id, manual_fn in MODELS:
            report_difference(model_id, manual_fn, image)
        report_gemma(image)


if __name__ == "__main__":
    main()
