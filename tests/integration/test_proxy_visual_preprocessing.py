from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.proxies.clip import CLIP_PREPROCESS
from primary_ml_cka.models.proxies.siglip2 import SIGLIP2_PREPROCESS
from primary_ml_cka.models.proxies.visual import (
    gemma_visual_inputs,
    internvl_visual_inputs,
)


def _image() -> tuple[Image.Image, torch.Tensor]:
    values = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
    array = (values % 256).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return image, tensor


@pytest.mark.parametrize(
    ("model_id", "manual_fn", "mean_tolerance", "maximum_tolerance"),
    (
        ("openai/clip-vit-large-patch14", CLIP_PREPROCESS, 1e-6, 1e-5),
        ("google/siglip2-so400m-patch14-384", SIGLIP2_PREPROCESS, 0.002, 0.01),
        (
            "OpenGVLab/InternVL3_5-2B-HF",
            lambda images: internvl_visual_inputs(images)["pixel_values"],
            0.006,
            0.16,
        ),
    ),
)
def test_differentiable_preprocessing_matches_native_processor(
    model_id: str,
    manual_fn,
    mean_tolerance: float,
    maximum_tolerance: float,
) -> None:
    snapshot = local_snapshot(Path(".hf-cache"), model_id, MODEL_REVISIONS[model_id])
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    image, tensor = _image()
    native = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
    manual = manual_fn(tensor).float()
    difference = native.float().cpu().sub(manual.cpu()).abs()
    assert tuple(native.shape) == tuple(manual.shape)
    assert float(difference.mean()) < mean_tolerance
    assert float(difference.max()) < maximum_tolerance


def test_gemma_preprocessing_matches_native_processor() -> None:
    model_id = "google/gemma-4-E2B-it"
    snapshot = local_snapshot(Path(".hf-cache"), model_id, MODEL_REVISIONS[model_id])
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    image, tensor = _image()
    native = processor.image_processor(images=image, return_tensors="pt")
    manual = gemma_visual_inputs(processor, tensor)
    difference = (
        native["pixel_values"].float().cpu()
        - manual["pixel_values"].float().cpu()
    ).abs()
    assert tuple(native["pixel_values"].shape) == tuple(manual["pixel_values"].shape)
    assert float(difference.mean()) < 0.002
    assert float(difference.max()) < 0.08
    assert torch.equal(
        native["image_position_ids"].cpu(), manual["image_position_ids"].cpu()
    )
