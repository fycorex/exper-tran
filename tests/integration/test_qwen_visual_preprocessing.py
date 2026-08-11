from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.proxies.visual import qwen_visual_inputs


def test_differentiable_qwen_preprocessing_matches_native_processor() -> None:
    model_id = "Qwen/Qwen3.5-2B"
    snapshot = local_snapshot(Path(".hf-cache"), model_id, MODEL_REVISIONS[model_id])
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    ramp = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
    image_array = (ramp % 256).astype(np.uint8)
    image = Image.fromarray(image_array, mode="RGB")
    native = processor.image_processor(images=image, return_tensors="pt")
    tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
    differentiable = qwen_visual_inputs(tensor)

    assert native["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert differentiable["image_grid_thw"].cpu().tolist() == [[1, 16, 16]]
    assert tuple(native["pixel_values"].shape) == (256, 1536)
    assert tuple(differentiable["pixel_values"].shape) == (256, 1536)
    difference = (
        native["pixel_values"].float().cpu()
        - differentiable["pixel_values"].float().cpu()
    ).abs()
    assert float(difference.mean()) < 0.005
    assert float(difference.max()) < 0.06
