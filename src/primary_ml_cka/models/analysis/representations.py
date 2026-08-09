import gc
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.common.loading import (
    freeze_module,
    load_prefixed_weights,
    local_snapshot,
    safetensor_files,
)


def _visual_tokens(
    model: object, processor: object, image: Image.Image, *, vision_only: bool = False
) -> torch.Tensor:
    image_processor = getattr(processor, "image_processor", processor)
    inputs = image_processor(images=image, return_tensors="pt")
    device = next(model.parameters()).device
    visual = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    if vision_only:
        output = model(
            pixel_values=visual["pixel_values"],
            pixel_position_ids=visual["image_position_ids"],
            return_dict=True,
        )
        tokens = output.last_hidden_state
        return tokens.unsqueeze(0) if tokens.ndim == 2 else tokens
    if hasattr(model, "vision_model"):
        output = model.vision_model(pixel_values=visual["pixel_values"], return_dict=True)
        tokens = output.last_hidden_state
        if "clip" in getattr(model, "name_or_path", "").lower():
            tokens = tokens[:, 1:, :]
        return tokens
    output = model.get_image_features(**visual, return_dict=True)
    tokens = getattr(output, "last_hidden_state", None)
    if tokens is None:
        tokens = getattr(output, "pooler_output", None)
    if not isinstance(tokens, torch.Tensor):
        raise RuntimeError("Model did not expose visual hidden states for analysis")
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    return tokens


def extract_representations(
    model_id: str,
    image_paths: Sequence[Path],
    hf_home: Path,
    device: torch.device,
) -> torch.Tensor:
    """Load one model, extract pooled visual states, then release it."""
    snapshot = local_snapshot(hf_home, model_id, MODEL_REVISIONS[model_id])
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    vision_only = model_id.startswith("google/gemma")
    if vision_only:
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        model = freeze_module(AutoModel.from_config(config.vision_config).to(device))
        load_prefixed_weights(
            model,
            safetensor_files(snapshot),
            ("model.vision_tower.",),
            device=device,
        )
    elif model_id.startswith(("openai/", "google/siglip")):
        model = freeze_module(
            AutoModel.from_pretrained(snapshot, local_files_only=True).to(device)
        )
    else:
        model = load_target_for_generation(snapshot, device)
    rows = []
    try:
        with torch.no_grad():
            for path in image_paths:
                with Image.open(path) as image:
                    tokens = _visual_tokens(
                        model, processor, image.convert("RGB"), vision_only=vision_only
                    )
                rows.append(functional.normalize(tokens.float().mean(dim=(0, 1)), dim=0).cpu())
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return torch.stack(rows)
