"""Black-box target generation backend.

This module is forbidden from proxy attack imports. It exposes decoded text only,
never target logits, hidden states, image representations, or gradients.
"""

from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

from primary_ml_cka.models.common.loading import freeze_module


def load_target_for_generation(snapshot: Path, device: torch.device):
    if device.type != "cuda":
        raise ValueError("Target generation requires CUDA")
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index or 0},
    )
    model.config.use_cache = True
    return freeze_module(model)
