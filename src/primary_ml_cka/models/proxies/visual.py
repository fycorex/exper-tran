from functools import partial

import torch

from primary_ml_cka.data.preprocessing import resize_crop_normalize
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.types import ImageEmbeddingOutput, TapContract
from primary_ml_cka.models.taps.pooling import masked_mean_l2

INTERNVL_PREPROCESS = partial(
    resize_crop_normalize,
    size=448,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def qwen_patchify(images: torch.Tensor, patch_size: int = 16, temporal_patch_size: int = 2):
    if images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError("Qwen canvas dimensions must be divisible by patch_size")
    normalized = (images - 0.5) / 0.5
    frames = normalized.unsqueeze(1).expand(-1, temporal_patch_size, -1, -1, -1)
    batch, _, channels, height, width = frames.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = (
        frames.reshape(
            batch, 1, temporal_patch_size, channels, grid_h, patch_size, grid_w, patch_size
        )
        .permute(0, 1, 4, 6, 3, 2, 5, 7)
        .reshape(batch * grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size)
    )
    grid = torch.tensor([[1, grid_h, grid_w]] * batch, device=images.device, dtype=torch.long)
    return patches, grid


def qwen_visual_inputs(images: torch.Tensor) -> dict[str, torch.Tensor]:
    patches, grid = qwen_patchify(images)
    return {"pixel_values": patches.to(torch.bfloat16), "image_grid_thw": grid}


def internvl_visual_inputs(images: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"pixel_values": INTERNVL_PREPROCESS(images).to(torch.bfloat16)}


def qwen_proxy_embeddings(model_id: str, visual: torch.nn.Module, images: torch.Tensor):
    inputs = qwen_visual_inputs(images)
    captured: list[torch.Tensor] = []

    def capture(_module, args):
        captured.append(args[0])

    handle = visual.merger.register_forward_pre_hook(capture)
    try:
        visual(hidden_states=inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("Qwen proxy pre-merger tap did not fire exactly once")
    tokens_flat = visual.merger.norm(captured[0])
    tokens = tokens_flat.reshape(images.shape[0], -1, tokens_flat.shape[-1])
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    embeddings = masked_mean_l2(tokens, mask)
    tap = TapContract(
        model_id,
        MODEL_REVISIONS[model_id],
        "model.visual.merger.norm",
        "proxy final visual patch tokens after normalization, before merger projection",
        "masked mean over proxy image patch tokens",
        "per-image L2; FP32 before CKA",
        str(tokens.dtype).removeprefix("torch."),
        "validated_by_forward",
        tuple(tokens.shape),
        "grid_thw image patches only",
    )
    return ImageEmbeddingOutput(embeddings, tokens, mask, tap)


def internvl_proxy_embeddings(model_id: str, vision_tower: torch.nn.Module, images: torch.Tensor):
    pixel_values = internvl_visual_inputs(images)["pixel_values"]
    output = vision_tower(pixel_values=pixel_values, return_dict=True)
    tokens = output.last_hidden_state[:, 1:, :]
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    embeddings = masked_mean_l2(tokens, mask)
    tap = TapContract(
        model_id,
        MODEL_REVISIONS[model_id],
        "model.vision_tower.layernorm",
        "proxy final InternViT patch tokens after final normalization",
        "masked mean over proxy image patch tokens",
        "per-image L2; FP32 before CKA",
        str(tokens.dtype).removeprefix("torch."),
        "validated_by_forward",
        tuple(tokens.shape),
        "non-CLS tokens from one valid 448x448 tile per image",
    )
    return ImageEmbeddingOutput(embeddings, tokens, mask, tap)
