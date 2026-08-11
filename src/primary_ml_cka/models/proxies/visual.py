from functools import partial

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from primary_ml_cka.data.preprocessing import resize_crop_normalize, ste_quantize_8bit
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.types import ImageEmbeddingOutput, TapContract
from primary_ml_cka.models.taps.pooling import masked_mean_l2

INTERNVL_PREPROCESS = partial(
    resize_crop_normalize,
    size=448,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def qwen_patchify(
    images: torch.Tensor,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
):
    if images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError("Qwen canvas dimensions must be divisible by patch_size")
    normalized = (images - 0.5) / 0.5
    batch, channels, height, width = normalized.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Qwen patch grid must be divisible by merge_size")
    patches = normalized.reshape(
        batch,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            batch * grid_h * grid_w,
            channels * temporal_patch_size * patch_size * patch_size,
        )
    )
    grid = torch.tensor([[1, grid_h, grid_w]] * batch, device=images.device, dtype=torch.long)
    return patches, grid


def qwen_visual_inputs(images: torch.Tensor) -> dict[str, torch.Tensor]:
    # Qwen's native processor smart-resizes a canonical 224x224 image to the
    # configured minimum 256x256 canvas (16x16 patches, merge size two).
    resized = functional.interpolate(
        ste_quantize_8bit(images),
        size=(256, 256),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0, 1)
    resized = ste_quantize_8bit(resized)
    patches, grid = qwen_patchify(resized)
    return {"pixel_values": patches.to(torch.bfloat16), "image_grid_thw": grid}


def internvl_visual_inputs(images: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"pixel_values": INTERNVL_PREPROCESS(images).to(torch.bfloat16)}


def gemma_visual_inputs(processor: object, images: torch.Tensor) -> dict[str, torch.Tensor]:
    quantized = ste_quantize_8bit(images)
    surrogate = processor.image_processor(
        images=quantized,
        do_rescale=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        native = processor.image_processor(
            images=quantized.mul(255).round().to(torch.uint8),
            return_tensors="pt",
        )
    pixel_values = surrogate["pixel_values"] + (
        native["pixel_values"].to(surrogate["pixel_values"])
        - surrogate["pixel_values"]
    ).detach()
    return {
        "pixel_values": pixel_values.to(images.device, torch.bfloat16),
        "image_position_ids": native["image_position_ids"].to(images.device),
    }


def qwen_proxy_embeddings(model_id: str, model: torch.nn.Module, images: torch.Tensor):
    inputs = qwen_visual_inputs(images)
    # Use the same post-merger soft tokens that replace image placeholders in
    # the language model.  Tapping merger.norm (the old implementation) put
    # CKA before the spatial merger and projection while classification acted
    # after them, so the two attack terms optimized different representations.
    output = model.get_image_features(**inputs, return_dict=True)
    packed = output.pooler_output
    if not isinstance(packed, (tuple, list)) or len(packed) != images.shape[0]:
        raise RuntimeError("Expected one Qwen projected-token tensor per image")
    tokens_flat = torch.cat(tuple(packed), dim=0)
    tokens = tokens_flat.reshape(images.shape[0], -1, tokens_flat.shape[-1])
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    embeddings = masked_mean_l2(tokens, mask)
    tap = TapContract(
        model_id,
        MODEL_REVISIONS[model_id],
        "model.get_image_features.pooler_output",
        "post-merger Qwen visual soft tokens passed to the language model",
        "masked mean over projected proxy image tokens",
        "per-image L2; FP32 before CKA",
        str(tokens.dtype).removeprefix("torch."),
        "validated_by_forward",
        tuple(tokens.shape),
        "grid_thw image patches only",
    )
    return ImageEmbeddingOutput(embeddings, tokens, mask, tap)


def internvl_proxy_embeddings(
    model_id: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    microbatch_size: int = 4,
):
    def extract(image_chunk: torch.Tensor) -> torch.Tensor:
        pixel_values = internvl_visual_inputs(image_chunk)["pixel_values"]
        output = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            return_dict=True,
        )
        tokens = output.pooler_output
        if tokens.ndim != 3 or tokens.shape[0] != image_chunk.shape[0]:
            raise RuntimeError(
                f"Expected InternVL projected tokens [B,T,D], got {tokens.shape}"
            )
        return tokens

    token_chunks = []
    for image_chunk in images.split(microbatch_size):
        if torch.is_grad_enabled() and images.requires_grad:
            token_chunks.append(checkpoint(extract, image_chunk, use_reentrant=False))
        else:
            token_chunks.append(extract(image_chunk))
    tokens = torch.cat(token_chunks)
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    embeddings = masked_mean_l2(tokens, mask)
    tap = TapContract(
        model_id,
        MODEL_REVISIONS[model_id],
        "model.get_image_features.pooler_output",
        "post-pixel-shuffle InternVL visual soft tokens passed to the language model",
        "masked mean over projected proxy image tokens",
        "per-image L2; FP32 before CKA",
        str(tokens.dtype).removeprefix("torch."),
        "validated_by_forward",
        tuple(tokens.shape),
        "non-CLS tokens from one valid 448x448 tile per image",
    )
    return ImageEmbeddingOutput(embeddings, tokens, mask, tap)


def gemma_proxy_embeddings(
    model_id: str, model: torch.nn.Module, processor: object, images: torch.Tensor
):
    def extract(image_chunk: torch.Tensor) -> torch.Tensor:
        inputs = gemma_visual_inputs(processor, image_chunk)
        output = model.get_image_features(**inputs, return_dict=True)
        tokens_flat = output.pooler_output
        if tokens_flat.ndim != 2 or tokens_flat.shape[0] % image_chunk.shape[0]:
            raise RuntimeError(
                f"Expected evenly packed Gemma visual tokens [B*T,D], got {tokens_flat.shape}"
            )
        return tokens_flat.reshape(image_chunk.shape[0], -1, tokens_flat.shape[-1])

    token_chunks = []
    for image_chunk in images.split(1):
        if torch.is_grad_enabled() and images.requires_grad:
            token_chunks.append(checkpoint(extract, image_chunk, use_reentrant=False))
        else:
            token_chunks.append(extract(image_chunk))
    tokens = torch.cat(token_chunks)
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    embeddings = masked_mean_l2(tokens, mask)
    tap = TapContract(
        model_id,
        MODEL_REVISIONS[model_id],
        "model.embed_vision",
        "pooled Gemma visual soft tokens passed to the language model",
        "masked mean over non-padding visual soft tokens",
        "per-image L2; FP32 before CKA",
        str(tokens.dtype).removeprefix("torch."),
        "validated_by_forward",
        tuple(tokens.shape),
        "derived from non-padding image_position_ids and pooling kernel",
    )
    return ImageEmbeddingOutput(embeddings, tokens, mask, tap)
