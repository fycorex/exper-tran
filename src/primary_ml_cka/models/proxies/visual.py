from functools import partial

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from primary_ml_cka.data.preprocessing import resize_crop_normalize, ste_quantize_8bit
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.types import ImageEmbeddingOutput, TapContract
from primary_ml_cka.models.representations import RepresentationSpec, resolve_vision_layer
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
    pixel_values = (
        surrogate["pixel_values"]
        + (
            native["pixel_values"].to(surrogate["pixel_values"]) - surrogate["pixel_values"]
        ).detach()
    )
    return {
        "pixel_values": pixel_values.to(images.device, torch.bfloat16),
        "image_position_ids": native["image_position_ids"].to(images.device),
    }


def _selected_output(
    model_id: str,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    module_path: str,
    extraction: str,
    spec: RepresentationSpec,
    resolved_layer: int | None,
    total_layers: int | None,
) -> ImageEmbeddingOutput[torch.Tensor]:
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("Selected visual tokens and mask must be [B,T,D] and [B,T]")
    pooled = masked_mean_l2(tokens, mask)
    embeddings = tokens if spec.pooling == "none" else pooled
    tap = TapContract(
        model_id,
        MODEL_REVISIONS[model_id],
        module_path,
        extraction,
        "none" if spec.pooling == "none" else "masked mean over valid visual tokens",
        "per-image L2 after mean pooling; FP32 loss computation",
        str(tokens.dtype).removeprefix("torch."),
        "validated_by_forward",
        tuple(tokens.shape),
        "valid visual tokens only",
        representation_type=spec.representation_type,
        requested_layer=spec.layer,
        resolved_layer=resolved_layer,
        total_vision_layers=total_layers,
    )
    return ImageEmbeddingOutput(
        embeddings,
        tokens,
        mask,
        tap,
        semantic_embeddings=pooled if spec.pooling == "mean" else None,
    )


def qwen_proxy_embeddings(
    model_id: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    representation_type: str = "legacy_projected",
    layer: int = -1,
    pooling: str = "mean",
):
    spec = RepresentationSpec(representation_type, layer, pooling)
    spec.validate()
    inputs = qwen_visual_inputs(images)
    if representation_type == "vision_encoder":
        total_layers = int(model.config.vision_config.depth)
        resolved = resolve_vision_layer(layer, total_layers)
        output = model.model.visual(
            inputs["pixel_values"],
            grid_thw=inputs["image_grid_thw"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[resolved + 1]
        if hidden.ndim != 2 or hidden.shape[0] % images.shape[0]:
            raise RuntimeError(f"Unexpected Qwen vision hidden shape: {tuple(hidden.shape)}")
        tokens = hidden.reshape(images.shape[0], -1, hidden.shape[-1])
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return _selected_output(
            model_id,
            tokens,
            mask,
            module_path=f"model.model.visual.blocks.{resolved}",
            extraction="Qwen Vision Encoder block output before spatial merger",
            spec=spec,
            resolved_layer=resolved,
            total_layers=total_layers,
        )
    # Use the same post-merger soft tokens that replace image placeholders in
    # the language model.  Tapping merger.norm (the old implementation) put
    # CKA before the spatial merger and projection while classification acted
    # after them, so the two attack terms optimized different representations.
    output = model.get_image_features(**inputs, return_dict=True)
    packed = output.pooler_output
    if not isinstance(packed, tuple | list) or len(packed) != images.shape[0]:
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
    return ImageEmbeddingOutput(
        tokens if pooling == "none" else embeddings,
        tokens,
        mask,
        tap,
        semantic_embeddings=embeddings if pooling == "mean" else None,
    )


def internvl_proxy_embeddings(
    model_id: str,
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    microbatch_size: int = 4,
    representation_type: str = "legacy_projected",
    layer: int = -1,
    pooling: str = "mean",
):
    spec = RepresentationSpec(representation_type, layer, pooling)
    spec.validate()
    total_layers = (
        int(model.config.vision_config.num_hidden_layers)
        if representation_type == "vision_encoder"
        else None
    )
    resolved = (
        resolve_vision_layer(layer, total_layers)
        if representation_type == "vision_encoder"
        else None
    )

    def extract(image_chunk: torch.Tensor) -> torch.Tensor:
        pixel_values = internvl_visual_inputs(image_chunk)["pixel_values"]
        if representation_type == "vision_encoder":
            output = model.model.vision_tower(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            # Hidden-state tuple includes the patch embedding at position zero.
            return output.hidden_states[resolved + 1][:, 1:, :]
        output = model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=-1,
            vision_feature_select_strategy="default",
            return_dict=True,
        )
        tokens = output.pooler_output
        if tokens.ndim != 3 or tokens.shape[0] != image_chunk.shape[0]:
            raise RuntimeError(f"Expected InternVL projected tokens [B,T,D], got {tokens.shape}")
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
    if representation_type == "vision_encoder":
        return _selected_output(
            model_id,
            tokens,
            mask,
            module_path=f"model.model.vision_tower.encoder.layer.{resolved}",
            extraction="InternVL Vision Encoder block output with CLS token excluded",
            spec=spec,
            resolved_layer=resolved,
            total_layers=total_layers,
        )
    return ImageEmbeddingOutput(
        tokens if pooling == "none" else embeddings,
        tokens,
        mask,
        tap,
        semantic_embeddings=embeddings if pooling == "mean" else None,
    )


def gemma_proxy_embeddings(
    model_id: str,
    model: torch.nn.Module,
    processor: object,
    images: torch.Tensor,
    *,
    representation_type: str = "legacy_projected",
    layer: int = -1,
    pooling: str = "mean",
):
    spec = RepresentationSpec(representation_type, layer, pooling)
    spec.validate()
    total_layers = (
        int(model.config.vision_config.num_hidden_layers)
        if representation_type == "vision_encoder"
        else None
    )
    resolved = (
        resolve_vision_layer(layer, total_layers)
        if representation_type == "vision_encoder"
        else None
    )

    def extract(image_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = gemma_visual_inputs(processor, image_chunk)
        if representation_type == "vision_encoder":
            captured: list[torch.Tensor] = []

            def capture(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("Gemma Vision Encoder hook did not capture a Tensor")
                captured.append(hidden)

            layer_module = model.model.vision_tower.encoder.layers[resolved]
            handle = layer_module.register_forward_hook(capture)
            try:
                model.model.vision_tower(
                    pixel_values=inputs["pixel_values"],
                    pixel_position_ids=inputs["image_position_ids"],
                    return_dict=True,
                )
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError(
                    f"Expected one Gemma layer-{resolved} activation, got {len(captured)}"
                )
            hidden = captured[0]
            mask = ~(inputs["image_position_ids"].eq(-1).all(dim=-1))
            return hidden, mask
        output = model.get_image_features(**inputs, return_dict=True)
        tokens_flat = output.pooler_output
        if tokens_flat.ndim != 2 or tokens_flat.shape[0] % image_chunk.shape[0]:
            raise RuntimeError(
                f"Expected evenly packed Gemma visual tokens [B*T,D], got {tokens_flat.shape}"
            )
        tokens = tokens_flat.reshape(image_chunk.shape[0], -1, tokens_flat.shape[-1])
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)

    token_chunks = []
    mask_chunks = []
    for image_chunk in images.split(1):
        if torch.is_grad_enabled() and images.requires_grad:
            tokens, mask = checkpoint(extract, image_chunk, use_reentrant=False)
        else:
            tokens, mask = extract(image_chunk)
        token_chunks.append(tokens)
        mask_chunks.append(mask)
    tokens = torch.cat(token_chunks)
    mask = torch.cat(mask_chunks)
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
    if representation_type == "vision_encoder":
        return _selected_output(
            model_id,
            tokens,
            mask,
            module_path=f"model.model.vision_tower.encoder.layers.{resolved}",
            extraction="Gemma Vision Encoder block output before visual pooling",
            spec=spec,
            resolved_layer=resolved,
            total_layers=total_layers,
        )
    return ImageEmbeddingOutput(
        tokens if pooling == "none" else embeddings,
        tokens,
        mask,
        tap,
        semantic_embeddings=embeddings if pooling == "mean" else None,
    )
