from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image

from primary_ml_cka.data.manifests import ImageRecord


def ste_quantize_8bit(images: torch.Tensor) -> torch.Tensor:
    """Match PNG uint8 values in the forward pass while preserving gradients."""
    quantized = images.mul(255).round().clamp(0, 255).div(255)
    return images + (quantized - images).detach()


def resize_crop_normalize(
    images: torch.Tensor,
    *,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    interpolation_mode: str = "bicubic",
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
    quantized_images = ste_quantize_8bit(images)
    resized = functional.interpolate(
        quantized_images,
        size=(size, size),
        mode=interpolation_mode,
        align_corners=False,
        antialias=True,
    )
    resized = ste_quantize_8bit(resized)
    mean_tensor = resized.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = resized.new_tensor(std).view(1, 3, 1, 1)
    return (resized - mean_tensor) / std_tensor


def ensure_canvas(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    return functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )


def canonicalize_records(
    records: tuple[ImageRecord, ...],
    source_root: Path,
    canonical_root: Path,
    group: str,
    size: int = 224,
) -> tuple[ImageRecord, ...]:
    """Materialize the single PNG input used by every later experiment stage."""
    source_root = Path(source_root)
    canonical_root = Path(canonical_root)
    output_dir = canonical_root / group
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = []
    for index, record in enumerate(records):
        relative_path = Path(group) / f"{index:04d}_{Path(record.image_id).stem}.png"
        output_path = canonical_root / relative_path
        with Image.open(source_root / record.relative_path) as image:
            rgb = image.convert("RGB")
            resized = rgb.resize((size, size), resample=Image.Resampling.BICUBIC)
            resized.save(output_path, format="PNG", compress_level=9)
        canonical.append(
            ImageRecord(
                record.image_id,
                relative_path,
                record.human_label,
                record.class_name,
                record.synset,
            )
        )
    return tuple(canonical)
