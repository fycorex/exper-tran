import torch
import torch.nn.functional as functional


def resize_crop_normalize(
    images: torch.Tensor,
    *,
    size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}")
    resized = functional.interpolate(
        images, size=(size, size), mode="bicubic", align_corners=False, antialias=True
    )
    mean_tensor = resized.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = resized.new_tensor(std).view(1, 3, 1, 1)
    return (resized - mean_tensor) / std_tensor


def ensure_canvas(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    return functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
