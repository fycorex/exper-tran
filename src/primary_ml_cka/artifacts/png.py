from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def save_png_tensor(image: torch.Tensor, path: Path) -> None:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(image.shape)}")
    quantized = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    array = quantized.permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=9)


def load_png_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div(255.0)


def assert_png_linf(clean_path: Path, adversarial_path: Path, epsilon: float = 16 / 255) -> float:
    clean = load_png_tensor(clean_path)
    adversarial = load_png_tensor(adversarial_path)
    linf = float((adversarial - clean).abs().max())
    if linf > epsilon + 1e-7:
        raise ValueError(f"PNG L-inf {linf} exceeds {epsilon}")
    return linf
