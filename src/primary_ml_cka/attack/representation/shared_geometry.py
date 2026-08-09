from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from primary_ml_cka.attack.cka.linear import linear_cka


@dataclass(frozen=True, slots=True)
class SharedGeometryLoss:
    total: torch.Tensor
    clean_alignment: torch.Tensor
    view_alignment: torch.Tensor


def center_crop_view(images: torch.Tensor, scale: float) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError("Images must have shape [N,C,H,W]")
    if not 0 < scale <= 1:
        raise ValueError("View scale must be in (0,1]")
    height, width = images.shape[-2:]
    crop_height = max(1, round(height * scale))
    crop_width = max(1, round(width * scale))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    crop = images[:, :, top : top + crop_height, left : left + crop_width]
    return functional.interpolate(
        crop,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def random_resized_crop_view(
    images: torch.Tensor,
    scale: float,
    *,
    seed: int,
) -> torch.Tensor:
    """Create one independently translated square crop per image on its device."""
    if images.ndim != 4:
        raise ValueError("Images must have shape [N,C,H,W]")
    if not 0 < scale <= 1:
        raise ValueError("View scale must be in (0,1]")
    generator = torch.Generator(device=images.device)
    generator.manual_seed(seed)
    translations = (
        2.0
        * (1.0 - scale)
        * (torch.rand((images.shape[0], 2), generator=generator, device=images.device) - 0.5)
    )
    transform = torch.zeros(
        (images.shape[0], 2, 3), dtype=images.dtype, device=images.device
    )
    transform[:, 0, 0] = scale
    transform[:, 1, 1] = scale
    transform[:, :, 2] = translations
    grid = functional.affine_grid(transform, images.shape, align_corners=False)
    return functional.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def shared_geometry_loss(
    adversarial_embeddings: torch.Tensor,
    clean_embeddings: torch.Tensor,
    view_embeddings: torch.Tensor,
    *,
    clean_weight: float,
    view_weight: float,
    adversarial_patch_embeddings: torch.Tensor | None = None,
    clean_patch_embeddings: torch.Tensor | None = None,
) -> SharedGeometryLoss:
    if clean_weight < 0 or view_weight < 0:
        raise ValueError("Shared-geometry weights must be non-negative")
    if (adversarial_patch_embeddings is None) != (clean_patch_embeddings is None):
        raise ValueError("Adversarial and clean patch embeddings must be provided together")
    if adversarial_patch_embeddings is None:
        clean_alignment = linear_cka(adversarial_embeddings, clean_embeddings)
    else:
        if (
            adversarial_patch_embeddings.ndim != 3
            or clean_patch_embeddings.ndim != 3
            or adversarial_patch_embeddings.shape[:2] != clean_patch_embeddings.shape[:2]
        ):
            raise ValueError("Patch embeddings must align as [N_images,N_patches,D_model]")
        clean_alignment = torch.stack(
            [
                linear_cka(adversarial, clean)
                for adversarial, clean in zip(
                    adversarial_patch_embeddings,
                    clean_patch_embeddings,
                    strict=True,
                )
            ]
        ).mean()
    view_alignment = functional.cosine_similarity(
        adversarial_embeddings.float(), view_embeddings.float(), dim=-1
    ).mean()
    total = clean_weight * (1.0 - clean_alignment) + view_weight * (
        1.0 - view_alignment
    )
    return SharedGeometryLoss(total, clean_alignment, view_alignment)
