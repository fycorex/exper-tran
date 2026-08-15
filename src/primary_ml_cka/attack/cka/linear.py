import torch
import torch.nn.functional as functional


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("CKA inputs must be two-dimensional")
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"CKA inputs must have the same batch size, got {x.shape[0]} and {y.shape[0]}"
        )
    if x.shape[0] < 2:
        raise ValueError("CKA requires at least two image rows")
    x = x.float()
    y = y.float()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("CKA inputs must be finite")
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    # Dual/Gram form is equivalent to the feature-space form and avoids D x D
    # matrices for high-width visual tokens.
    xx = x_centered @ x_centered.T
    yy = y_centered @ y_centered.T
    numerator = (xx * yy).sum()
    denominator = torch.linalg.matrix_norm(xx) * torch.linalg.matrix_norm(yy)
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("CKA denominator must be finite and positive")
    return numerator / (denominator + 1e-12)


def paired_token_cka(
    x: torch.Tensor,
    y: torch.Tensor,
    x_mask: torch.Tensor | None = None,
    y_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one token-level CKA value per image.

    The observations are patch/image tokens, never batch rows. Feature widths
    may differ, but paired inputs must expose the same token positions.
    """
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError("Token CKA inputs must have shape [B,T,D]")
    if x.shape[:2] != y.shape[:2]:
        raise ValueError(
            "Paired token CKA inputs must have equal batch and token dimensions, "
            f"got {tuple(x.shape[:2])} and {tuple(y.shape[:2])}"
        )
    if x_mask is None:
        x_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if y_mask is None:
        y_mask = torch.ones(y.shape[:2], dtype=torch.bool, device=y.device)
    if x_mask.shape != x.shape[:2] or y_mask.shape != y.shape[:2]:
        raise ValueError("Token masks must match [B,T]")
    values = []
    for index in range(x.shape[0]):
        valid = x_mask[index].bool() & y_mask[index].bool()
        if int(valid.sum()) < 2:
            raise ValueError("Every image pair needs at least two shared valid tokens")
        values.append(linear_cka(x[index, valid], y[index, valid]))
    return torch.stack(values)


def token_cka_against_bank(
    images: torch.Tensor,
    references: torch.Tensor,
    image_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B,K] token CKA for every image/reference combination."""
    if images.ndim != 3 or references.ndim != 3:
        raise ValueError("Token CKA bank inputs must have shape [B,T,D] and [K,T,D]")
    if images.shape[1] != references.shape[1]:
        raise ValueError("Image and reference token counts must match")
    batch_size, reference_count = images.shape[0], references.shape[0]
    if image_mask is None:
        image_mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
    if reference_mask is None:
        reference_mask = torch.ones(
            references.shape[:2], dtype=torch.bool, device=references.device
        )
    if bool(image_mask.all()) and bool(reference_mask.all()):
        # Compute every token Gram once. Expanding [B,K,T,D] and reshaping
        # materializes hundreds of MiB for wide VLM embeddings (Gemma D=2560).
        images_fp32 = images.float()
        references_fp32 = references.float()
        centered_images = images_fp32 - images_fp32.mean(dim=1, keepdim=True)
        centered_references = references_fp32 - references_fp32.mean(
            dim=1, keepdim=True
        )
        image_grams = centered_images @ centered_images.transpose(1, 2)
        reference_grams = centered_references @ centered_references.transpose(1, 2)
        numerator = torch.einsum("bij,kij->bk", image_grams, reference_grams)
        image_norms = torch.linalg.vector_norm(image_grams.flatten(1), dim=1)
        reference_norms = torch.linalg.vector_norm(reference_grams.flatten(1), dim=1)
        denominator = image_norms[:, None] * reference_norms[None, :]
        if not torch.isfinite(denominator).all() or bool((denominator <= 0).any()):
            raise ValueError("CKA denominator must be finite and positive")
        return numerator / (denominator + 1e-12)

    values = []
    for image_index in range(batch_size):
        row = []
        for reference_index in range(reference_count):
            valid = image_mask[image_index].bool() & reference_mask[
                reference_index
            ].bool()
            if int(valid.sum()) < 2:
                raise ValueError("Every image/reference pair needs two valid tokens")
            row.append(
                linear_cka(
                    images[image_index, valid], references[reference_index, valid]
                )
            )
        values.append(torch.stack(row))
    return torch.stack(values)


def clean_anchored_reference_prototype(
    clean: torch.Tensor,
    references: torch.Tensor,
    clean_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align target tokens to each clean image without spatial-index pairing.

    Target images do not have a defensible patch-position correspondence with
    a source image.  This routine establishes a fixed, differentiability-free
    correspondence in the proxy feature space: every clean token softly
    retrieves semantically similar tokens from each target reference, then the
    aligned references are averaged into one per-source target prototype.

    The returned prototype is detached.  During an attack, gradients therefore
    flow only through the adversarial representation, never through clean or
    target-reference images.
    """
    if clean.ndim != 3 or references.ndim != 3:
        raise ValueError("Aligned target inputs must have shape [B,T,D] and [K,T,D]")
    if clean.shape[2] != references.shape[2]:
        raise ValueError("Clean and target tokens must share the proxy feature width")
    if temperature <= 0:
        raise ValueError("Alignment temperature must be positive")
    if clean_mask is None:
        clean_mask = torch.ones(clean.shape[:2], dtype=torch.bool, device=clean.device)
    if reference_mask is None:
        reference_mask = torch.ones(
            references.shape[:2], dtype=torch.bool, device=references.device
        )
    if clean_mask.shape != clean.shape[:2] or reference_mask.shape != references.shape[:2]:
        raise ValueError("Alignment masks must match the token dimensions")

    with torch.no_grad():
        clean_fp32 = clean.detach().float()
        references_fp32 = references.detach().float()
        prototypes = torch.zeros_like(clean_fp32)
        for image_index in range(clean.shape[0]):
            clean_valid = clean_mask[image_index].bool()
            if int(clean_valid.sum()) < 2:
                raise ValueError("Every clean image needs at least two valid tokens")
            queries = functional.normalize(clean_fp32[image_index, clean_valid], dim=-1)
            aligned_sum = torch.zeros_like(clean_fp32[image_index, clean_valid])
            valid_reference_count = 0
            for reference_index in range(references.shape[0]):
                reference_valid = reference_mask[reference_index].bool()
                if int(reference_valid.sum()) < 2:
                    continue
                reference_tokens = references_fp32[reference_index, reference_valid]
                keys = functional.normalize(reference_tokens, dim=-1)
                weights = torch.softmax((queries @ keys.T) / temperature, dim=-1)
                aligned_sum.add_(weights @ reference_tokens)
                valid_reference_count += 1
            if valid_reference_count == 0:
                raise ValueError("Target bank has no reference with two valid tokens")
            prototypes[image_index, clean_valid] = aligned_sum / valid_reference_count
    return prototypes.detach(), clean_mask.detach().bool()
