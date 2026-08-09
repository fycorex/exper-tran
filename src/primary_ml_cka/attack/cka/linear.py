import torch


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
    expanded_images = images[:, None].expand(-1, reference_count, -1, -1)
    expanded_references = references[None].expand(batch_size, -1, -1, -1)
    if image_mask is None:
        image_mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
    if reference_mask is None:
        reference_mask = torch.ones(
            references.shape[:2], dtype=torch.bool, device=references.device
        )
    expanded_image_mask = image_mask[:, None].expand(-1, reference_count, -1)
    expanded_reference_mask = reference_mask[None].expand(batch_size, -1, -1)
    return paired_token_cka(
        expanded_images.reshape(-1, images.shape[1], images.shape[2]),
        expanded_references.reshape(-1, references.shape[1], references.shape[2]),
        expanded_image_mask.reshape(-1, images.shape[1]),
        expanded_reference_mask.reshape(-1, references.shape[1]),
    ).reshape(batch_size, reference_count)
