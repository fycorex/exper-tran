import torch


def lengths_to_mask(lengths: torch.Tensor, maximum: int | None = None) -> torch.Tensor:
    if lengths.ndim != 1 or torch.any(lengths <= 0):
        raise ValueError("Lengths must be a positive one-dimensional tensor")
    maximum = maximum or int(lengths.max())
    return torch.arange(maximum, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def gemma_position_mask(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.ndim != 3 or position_ids.shape[-1] != 2:
        raise ValueError("Gemma image positions must have shape [B,T,2]")
    return (position_ids >= 0).all(dim=-1)
