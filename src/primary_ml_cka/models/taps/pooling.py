import torch
import torch.nn.functional as functional


def masked_mean_l2(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,T,D] tokens, got {tuple(tokens.shape)}")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"Mask {tuple(mask.shape)} does not match tokens {tuple(tokens.shape)}")
    if mask.dtype is not torch.bool:
        raise TypeError("Token mask must be boolean")
    counts = mask.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Every image must have at least one valid token")
    pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / counts
    return functional.normalize(pooled.float(), dim=-1)
