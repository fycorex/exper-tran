import torch


def project_linf(adversarial: torch.Tensor, clean: torch.Tensor, epsilon: float) -> torch.Tensor:
    projected = torch.maximum(torch.minimum(adversarial, clean + epsilon), clean - epsilon)
    return projected.clamp(0.0, 1.0)
