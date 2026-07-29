import torch

from primary_ml_cka.attack.optimization.projection import project_linf


def shared_random_start(clean: torch.Tensor, epsilon: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(seed)
    noise = torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=generator)
    return project_linf(clean + noise, clean, epsilon).detach().requires_grad_(True)
