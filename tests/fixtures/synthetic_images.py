import torch


def image_batch(seed: int = 11) -> torch.Tensor:
    return torch.rand(
        8, 3, 224, 224, generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda"
    )
