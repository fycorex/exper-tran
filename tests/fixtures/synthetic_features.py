import torch


def distinct_features(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(8, 12, generator=generator)
    target = torch.randn(8, 12, generator=generator)
    return source, target
