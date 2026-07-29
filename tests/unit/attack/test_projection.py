import torch

from primary_ml_cka.attack.optimization.projection import project_linf


def test_linf_and_pixel_projection() -> None:
    clean = torch.full((2, 3, 4, 4), 0.5)
    adversarial = clean + torch.linspace(-1, 1, clean.numel()).reshape_as(clean)
    projected = project_linf(adversarial, clean, 16 / 255)
    assert (projected - clean).abs().max() <= 16 / 255 + 1e-7
    assert projected.min() >= 0 and projected.max() <= 1
