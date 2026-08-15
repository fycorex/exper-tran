import torch

from primary_ml_cka.attack.cka.linear import linear_cka


def test_total_loss_autograd_matches_centered_finite_difference() -> None:
    torch.manual_seed(3)
    pixels = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    clean = torch.randn(8, 5, dtype=torch.float64)
    target = torch.randn(8, 5, dtype=torch.float64)
    projection = torch.randn(6, 5, dtype=torch.float64)

    def total(value: torch.Tensor) -> torch.Tensor:
        z_adv = value @ projection
        ml = (value.square()).mean()
        return ml + 0.1 * (linear_cka(z_adv, clean) - linear_cka(z_adv, target))

    direction = torch.randn_like(pixels)
    direction /= direction.norm()
    gradient = torch.autograd.grad(total(pixels), pixels)[0]
    autograd_directional = (gradient * direction).sum()
    # linear_cka deliberately evaluates high-width model features in FP32.
    # A 1e-4 float64 probe falls below a stable FP32 difference on CPU.
    step = 1e-3
    finite = (total(pixels + step * direction) - total(pixels - step * direction)) / (2 * step)
    torch.testing.assert_close(autograd_directional, finite, atol=2e-5, rtol=2e-4)
