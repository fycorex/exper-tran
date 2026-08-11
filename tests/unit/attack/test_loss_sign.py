import torch

from primary_ml_cka.attack.cka.linear import linear_cka
from primary_ml_cka.attack.losses.primary import _semantic_centroid_loss, primary_loss


def cka_loss(z_adv: torch.Tensor, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return linear_cka(z_adv, source) - linear_cka(z_adv, target)


def test_target_has_lower_loss_than_source() -> None:
    source = torch.randn(8, 16)
    target = torch.randn(8, 16)
    assert cka_loss(target, source, target) < cka_loss(source, source, target)


def test_gradient_descent_reduces_cka_loss() -> None:
    source = torch.randn(8, 16)
    target = torch.randn(8, 16)
    adversarial = (source + 0.1 * torch.randn_like(source)).requires_grad_()
    before = cka_loss(adversarial, source, target)
    gradient = torch.autograd.grad(before, adversarial)[0]
    after = cka_loss(adversarial - 0.05 * gradient, source, target)
    assert after < before


def test_target_cka_weight_changes_internal_balance() -> None:
    source = torch.randn(8, 12, 16)
    target = torch.randn(8, 12, 16)
    adversarial = torch.randn(8, 12, 16)
    loss = primary_loss(
        torch.tensor(0.0),
        1.0,
        adversarial,
        source,
        target,
        target_cka_weight=5.0,
    )
    from primary_ml_cka.attack.cka.linear import paired_token_cka, token_cka_against_bank

    expected = paired_token_cka(adversarial, source).mean() - 5.0 * token_cka_against_bank(
        adversarial, target
    ).mean()
    assert torch.allclose(loss.cka, expected)


def test_semantic_anchor_prefers_target_centroid() -> None:
    target = torch.zeros(3, 6, 8)
    target[..., 0] = 2.0
    near_target = torch.zeros(2, 6, 8)
    near_target[..., 0] = 1.0
    near_source = torch.zeros(2, 6, 8)
    near_source[..., 1] = 1.0
    target_loss = _semantic_centroid_loss(near_target, target, None, None)
    source_loss = _semantic_centroid_loss(near_source, target, None, None)
    assert target_loss < source_loss
