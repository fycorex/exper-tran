import torch

from primary_ml_cka.attack.cka.linear import linear_cka
from primary_ml_cka.attack.losses import primary as primary_module
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


def test_target_only_objective_removes_source_repulsion() -> None:
    source = torch.randn(2, 6, 8)
    target = torch.randn(4, 6, 8)
    adversarial = torch.randn(2, 6, 8)
    loss = primary_loss(
        torch.tensor(0.0),
        1.0,
        adversarial,
        source,
        target,
        source_cka_weight=0.0,
        target_cka_weight=1.0,
    )
    from primary_ml_cka.attack.cka.linear import token_cka_against_bank

    expected = -token_cka_against_bank(adversarial, target).mean()
    assert torch.allclose(loss.cka, expected)


def test_semantic_only_objective_allows_zero_cka_weights() -> None:
    source = torch.randn(2, 6, 8)
    target = torch.randn(4, 6, 8)
    adversarial = torch.randn(2, 6, 8)
    loss = primary_loss(
        torch.tensor(0.0),
        1.0,
        adversarial,
        source,
        target,
        source_cka_weight=0.0,
        target_cka_weight=0.0,
        semantic_target_weight=1.0,
    )
    assert torch.allclose(
        loss.cka, _semantic_centroid_loss(adversarial, target, None, None)
    )


def test_zero_weight_cka_components_are_not_computed(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("zero-weight CKA component was evaluated")

    monkeypatch.setattr(primary_module, "paired_token_cka", unexpected)
    monkeypatch.setattr(primary_module, "token_cka_against_bank", unexpected)
    source = torch.randn(2, 6, 8)
    target = torch.randn(4, 6, 8)
    adversarial = torch.randn(2, 6, 8)

    loss = primary_module.primary_loss(
        torch.tensor(0.0),
        1.0,
        adversarial,
        source,
        target,
        source_cka_weight=0.0,
        target_cka_weight=0.0,
        semantic_target_weight=1.0,
    )

    assert torch.isfinite(loss.total)
