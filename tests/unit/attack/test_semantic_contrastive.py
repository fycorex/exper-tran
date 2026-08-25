import pytest
import torch

from primary_ml_cka.attack.losses.semantic_contrastive import semantic_representation_loss


@pytest.mark.parametrize("mode", ["prototype", "mean_reference"])
def test_contrastive_loss_prefers_target_direction(mode: str) -> None:
    target = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    source = torch.tensor([[0.0, 1.0], [0.1, 0.9]])
    near_target = torch.tensor([[1.0, 0.05]], requires_grad=True)
    near_source = torch.tensor([[0.05, 1.0]])
    target_output = semantic_representation_loss(near_target, target, source, mode=mode, tau=0.1)
    source_output = semantic_representation_loss(near_source, target, source, mode=mode, tau=0.1)
    assert target_output.loss < source_output.loss
    gradient = torch.autograd.grad(target_output.loss, near_target)[0]
    assert torch.isfinite(gradient).all()


def test_target_only_matches_centroid_cosine_attraction() -> None:
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    adversarial = torch.tensor([[1.0, 0.0]])
    output = semantic_representation_loss(adversarial, target, mode="target_only")
    torch.testing.assert_close(output.loss, torch.tensor(0.0))


def test_target_dominant_weight_increases_target_gradient_share() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0]])
    source = torch.tensor([[0.0, 1.0, 0.0]])
    adversarial = torch.tensor([[0.4, 0.5, 0.768]], requires_grad=True)
    balanced = semantic_representation_loss(
        adversarial,
        target,
        source,
        mode="prototype",
        target_logit_weight=1.0,
        source_logit_weight=1.0,
    )
    target_dominant = semantic_representation_loss(
        adversarial,
        target,
        source,
        mode="prototype",
        target_logit_weight=1.0,
        source_logit_weight=0.25,
    )
    balanced_gradient = torch.autograd.grad(
        balanced.loss, adversarial, retain_graph=True
    )[0]
    target_gradient = torch.autograd.grad(target_dominant.loss, adversarial)[0]
    # Lower source weight rotates the descent direction toward the target
    # direction projected onto the normalized embedding's tangent plane.
    balanced_descent = -balanced_gradient
    target_descent = -target_gradient
    normalized_adversarial = torch.nn.functional.normalize(adversarial.detach(), dim=-1)
    projected_target = target - (
        (target * normalized_adversarial).sum(dim=-1, keepdim=True)
        * normalized_adversarial
    )
    balanced_alignment = torch.nn.functional.cosine_similarity(
        balanced_descent, projected_target
    )
    target_alignment = torch.nn.functional.cosine_similarity(
        target_descent, projected_target
    )
    assert target_alignment > balanced_alignment


@pytest.mark.parametrize(
    ("target_weight", "source_weight"),
    [(-1.0, 1.0), (1.0, -1.0), (float("nan"), 1.0), (0.0, 0.0)],
)
def test_semantic_logit_weights_are_validated(
    target_weight: float, source_weight: float
) -> None:
    with pytest.raises(ValueError):
        semantic_representation_loss(
            torch.ones(1, 2),
            torch.ones(2, 2),
            torch.ones(2, 2),
            mode="prototype",
            target_logit_weight=target_weight,
            source_logit_weight=source_weight,
        )


@pytest.mark.parametrize("tau", [0.0, -0.1, float("nan")])
def test_tau_must_be_positive_and_finite(tau: float) -> None:
    with pytest.raises(ValueError):
        semantic_representation_loss(
            torch.ones(1, 2),
            torch.ones(2, 2),
            torch.ones(2, 2),
            mode="prototype",
            tau=tau,
        )
