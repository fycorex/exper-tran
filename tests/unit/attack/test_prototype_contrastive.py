import torch

from primary_ml_cka.attack.representation.prototype_contrastive import (
    normalized_prototype,
    prototype_contrastive_loss,
)


def test_target_aligned_embeddings_have_lower_prototype_loss() -> None:
    target = normalized_prototype(torch.eye(4)[:2])
    source = normalized_prototype(-torch.eye(4)[:2])
    aligned = target.repeat(3, 1)
    source_aligned = source.repeat(3, 1)
    target_loss = prototype_contrastive_loss(
        aligned,
        target,
        source,
        margin=0.2,
        separation_weight=1.0,
    )
    source_loss = prototype_contrastive_loss(
        source_aligned,
        target,
        source,
        margin=0.2,
        separation_weight=1.0,
    )
    assert target_loss.total < source_loss.total
    assert torch.all(target_loss.target_source_margin > 0)


def test_gradient_step_increases_target_source_margin() -> None:
    target = torch.tensor([1.0, 0.0, 0.0])
    source = torch.tensor([0.0, 1.0, 0.0])
    embeddings = torch.tensor([[0.2, 0.8, 0.1]], requires_grad=True)
    before = prototype_contrastive_loss(
        embeddings,
        target,
        source,
        margin=0.2,
        separation_weight=1.0,
    )
    gradient = torch.autograd.grad(before.total, embeddings)[0]
    after = prototype_contrastive_loss(
        embeddings - 0.1 * gradient,
        target,
        source,
        margin=0.2,
        separation_weight=1.0,
    )
    assert after.target_source_margin.item() > before.target_source_margin.item()
