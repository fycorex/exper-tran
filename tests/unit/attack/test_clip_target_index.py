import torch

from primary_ml_cka.attack.likelihood.contrastive_ce import (
    contrastive_target_ce,
    proxy_classification_loss,
)


def test_contrastive_ce_uses_requested_target_index() -> None:
    classes = torch.eye(10)
    target_index = 2
    images = classes[target_index].repeat(3, 1)
    loss_correct, probability = contrastive_target_ce(
        images, classes, torch.tensor(20.0), target_index=target_index
    )
    loss_wrong, _ = contrastive_target_ce(images, classes, torch.tensor(20.0), target_index=7)
    assert loss_correct < loss_wrong
    assert probability > 0.99


def test_ranking_and_suppression_enforce_target_dominance() -> None:
    weak = torch.zeros(2, 10)
    strong = weak.clone()
    target_index = 4
    strong[:, target_index] = 6.0
    weak_loss = proxy_classification_loss(weak, target_index=target_index)
    strong_loss = proxy_classification_loss(strong, target_index=target_index)
    assert strong_loss.total < weak_loss.total
    assert strong_loss.target_probability > strong_loss.max_other_probability
    assert strong_loss.other_suppression < weak_loss.other_suppression
