import torch

from primary_ml_cka.attack.likelihood.contrastive_ce import (
    contrastive_target_ce,
    proxy_classification_loss,
    proxy_target_diagnostics,
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


def test_ce_and_margin_penalty_enforce_target_dominance() -> None:
    weak = torch.zeros(2, 10)
    strong = weak.clone()
    target_index = 4
    strong[:, target_index] = 6.0
    weak_loss = proxy_classification_loss(weak, target_index=target_index)
    strong_loss = proxy_classification_loss(strong, target_index=target_index)
    assert strong_loss.total < weak_loss.total
    assert strong_loss.target_probability > strong_loss.max_other_probability
    assert strong_loss.margin_penalty < weak_loss.margin_penalty


def test_proxy_target_gate_requires_every_image_to_reach_target() -> None:
    logits = torch.zeros(3, 10)
    logits[:, 2] = torch.tensor([7.0, 6.0, 0.5])
    logits[2, 4] = 0.6
    failed = proxy_target_diagnostics(logits, target_index=2)
    assert failed.hit_count == 2
    assert failed.denominator == 3
    assert not failed.all_hit
    assert failed.minimum_logit_margin < 0
    assert failed.hit_mask == (True, True, False)

    logits[2, 2] = 7.0
    passed = proxy_target_diagnostics(logits, target_index=2)
    assert passed.all_hit
    assert passed.hit_count == passed.denominator == 3
    assert passed.hit_mask == (True, True, True)


def test_proxy_target_gate_checks_margin_probability_and_free_generation() -> None:
    logits = torch.zeros(2, 10)
    logits[:, 4] = 7.0
    assert proxy_target_diagnostics(logits, target_index=4).all_hit
    assert not proxy_target_diagnostics(
        logits, target_index=4, free_generated_labels=(5, 3)
    ).all_hit
    low_probability = torch.zeros(1, 10)
    low_probability[:, 4] = 2.1
    assert not proxy_target_diagnostics(low_probability, target_index=4).all_hit
