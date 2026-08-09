import torch
import torch.nn.functional as functional

from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_classification_loss


def test_target_nll_is_differentiable_target_class_cross_entropy() -> None:
    logits = torch.randn(6, 10, requires_grad=True)
    target_index = 4
    output = proxy_classification_loss(logits, target_index=target_index)
    expected = functional.cross_entropy(
        logits,
        torch.full((logits.shape[0],), target_index, device=logits.device),
    )
    assert torch.allclose(output.cross_entropy, expected)
    gradient = torch.autograd.grad(output.cross_entropy, logits)[0]
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient[:, target_index] < 0)
