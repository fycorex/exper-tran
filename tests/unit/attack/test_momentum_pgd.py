import torch

from primary_ml_cka.attack.optimization.momentum_pgd import MomentumPGDState, descent_step


def test_optimizer_uses_gradient_descent_once() -> None:
    clean = torch.full((1, 3, 2, 2), 0.5)
    adversarial = clean.clone().requires_grad_()
    state = MomentumPGDState(adversarial, torch.zeros_like(adversarial))
    loss = adversarial.sum()
    updated = descent_step(loss, state, clean, epsilon=0.2, step_size=0.1, momentum=1.0)
    torch.testing.assert_close(updated.adversarial, clean - 0.1)
