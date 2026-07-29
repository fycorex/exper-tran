from dataclasses import dataclass

import torch

from primary_ml_cka.attack.optimization.projection import project_linf


@dataclass(frozen=True, slots=True)
class MomentumPGDState:
    adversarial: torch.Tensor
    momentum_buffer: torch.Tensor


def descent_step(
    loss_total: torch.Tensor,
    state: MomentumPGDState,
    clean: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    momentum: float,
) -> MomentumPGDState:
    gradient = torch.autograd.grad(loss_total, state.adversarial, only_inputs=True)[0]
    gradient = gradient / (gradient.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
    buffer = momentum * state.momentum_buffer + gradient
    # The single optimizer-direction sign: gradient descent.
    adversarial = state.adversarial - step_size * buffer.sign()
    adversarial = project_linf(adversarial, clean, epsilon)
    return MomentumPGDState(adversarial.detach().requires_grad_(True), buffer.detach())
