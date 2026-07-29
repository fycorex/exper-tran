from collections.abc import Callable
from dataclasses import dataclass

import torch

from primary_ml_cka.attack.losses.primary import PrimaryLoss
from primary_ml_cka.attack.optimization.momentum_pgd import MomentumPGDState, descent_step


@dataclass(slots=True)
class AttackRunner:
    epsilon: float
    step_size: float
    momentum: float

    def run(
        self,
        clean: torch.Tensor,
        initial: torch.Tensor,
        steps: int,
        loss_function: Callable[[torch.Tensor], PrimaryLoss],
        callback: Callable[[int, PrimaryLoss, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        state = MomentumPGDState(initial, torch.zeros_like(initial))
        for step in range(steps):
            losses = loss_function(state.adversarial)
            if not torch.isfinite(losses.total):
                raise RuntimeError(f"Non-finite total loss at step {step}")
            if callback is not None:
                callback(step, losses, state.adversarial)
            state = descent_step(
                losses.total,
                state,
                clean,
                epsilon=self.epsilon,
                step_size=self.step_size,
                momentum=self.momentum,
            )
        return state.adversarial.detach()
