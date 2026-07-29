from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AttackState:
    step: int
    adversarial: torch.Tensor
    momentum_buffer: torch.Tensor
