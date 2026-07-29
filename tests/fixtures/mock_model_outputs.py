from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MockOutput:
    logits: torch.Tensor
