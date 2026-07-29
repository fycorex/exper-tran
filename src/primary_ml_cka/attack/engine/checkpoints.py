from pathlib import Path

import torch

from primary_ml_cka.attack.engine.state import AttackState


def save_state(path: Path, state: AttackState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": state.step,
            "adversarial": state.adversarial.detach().cpu(),
            "momentum_buffer": state.momentum_buffer.detach().cpu(),
        },
        path,
    )
