import runpy
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "2026-08-qwen-transfer-diagnostics"
    / "src"
    / "run_decision_geometry.py"
)


class QuadraticAdapter:
    def target_loss(self, images, target_human_label, prompt):
        del target_human_label, prompt
        batch = images.shape[0]
        score = images.square().flatten(1).sum(dim=1)
        logits = torch.zeros(batch, 10, device=images.device)
        logits[:, 6] = score
        logits[:, 7] = -score
        return type("Output", (), {"class_logits": logits})()


def test_decision_gradients_are_computed_per_image_without_changing_values() -> None:
    module = runpy.run_path(str(SCRIPT))
    images = torch.randn(3, 3, 4, 4, device="cuda")

    gradients, margins = module["_gradient"](QuadraticAdapter(), images, "source_target")

    assert gradients.shape == images.shape
    assert torch.allclose(gradients, 4 * images.cpu())
    expected_margins = 2 * images.square().flatten(1).sum(dim=1).cpu()
    assert torch.allclose(margins, expected_margins)
