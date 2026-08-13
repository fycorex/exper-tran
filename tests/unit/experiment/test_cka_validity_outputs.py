import runpy
from dataclasses import dataclass
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "2026-08-qwen-transfer-diagnostics"
    / "src"
    / "run_cka_validity.py"
)


@dataclass
class StructuredImageFeatures:
    pooler_output: torch.Tensor


def test_pooled_row_accepts_structured_contrastive_image_features() -> None:
    pooled_row = runpy.run_path(str(SCRIPT))["_pooled_row"]
    features = torch.randn(1, 16)

    result = pooled_row(StructuredImageFeatures(features))

    assert result.shape == (16,)
    assert torch.isclose(result.norm(), torch.tensor(1.0))
