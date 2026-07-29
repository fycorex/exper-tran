from dataclasses import dataclass

import pytest
import torch

from primary_ml_cka.models.proxies.contrastive import _pooled_feature


@dataclass
class StructuredOutput:
    pooler_output: torch.Tensor


def test_extracts_tensor_and_structured_pooler_output() -> None:
    tensor = torch.randn(2, 3)
    assert _pooled_feature(tensor) is tensor
    assert _pooled_feature(StructuredOutput(tensor)) is tensor


def test_rejects_missing_pooler_output() -> None:
    with pytest.raises(TypeError):
        _pooled_feature(object())
