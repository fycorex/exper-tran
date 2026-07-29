import torch

from primary_ml_cka.domain.types import ImageEmbeddingOutput, TapContract
from primary_ml_cka.models.taps.pooling import masked_mean_l2
from primary_ml_cka.models.taps.validation import validate_proxy_gradient_path


class MockTarget:
    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module

    def image_embeddings(self, images: torch.Tensor):
        tokens = self.module(images.mean(dim=(-1, -2))).unsqueeze(1).expand(-1, 3, -1)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool)
        tap = TapContract("mock", "r", "module", "mock", "mean", "l2", "float32", "test")
        return ImageEmbeddingOutput(masked_mean_l2(tokens, mask), tokens, mask, tap)


def test_frozen_target_has_only_input_gradient() -> None:
    module = torch.nn.Linear(3, 5)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    images = torch.rand(8, 3, 4, 4, requires_grad=True)
    output = validate_proxy_gradient_path(MockTarget(module), module, images)
    assert tuple(output.tokens.shape) == (8, 3, 5)
    assert all(parameter.grad is None for parameter in module.parameters())
