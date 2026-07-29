import torch

from primary_ml_cka.domain.types import ImageEmbeddingOutput
from primary_ml_cka.models.common.gradients import (
    assert_frozen,
    assert_input_gradient,
    assert_parameter_gradients_none,
)


def validate_proxy_gradient_path(
    adapter: object, module: torch.nn.Module, images: torch.Tensor
) -> ImageEmbeddingOutput[torch.Tensor]:
    assert_frozen(module)
    output = adapter.image_embeddings(images)
    if not torch.isfinite(output.embeddings).all():
        raise RuntimeError("Proxy embeddings contain non-finite values")
    weights = torch.linspace(
        0.5,
        1.5,
        output.embeddings.numel(),
        device=output.embeddings.device,
        dtype=output.embeddings.dtype,
    ).reshape_as(output.embeddings)
    assert_input_gradient((output.embeddings * weights).sum(), images)
    assert_parameter_gradients_none(module)
    return output
