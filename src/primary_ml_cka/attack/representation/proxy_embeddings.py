import torch

from primary_ml_cka.models.common.protocols import ProxyModel


def detached_proxy_embedding(model: ProxyModel, images: torch.Tensor) -> torch.Tensor:
    if images.device.type != "cuda":
        raise ValueError("Proxy embeddings require CUDA images")
    with torch.no_grad():
        output = model.image_embeddings(images)
    return output.embeddings.detach().float()
