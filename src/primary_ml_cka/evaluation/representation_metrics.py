from dataclasses import dataclass

import torch

from primary_ml_cka.attack.cka.linear import linear_cka


@dataclass(frozen=True, slots=True)
class RepresentationMetrics:
    cka_clean_reference: float
    cka_adv_source: float
    cka_adv_reference: float
    reference_cka_gain: float
    source_cka_drop: float
    proxy_representation_shift: float


@dataclass(frozen=True, slots=True)
class CrossModelCKA:
    image_count: int
    proxy_embedding_dimension: int
    target_embedding_dimension: int
    value: float


def cross_model_cka(
    proxy_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
) -> CrossModelCKA:
    """Compare two models on the same ordered set of images."""
    if proxy_embeddings.ndim != 2 or target_embeddings.ndim != 2:
        raise ValueError("Cross-model CKA expects [N_images, D_model] tensors")
    if proxy_embeddings.shape[0] != target_embeddings.shape[0]:
        raise ValueError("Proxy and target must represent the same number of images")
    value = linear_cka(proxy_embeddings, target_embeddings)
    return CrossModelCKA(
        image_count=proxy_embeddings.shape[0],
        proxy_embedding_dimension=proxy_embeddings.shape[1],
        target_embedding_dimension=target_embeddings.shape[1],
        value=float(value),
    )


def representation_metrics(
    z_clean: torch.Tensor, z_adv: torch.Tensor, z_reference: torch.Tensor
) -> RepresentationMetrics:
    clean_reference = linear_cka(z_clean, z_reference)
    adv_source = linear_cka(z_adv, z_clean)
    adv_reference = linear_cka(z_adv, z_reference)
    reference_gain = adv_reference - clean_reference
    source_drop = linear_cka(z_clean, z_clean) - adv_source
    shift = reference_gain + source_drop
    return RepresentationMetrics(
        float(clean_reference),
        float(adv_source),
        float(adv_reference),
        float(reference_gain),
        float(source_drop),
        float(shift),
    )
