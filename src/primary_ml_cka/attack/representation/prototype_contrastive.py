from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class PrototypeContrastiveLoss:
    total: torch.Tensor
    attraction: torch.Tensor
    separation: torch.Tensor
    clean_separation: torch.Tensor
    target_similarity: torch.Tensor
    source_similarity: torch.Tensor
    clean_similarity: torch.Tensor
    target_source_margin: torch.Tensor
    target_clean_margin: torch.Tensor


def normalized_prototype(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim != 2 or embeddings.shape[0] < 1:
        raise ValueError("Prototype embeddings must have shape [N,D] with N >= 1")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Prototype embeddings contain non-finite values")
    return functional.normalize(embeddings.float().mean(dim=0), dim=0)


def prototype_contrastive_loss(
    adversarial_embeddings: torch.Tensor,
    target_prototype: torch.Tensor,
    source_prototype: torch.Tensor,
    *,
    margin: float,
    separation_weight: float,
    clean_embeddings: torch.Tensor | None = None,
    clean_separation_weight: float = 0.0,
) -> PrototypeContrastiveLoss:
    if adversarial_embeddings.ndim != 2:
        raise ValueError("Adversarial embeddings must have shape [B,D]")
    if target_prototype.ndim != 1 or source_prototype.ndim != 1:
        raise ValueError("Class prototypes must be one-dimensional")
    if target_prototype.shape != source_prototype.shape:
        raise ValueError("Source and target prototypes must have equal shape")
    if adversarial_embeddings.shape[1] != target_prototype.shape[0]:
        raise ValueError("Embedding and prototype dimensions do not match")
    if margin < 0 or separation_weight < 0 or clean_separation_weight < 0:
        raise ValueError("Margin and separation weight must be non-negative")
    if clean_embeddings is not None and clean_embeddings.shape != adversarial_embeddings.shape:
        raise ValueError("Clean and adversarial embeddings must have equal shape")
    if clean_embeddings is None and clean_separation_weight > 0:
        raise ValueError("Clean embeddings are required for own-clean separation")

    embeddings = functional.normalize(adversarial_embeddings.float(), dim=-1)
    target = functional.normalize(target_prototype.float(), dim=0)
    source = functional.normalize(source_prototype.float(), dim=0)
    target_similarity = embeddings @ target
    source_similarity = embeddings @ source
    target_source_margin = target_similarity - source_similarity
    if clean_embeddings is None:
        clean_similarity = torch.zeros_like(target_similarity)
        target_clean_margin = torch.zeros_like(target_similarity)
        clean_separation = torch.zeros((), device=embeddings.device)
    else:
        clean = functional.normalize(clean_embeddings.float(), dim=-1)
        clean_similarity = (embeddings * clean).sum(dim=-1)
        target_clean_margin = target_similarity - clean_similarity
        clean_separation = functional.softplus(
            clean_similarity - target_similarity + margin
        ).mean()
    attraction = (1.0 - target_similarity).mean()
    separation = functional.softplus(source_similarity - target_similarity + margin).mean()
    total = (
        attraction
        + separation_weight * separation
        + clean_separation_weight * clean_separation
    )
    return PrototypeContrastiveLoss(
        total=total,
        attraction=attraction,
        separation=separation,
        clean_separation=clean_separation,
        target_similarity=target_similarity,
        source_similarity=source_similarity,
        clean_similarity=clean_similarity,
        target_source_margin=target_source_margin,
        target_clean_margin=target_clean_margin,
    )
