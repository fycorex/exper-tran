from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from primary_ml_cka.domain.constants import (
    CLASS_MARGIN,
    OTHER_SUPPRESSION_WEIGHT,
    RANK_WEIGHT,
)


@dataclass(frozen=True, slots=True)
class ProxyClassificationLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    rank: torch.Tensor
    other_suppression: torch.Tensor
    target_probability: torch.Tensor
    max_other_probability: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProxyTargetDiagnostics:
    hit_count: int
    denominator: int
    all_hit: bool
    minimum_logit_margin: float


def proxy_target_diagnostics(
    logits: torch.Tensor,
    *,
    target_index: int,
) -> ProxyTargetDiagnostics:
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError(f"Expected ten-class logits [B,10], got {tuple(logits.shape)}")
    logits_fp32 = logits.float()
    target_logits = logits_fp32[:, target_index]
    other_mask = torch.arange(10, device=logits.device) != target_index
    maximum_other = logits_fp32[:, other_mask].max(dim=1).values
    margins = target_logits - maximum_other
    hits = margins > 0
    hit_count = int(hits.sum().item())
    denominator = logits.shape[0]
    return ProxyTargetDiagnostics(
        hit_count=hit_count,
        denominator=denominator,
        all_hit=hit_count == denominator,
        minimum_logit_margin=float(margins.min().item()),
    )


def proxy_classification_loss(
    logits: torch.Tensor,
    *,
    target_index: int,
    margin: float = CLASS_MARGIN,
    rank_weight: float = RANK_WEIGHT,
    suppression_weight: float = OTHER_SUPPRESSION_WEIGHT,
) -> ProxyClassificationLoss:
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError(f"Expected ten-class logits [B,10], got {tuple(logits.shape)}")
    targets = torch.full((logits.shape[0],), target_index, dtype=torch.long, device=logits.device)
    logits_fp32 = logits.float()
    cross_entropy = functional.cross_entropy(logits_fp32, targets)
    target_logits = logits_fp32[:, target_index]
    other_mask = torch.arange(10, device=logits.device) != target_index
    other_logits = logits_fp32[:, other_mask]
    max_other_logits = other_logits.max(dim=1).values
    rank = functional.softplus(max_other_logits - target_logits + margin).mean()
    probabilities = logits_fp32.softmax(dim=-1)
    target_probability = probabilities[:, target_index].mean()
    other_probabilities = probabilities[:, other_mask]
    other_suppression = other_probabilities.mean()
    max_other_probability = other_probabilities.max(dim=1).values.mean()
    total = cross_entropy + rank_weight * rank + suppression_weight * other_suppression
    return ProxyClassificationLoss(
        total,
        cross_entropy,
        rank,
        other_suppression,
        target_probability,
        max_other_probability,
        logits_fp32,
    )


def contrastive_target_ce(
    image_embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
    target_index: int,
    logit_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_embeddings.ndim != 2 or class_embeddings.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    images = functional.normalize(image_embeddings, dim=-1)
    classes = functional.normalize(class_embeddings, dim=-1)
    logits = logit_scale * (images @ classes.T)
    if logit_bias is not None:
        logits = logits + logit_bias
    output = proxy_classification_loss(logits, target_index=target_index)
    return output.total, output.target_probability
