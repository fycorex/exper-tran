"""Semantic attraction and source-vs-target contrastive objectives."""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

SEMANTIC_MODES = ("target_only", "prototype", "mean_reference")


@dataclass(frozen=True, slots=True)
class SemanticContrastiveOutput:
    loss: torch.Tensor
    target_similarity: torch.Tensor
    source_similarity: torch.Tensor
    semantic_gap: torch.Tensor


def _normalized_rows(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] < 1:
        raise ValueError(f"{name} must have shape [N,D] with N >= 1")
    return functional.normalize(value.float(), dim=-1)


def semantic_representation_loss(
    adversarial: torch.Tensor,
    target_references: torch.Tensor,
    source_references: torch.Tensor | None = None,
    *,
    mode: str = "target_only",
    tau: float = 0.1,
    target_logit_weight: float = 1.0,
    source_logit_weight: float = 1.0,
) -> SemanticContrastiveOutput:
    """Compute target-only or source-vs-target semantic loss.

    All inputs are pooled per-image embeddings. Reference rows are normalized
    before either constructing a prototype or averaging pairwise similarities.
    """
    if mode not in SEMANTIC_MODES:
        raise ValueError(f"Unknown semantic mode: {mode}")
    if not torch.isfinite(torch.tensor(tau)) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    logit_weights = torch.tensor((target_logit_weight, source_logit_weight))
    if not torch.isfinite(logit_weights).all() or (logit_weights < 0).any():
        raise ValueError("Semantic logit weights must be finite and non-negative")
    adv = _normalized_rows(adversarial, "adversarial")
    target = _normalized_rows(target_references, "target_references")
    if adv.shape[1] != target.shape[1]:
        raise ValueError("Adversarial and target reference dimensions must match")

    if mode == "target_only":
        target_center = functional.normalize(target.mean(dim=0, keepdim=True), dim=-1)
        target_similarity = (adv * target_center).sum(dim=-1)
        source_similarity = torch.full_like(target_similarity, float("nan"))
        return SemanticContrastiveOutput(
            (1.0 - target_similarity).mean(),
            target_similarity.mean(),
            source_similarity.mean(),
            source_similarity.mean(),
        )

    if source_references is None:
        raise ValueError(f"semantic mode {mode!r} requires source references")
    if target_logit_weight == 0 and source_logit_weight == 0:
        raise ValueError("At least one semantic logit weight must be positive")
    source = _normalized_rows(source_references, "source_references")
    if source.shape[1] != adv.shape[1]:
        raise ValueError("Adversarial and source reference dimensions must match")

    if mode == "prototype":
        target_center = functional.normalize(target.mean(dim=0, keepdim=True), dim=-1)
        source_center = functional.normalize(source.mean(dim=0, keepdim=True), dim=-1)
        target_similarity = (adv * target_center).sum(dim=-1)
        source_similarity = (adv * source_center).sum(dim=-1)
    else:
        target_similarity = (adv @ target.T).mean(dim=-1)
        source_similarity = (adv @ source.T).mean(dim=-1)

    # With unit weights this is the original two-class InfoNCE objective.
    # Independent weights expose the gradient-direction ratio explicitly:
    # increasing target_logit_weight strengthens attraction, while increasing
    # source_logit_weight strengthens repulsion. This is intentionally applied
    # inside the logits rather than as an undocumented outer loss multiplier.
    logits = torch.stack(
        (
            target_logit_weight * target_similarity,
            source_logit_weight * source_similarity,
        ),
        dim=-1,
    ) / tau
    labels = torch.zeros(adv.shape[0], dtype=torch.long, device=adv.device)
    loss = functional.cross_entropy(logits, labels)
    return SemanticContrastiveOutput(
        loss,
        target_similarity.mean(),
        source_similarity.mean(),
        (target_similarity - source_similarity).mean(),
    )
