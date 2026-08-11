from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from primary_ml_cka.domain.constants import CLASS_MARGIN, MARGIN_TEMPERATURE, MARGIN_WEIGHT


@dataclass(frozen=True, slots=True)
class ProxyClassificationLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    margin_penalty: torch.Tensor
    target_probability: torch.Tensor
    max_other_probability: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ProxyTargetDiagnostics:
    hit_count: int
    denominator: int
    all_hit: bool
    minimum_logit_margin: float
    minimum_target_probability: float
    hit_mask: tuple[bool, ...]


def proxy_target_diagnostics(
    logits: torch.Tensor,
    *,
    target_index: int,
    required_margin: float = CLASS_MARGIN,
    required_probability: float = 0.9,
    free_generated_labels: tuple[int | None, ...] | None = None,
) -> ProxyTargetDiagnostics:
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError(f"Expected ten-class logits [B,10], got {tuple(logits.shape)}")
    logits_fp32 = logits.float()
    target_logits = logits_fp32[:, target_index]
    other_mask = torch.arange(10, device=logits.device) != target_index
    maximum_other = logits_fp32[:, other_mask].max(dim=1).values
    margins = target_logits - maximum_other
    probabilities = logits_fp32.softmax(dim=-1)[:, target_index]
    hits = (
        logits_fp32.argmax(dim=-1).eq(target_index)
        & margins.ge(required_margin)
        & probabilities.ge(required_probability)
    )
    if free_generated_labels is not None:
        if len(free_generated_labels) != logits.shape[0]:
            raise ValueError("Free-generation labels must match the batch size")
        free_hits = torch.tensor(
            tuple(label == target_index + 1 for label in free_generated_labels),
            device=logits.device,
            dtype=torch.bool,
        )
        hits = hits & free_hits
    hit_count = int(hits.sum().item())
    denominator = logits.shape[0]
    return ProxyTargetDiagnostics(
        hit_count=hit_count,
        denominator=denominator,
        all_hit=hit_count == denominator,
        minimum_logit_margin=float(margins.min().item()),
        minimum_target_probability=float(probabilities.min().item()),
        hit_mask=tuple(bool(value) for value in hits.detach().cpu().tolist()),
    )


def proxy_classification_loss(
    logits: torch.Tensor,
    *,
    target_index: int,
    margin: float = CLASS_MARGIN,
    margin_weight: float = MARGIN_WEIGHT,
    temperature: float = MARGIN_TEMPERATURE,
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
    if temperature <= 0:
        raise ValueError("Margin temperature must be positive")
    margin_penalty = functional.softplus(
        (max_other_logits - target_logits + margin) / temperature
    ).mean()
    probabilities = logits_fp32.softmax(dim=-1)
    target_probability = probabilities[:, target_index].mean()
    other_probabilities = probabilities[:, other_mask]
    max_other_probability = other_probabilities.max(dim=1).values.mean()
    total = cross_entropy + margin_weight * margin_penalty
    return ProxyClassificationLoss(
        total,
        cross_entropy,
        margin_penalty,
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
