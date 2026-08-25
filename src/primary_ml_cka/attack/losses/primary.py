from dataclasses import dataclass

import torch

from primary_ml_cka.attack.cka.linear import paired_token_cka, token_cka_against_bank
from primary_ml_cka.attack.losses.semantic_contrastive import semantic_representation_loss


@dataclass(frozen=True, slots=True)
class PrimaryLoss:
    total: torch.Tensor
    ml: torch.Tensor
    cka: torch.Tensor
    cka_source: torch.Tensor
    cka_reference: torch.Tensor
    semantic_reference: torch.Tensor


def _semantic_centroid_loss(
    adversarial: torch.Tensor,
    references: torch.Tensor,
    adversarial_mask: torch.Tensor | None,
    reference_mask: torch.Tensor | None,
) -> torch.Tensor:
    if adversarial_mask is None:
        adversarial_mask = torch.ones(
            adversarial.shape[:2], dtype=torch.bool, device=adversarial.device
        )
    if reference_mask is None:
        reference_mask = torch.ones(
            references.shape[:2], dtype=torch.bool, device=references.device
        )

    def pooled(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        means = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return torch.nn.functional.normalize(means.float(), dim=-1)

    adversarial_embeddings = pooled(adversarial, adversarial_mask)
    reference_embeddings = pooled(references, reference_mask)
    centroid = torch.nn.functional.normalize(reference_embeddings.mean(dim=0, keepdim=True), dim=-1)
    return (1.0 - (adversarial_embeddings * centroid).sum(dim=-1)).mean()


def _semantic_embedding_centroid_loss(
    adversarial: torch.Tensor, references: torch.Tensor
) -> torch.Tensor:
    adversarial = torch.nn.functional.normalize(adversarial.float(), dim=-1)
    references = torch.nn.functional.normalize(references.float(), dim=-1)
    centroid = torch.nn.functional.normalize(references.mean(dim=0, keepdim=True), dim=-1)
    return (1.0 - (adversarial * centroid).sum(dim=-1)).mean()


def primary_loss(
    loss_ml: torch.Tensor,
    lambda_cka: float,
    z_adv: torch.Tensor | None = None,
    z_clean: torch.Tensor | None = None,
    z_reference: torch.Tensor | None = None,
    adv_mask: torch.Tensor | None = None,
    clean_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    source_cka_weight: float = 1.0,
    target_cka_weight: float = 1.0,
    semantic_target_weight: float = 0.0,
    semantic_adv: torch.Tensor | None = None,
    semantic_reference: torch.Tensor | None = None,
    semantic_source_reference: torch.Tensor | None = None,
    semantic_mode: str = "target_only",
    semantic_temperature: float = 0.1,
    semantic_target_logit_weight: float = 1.0,
    semantic_source_logit_weight: float = 1.0,
    lambda_cls: float = 1.0,
    aligned_target: torch.Tensor | None = None,
    aligned_target_mask: torch.Tensor | None = None,
) -> PrimaryLoss:
    weights = (source_cka_weight, target_cka_weight, semantic_target_weight, lambda_cls)
    if any(weight < 0 for weight in weights):
        raise ValueError("CKA and semantic component weights must be non-negative")
    if lambda_cka == 0:
        zero = loss_ml.new_zeros(())
        weighted_ml = lambda_cls * loss_ml
        return PrimaryLoss(weighted_ml, loss_ml, zero, zero, zero, zero)
    needs_token_representations = source_cka_weight > 0 or target_cka_weight > 0
    needs_token_semantic_fallback = (
        semantic_target_weight > 0
        and (semantic_adv is None or semantic_reference is None)
    )
    if (needs_token_representations or needs_token_semantic_fallback) and (
        z_adv is None or z_clean is None or z_reference is None
    ):
        raise ValueError(
            "Positive lambda requires adversarial, clean, and proxy-reference embeddings"
        )
    zero = loss_ml.new_zeros(())
    cka_source = (
        paired_token_cka(z_adv.float(), z_clean.float(), adv_mask, clean_mask).mean()
        if source_cka_weight > 0
        else zero
    )
    if target_cka_weight > 0:
        cka_reference = (
            paired_token_cka(
                z_adv.float(),
                aligned_target.float(),
                adv_mask,
                aligned_target_mask,
            ).mean()
            if aligned_target is not None
            else token_cka_against_bank(
                z_adv.float(), z_reference.float(), adv_mask, reference_mask
            ).mean()
        )
    else:
        cka_reference = zero
    semantic_reference = (
        semantic_representation_loss(
            semantic_adv,
            semantic_reference,
            semantic_source_reference,
            mode=semantic_mode,
            tau=semantic_temperature,
            target_logit_weight=semantic_target_logit_weight,
            source_logit_weight=semantic_source_logit_weight,
        ).loss
        if semantic_target_weight > 0
        and semantic_adv is not None
        and semantic_reference is not None
        else _semantic_centroid_loss(z_adv.float(), z_reference.float(), adv_mask, reference_mask)
        if semantic_target_weight > 0
        else zero
    )
    loss_cka = (
        source_cka_weight * cka_source
        - target_cka_weight * cka_reference
        + semantic_target_weight * semantic_reference
    )
    return PrimaryLoss(
        lambda_cls * loss_ml + lambda_cka * loss_cka,
        loss_ml,
        loss_cka,
        cka_source,
        cka_reference,
        semantic_reference,
    )
