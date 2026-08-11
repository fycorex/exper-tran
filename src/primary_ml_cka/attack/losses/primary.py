from dataclasses import dataclass

import torch

from primary_ml_cka.attack.cka.linear import paired_token_cka, token_cka_against_bank


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
    centroid = torch.nn.functional.normalize(
        reference_embeddings.mean(dim=0, keepdim=True), dim=-1
    )
    return (1.0 - (adversarial_embeddings * centroid).sum(dim=-1)).mean()


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
) -> PrimaryLoss:
    weights = (source_cka_weight, target_cka_weight, semantic_target_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("CKA and semantic component weights must be non-negative")
    if lambda_cka == 0:
        zero = loss_ml.new_zeros(())
        return PrimaryLoss(loss_ml, loss_ml, zero, zero, zero, zero)
    if z_adv is None or z_clean is None or z_reference is None:
        raise ValueError(
            "Positive lambda requires adversarial, clean, and proxy-reference embeddings"
        )
    cka_source = paired_token_cka(
        z_adv.float(), z_clean.float(), adv_mask, clean_mask
    ).mean()
    cka_reference = token_cka_against_bank(
        z_adv.float(), z_reference.float(), adv_mask, reference_mask
    ).mean()
    semantic_reference = _semantic_centroid_loss(
        z_adv.float(), z_reference.float(), adv_mask, reference_mask
    )
    loss_cka = (
        source_cka_weight * cka_source
        - target_cka_weight * cka_reference
        + semantic_target_weight * semantic_reference
    )
    return PrimaryLoss(
        loss_ml + lambda_cka * loss_cka,
        loss_ml,
        loss_cka,
        cka_source,
        cka_reference,
        semantic_reference,
    )
