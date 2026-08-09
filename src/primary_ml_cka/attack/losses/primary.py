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


def primary_loss(
    loss_ml: torch.Tensor,
    lambda_cka: float,
    z_adv: torch.Tensor | None = None,
    z_clean: torch.Tensor | None = None,
    z_reference: torch.Tensor | None = None,
    adv_mask: torch.Tensor | None = None,
    clean_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
    target_cka_weight: float = 1.0,
) -> PrimaryLoss:
    if target_cka_weight <= 0:
        raise ValueError("target_cka_weight must be positive")
    if lambda_cka == 0:
        zero = loss_ml.new_zeros(())
        return PrimaryLoss(loss_ml, loss_ml, zero, zero, zero)
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
    loss_cka = cka_source - target_cka_weight * cka_reference
    return PrimaryLoss(
        loss_ml + lambda_cka * loss_cka,
        loss_ml,
        loss_cka,
        cka_source,
        cka_reference,
    )
