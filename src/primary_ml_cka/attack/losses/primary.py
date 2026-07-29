from dataclasses import dataclass

import torch

from primary_ml_cka.attack.cka.linear import linear_cka


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
) -> PrimaryLoss:
    if lambda_cka == 0:
        zero = loss_ml.new_zeros(())
        return PrimaryLoss(loss_ml, loss_ml, zero, zero, zero)
    if z_adv is None or z_clean is None or z_reference is None:
        raise ValueError(
            "Positive lambda requires adversarial, clean, and proxy-reference embeddings"
        )
    cka_source = linear_cka(z_adv.float(), z_clean.float())
    cka_reference = linear_cka(z_adv.float(), z_reference.float())
    loss_cka = cka_source - cka_reference
    return PrimaryLoss(
        loss_ml + lambda_cka * loss_cka,
        loss_ml,
        loss_cka,
        cka_source,
        cka_reference,
    )
