from dataclasses import dataclass

import torch

from primary_ml_cka.attack.cka.linear import paired_token_cka, token_cka_against_bank


@dataclass(frozen=True, slots=True)
class RepresentationMetrics:
    cka_clean_reference: float
    cka_adv_source: float
    cka_adv_reference: float
    reference_cka_gain: float
    source_cka_drop: float
    proxy_representation_shift: float


def representation_metrics(
    z_clean: torch.Tensor,
    z_adv: torch.Tensor,
    z_reference: torch.Tensor,
    clean_mask: torch.Tensor | None = None,
    adv_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> RepresentationMetrics:
    """Summarize the same per-image token CKA geometry used by the attack."""
    clean_reference = token_cka_against_bank(
        z_clean, z_reference, clean_mask, reference_mask
    ).mean()
    adv_source = paired_token_cka(z_adv, z_clean, adv_mask, clean_mask).mean()
    adv_reference = token_cka_against_bank(
        z_adv, z_reference, adv_mask, reference_mask
    ).mean()
    reference_gain = adv_reference - clean_reference
    source_drop = 1.0 - adv_source
    shift = reference_gain + source_drop
    return RepresentationMetrics(
        float(clean_reference),
        float(adv_source),
        float(adv_reference),
        float(reference_gain),
        float(source_drop),
        float(shift),
    )
