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
