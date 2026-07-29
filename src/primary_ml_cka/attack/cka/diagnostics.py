import torch


def cka_direction_metrics(
    z_clean: torch.Tensor,
    z_adv: torch.Tensor,
    z_reference: torch.Tensor,
    linear_cka_fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_reference = linear_cka_fn(z_clean, z_reference)
    adv_source = linear_cka_fn(z_adv, z_clean)
    adv_reference = linear_cka_fn(z_adv, z_reference)
    return clean_reference, adv_source, adv_reference
