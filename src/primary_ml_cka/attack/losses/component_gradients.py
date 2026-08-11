from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class ComponentGradientDiagnostics:
    grad_ml_l1: float
    grad_cka_weighted_l1: float
    cosine: float


def calibrate_gradient_ratio(
    grad_ml: torch.Tensor,
    grad_aux: torch.Tensor,
    ratio: float,
) -> tuple[float, ComponentGradientDiagnostics]:
    if ratio <= 0:
        raise ValueError("Gradient ratio must be positive")
    ml_l1 = grad_ml.abs().mean()
    aux_l1 = grad_aux.abs().mean()
    if not torch.isfinite(ml_l1) or ml_l1 <= 0:
        raise ValueError("Classification input gradient must be finite and non-zero")
    if not torch.isfinite(aux_l1) or aux_l1 <= 0:
        raise ValueError("Auxiliary input gradient must be finite and non-zero")
    weight = float(ratio * ml_l1 / aux_l1)
    weighted = weight * grad_aux
    diagnostics = ComponentGradientDiagnostics(
        float(ml_l1),
        float(weighted.abs().mean()),
        float(functional.cosine_similarity(grad_ml.flatten(), weighted.flatten(), dim=0)),
    )
    return weight, diagnostics


def component_gradient_diagnostics(
    loss_ml: torch.Tensor,
    loss_cka: torch.Tensor,
    images: torch.Tensor,
    lambda_cka: float,
) -> ComponentGradientDiagnostics:
    grad_ml = torch.autograd.grad(loss_ml, images, retain_graph=True, only_inputs=True)[0]
    grad_cka = torch.autograd.grad(loss_cka, images, retain_graph=True, only_inputs=True)[0]
    weighted = lambda_cka * grad_cka
    cosine = functional.cosine_similarity(grad_ml.flatten(), weighted.flatten(), dim=0)
    return ComponentGradientDiagnostics(
        float(grad_ml.abs().mean()),
        float(weighted.abs().mean()),
        float(cosine),
    )
