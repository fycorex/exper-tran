from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class ComponentGradientDiagnostics:
    grad_ml_l1: float
    grad_cka_weighted_l1: float
    cosine: float


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
