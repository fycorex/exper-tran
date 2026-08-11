import pytest
import torch

from primary_ml_cka.attack.losses.component_gradients import calibrate_gradient_ratio


def test_calibration_matches_requested_gradient_ratio() -> None:
    grad_cls = torch.full((2, 3, 4), 2.0)
    grad_aux = torch.full((2, 3, 4), 0.25)
    weight, diagnostics = calibrate_gradient_ratio(grad_cls, grad_aux, 0.3)

    assert weight == pytest.approx(2.4)
    assert diagnostics.grad_ml_l1 == pytest.approx(2.0)
    assert diagnostics.grad_cka_weighted_l1 == pytest.approx(0.6)
    assert diagnostics.grad_cka_weighted_l1 / diagnostics.grad_ml_l1 == pytest.approx(0.3)


def test_calibration_rejects_zero_auxiliary_gradient() -> None:
    with pytest.raises(ValueError, match="Auxiliary"):
        calibrate_gradient_ratio(torch.ones(2), torch.zeros(2), 0.3)
