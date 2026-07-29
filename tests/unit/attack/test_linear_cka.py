import pytest
import torch

from primary_ml_cka.attack.cka.linear import linear_cka


def test_self_cka_is_one_and_finite() -> None:
    x = torch.randn(5, 20)
    value = linear_cka(x, x)
    torch.testing.assert_close(value, torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(value)


def test_orthogonal_rotation_invariance() -> None:
    x = torch.randn(11, 12)
    q, _ = torch.linalg.qr(torch.randn(12, 12))
    torch.testing.assert_close(linear_cka(x, x @ q), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_batch_size_is_generic_but_must_match() -> None:
    value = linear_cka(torch.randn(3, 4), torch.randn(3, 9))
    assert torch.isfinite(value)
    with pytest.raises(ValueError, match="same batch size"):
        linear_cka(torch.randn(3, 4), torch.randn(4, 4))
