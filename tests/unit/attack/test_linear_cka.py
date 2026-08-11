import pytest
import torch

from primary_ml_cka.attack.cka.linear import (
    linear_cka,
    paired_token_cka,
    token_cka_against_bank,
)


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


def test_token_cka_returns_one_value_per_image_not_one_per_batch() -> None:
    images = torch.randn(3, 6, 4)
    values = paired_token_cka(images, images)
    assert values.shape == (3,)
    torch.testing.assert_close(values, torch.ones_like(values), atol=1e-5, rtol=1e-5)


def test_token_cka_reference_bank_averages_independent_references() -> None:
    images = torch.randn(2, 6, 4)
    references = torch.randn(3, 6, 7)
    values = token_cka_against_bank(images, references)
    assert values.shape == (2, 3)
    assert torch.isfinite(values).all()


def test_token_cka_bank_matches_explicit_pairwise_values() -> None:
    images = torch.randn(2, 5, 7)
    references = torch.randn(3, 5, 4)
    values = token_cka_against_bank(images, references)
    expected = torch.stack(
        [
            torch.stack([linear_cka(image, reference) for reference in references])
            for image in images
        ]
    )
    assert torch.allclose(values, expected, atol=1e-6, rtol=1e-5)
