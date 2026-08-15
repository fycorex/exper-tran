import pytest
import torch

from primary_ml_cka.attack.cka.linear import (
    paired_token_cka,
    token_cka_against_bank,
)
from primary_ml_cka.evaluation.representation_metrics import representation_metrics


def test_metrics_use_per_image_token_cka_and_reference_bank() -> None:
    torch.manual_seed(11)
    clean = torch.randn(3, 5, 4)
    adversarial = clean + 0.2 * torch.randn(3, 5, 4)
    references = torch.randn(2, 5, 4)

    result = representation_metrics(clean, adversarial, references)

    clean_reference = token_cka_against_bank(clean, references).mean()
    adv_source = paired_token_cka(adversarial, clean).mean()
    adv_reference = token_cka_against_bank(adversarial, references).mean()
    assert result.cka_clean_reference == pytest.approx(float(clean_reference))
    assert result.cka_adv_source == pytest.approx(float(adv_source))
    assert result.cka_adv_reference == pytest.approx(float(adv_reference))
    assert result.reference_cka_gain == pytest.approx(
        float(adv_reference - clean_reference)
    )
    assert result.source_cka_drop == pytest.approx(float(1.0 - adv_source))


def test_metrics_use_clean_aligned_target_when_provided() -> None:
    clean = torch.randn(2, 6, 8)
    adversarial = torch.randn(2, 6, 8)
    references = torch.randn(3, 6, 8)
    aligned = torch.randn(2, 6, 8)

    result = representation_metrics(
        clean,
        adversarial,
        references,
        aligned_target=aligned,
    )

    assert result.cka_clean_reference == pytest.approx(
        float(paired_token_cka(clean, aligned).mean())
    )
    assert result.cka_adv_reference == pytest.approx(
        float(paired_token_cka(adversarial, aligned).mean())
    )
