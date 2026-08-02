import torch

from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.evaluation.representation_metrics import cross_model_cka


def test_clean_conditioned_hit_counts_and_denominator() -> None:
    rates = attack_rates(
        (4, 4, 7, None),
        (9, 6, 9, 9),
        source_human_label=4,
        target_human_label=9,
    )
    assert rates.clean_valid_count == 2
    assert rates.targeted_hit_count == 1
    assert rates.untargeted_hit_count == 2
    assert rates.tasr_percent == 50
    assert rates.asr_percent == 100


def test_cross_model_cka_has_explicit_dynamic_image_dimension() -> None:
    proxy = torch.randn(7, 13)
    target = torch.randn(7, 5)
    result = cross_model_cka(proxy, target)
    assert result.image_count == 7
    assert result.proxy_embedding_dimension == 13
    assert result.target_embedding_dimension == 5
    assert 0.0 <= result.value <= 1.0 + 1e-6
