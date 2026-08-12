import torch

from primary_ml_cka.evaluation.model_similarity import (
    cka_permutation_baseline,
    proxy_target_similarity,
)


def test_global_and_local_similarity_use_matching_calibration_rows() -> None:
    proxy = torch.randn(12, 5)
    target = torch.cat((proxy, torch.zeros(12, 3, device=proxy.device)), dim=1)
    result = proxy_target_similarity(proxy, target, proxy[:3], neighbor_count=4)
    assert result.global_cka > 0.99
    assert len(result.local_cka) == 3
    assert all(value > 0.99 for value in result.local_cka)
    assert all(len(indices) == 4 for indices in result.neighbor_indices)


def test_cka_permutation_baseline_calibrates_small_sample_exactly() -> None:
    torch.manual_seed(7)
    proxy = torch.randn(5, 32)
    target = torch.cat((proxy, torch.zeros(5, 4)), dim=1)

    result = cka_permutation_baseline(proxy, target)

    assert result.true_cka > 0.999
    assert result.exact
    assert result.permutations_evaluated == 119
    assert result.null_mean < result.true_cka
    assert result.empirical_p_value == 1 / 120


def test_cka_permutation_baseline_monte_carlo_is_seeded() -> None:
    torch.manual_seed(11)
    proxy = torch.randn(10, 8)
    target = torch.randn(10, 12)

    first = cka_permutation_baseline(proxy, target, permutation_count=25, seed=9)
    second = cka_permutation_baseline(proxy, target, permutation_count=25, seed=9)

    assert not first.exact
    assert first.permutations_evaluated == 25
    assert first == second
