import torch

from primary_ml_cka.evaluation.model_similarity import proxy_target_similarity


def test_global_and_local_similarity_use_matching_calibration_rows() -> None:
    proxy = torch.randn(12, 5)
    target = torch.cat((proxy, torch.zeros(12, 3, device=proxy.device)), dim=1)
    result = proxy_target_similarity(proxy, target, proxy[:3], neighbor_count=4)
    assert result.global_cka > 0.99
    assert len(result.local_cka) == 3
    assert all(value > 0.99 for value in result.local_cka)
    assert all(len(indices) == 4 for indices in result.neighbor_indices)
