from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from primary_ml_cka.attack.cka.linear import linear_cka


@dataclass(frozen=True, slots=True)
class ModelSimilarity:
    global_cka: float
    local_cka: tuple[float, ...]
    neighbor_indices: tuple[tuple[int, ...], ...]


def proxy_target_similarity(
    proxy_calibration: torch.Tensor,
    target_calibration: torch.Tensor,
    proxy_queries: torch.Tensor,
    *,
    neighbor_count: int,
) -> ModelSimilarity:
    """Compute pair-level global and query-level local CKA on identical images."""
    if proxy_calibration.ndim != 2 or target_calibration.ndim != 2:
        raise ValueError("Calibration representations must have shape [N,D]")
    if proxy_calibration.shape[0] != target_calibration.shape[0]:
        raise ValueError("Proxy and target calibration rows must identify the same images")
    if proxy_queries.ndim != 2 or proxy_queries.shape[1] != proxy_calibration.shape[1]:
        raise ValueError("Proxy queries must use the proxy calibration feature space")
    if not 2 <= neighbor_count <= proxy_calibration.shape[0]:
        raise ValueError("neighbor_count must be between 2 and calibration size")
    proxy_bank = functional.normalize(proxy_calibration.float(), dim=-1)
    queries = functional.normalize(proxy_queries.float(), dim=-1)
    neighbors = (queries @ proxy_bank.T).topk(neighbor_count, dim=1).indices
    local_values = []
    for indices in neighbors:
        local_values.append(
            float(linear_cka(proxy_calibration[indices], target_calibration[indices]))
        )
    return ModelSimilarity(
        float(linear_cka(proxy_calibration, target_calibration)),
        tuple(local_values),
        tuple(tuple(int(index) for index in row) for row in neighbors.tolist()),
    )
