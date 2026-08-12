import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from primary_ml_cka.attack.cka.linear import linear_cka


@dataclass(frozen=True, slots=True)
class ModelSimilarity:
    global_cka: float
    local_cka: tuple[float, ...]
    neighbor_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class CkaPermutationBaseline:
    true_cka: float
    null_mean: float
    null_std: float
    z_score: float
    empirical_p_value: float
    permutations_evaluated: int
    exact: bool


def cka_permutation_baseline(
    proxy_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    permutation_count: int = 1_000,
    seed: int = 42,
) -> CkaPermutationBaseline:
    """Calibrate paired-image linear CKA against shuffled correspondence.

    For at most six observations, every non-identity permutation is evaluated.
    Larger samples use a deterministic Monte Carlo null. The returned p-value
    is the upper-tail probability that shuffled correspondence reaches the
    observed CKA.
    """
    if proxy_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("CKA permutation inputs must have shape [N,D]")
    if proxy_features.shape[0] != target_features.shape[0]:
        raise ValueError("Permutation inputs must identify the same images")
    if proxy_features.shape[0] < 2:
        raise ValueError("CKA permutation calibration needs at least two images")
    if permutation_count < 1:
        raise ValueError("permutation_count must be positive")

    proxy = proxy_features.float()
    target = target_features.float()
    proxy = proxy - proxy.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    proxy_gram = proxy @ proxy.T
    target_gram = target @ target.T
    denominator = torch.linalg.matrix_norm(proxy_gram) * torch.linalg.matrix_norm(target_gram)
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("CKA permutation denominator must be finite and positive")

    observation_count = proxy.shape[0]
    identity = tuple(range(observation_count))
    exact = observation_count <= 6
    if exact:
        permutations = (
            permutation
            for permutation in itertools.permutations(range(observation_count))
            if permutation != identity
        )
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def sampled_permutations():
            emitted = 0
            while emitted < permutation_count:
                permutation = tuple(
                    torch.randperm(observation_count, generator=generator, device="cpu").tolist()
                )
                if permutation == identity:
                    continue
                emitted += 1
                yield permutation

        permutations = sampled_permutations()

    true_cka = float((proxy_gram * target_gram).sum() / denominator)
    null_values = []
    for permutation in permutations:
        indices = torch.tensor(permutation, dtype=torch.long, device=target_gram.device)
        permuted = target_gram[indices][:, indices]
        null_values.append(float((proxy_gram * permuted).sum() / denominator))
    null = torch.tensor(null_values, dtype=torch.float64)
    null_mean = float(null.mean())
    null_std = float(null.std(unbiased=False))
    z_score = (
        (true_cka - null_mean) / null_std
        if null_std > 0
        else math.copysign(math.inf, true_cka - null_mean)
    )
    empirical_p = (1 + int((null >= true_cka).sum())) / (1 + len(null_values))
    return CkaPermutationBaseline(
        true_cka=true_cka,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        empirical_p_value=empirical_p,
        permutations_evaluated=len(null_values),
        exact=exact,
    )


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
