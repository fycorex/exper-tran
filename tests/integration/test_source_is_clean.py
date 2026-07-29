import torch

from primary_ml_cka.attack.representation.cache import EmbeddingCache


def test_source_representation_is_exact_clean_cache() -> None:
    z_clean = torch.randn(8, 12, device="cuda").detach().float()
    z_reference = torch.randn(8, 12, device="cuda").detach().float()
    cache = EmbeddingCache(
        z_clean,
        z_reference,
        tuple(f"source-{index}" for index in range(8)),
        tuple(f"target-{index}" for index in range(8)),
    )
    device = torch.device("cuda")
    z_source, _ = cache.to(device)
    torch.testing.assert_close(z_source, z_clean, rtol=0, atol=0)
