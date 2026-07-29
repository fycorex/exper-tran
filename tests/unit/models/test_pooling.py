import torch

from primary_ml_cka.models.taps.pooling import masked_mean_l2


def test_masked_pooling_excludes_invalid_tokens() -> None:
    tokens = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    pooled = masked_mean_l2(tokens, mask)
    torch.testing.assert_close(pooled, torch.tensor([[1.0, 0.0]]))
    assert pooled.dtype == torch.float32
