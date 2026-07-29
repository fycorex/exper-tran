import torch

from primary_ml_cka.models.taps.masks import gemma_position_mask, lengths_to_mask


def test_lengths_mask() -> None:
    actual = lengths_to_mask(torch.tensor([2, 1]), maximum=3)
    expected = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.equal(actual, expected)


def test_gemma_position_mask() -> None:
    positions = torch.tensor([[[0, 0], [1, 0], [-1, -1]]])
    assert torch.equal(gemma_position_mask(positions), torch.tensor([[True, True, False]]))
