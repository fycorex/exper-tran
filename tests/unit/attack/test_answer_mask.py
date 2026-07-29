import torch

from primary_ml_cka.attack.likelihood.answer_mask import build_answer_mask


def test_scores_only_multitoken_answer() -> None:
    full = torch.tensor([1, 2, 30, 31, 99])
    prompt = torch.tensor([1, 2])
    result = build_answer_mask(full, prompt, ((7,), (30, 31), (8, 30, 31)))
    assert result.answer_token_ids == (30, 31)
    assert result.label_positions == (2, 3)
    assert result.labels.tolist() == [-100, -100, 30, 31, -100]


def test_leading_newline_candidate_is_supported() -> None:
    full = torch.tensor([1, 55, 7, 2])
    prompt = torch.tensor([1])
    result = build_answer_mask(full, prompt, ((7,), (55, 7)))
    assert result.answer_token_ids == (55, 7)
