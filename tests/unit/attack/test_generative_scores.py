import torch

from primary_ml_cka.attack.likelihood.generative_nll import (
    mean_answer_token_log_probability,
)


def test_mean_answer_score_is_token_length_normalized() -> None:
    logits = torch.zeros(2, 4, 3)
    labels = torch.tensor([[-100, 1, -100, -100], [-100, 1, 1, -100]])
    scores = mean_answer_token_log_probability(logits, labels)
    torch.testing.assert_close(scores[0], scores[1])
