import torch
import torch.nn.functional as functional


def answer_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("Expected logits [B,S,V] and labels [B,S]")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    token_losses = functional.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, ignore_index=-100, reduction="none"
    )
    scored = shifted_labels.ne(-100)
    counts = scored.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every item must score at least one answer token")
    return (token_losses.sum(dim=1) / counts).mean()


def answer_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    log_probabilities = shifted_logits.log_softmax(dim=-1)
    scored = shifted_labels.ne(-100)
    gathered = log_probabilities.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    sequence_log_probability = (gathered * scored).sum(dim=1)
    return sequence_log_probability.exp().mean()


def answer_sequence_log_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    scored = shifted_labels.ne(-100)
    if torch.any(scored.sum(dim=1) == 0):
        raise ValueError("Every item must score at least one answer token")
    log_probabilities = shifted_logits.log_softmax(dim=-1)
    gathered = log_probabilities.gather(-1, shifted_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (gathered * scored).sum(dim=1)
