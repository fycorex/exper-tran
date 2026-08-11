from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class AnswerMask:
    labels: torch.Tensor
    answer_token_ids: tuple[int, ...]
    label_positions: tuple[int, ...]


def _find_subsequence(
    sequence: list[int], candidate: list[int], start: int
) -> tuple[int, ...] | None:
    if not candidate:
        return None
    for index in range(start, len(sequence) - len(candidate) + 1):
        if sequence[index : index + len(candidate)] == candidate:
            return tuple(range(index, index + len(candidate)))
    return None


def build_answer_mask(
    full_input_ids: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    candidate_answer_ids: Iterable[Iterable[int]],
) -> AnswerMask:
    if full_input_ids.ndim != 1 or prompt_input_ids.ndim != 1:
        raise ValueError("Answer masking expects unbatched token ID tensors")
    full = full_input_ids.tolist()
    prompt = prompt_input_ids.tolist()
    common = 0
    for left, right in zip(full, prompt, strict=False):
        if left != right:
            break
        common += 1
    matches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for candidate in candidate_answer_ids:
        token_ids = tuple(int(token) for token in candidate)
        # Never allow a class-number occurrence in the user prompt to become a
        # supervised answer token. All supported chat templates retain the
        # generation prefix through ``common``; assistant content starts after it.
        positions = _find_subsequence(full, list(token_ids), common)
        if positions is not None:
            matches.append((token_ids, positions))
    if not matches:
        raise ValueError("Exact assistant answer tokens were not found after the prompt boundary")
    # Prefer the shortest exact tokenization so chat-template boundary tokens
    # (for example a shared assistant newline) are not scored as answer content.
    token_ids, positions = min(matches, key=lambda item: (len(item[0]), item[1][0]))
    labels = torch.full_like(full_input_ids, -100)
    labels[list(positions)] = full_input_ids[list(positions)]
    return AnswerMask(labels, token_ids, positions)


def answer_tokenization_candidates(tokenizer: object, answer: str) -> tuple[tuple[int, ...], ...]:
    candidates = []
    for text in (answer, f" {answer}", f"\n{answer}", f"\n\n{answer}"):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_ids = tuple(int(item) for item in encoded)
        if token_ids and token_ids not in candidates:
            candidates.append(token_ids)
    return tuple(candidates)
