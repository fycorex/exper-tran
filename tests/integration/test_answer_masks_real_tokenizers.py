from pathlib import Path

import pytest
import torch
from transformers import AutoProcessor

from primary_ml_cka.attack.likelihood.answer_mask import (
    answer_tokenization_candidates,
    build_answer_mask,
)
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.prompts.chat_templates import classification_messages
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

GENERATIVE_MODELS = tuple(
    model_id
    for model_id in MODEL_REVISIONS
    if not model_id.startswith("openai/clip") and not model_id.startswith("google/siglip")
)


@pytest.mark.parametrize("model_id", GENERATIVE_MODELS)
def test_exact_answer_positions_for_cached_processor(model_id: str) -> None:
    snapshot = local_snapshot(Path(".hf-cache"), model_id)
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    tokenizer = processor.tokenizer
    for answer in (str(label) for label in range(1, 11)):
        messages = classification_messages(CLASSIFICATION_PROMPT, answer)
        prompt = processor.apply_chat_template(
            list(messages.prompt_only), tokenize=False, add_generation_prompt=True
        )
        full = processor.apply_chat_template(
            list(messages.with_answer), tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids[0]
        mask = build_answer_mask(
            full_ids,
            prompt_ids,
            answer_tokenization_candidates(tokenizer, answer),
        )
        scored = torch.where(mask.labels != -100)[0]
        assert tuple(scored.tolist()) == mask.label_positions
        assert tuple(full_ids[scored].tolist()) == mask.answer_token_ids
