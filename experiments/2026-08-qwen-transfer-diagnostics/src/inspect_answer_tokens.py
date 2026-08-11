from pathlib import Path

from primary_ml_cka.attack.likelihood.answer_mask import (
    answer_tokenization_candidates,
    build_answer_mask,
)
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.prompts.chat_templates import (
    classification_messages,
    render_chat_template,
)
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def inspect_model(model_id: str) -> None:
    snapshot = local_snapshot(Path(".hf-cache"), model_id, MODEL_REVISIONS[model_id])
    processor = load_processor(snapshot)
    tokenizer = processor.tokenizer
    print(f"MODEL {model_id}")
    for answer in ("7", "10"):
        messages = classification_messages(CLASSIFICATION_PROMPT, answer)
        prompt = render_chat_template(
            processor, messages.prompt_only, add_generation_prompt=True
        )
        full = render_chat_template(
            processor, messages.with_answer, add_generation_prompt=False
        )
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        full_ids = tokenizer(
            full, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        mask = build_answer_mask(
            full_ids,
            prompt_ids,
            answer_tokenization_candidates(tokenizer, answer),
        )
        decoded = tuple(tokenizer.decode([token_id]) for token_id in mask.answer_token_ids)
        plain_ids = tuple(tokenizer(answer, add_special_tokens=False).input_ids)
        plain_decoded = tuple(tokenizer.decode([token_id]) for token_id in plain_ids)
        common_prefix = 0
        for prompt_id, full_id in zip(prompt_ids, full_ids, strict=False):
            if prompt_id != full_id:
                break
            common_prefix += 1
        print(
            {
                "answer": answer,
                "prompt_length": len(prompt_ids),
                "full_length": len(full_ids),
                "common_prefix": common_prefix,
                "mask_positions": mask.label_positions,
                "mask_ids": mask.answer_token_ids,
                "mask_decoded": decoded,
                "plain_ids": plain_ids,
                "plain_decoded": plain_decoded,
            }
        )


def main() -> None:
    inspect_model("Qwen/Qwen3.5-2B")
    inspect_model("Qwen/Qwen3.5-4B")


if __name__ == "__main__":
    main()
