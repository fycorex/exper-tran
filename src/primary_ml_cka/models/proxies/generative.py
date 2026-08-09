from collections.abc import Callable

import torch
from torch.utils.checkpoint import checkpoint

from primary_ml_cka.attack.likelihood.answer_mask import (
    answer_tokenization_candidates,
    build_answer_mask,
)
from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_classification_loss
from primary_ml_cka.attack.likelihood.generative_nll import answer_sequence_log_probability
from primary_ml_cka.domain.labels import human_label_to_index
from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput
from primary_ml_cka.models.proxies.base import BaseProxy
from primary_ml_cka.prompts.chat_templates import classification_messages


class GenerativeProxy(BaseProxy):
    def __init__(
        self,
        model: torch.nn.Module,
        processor: object,
        visual_inputs: Callable[[torch.Tensor], dict[str, torch.Tensor]],
        image_embedding_fn: Callable[[torch.Tensor], ImageEmbeddingOutput[torch.Tensor]],
        *,
        class_margin: float = 2.0,
        rank_weight: float = 1.0,
        suppression_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.processor = processor
        self.visual_inputs = visual_inputs
        self.image_embedding_fn = image_embedding_fn
        self.class_margin = class_margin
        self.rank_weight = rank_weight
        self.suppression_weight = suppression_weight

    def image_embeddings(self, images: torch.Tensor) -> ImageEmbeddingOutput[torch.Tensor]:
        return self.image_embedding_fn(images)

    def target_loss(
        self, images: torch.Tensor, human_target_label: int, prompt: str
    ) -> ProxyLossOutput[torch.Tensor]:
        target_index = human_label_to_index(human_target_label)
        class_scores = []
        target_mask = None
        target_rendered = ""
        for label in range(1, 11):
            score, rendered, mask = self._answer_score(images, prompt, str(label))
            class_scores.append(score)
            if label == human_target_label:
                target_mask = mask
                target_rendered = rendered
        logits = torch.stack(class_scores, dim=1)
        output = proxy_classification_loss(
            logits,
            target_index=target_index,
            margin=self.class_margin,
            rank_weight=self.rank_weight,
            suppression_weight=self.suppression_weight,
        )
        if target_mask is None:
            raise AssertionError("Target answer mask was not constructed")
        return ProxyLossOutput(
            loss=output.total,
            target_nll=output.cross_entropy,
            target_probability=output.target_probability.detach(),
            class_logits=output.logits,
            classification_ce=output.cross_entropy.detach(),
            rank_loss=output.rank.detach(),
            other_suppression_loss=output.other_suppression.detach(),
            max_other_probability=output.max_other_probability.detach(),
            answer_token_ids=target_mask.answer_token_ids,
            label_positions=target_mask.label_positions,
            rendered_prompt=target_rendered,
        )

    def _answer_score(
        self, images: torch.Tensor, prompt: str, answer: str
    ) -> tuple[torch.Tensor, str, object]:
        messages = classification_messages(prompt, answer)
        tokenizer = self.processor.tokenizer
        prompt_text = self.processor.apply_chat_template(
            list(messages.prompt_only), tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            list(messages.with_answer), tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
        mask = build_answer_mask(
            full_ids, prompt_ids, answer_tokenization_candidates(tokenizer, answer)
        )
        input_ids = full_ids.unsqueeze(0).expand(images.shape[0], -1).to(images.device)
        labels = mask.labels.unsqueeze(0).expand(images.shape[0], -1).to(images.device)
        attention_mask = torch.ones_like(input_ids)

        def score(image_tensor: torch.Tensor) -> torch.Tensor:
            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **self.visual_inputs(image_tensor),
            )
            return answer_sequence_log_probability(model_output.logits, labels)

        if torch.is_grad_enabled() and images.requires_grad:
            sequence_score = checkpoint(score, images, use_reentrant=False)
        else:
            sequence_score = score(images)
        return sequence_score, full_text, mask
