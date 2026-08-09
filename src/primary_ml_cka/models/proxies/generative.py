from collections.abc import Callable

import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from primary_ml_cka.attack.likelihood.answer_mask import (
    answer_tokenization_candidates,
    build_answer_mask,
)
from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_classification_loss
from primary_ml_cka.attack.likelihood.generative_nll import mean_answer_token_log_probability
from primary_ml_cka.domain.labels import human_label_to_index
from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput
from primary_ml_cka.models.proxies.base import BaseProxy
from primary_ml_cka.prompts.chat_templates import classification_messages
from primary_ml_cka.prompts.parser import parse_exact_label


class GenerativeProxy(BaseProxy):
    def __init__(
        self,
        model: torch.nn.Module,
        processor: object,
        visual_inputs: Callable[[torch.Tensor], dict[str, torch.Tensor]],
        image_embedding_fn: Callable[[torch.Tensor], ImageEmbeddingOutput[torch.Tensor]],
        *,
        class_margin: float = 2.0,
        margin_weight: float = 1.0,
        margin_temperature: float = 0.5,
        microbatch_size: int | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.visual_inputs = visual_inputs
        self.image_embedding_fn = image_embedding_fn
        self.class_margin = class_margin
        self.margin_weight = margin_weight
        self.margin_temperature = margin_temperature
        self.microbatch_size = microbatch_size

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
            margin_weight=self.margin_weight,
            temperature=self.margin_temperature,
        )
        if target_mask is None:
            raise AssertionError("Target answer mask was not constructed")
        target_nll = -logits[:, target_index].mean()
        return ProxyLossOutput(
            loss=output.total,
            target_nll=target_nll.detach(),
            target_probability=output.target_probability.detach(),
            class_logits=output.logits,
            classification_ce=output.cross_entropy.detach(),
            margin_loss=output.margin_penalty.detach(),
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
        with torch.no_grad():
            visual_token_count = self._visual_token_count(self.visual_inputs(images.detach()))
        prompt_ids, _ = self._multimodal_token_ids(prompt_text, visual_token_count)
        full_ids, full_mm_types = self._multimodal_token_ids(full_text, visual_token_count)
        mask = build_answer_mask(
            full_ids, prompt_ids, answer_tokenization_candidates(tokenizer, answer)
        )
        def score(image_tensor: torch.Tensor) -> torch.Tensor:
            item_count = image_tensor.shape[0]
            input_ids = full_ids.unsqueeze(0).expand(item_count, -1).to(image_tensor.device)
            labels = mask.labels.unsqueeze(0).expand(item_count, -1).to(image_tensor.device)
            attention_mask = torch.ones_like(input_ids)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                **self.visual_inputs(image_tensor),
            }
            if full_mm_types is not None:
                model_inputs["mm_token_type_ids"] = (
                    full_mm_types.unsqueeze(0)
                    .expand(item_count, -1)
                    .to(image_tensor.device)
                )
            model_output = self.model(
                **model_inputs,
            )
            return mean_answer_token_log_probability(model_output.logits, labels)

        chunk_size = getattr(self, "microbatch_size", None) or images.shape[0]
        score_chunks = []
        for image_chunk in images.split(chunk_size):
            if torch.is_grad_enabled() and images.requires_grad:
                score_chunks.append(checkpoint(score, image_chunk, use_reentrant=False))
            else:
                score_chunks.append(score(image_chunk))
        sequence_score = torch.cat(score_chunks)
        return sequence_score, full_text, mask

    def _multimodal_token_ids(
        self, text: str, visual_token_count: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Let the native processor expand one image placeholder into visual tokens."""
        dummy = Image.new("RGB", (224, 224))
        encoded = self.processor(text=text, images=dummy, return_tensors="pt")
        input_ids = encoded.input_ids[0]
        has_mm_types = "mm_token_type_ids" in encoded
        if visual_token_count is None:
            mm_types = encoded["mm_token_type_ids"][0] if has_mm_types else None
            return input_ids, mm_types
        image_token_id = int(self.model.config.image_token_id)
        positions = input_ids.eq(image_token_id).nonzero(as_tuple=False).flatten()
        if len(positions) == visual_token_count:
            mm_types = encoded["mm_token_type_ids"][0] if has_mm_types else None
            return input_ids, mm_types
        if len(positions) == 0:
            raise ValueError("Native processor did not emit image tokens")
        first, last = int(positions[0]), int(positions[-1]) + 1
        if not torch.all(input_ids[first:last].eq(image_token_id)):
            raise ValueError("Image tokens must form one contiguous run")
        replacement = input_ids.new_full((visual_token_count,), image_token_id)
        input_ids = torch.cat((input_ids[:first], replacement, input_ids[last:]))
        mm_types = input_ids.eq(image_token_id).to(dtype=input_ids.dtype) if has_mm_types else None
        return input_ids, mm_types

    def _visual_token_count(self, visual_inputs: dict[str, torch.Tensor]) -> int | None:
        grid = visual_inputs.get("image_grid_thw")
        if grid is None:
            return None
        merge_size = int(self.model.config.vision_config.spatial_merge_size)
        counts = grid.prod(dim=1) // (merge_size * merge_size)
        if not torch.all(counts.eq(counts[0])):
            raise ValueError("A proxy batch must use equal visual token counts")
        return int(counts[0])

    def free_generate_labels(
        self, images: torch.Tensor, prompt: str
    ) -> tuple[int | None, ...]:
        messages = classification_messages(prompt).prompt_only
        try:
            rendered = self.processor.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.processor.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        visual_inputs = self.visual_inputs(images)
        visual_token_count = self._visual_token_count(visual_inputs)
        input_ids, mm_types = self._multimodal_token_ids(rendered, visual_token_count)
        input_ids = input_ids.to(images.device)
        input_ids = input_ids.unsqueeze(0).expand(images.shape[0], -1)
        text_inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        if mm_types is not None:
            text_inputs["mm_token_type_ids"] = mm_types.to(images.device).unsqueeze(0).expand(
                images.shape[0], -1
            )
        generated = self.model.generate(
            **text_inputs,
            **visual_inputs,
            do_sample=False,
            temperature=0,
            max_new_tokens=4,
            use_cache=False,
        )
        prompt_length = text_inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        return tuple(parse_exact_label(raw).label for raw in decoded)
