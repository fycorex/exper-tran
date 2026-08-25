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
from primary_ml_cka.domain.output_codes import human_label_to_output_code
from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput
from primary_ml_cka.models.proxies.base import BaseProxy
from primary_ml_cka.prompts.chat_templates import (
    classification_messages,
    render_chat_template,
)
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

    def image_embeddings(
        self,
        images: torch.Tensor,
        *,
        representation_type: str = "legacy_projected",
        layer: int = -1,
        pooling: str = "mean",
    ) -> ImageEmbeddingOutput[torch.Tensor]:
        return self.image_embedding_fn(
            images,
            representation_type=representation_type,
            layer=layer,
            pooling=pooling,
        )

    def target_loss(
        self,
        images: torch.Tensor,
        human_target_label: int,
        prompt: str,
        cls_loss_mode: str = "ce_margin",
    ) -> ProxyLossOutput[torch.Tensor]:
        target_index = human_label_to_index(human_target_label)
        joint = self._joint_answer_scores(images, prompt)
        if joint is None:
            scored = [
                self._answer_score(images, prompt, human_label_to_output_code(label))
                for label in range(1, 11)
            ]
            logits = torch.stack([item[0] for item in scored], dim=1)
            target_rendered, target_mask = scored[target_index][1:]
        else:
            logits, rendered_prompts, masks = joint
            target_rendered = rendered_prompts[target_index]
            target_mask = masks[target_index]
        output = proxy_classification_loss(
            logits,
            target_index=target_index,
            margin=self.class_margin,
            margin_weight=self.margin_weight,
            temperature=self.margin_temperature,
        )
        target_nll = -logits[:, target_index].mean()
        objectives = {
            "none": logits.sum() * 0.0,
            "target_token_nll": target_nll,
            "closedset_ce": output.cross_entropy,
            "margin_only": output.margin_penalty,
            "ce_margin": output.total,
        }
        if cls_loss_mode not in objectives:
            raise ValueError(f"Unknown classification loss mode: {cls_loss_mode}")
        return ProxyLossOutput(
            loss=objectives[cls_loss_mode],
            target_nll=target_nll,
            target_probability=output.target_probability.detach(),
            class_logits=output.logits,
            classification_ce=output.cross_entropy,
            margin_loss=output.margin_penalty,
            max_other_probability=output.max_other_probability.detach(),
            answer_token_ids=target_mask.answer_token_ids,
            label_positions=target_mask.label_positions,
            rendered_prompt=target_rendered,
        )

    def _joint_answer_scores(self, images: torch.Tensor, prompt: str):
        """Score a shared answer prefix and ten final class tokens in one forward."""
        with torch.no_grad():
            visual_token_count = self._visual_token_count(self.visual_inputs(images.detach()))
        encodings = [
            self._answer_encoding(prompt, human_label_to_output_code(label), visual_token_count)
            for label in range(1, 11)
        ]
        rendered_prompts = tuple(item[0] for item in encodings)
        full_ids = tuple(item[1] for item in encodings)
        mm_types = tuple(item[2] for item in encodings)
        masks = tuple(item[3] for item in encodings)
        positions = tuple(tuple(mask.label_positions) for mask in masks)
        answer_ids = tuple(tuple(mask.answer_token_ids) for mask in masks)
        if any(not item for item in answer_ids):
            return None
        groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for index, (item_positions, item_ids) in enumerate(zip(positions, answer_ids, strict=True)):
            groups.setdefault((item_positions, item_ids[:-1]), []).append(index)
        chunk_size = getattr(self, "microbatch_size", None) or images.shape[0]
        class_scores: list[torch.Tensor | None] = [None] * 10
        for indices in groups.values():
            representative_ids = full_ids[indices[0]]
            representative_mm = mm_types[indices[0]]
            scored_positions = torch.tensor(
                [position - 1 for position in positions[indices[0]]], dtype=torch.long
            )
            candidate_ids = torch.tensor([answer_ids[index] for index in indices])

            def score(
                image_tensor: torch.Tensor,
                representative_ids: torch.Tensor = representative_ids,
                representative_mm: torch.Tensor | None = representative_mm,
                scored_positions: torch.Tensor = scored_positions,
                candidate_ids: torch.Tensor = candidate_ids,
            ) -> torch.Tensor:
                item_count = image_tensor.shape[0]
                input_ids = (
                    representative_ids.unsqueeze(0).expand(item_count, -1).to(image_tensor.device)
                )
                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "use_cache": False,
                    **self.visual_inputs(image_tensor),
                }
                if representative_mm is not None:
                    model_inputs["mm_token_type_ids"] = (
                        representative_mm.unsqueeze(0)
                        .expand(item_count, -1)
                        .to(image_tensor.device)
                    )
                output = self.model(**model_inputs)
                selected = output.logits[:, scored_positions.to(image_tensor.device)].float()
                log_probabilities = selected.log_softmax(dim=-1)
                scores = []
                for ids in candidate_ids.to(image_tensor.device):
                    gathered = log_probabilities.gather(
                        -1, ids.view(1, -1, 1).expand(item_count, -1, -1)
                    ).squeeze(-1)
                    scores.append(gathered.mean(dim=1))
                return torch.stack(scores, dim=1)

            chunks = []
            for image_chunk in images.split(chunk_size):
                if torch.is_grad_enabled() and images.requires_grad:
                    chunks.append(checkpoint(score, image_chunk, use_reentrant=False))
                else:
                    chunks.append(score(image_chunk))
            group_scores = torch.cat(chunks)
            for column, class_index in enumerate(indices):
                class_scores[class_index] = group_scores[:, column]
        if any(item is None for item in class_scores):
            raise AssertionError("Every closed-set class must receive a score")
        return torch.stack(class_scores, dim=1), rendered_prompts, masks  # type: ignore[arg-type]

    def _answer_encoding(
        self, prompt: str, answer: str, visual_token_count: int | None
    ) -> tuple[str, torch.Tensor, torch.Tensor | None, object]:
        messages = classification_messages(prompt, answer)
        tokenizer = self.processor.tokenizer
        prompt_text = render_chat_template(
            self.processor,
            messages.prompt_only,
            add_generation_prompt=True,
        )
        full_text = render_chat_template(
            self.processor,
            messages.with_answer,
            add_generation_prompt=False,
        )
        prompt_ids, _ = self._multimodal_token_ids(prompt_text, visual_token_count)
        full_ids, full_mm_types = self._multimodal_token_ids(full_text, visual_token_count)
        mask = build_answer_mask(
            full_ids, prompt_ids, answer_tokenization_candidates(tokenizer, answer)
        )
        return full_text, full_ids, full_mm_types, mask

    def _answer_score(
        self, images: torch.Tensor, prompt: str, answer: str
    ) -> tuple[torch.Tensor, str, object]:
        with torch.no_grad():
            visual_token_count = self._visual_token_count(self.visual_inputs(images.detach()))
        full_text, full_ids, full_mm_types, mask = self._answer_encoding(
            prompt, answer, visual_token_count
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
                    full_mm_types.unsqueeze(0).expand(item_count, -1).to(image_tensor.device)
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

    def free_generate_labels(self, images: torch.Tensor, prompt: str) -> tuple[int | None, ...]:
        messages = classification_messages(prompt).prompt_only
        rendered = render_chat_template(self.processor, messages, add_generation_prompt=True)
        visual_inputs = self.visual_inputs(images)
        visual_token_count = self._visual_token_count(visual_inputs)
        input_ids, mm_types = self._multimodal_token_ids(rendered, visual_token_count)
        input_ids = input_ids.to(images.device)
        input_ids = input_ids.unsqueeze(0).expand(images.shape[0], -1)
        text_inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        if mm_types is not None:
            text_inputs["mm_token_type_ids"] = (
                mm_types.to(images.device).unsqueeze(0).expand(images.shape[0], -1)
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
