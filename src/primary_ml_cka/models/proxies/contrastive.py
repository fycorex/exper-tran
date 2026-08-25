from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_classification_loss
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.labels import CLASS_NAMES, human_label_to_index
from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput, TapContract
from primary_ml_cka.models.proxies.base import BaseProxy
from primary_ml_cka.models.representations import RepresentationSpec, resolve_vision_layer
from primary_ml_cka.models.taps.pooling import masked_mean_l2

TEMPLATES = (
    "a photo of a {class_name}",
    "an image of a {class_name}",
    "the main object is a {class_name}",
    "a vehicle classified as a {class_name}",
)


def _pooled_feature(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if not isinstance(pooled, torch.Tensor):
        raise TypeError(
            f"Expected Tensor or structured output with Tensor pooler_output, got {type(output)!r}"
        )
    return pooled


class ContrastiveProxy(BaseProxy):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        image_preprocess: Callable[[torch.Tensor], torch.Tensor],
        model_id: str,
        *,
        drop_cls_token: bool,
        class_margin: float = 2.0,
        margin_weight: float = 1.0,
        margin_temperature: float = 0.5,
        microbatch_size: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_preprocess = image_preprocess
        self.model_id = model_id
        self.drop_cls_token = drop_cls_token
        self.class_margin = class_margin
        self.margin_weight = margin_weight
        self.margin_temperature = margin_temperature
        self.microbatch_size = microbatch_size
        self.class_embeddings = self._build_class_embeddings()

    def image_embeddings(
        self,
        images: torch.Tensor,
        *,
        representation_type: str = "legacy_projected",
        layer: int = -1,
        pooling: str = "mean",
    ) -> ImageEmbeddingOutput[torch.Tensor]:
        spec = RepresentationSpec(representation_type, layer, pooling)
        spec.validate()
        total_layers = int(self.model.config.vision_config.num_hidden_layers)
        resolved = (
            resolve_vision_layer(layer, total_layers)
            if representation_type == "vision_encoder"
            else None
        )

        def extract(image_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            pixel_values = self.image_preprocess(image_chunk)
            vision_output = self.model.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=representation_type == "vision_encoder",
                return_dict=True,
            )
            selected = (
                vision_output.hidden_states[resolved + 1]
                if representation_type == "vision_encoder"
                else vision_output.last_hidden_state
            )
            pooled = vision_output.pooler_output
            projection = getattr(self.model, "visual_projection", None)
            classifier_facing = projection(pooled) if projection is not None else pooled
            return selected, classifier_facing

        chunk_size = self.microbatch_size or images.shape[0]
        token_chunks = []
        semantic_chunks = []
        for image_chunk in images.split(chunk_size):
            if torch.is_grad_enabled() and images.requires_grad:
                tokens, semantic = checkpoint(extract, image_chunk, use_reentrant=False)
            else:
                tokens, semantic = extract(image_chunk)
            token_chunks.append(tokens)
            semantic_chunks.append(semantic)
        tokens = torch.cat(token_chunks)
        semantic_embeddings = torch.cat(semantic_chunks)
        if self.drop_cls_token:
            tokens = tokens[:, 1:, :]
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        embeddings = masked_mean_l2(tokens, mask)
        tap = TapContract(
            self.model_id,
            MODEL_REVISIONS[self.model_id],
            (
                f"model.vision_model.encoder.layers.{resolved}"
                if representation_type == "vision_encoder"
                else "vision_model.last_hidden_state"
            ),
            "selected Vision Encoder block output"
            if representation_type == "vision_encoder"
            else "final visual patch tokens (CLS excluded when present)",
            "none" if pooling == "none" else "masked mean over real visual patch tokens",
            "per-image L2; FP32 before CKA",
            str(tokens.dtype).removeprefix("torch."),
            "validated_by_forward",
            tuple(tokens.shape),
            "all spatial patch tokens; no synthetic global token",
            representation_type=representation_type,
            requested_layer=layer,
            resolved_layer=resolved,
            total_vision_layers=total_layers,
        )
        return ImageEmbeddingOutput(
            tokens if pooling == "none" else embeddings,
            tokens,
            mask,
            tap,
            semantic_embeddings=(
                embeddings
                if representation_type == "vision_encoder" and pooling == "mean"
                else semantic_embeddings
                if pooling == "mean"
                else None
            ),
        )

    def _build_class_embeddings(self) -> torch.Tensor:
        prompts = [
            template.format(class_name=class_name)
            for class_name in CLASS_NAMES
            for template in TEMPLATES
        ]
        if self.model_id.startswith("google/siglip"):
            tokens = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokens = self.tokenizer(prompts, padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            text = functional.normalize(
                _pooled_feature(self.model.get_text_features(**tokens)).float(),
                dim=-1,
            )
        return functional.normalize(text.reshape(10, 4, -1).mean(dim=1), dim=-1)

    def target_loss(
        self,
        images: torch.Tensor,
        human_target_label: int,
        prompt: str,
        cls_loss_mode: str = "ce_margin",
    ) -> ProxyLossOutput[torch.Tensor]:
        target_index = human_label_to_index(human_target_label)

        def score(image_chunk: torch.Tensor) -> torch.Tensor:
            image_features = _pooled_feature(
                self.model.get_image_features(pixel_values=self.image_preprocess(image_chunk))
            )
            scale = self.model.logit_scale.exp()
            bias_parameter = getattr(self.model, "logit_bias", None)
            normalized_images = functional.normalize(image_features, dim=-1)
            classes = functional.normalize(self.class_embeddings, dim=-1)
            chunk_logits = scale * (normalized_images @ classes.T)
            if bias_parameter is not None:
                chunk_logits = chunk_logits + bias_parameter
            return chunk_logits

        chunk_size = self.microbatch_size or images.shape[0]
        logit_chunks = []
        for image_chunk in images.split(chunk_size):
            if torch.is_grad_enabled() and images.requires_grad:
                logit_chunks.append(checkpoint(score, image_chunk, use_reentrant=False))
            else:
                logit_chunks.append(score(image_chunk))
        logits = torch.cat(logit_chunks)
        output = proxy_classification_loss(
            logits,
            target_index=target_index,
            margin=self.class_margin,
            margin_weight=self.margin_weight,
            temperature=self.margin_temperature,
        )
        target_nll = -output.logits.log_softmax(dim=-1)[:, target_index].mean()
        objectives = {
            "none": logits.sum() * 0.0,
            "closedset_ce": output.cross_entropy,
            "margin_only": output.margin_penalty,
            "ce_margin": output.total,
        }
        if cls_loss_mode == "target_token_nll":
            raise ValueError("target_token_nll is only defined for generative proxies")
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
        )

    def free_generate_labels(self, images: torch.Tensor, prompt: str) -> tuple[int | None, ...]:
        with torch.no_grad():
            output = self.target_loss(images, human_target_label=1, prompt=prompt)
        if output.class_logits is None:
            raise RuntimeError("Contrastive proxy logits are unavailable")
        return tuple(int(index) + 1 for index in output.class_logits.argmax(dim=1).tolist())
