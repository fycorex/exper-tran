from collections.abc import Callable

import torch
import torch.nn.functional as functional

from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_classification_loss
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.labels import CLASS_NAMES, human_label_to_index
from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput, TapContract
from primary_ml_cka.models.proxies.base import BaseProxy

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
        class_margin: float = 2.0,
        rank_weight: float = 1.0,
        suppression_weight: float = 1.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.image_preprocess = image_preprocess
        self.model_id = model_id
        self.class_margin = class_margin
        self.rank_weight = rank_weight
        self.suppression_weight = suppression_weight
        self.class_embeddings = self._build_class_embeddings()

    def image_embeddings(self, images: torch.Tensor) -> ImageEmbeddingOutput[torch.Tensor]:
        features = _pooled_feature(
            self.model.get_image_features(pixel_values=self.image_preprocess(images))
        )
        return self._embedding_output(features, images.shape[0])

    def image_embeddings_with_patches(
        self, images: torch.Tensor
    ) -> tuple[ImageEmbeddingOutput[torch.Tensor], torch.Tensor]:
        """Return projected global and patch embeddings from one CLIP vision pass."""
        vision_model = getattr(self.model, "vision_model", None)
        projection = getattr(self.model, "visual_projection", None)
        if vision_model is None or projection is None:
            raise TypeError("Patch-token CKA requires a CLIP vision model and projection")
        output = vision_model(pixel_values=self.image_preprocess(images))
        hidden = getattr(output, "last_hidden_state", None)
        pooled = getattr(output, "pooler_output", None)
        if not isinstance(hidden, torch.Tensor) or not isinstance(pooled, torch.Tensor):
            raise TypeError("CLIP vision output must expose last_hidden_state and pooler_output")
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            raise ValueError("CLIP patch-token output must have shape [N,1+P,D]")
        global_features = projection(pooled)
        patch_features = functional.normalize(projection(hidden[:, 1:]).float(), dim=-1)
        return self._embedding_output(global_features, images.shape[0]), patch_features

    def _embedding_output(
        self, features: torch.Tensor, image_count: int
    ) -> ImageEmbeddingOutput[torch.Tensor]:
        embeddings = functional.normalize(features.float(), dim=-1)
        tokens = embeddings.unsqueeze(1)
        mask = torch.ones((image_count, 1), dtype=torch.bool, device=features.device)
        tap = TapContract(
            self.model_id,
            MODEL_REVISIONS[self.model_id],
            "get_image_features",
            "native projected image embedding",
            "checkpoint-native global pooling",
            "per-image L2; FP32 before CKA",
            str(features.dtype).removeprefix("torch."),
            "validated_by_forward",
            tuple(tokens.shape),
            "one valid global image token per image",
        )
        return ImageEmbeddingOutput(embeddings, tokens, mask, tap)

    def _build_class_embeddings(self) -> torch.Tensor:
        prompts = [
            template.format(class_name=class_name)
            for class_name in CLASS_NAMES
            for template in TEMPLATES
        ]
        padding = "max_length" if "siglip" in self.model_id.lower() else True
        tokens = self.tokenizer(prompts, padding=padding, return_tensors="pt")
        device = next(self.model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            text = functional.normalize(
                _pooled_feature(self.model.get_text_features(**tokens)).float(),
                dim=-1,
            )
        return functional.normalize(text.reshape(10, 4, -1).mean(dim=1), dim=-1)

    def target_loss(
        self, images: torch.Tensor, human_target_label: int, prompt: str
    ) -> ProxyLossOutput[torch.Tensor]:
        image_features = _pooled_feature(
            self.model.get_image_features(pixel_values=self.image_preprocess(images))
        )
        return self.target_loss_from_embeddings(image_features, human_target_label)

    def target_loss_from_embeddings(
        self,
        image_features: torch.Tensor,
        human_target_label: int,
    ) -> ProxyLossOutput[torch.Tensor]:
        target_index = human_label_to_index(human_target_label)
        scale = self.model.logit_scale.exp()
        bias_parameter = getattr(self.model, "logit_bias", None)
        bias = bias_parameter if bias_parameter is not None else None
        images = functional.normalize(image_features, dim=-1)
        classes = functional.normalize(self.class_embeddings, dim=-1)
        logits = scale * (images @ classes.T)
        if bias is not None:
            logits = logits + bias
        output = proxy_classification_loss(
            logits,
            target_index=target_index,
            margin=self.class_margin,
            rank_weight=self.rank_weight,
            suppression_weight=self.suppression_weight,
        )
        return ProxyLossOutput(
            loss=output.total,
            target_nll=output.cross_entropy,
            target_probability=output.target_probability.detach(),
            class_logits=output.logits,
            classification_ce=output.cross_entropy.detach(),
            rank_loss=output.rank.detach(),
            other_suppression_loss=output.other_suppression.detach(),
            max_other_probability=output.max_other_probability.detach(),
        )
