from abc import ABC, abstractmethod

from torch import Tensor

from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput


class BaseProxy(ABC):
    @abstractmethod
    def target_loss(
        self,
        images: Tensor,
        human_target_label: int,
        prompt: str,
        cls_loss_mode: str = "ce_margin",
    ) -> ProxyLossOutput[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def image_embeddings(
        self,
        images: Tensor,
        *,
        representation_type: str = "legacy_projected",
        layer: int = -1,
        pooling: str = "mean",
    ) -> ImageEmbeddingOutput[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def free_generate_labels(self, images: Tensor, prompt: str) -> tuple[int | None, ...]:
        """Return strict parsed labels from an unconstrained proxy prediction."""
        raise NotImplementedError
