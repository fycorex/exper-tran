from abc import ABC, abstractmethod

from torch import Tensor

from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput


class BaseProxy(ABC):
    @abstractmethod
    def target_loss(
        self, images: Tensor, human_target_label: int, prompt: str
    ) -> ProxyLossOutput[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def image_embeddings(self, images: Tensor) -> ImageEmbeddingOutput[Tensor]:
        raise NotImplementedError

    @abstractmethod
    def free_generate_labels(self, images: Tensor, prompt: str) -> tuple[int | None, ...]:
        """Return strict parsed labels from an unconstrained proxy prediction."""
        raise NotImplementedError
