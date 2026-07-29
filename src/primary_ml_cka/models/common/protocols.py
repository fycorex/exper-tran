from pathlib import Path
from typing import Protocol

from torch import Tensor

from primary_ml_cka.domain.types import ImageEmbeddingOutput, ProxyLossOutput
from primary_ml_cka.models.common.outputs import GenerationOutput


class ProxyModel(Protocol):
    def target_loss(
        self, images: Tensor, human_target_label: int, prompt: str
    ) -> ProxyLossOutput[Tensor]: ...

    def image_embeddings(self, images: Tensor) -> ImageEmbeddingOutput[Tensor]: ...


class TargetGenerator(Protocol):
    def generate_label(self, image_path: Path, prompt: str) -> GenerationOutput: ...
