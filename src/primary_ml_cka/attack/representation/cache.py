from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class EmbeddingCache:
    z_clean: torch.Tensor
    z_reference: torch.Tensor
    source_image_ids: tuple[str, ...]
    target_reference_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, tensor in (("z_clean", self.z_clean), ("z_reference", self.z_reference)):
            if tensor.device.type != "cuda" or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be detached FP32 on CUDA")
            if tensor.requires_grad:
                raise ValueError(f"{name} must be detached")
        batch_size = self.z_clean.shape[0]
        if batch_size < 2 or self.z_reference.shape[0] != batch_size:
            raise ValueError("Cached CKA tensors must have the same batch size of at least two")
        if len(self.source_image_ids) != batch_size or len(self.target_reference_ids) != batch_size:
            raise ValueError("Embedding rows and image IDs must align")

    def to(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device.type != "cuda":
            raise ValueError("CPU embedding use is forbidden")
        return self.z_clean.to(device), self.z_reference.to(device)
