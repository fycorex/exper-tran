from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.data.manifests import ImageRecord


class ManifestImageDataset(Dataset[tuple[str, torch.Tensor]]):
    def __init__(self, root: Path, records: tuple[ImageRecord, ...]) -> None:
        self.root = root
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(self.root / record.relative_path) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div(255.0)
        return record.image_id, tensor
