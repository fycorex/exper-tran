from dataclasses import dataclass

import pytest
import torch

from primary_ml_cka.models.proxies.contrastive import ContrastiveProxy, _pooled_feature


@dataclass
class StructuredOutput:
    pooler_output: torch.Tensor


def test_extracts_tensor_and_structured_pooler_output() -> None:
    tensor = torch.randn(2, 3)
    assert _pooled_feature(tensor) is tensor
    assert _pooled_feature(StructuredOutput(tensor)) is tensor


def test_rejects_missing_pooler_output() -> None:
    with pytest.raises(TypeError):
        _pooled_feature(object())


class RecordingTokenizer:
    def __init__(self) -> None:
        self.padding = None
        self.max_length = None

    def __call__(
        self,
        prompts,
        *,
        padding,
        return_tensors,
        max_length=None,
        truncation=False,
    ):
        self.padding = padding
        self.max_length = max_length
        return {"input_ids": torch.ones(len(prompts), 2, dtype=torch.long)}


class TextFeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_text_features(self, input_ids):
        return torch.nn.functional.one_hot(
            torch.arange(input_ids.shape[0]) % 10, num_classes=10
        ).float()


def test_siglip_class_prompts_use_training_time_max_length_padding() -> None:
    tokenizer = RecordingTokenizer()
    ContrastiveProxy(
        TextFeatureModel(),
        tokenizer,
        lambda images: images,
        "google/siglip2-so400m-patch14-384",
        drop_cls_token=False,
    )
    assert tokenizer.padding == "max_length"
    assert tokenizer.max_length == 64
