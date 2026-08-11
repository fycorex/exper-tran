from types import SimpleNamespace

import torch
import torch.nn.functional as functional

from primary_ml_cka.models.proxies.visual import (
    internvl_proxy_embeddings,
    qwen_proxy_embeddings,
)


class _QwenModel(torch.nn.Module):
    def get_image_features(self, pixel_values, image_grid_thw, return_dict):
        assert return_dict
        batch_size = image_grid_thw.shape[0]
        patches = pixel_values.reshape(batch_size, 256, -1)
        projected = patches.reshape(batch_size, 64, 4, -1).mean(dim=2)[..., :7]
        return SimpleNamespace(pooler_output=tuple(projected.unbind(dim=0)))


class _InternVLModel(torch.nn.Module):
    def get_image_features(
        self,
        pixel_values,
        vision_feature_layer,
        vision_feature_select_strategy,
        return_dict,
    ):
        assert vision_feature_layer == -1
        assert vision_feature_select_strategy == "default"
        assert return_dict
        pooled = functional.adaptive_avg_pool2d(pixel_values.float(), (4, 4))
        tokens = pooled.flatten(2).transpose(1, 2)
        return SimpleNamespace(pooler_output=tokens)


def test_qwen_cka_tap_uses_projected_language_model_tokens() -> None:
    images = torch.rand(2, 3, 224, 224, requires_grad=True)
    output = qwen_proxy_embeddings("Qwen/Qwen3.5-2B", _QwenModel(), images)

    assert output.tokens.shape == (2, 64, 7)
    assert output.tap.module_path == "model.get_image_features.pooler_output"
    output.tokens.sum().backward()
    assert images.grad is not None and bool(images.grad.abs().gt(0).any())


def test_internvl_cka_tap_uses_projected_language_model_tokens() -> None:
    images = torch.rand(2, 3, 224, 224, requires_grad=True)
    output = internvl_proxy_embeddings(
        "OpenGVLab/InternVL3_5-2B-HF",
        _InternVLModel(),
        images,
        microbatch_size=1,
    )

    assert output.tokens.shape == (2, 16, 3)
    assert output.tap.module_path == "model.get_image_features.pooler_output"
    output.tokens.sum().backward()
    assert images.grad is not None and bool(images.grad.abs().gt(0).any())
