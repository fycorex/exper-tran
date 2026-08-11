import torch

from primary_ml_cka.data.preprocessing import ensure_canvas, ste_quantize_8bit


def test_ste_quantization_matches_uint8_and_preserves_gradient() -> None:
    images = torch.tensor([0.1, 0.501, 0.999], requires_grad=True)
    quantized = ste_quantize_8bit(images)
    assert torch.equal(quantized.detach(), images.detach().mul(255).round().div(255))
    quantized.sum().backward()
    assert torch.equal(images.grad, torch.ones_like(images))


def test_different_source_sizes_can_be_resized_before_batching() -> None:
    images = (
        torch.rand(1, 3, 375, 500, device="cuda"),
        torch.rand(1, 3, 333, 500, device="cuda"),
    )
    canvas = torch.stack([ensure_canvas(image, size=224).squeeze(0) for image in images])
    assert canvas.shape == (2, 3, 224, 224)
    assert torch.isfinite(canvas).all()
