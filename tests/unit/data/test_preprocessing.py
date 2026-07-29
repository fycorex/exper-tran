import torch

from primary_ml_cka.data.preprocessing import ensure_canvas


def test_different_source_sizes_can_be_resized_before_batching() -> None:
    images = (
        torch.rand(1, 3, 375, 500, device="cuda"),
        torch.rand(1, 3, 333, 500, device="cuda"),
    )
    canvas = torch.stack([ensure_canvas(image, size=224).squeeze(0) for image in images])
    assert canvas.shape == (2, 3, 224, 224)
    assert torch.isfinite(canvas).all()
