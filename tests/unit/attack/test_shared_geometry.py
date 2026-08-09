import torch

from primary_ml_cka.attack.representation.shared_geometry import (
    center_crop_view,
    random_resized_crop_view,
    shared_geometry_loss,
)


def test_shared_geometry_prefers_aligned_same_image_rows() -> None:
    clean = torch.randn(7, 11)
    view = clean + 0.01 * torch.randn_like(clean)
    aligned = shared_geometry_loss(
        clean.clone().requires_grad_(True),
        clean,
        view,
        clean_weight=1.0,
        view_weight=1.0,
    )
    shuffled = shared_geometry_loss(
        clean.flip(0),
        clean,
        view,
        clean_weight=1.0,
        view_weight=1.0,
    )
    assert aligned.total < shuffled.total


def test_center_crop_view_is_differentiable_and_shape_preserving() -> None:
    images = torch.rand(3, 3, 224, 224, requires_grad=True)
    view = center_crop_view(images, 0.875)
    assert view.shape == images.shape
    gradient = torch.autograd.grad(view.square().mean(), images)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_random_resized_crop_is_seeded_per_image_and_differentiable() -> None:
    image = torch.rand(1, 3, 32, 32)
    images = image.repeat(4, 1, 1, 1).requires_grad_()
    first = random_resized_crop_view(images, 0.75, seed=17)
    repeated = random_resized_crop_view(images, 0.75, seed=17)
    assert first.shape == images.shape
    assert torch.equal(first, repeated)
    assert not torch.equal(first[0], first[1])
    gradient = torch.autograd.grad(first.square().mean(), images)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_view_term_uses_per_image_cosine_not_batch_geometry() -> None:
    embeddings = torch.eye(5)
    aligned = shared_geometry_loss(
        embeddings,
        embeddings,
        embeddings,
        clean_weight=0.0,
        view_weight=1.0,
    )
    shuffled = shared_geometry_loss(
        embeddings,
        embeddings,
        embeddings.roll(1, dims=0),
        clean_weight=0.0,
        view_weight=1.0,
    )
    assert torch.allclose(aligned.view_alignment, torch.tensor(1.0))
    assert aligned.total < shuffled.total


def test_clean_term_uses_patch_token_cka_per_image_when_provided() -> None:
    global_embeddings = torch.randn(3, 7)
    clean_patches = torch.randn(3, 9, 7)
    aligned = shared_geometry_loss(
        global_embeddings,
        global_embeddings,
        global_embeddings,
        clean_weight=1.0,
        view_weight=0.0,
        adversarial_patch_embeddings=clean_patches,
        clean_patch_embeddings=clean_patches,
    )
    shuffled = shared_geometry_loss(
        global_embeddings,
        global_embeddings,
        global_embeddings,
        clean_weight=1.0,
        view_weight=0.0,
        adversarial_patch_embeddings=clean_patches.roll(1, dims=1),
        clean_patch_embeddings=clean_patches,
    )
    assert torch.allclose(aligned.clean_alignment, torch.tensor(1.0), atol=1e-5)
    assert aligned.total < shuffled.total
