import torch

from primary_ml_cka.artifacts.png import load_png_tensor, save_png_tensor


def test_png_roundtrip_preserves_linf_bound(tmp_path) -> None:
    clean = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8).float() / 255
    delta = torch.randint(-16, 17, clean.shape).float() / 255
    adversarial = (clean + delta).clamp(0, 1)
    clean_path = tmp_path / "clean.png"
    adversarial_path = tmp_path / "adv.png"
    save_png_tensor(clean, clean_path)
    save_png_tensor(adversarial, adversarial_path)
    reloaded_clean = load_png_tensor(clean_path).cuda()
    reloaded_adversarial = load_png_tensor(adversarial_path).cuda()
    assert (reloaded_adversarial - reloaded_clean).abs().max() <= 16 / 255 + 1e-7
