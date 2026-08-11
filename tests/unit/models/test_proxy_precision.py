from primary_ml_cka.models.proxies.registry import vision_precision_skip_modules


def test_large_vlm_vision_modules_can_be_excluded_from_nf4() -> None:
    assert vision_precision_skip_modules("Qwen/Qwen3.5-4B") == ("visual",)
    assert vision_precision_skip_modules("OpenGVLab/InternVL3_5-4B-HF") == (
        "vision_tower",
        "multi_modal_projector",
    )
    assert vision_precision_skip_modules("google/gemma-4-E4B-it") == (
        "vision_tower",
        "embed_vision",
    )
