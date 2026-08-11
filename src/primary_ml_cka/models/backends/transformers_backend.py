from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from primary_ml_cka.models.common.loading import freeze_module


def load_processor(snapshot: Path):
    return AutoProcessor.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def load_generative_proxy(
    snapshot: Path,
    device: torch.device,
    *,
    modules_to_not_convert: tuple[str, ...] = (),
):
    if device.type != "cuda":
        raise ValueError("Generative proxies require CUDA")
    quantization_kwargs = {}
    if modules_to_not_convert:
        quantization_kwargs["llm_int8_skip_modules"] = list(modules_to_not_convert)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        **quantization_kwargs,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map={"": device.index or 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    return freeze_module(model)
