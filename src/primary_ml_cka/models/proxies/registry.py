from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.backends.transformers_backend import (
    load_generative_proxy,
    load_processor,
)
from primary_ml_cka.models.common.loading import freeze_module, local_snapshot
from primary_ml_cka.models.proxies.clip import CLIP_PREPROCESS
from primary_ml_cka.models.proxies.contrastive import ContrastiveProxy
from primary_ml_cka.models.proxies.generative import GenerativeProxy
from primary_ml_cka.models.proxies.siglip2 import SIGLIP2_PREPROCESS
from primary_ml_cka.models.proxies.visual import (
    gemma_proxy_embeddings,
    gemma_visual_inputs,
    internvl_proxy_embeddings,
    internvl_visual_inputs,
    qwen_proxy_embeddings,
    qwen_visual_inputs,
)


def load_proxy(
    model_id: str,
    hf_home: Path,
    device: torch.device,
    attack_config: AttackConfig | None = None,
):
    attack_config = attack_config or AttackConfig()
    snapshot = local_snapshot(hf_home, model_id, MODEL_REVISIONS[model_id])
    if model_id == "openai/clip-vit-large-patch14":
        model = freeze_module(AutoModel.from_pretrained(snapshot, local_files_only=True).to(device))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        return ContrastiveProxy(
            model,
            tokenizer,
            CLIP_PREPROCESS,
            model_id,
            drop_cls_token=True,
            class_margin=attack_config.class_margin,
            margin_weight=attack_config.margin_weight,
            margin_temperature=attack_config.margin_temperature,
        )
    if model_id == "google/siglip2-so400m-patch14-384":
        model = freeze_module(AutoModel.from_pretrained(snapshot, local_files_only=True).to(device))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        return ContrastiveProxy(
            model,
            tokenizer,
            SIGLIP2_PREPROCESS,
            model_id,
            drop_cls_token=False,
            class_margin=attack_config.class_margin,
            margin_weight=attack_config.margin_weight,
            margin_temperature=attack_config.margin_temperature,
        )
    model = load_generative_proxy(snapshot, device)
    processor = load_processor(snapshot)
    if model_id.startswith("Qwen/"):
        visual_inputs = qwen_visual_inputs

        def image_embedding_fn(images: torch.Tensor):
            return qwen_proxy_embeddings(model_id, model.model.visual, images)

    elif model_id.startswith("OpenGVLab/InternVL"):
        visual_inputs = internvl_visual_inputs

        def image_embedding_fn(images: torch.Tensor):
            return internvl_proxy_embeddings(model_id, model.model.vision_tower, images)

    elif model_id.startswith("google/gemma"):

        def visual_inputs(images: torch.Tensor):
            return gemma_visual_inputs(processor, images)

        def image_embedding_fn(images: torch.Tensor):
            return gemma_proxy_embeddings(model_id, model, processor, images)

    else:
        raise ValueError(f"Unsupported generative proxy: {model_id}")
    return GenerativeProxy(
        model,
        processor,
        visual_inputs,
        image_embedding_fn,
        class_margin=attack_config.class_margin,
        margin_weight=attack_config.margin_weight,
        margin_temperature=attack_config.margin_temperature,
        microbatch_size=1 if model_id.startswith("google/gemma") else None,
    )
