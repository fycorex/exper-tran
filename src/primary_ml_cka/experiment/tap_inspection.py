import gc
import traceback
from dataclasses import asdict
from pathlib import Path

import torch

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.models.common.gradients import (
    assert_input_gradient,
    assert_parameter_gradients_none,
)
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.models.taps.validation import validate_proxy_gradient_path
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def inspect_proxy_taps(
    hf_home: Path,
    output_dir: Path,
    pair_id: str | None,
    dry_run: bool,
    attack_config: AttackConfig,
    data_config: DataConfig,
) -> tuple[str, ...]:
    proxy_ids = tuple(
        dict.fromkeys(
            pair.proxy_model for pair in MODEL_PAIRS if pair_id is None or pair.pair_id == pair_id
        )
    )
    if dry_run:
        return tuple(f"dry-run: proxy {model_id}" for model_id in proxy_ids)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; proxy taps cannot be validated")
    results = []
    for model_id in proxy_ids:
        safe_name = model_id.replace("/", "__")
        adapter = None
        module = None
        images = None
        output = None
        loss_images = None
        proxy_loss = None
        free_labels = None
        try:
            adapter = load_proxy(model_id, hf_home, torch.device("cuda"), attack_config)
            module = adapter.model
            images = torch.rand(
                attack_config.batch_size,
                3,
                attack_config.canvas_size,
                attack_config.canvas_size,
                device="cuda",
                dtype=torch.float32,
                requires_grad=True,
            )
            output = validate_proxy_gradient_path(adapter, module, images)
            loss_images = images[:1].detach().requires_grad_(True)
            proxy_loss = adapter.target_loss(
                loss_images,
                data_config.target_human_label,
                CLASSIFICATION_PROMPT,
            )
            if not torch.isfinite(proxy_loss.loss):
                raise RuntimeError("Proxy classification loss is non-finite")
            assert_input_gradient(proxy_loss.loss, loss_images)
            assert_parameter_gradients_none(module)
            with torch.no_grad():
                free_labels = adapter.free_generate_labels(
                    loss_images.detach(), CLASSIFICATION_PROMPT
                )
            if len(free_labels) != 1:
                raise RuntimeError("Proxy free generation did not return one label per image")
            record = {
                **asdict(output.tap),
                "status": "validated",
                "classification_logits_shape": list(proxy_loss.class_logits.shape),
                "classification_loss": float(proxy_loss.loss.detach()),
                "target_probability": float(proxy_loss.target_probability),
                "target_human_label": data_config.target_human_label,
                "max_other_probability": float(proxy_loss.max_other_probability),
                "token_level_attack_cka": True,
                "answer_token_ids": list(proxy_loss.answer_token_ids),
                "answer_label_positions": list(proxy_loss.label_positions),
                "free_generation_probe_label": free_labels[0],
            }
            write_json(output_dir / "taps" / f"{safe_name}.json", record)
            results.append(f"validated {model_id} {tuple(output.tokens.shape)}")
        except Exception as exc:
            failure = {
                "model_id": model_id,
                "status": "blocked",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(output_dir / "taps" / f"{safe_name}.json", failure)
            results.append(f"blocked {model_id}: {exc!r}")
        finally:
            adapter = None
            module = None
            images = None
            output = None
            loss_images = None
            proxy_loss = None
            free_labels = None
            gc.collect()
            torch.cuda.empty_cache()
    return tuple(results)
