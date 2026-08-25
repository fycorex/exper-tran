#!/usr/bin/env python3
"""Extract and document the actual configured Vision Encoder representation."""

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.config.loader import load_config
from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.models.common.gradients import assert_parameter_gradients_none
from primary_ml_cka.models.proxies.registry import load_proxy


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/2026-08-semantic-contrastive-v3/config/embedding_audit.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    parser.add_argument("--model", action="append", dest="models")
    return parser.parse_args()


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        value = pil_to_tensor(image.convert("RGB")).float().div(255)
    return value.unsqueeze(0).cuda().requires_grad_(True)


def stats(value: torch.Tensor) -> dict[str, object]:
    flat = value.detach().float().flatten()
    return {
        "first_8_values": flat[:8].cpu().tolist(),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_norm": float(flat.norm()),
    }


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for embedding audit")
    raw = load_config(args.config)
    models = tuple(args.models or raw["models"])
    records = read_manifest(args.output_dir / "evaluation/manifests/attack_images.jsonl")
    canonical = args.output_dir / "canonical_images" / records[0].relative_path
    diagnostics = args.output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in models:
        proxy = load_proxy(str(model_id), Path(".hf-cache"), torch.device("cuda"), AttackConfig())
        image = image_tensor(canonical)
        output = proxy.image_embeddings(
            image,
            representation_type=str(raw["representation_type"]),
            layer=int(raw["layer"]),
            pooling=str(raw["pooling"]),
        )
        pooled = (
            output.semantic_embeddings
            if output.semantic_embeddings is not None
            else output.embeddings
        )
        gradient = torch.autograd.grad(pooled.float().sum(), image, only_inputs=True)[0]
        if not torch.isfinite(gradient).all() or float(gradient.abs().max()) == 0:
            raise RuntimeError(f"Embedding gradient did not reach pixels for {model_id}")
        assert_parameter_gradients_none(proxy.model)
        tap = output.tap
        rows.append(
            {
                "model_id": model_id,
                "revision": MODEL_REVISIONS[str(model_id)],
                "representation_type": tap.representation_type,
                "requested_layer": tap.requested_layer,
                "resolved_layer": tap.resolved_layer,
                "total_vision_layers": tap.total_vision_layers,
                "exact_module_path": tap.module_path,
                "pooling": tap.pooling,
                "token_selection_rule": tap.token_mask,
                "cls_token_handling": (
                    "excluded" if "CLS" in tap.extraction else "not present or mask-controlled"
                ),
                "raw_token_shape": list(output.tokens.shape),
                "pooled_embedding_shape": list(pooled.shape),
                "dtype": str(output.tokens.dtype).removeprefix("torch."),
                "requires_grad": bool(pooled.requires_grad),
                "gradient_max_abs": float(gradient.abs().max()),
                **stats(pooled),
            }
        )
        del proxy, image, output, gradient
        gc.collect()
        torch.cuda.empty_cache()

    (diagnostics / "embedding_audit.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Embedding audit",
        "",
        "| Model | Component | Layer | Module path | Token shape | Pooling | Embedding shape |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_id']} | {row['representation_type']} | "
            f"{row['resolved_layer']}/{row['total_vision_layers']} | "
            f"`{row['exact_module_path']}` | `{row['raw_token_shape']}` | "
            f"{row['pooling']} | `{row['pooled_embedding_shape']}` |"
        )
    (diagnostics / "embedding_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} embedding audit rows to {diagnostics}", flush=True)


if __name__ == "__main__":
    main()
