#!/usr/bin/env python3
"""Measure semantic separability at a few relative Vision Encoder depths."""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.experiment.attack_generation import _cuda_images
from primary_ml_cka.models.proxies.registry import load_proxy


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", default="P20")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    return parser.parse_args()


def extract(proxy: object, images: torch.Tensor, layer: int) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for chunk in images.split(4):
            output = proxy.image_embeddings(
                chunk,
                representation_type="vision_encoder",
                layer=layer,
                pooling="mean",
            )
            rows.append(output.semantic_embeddings.detach().float())
    return functional.normalize(torch.cat(rows), dim=-1)


def main() -> None:
    args = arguments()
    pair = get_pair(args.pair_id)
    manifests = args.output_dir / "evaluation/manifests"
    canonical = args.output_dir / "canonical_images"
    attacked = _cuda_images(canonical, read_manifest(manifests / "attack_images.jsonl"), 224)
    source = _cuda_images(canonical, read_manifest(manifests / "source_references.jsonl"), 224)
    target = _cuda_images(canonical, read_manifest(manifests / "target_references.jsonl"), 224)
    proxy = load_proxy(pair.proxy_model, Path(".hf-cache"), torch.device("cuda"), AttackConfig())
    vision_config = proxy.model.config.vision_config
    total = int(
        getattr(vision_config, "depth", getattr(vision_config, "num_hidden_layers", 0))
    )
    if total < 1:
        raise RuntimeError(f"Cannot resolve Vision Encoder depth for {pair.proxy_model}")
    layers = sorted({round(0.5 * (total - 1)), round(0.75 * (total - 1)), total - 1})
    rows = []
    for layer in layers:
        source_embeddings = extract(proxy, source, layer)
        target_embeddings = extract(proxy, target, layer)
        attack_embeddings = extract(proxy, attacked, layer)
        source_center = functional.normalize(source_embeddings.mean(0), dim=0)
        target_center = functional.normalize(target_embeddings.mean(0), dim=0)
        source_similarity = attack_embeddings @ source_center
        target_similarity = attack_embeddings @ target_center
        rows.append(
            {
                "pair_id": pair.pair_id,
                "model_id": pair.proxy_model,
                "layer": layer,
                "total_vision_layers": total,
                "source_target_prototype_cosine": float(source_center @ target_center),
                "attack_source_similarity_mean": float(source_similarity.mean()),
                "attack_target_similarity_mean": float(target_similarity.mean()),
                "attack_semantic_gap_mean": float(
                    (target_similarity - source_similarity).mean()
                ),
                "attack_semantic_gap_std": float(
                    (target_similarity - source_similarity).std()
                ),
                "source_embedding_feature_std": float(source_embeddings.std(dim=0).mean()),
                "target_embedding_feature_std": float(target_embeddings.std(dim=0).mean()),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    diagnostics = args.output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / f"layer_depth_{pair.pair_id}.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del proxy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
