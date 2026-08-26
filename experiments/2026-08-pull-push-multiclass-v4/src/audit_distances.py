#!/usr/bin/env python3
"""Measure proxy-space distances between all ten class prototypes."""

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    class_names,
    load_experiment,
    pair_specs,
    transitions,
)
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.models.proxies.registry import load_proxy


def image_batch(paths: tuple[Path, ...]) -> torch.Tensor:
    values = []
    for path in paths:
        with Image.open(path) as image:
            values.append(pil_to_tensor(image.convert("RGB")).float().div(255))
    return torch.stack(values).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the embedding-distance audit")
    raw = load_experiment(args.config)
    names = class_names(raw)
    diagnostics = args.output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {}

    for pair_id, spec in pair_specs(raw).items():
        pair = get_pair(pair_id)
        attack_config = AttackConfig(
            representation_type=str(spec["representation_type"]),
            representation_layer=int(spec["representation_layer"]),
            representation_pooling=str(spec["pooling"]),
        )
        proxy = load_proxy(pair.proxy_model, Path(".hf-cache"), torch.device("cuda"), attack_config)
        centers = []
        try:
            for label in range(1, 11):
                records = read_manifest(
                    args.output_dir
                    / "evaluation"
                    / "manifests"
                    / f"class_references_{label:02d}.jsonl"
                )
                embeddings = []
                for start in range(0, len(records), args.batch_size):
                    batch_records = records[start : start + args.batch_size]
                    images = image_batch(
                        tuple(
                            args.output_dir / "canonical_images" / record.relative_path
                            for record in batch_records
                        )
                    )
                    with torch.no_grad():
                        output = proxy.image_embeddings(
                            images,
                            representation_type=str(spec["representation_type"]),
                            layer=int(spec["representation_layer"]),
                            pooling=str(spec["pooling"]),
                        )
                        value = (
                            output.semantic_embeddings
                            if output.semantic_embeddings is not None
                            else output.embeddings
                        )
                    embeddings.append(functional.normalize(value.float(), dim=-1).cpu())
                    del images, output, value
                class_embeddings = torch.cat(embeddings)
                centers.append(functional.normalize(class_embeddings.mean(dim=0), dim=0))
            center_matrix = torch.stack(centers)
            cosine = center_matrix @ center_matrix.T
            distance = 1.0 - cosine
            transition_lookup = {
                transition.transition_id: float(
                    distance[transition.source - 1, transition.target - 1]
                )
                for transition in transitions(raw)
            }
            payload[pair_id] = {
                "proxy_model": pair.proxy_model,
                "representation_type": spec["representation_type"],
                "representation_layer": spec["representation_layer"],
                "pooling": spec["pooling"],
                "class_names": names,
                "cosine_similarity": cosine.tolist(),
                "cosine_distance": distance.tolist(),
                "transition_distance": transition_lookup,
            }
            for transition in transitions(raw):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "proxy_model": pair.proxy_model,
                        "transition_id": transition.transition_id,
                        "source_label": transition.source,
                        "source_name": names[transition.source - 1],
                        "target_label": transition.target,
                        "target_name": names[transition.target - 1],
                        "prototype_cosine": float(
                            cosine[transition.source - 1, transition.target - 1]
                        ),
                        "prototype_cosine_distance": transition_lookup[
                            transition.transition_id
                        ],
                    }
                )
        finally:
            del proxy
            gc.collect()
            torch.cuda.empty_cache()

    json_path = diagnostics / "prototype_distances.json"
    csv_path = diagnostics / "prototype_distances.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} transition distances to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
