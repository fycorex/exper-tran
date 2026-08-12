import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from transformers import AutoProcessor

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.attack.cka.linear import linear_cka
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.model_similarity import proxy_target_similarity
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.common.loading import local_snapshot

PAIR_IDS = ("P20", "P21", "P22")
LAYERS = ("vision_mid", "vision_final", "projected")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _pooled_row(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    if isinstance(value, tuple):
        if len(value) != 1:
            raise RuntimeError("Expected exactly one image feature tensor")
        value = value[0]
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise RuntimeError("Visual layer did not return token features")
    tokens = value.reshape(-1, value.shape[-1]).float()
    return functional.normalize(tokens.mean(dim=0), dim=0).cpu()


def _extract_layers(model, processor, paths: list[Path]) -> dict[str, torch.Tensor]:
    rows = {layer: [] for layer in LAYERS}
    image_processor = getattr(processor, "image_processor", processor)
    device = next(model.parameters()).device
    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                inputs = image_processor(
                    images=image.convert("RGB"), return_tensors="pt"
                )
            visual = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            output = model.get_image_features(
                **visual, return_dict=True, output_hidden_states=True
            )
            hidden_states = output.hidden_states
            if not hidden_states:
                raise RuntimeError("Vision model did not expose hidden states")
            rows["vision_mid"].append(_pooled_row(hidden_states[len(hidden_states) // 2]))
            rows["vision_final"].append(_pooled_row(output.last_hidden_state))
            rows["projected"].append(_pooled_row(output.pooler_output))
            if index == 1 or index % 10 == 0:
                print(f"extracted={index}/{len(paths)}", flush=True)
    return {layer: torch.stack(values) for layer, values in rows.items()}


def _model_cache(
    model_id: str,
    paths: list[Path],
    cache_path: Path,
    hf_home: Path,
    *,
    resume: bool,
) -> dict[str, torch.Tensor]:
    if resume and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload["path_count"] == len(paths):
            print(f"resumed={model_id} rows={len(paths)}", flush=True)
            return {layer: payload[layer] for layer in LAYERS}
    snapshot = local_snapshot(hf_home, model_id)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = load_target_for_generation(snapshot, torch.device("cuda"))
    try:
        print(f"loading={model_id} rows={len(paths)}", flush=True)
        layers = _extract_layers(model, processor, paths)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"path_count": len(paths), **layers}, cache_path)
    return layers


def _trial_queries(diagnostics: Path, output_dir: Path, pair_id: str):
    paths = []
    metadata = []
    for state_path in sorted((diagnostics / "trials").glob(f"{pair_id}__*.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            continue
        rho = state.get("rho")
        objective_dir = str(state["objective"])
        if rho is not None:
            objective_dir += f"_rho_{float(rho):g}"
        artifact_dir = (
            output_dir
            / "attacks"
            / pair_id
            / state["attack"]["phase"]
            / "batch_00"
            / objective_dir
            / f"lambda_{float(state['lambda_cka']):g}"
        )
        for index, (image_id, target_hit, proxy_hit) in enumerate(
            zip(
                state["attack"]["source_image_ids"],
                state["target_hit_mask"],
                state["proxy_hit_mask"],
                strict=True,
            )
        ):
            paths.append(artifact_dir / f"{index:02d}_adv.png")
            metadata.append(
                {
                    "pair_id": pair_id,
                    "objective": state["objective"],
                    "image_id": image_id,
                    "target_hit": int(target_hit),
                    "proxy_hit": int(proxy_hit),
                }
            )
    return paths, metadata


def _rsa_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = functional.normalize(left.float(), dim=-1)
    right = functional.normalize(right.float(), dim=-1)
    indices = torch.triu_indices(left.shape[0], left.shape[0], offset=1)
    left_values = (left @ left.T)[indices[0], indices[1]]
    right_values = (right @ right.T)[indices[0], indices[1]]
    return float(torch.corrcoef(torch.stack((left_values, right_values)))[0, 1])


def _centroids(features: torch.Tensor, labels: list[int]) -> torch.Tensor:
    return torch.stack(
        [features[[label == class_id for label in labels]].mean(dim=0) for class_id in range(1, 11)]
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(path, buffer.getvalue())


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CKA validity extraction")
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[3]
    diagnostics = output_dir / "diagnostics" / "objective_split_common48_rho03"
    result_dir = output_dir / "diagnostics" / "cka_validity"
    calibration = read_manifest(
        output_dir / "evaluation" / "manifests" / "calibration.jsonl"
    )
    calibration_paths = [
        output_dir / "canonical_images" / record.relative_path for record in calibration
    ]
    labels = [record.human_label for record in calibration]
    layer_rows = []
    local_rows = []
    summaries = []
    for pair_id in PAIR_IDS:
        pair = get_pair(pair_id)
        query_paths, query_metadata = _trial_queries(diagnostics, output_dir, pair_id)
        proxy_paths = calibration_paths + query_paths
        proxy = _model_cache(
            pair.proxy_model,
            proxy_paths,
            result_dir / "cache" / f"{pair_id}_proxy.pt",
            project_root / ".hf-cache",
            resume=args.resume,
        )
        target = _model_cache(
            pair.target_model,
            calibration_paths,
            result_dir / "cache" / f"{pair_id}_target.pt",
            project_root / ".hf-cache",
            resume=args.resume,
        )
        calibration_count = len(calibration)
        class_indices = {
            "global": list(range(calibration_count)),
            "class_7": [index for index, label in enumerate(labels) if label == 7],
            "class_8": [index for index, label in enumerate(labels) if label == 8],
            "classes_7_8": [index for index, label in enumerate(labels) if label in {7, 8}],
        }
        for proxy_layer in LAYERS:
            for target_layer in LAYERS:
                for subset, indices in class_indices.items():
                    layer_rows.append(
                        {
                            "pair_id": pair_id,
                            "proxy_layer": proxy_layer,
                            "target_layer": target_layer,
                            "subset": subset,
                            "image_count": len(indices),
                            "cka": float(
                                linear_cka(
                                    proxy[proxy_layer][indices],
                                    target[target_layer][indices],
                                )
                            ),
                        }
                    )
        proxy_projected = proxy["projected"][:calibration_count]
        target_projected = target["projected"]
        similarity = proxy_target_similarity(
            proxy_projected,
            target_projected,
            proxy["projected"][calibration_count:],
            neighbor_count=8,
        )
        for metadata, local_cka, neighbors in zip(
            query_metadata,
            similarity.local_cka,
            similarity.neighbor_indices,
            strict=True,
        ):
            local_rows.append(
                {
                    **metadata,
                    "local_cka": local_cka,
                    "neighbor_indices": "|".join(map(str, neighbors)),
                }
            )
        proxy_centroids = _centroids(proxy_projected, labels)
        target_centroids = _centroids(target_projected, labels)
        summaries.append(
            {
                "pair_id": pair_id,
                "projected_global_cka": similarity.global_cka,
                "class_centroid_cka": float(linear_cka(proxy_centroids, target_centroids)),
                "class_geometry_rsa": _rsa_correlation(proxy_centroids, target_centroids),
            }
        )
        _write_csv(result_dir / "layer_cka.csv", layer_rows)
        _write_csv(result_dir / "local_cka.csv", local_rows)
        write_json(result_dir / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
