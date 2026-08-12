import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoModel, AutoProcessor

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.attack.cka.linear import linear_cka
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.model_similarity import (
    cka_permutation_baseline,
    proxy_target_similarity,
)
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.common.loading import freeze_module, local_snapshot
from primary_ml_cka.models.proxies.clip import CLIP_PREPROCESS
from primary_ml_cka.models.proxies.siglip2 import SIGLIP2_PREPROCESS

PAIR_IDS = ("P02", "P06", "P11", "P14", "P16", "P19", "P20", "P21", "P22")
LAYERS = ("vision_mid", "vision_final", "projected")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs", nargs="+", choices=PAIR_IDS, default=list(PAIR_IDS))
    parser.add_argument("--diagnostics-name", default="objective_split_common48_rho03")
    parser.add_argument("--result-name", default="cka_validity")
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        help="Calibration manifest, relative to the output directory unless absolute.",
    )
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
                inputs = image_processor(images=image.convert("RGB"), return_tensors="pt")
            visual = {
                key: value.to(device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            output = model.get_image_features(**visual, return_dict=True, output_hidden_states=True)
            hidden_states = output.hidden_states
            if not hidden_states:
                raise RuntimeError("Vision model did not expose hidden states")
            rows["vision_mid"].append(_pooled_row(hidden_states[len(hidden_states) // 2]))
            rows["vision_final"].append(_pooled_row(output.last_hidden_state))
            rows["projected"].append(_pooled_row(output.pooler_output))
            if index == 1 or index % 10 == 0:
                print(f"extracted={index}/{len(paths)}", flush=True)
    return {layer: torch.stack(values) for layer, values in rows.items()}


def _extract_contrastive_layers(model, preprocess, paths: list[Path]) -> dict[str, torch.Tensor]:
    rows = {layer: [] for layer in LAYERS}
    device = next(model.parameters()).device
    with torch.no_grad():
        for index, path in enumerate(paths, start=1):
            with Image.open(path) as image:
                tensor = pil_to_tensor(image.convert("RGB")).float().div(255).unsqueeze(0)
            pixel_values = preprocess(tensor.to(device))
            output = model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=True,
            )
            if not output.hidden_states:
                raise RuntimeError("Contrastive vision model did not expose hidden states")
            rows["vision_mid"].append(
                _pooled_row(output.hidden_states[len(output.hidden_states) // 2])
            )
            rows["vision_final"].append(_pooled_row(output.last_hidden_state))
            rows["projected"].append(
                _pooled_row(model.get_image_features(pixel_values=pixel_values))
            )
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
    path_fingerprint = hashlib.sha256(
        "\n".join(str(path.resolve()) for path in paths).encode("utf-8")
    ).hexdigest()
    if resume and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            payload.get("path_count") == len(paths)
            and payload.get("path_fingerprint") == path_fingerprint
        ):
            print(f"resumed={model_id} rows={len(paths)}", flush=True)
            return {layer: payload[layer] for layer in LAYERS}
    snapshot = local_snapshot(hf_home, model_id)
    is_clip = model_id.startswith("openai/clip")
    is_siglip = model_id.startswith("google/siglip")
    if is_clip or is_siglip:
        processor = CLIP_PREPROCESS if is_clip else SIGLIP2_PREPROCESS
        model = freeze_module(AutoModel.from_pretrained(snapshot, local_files_only=True).cuda())
    else:
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = load_target_for_generation(snapshot, torch.device("cuda"))
    try:
        print(f"loading={model_id} rows={len(paths)}", flush=True)
        layers = (
            _extract_contrastive_layers(model, processor, paths)
            if is_clip or is_siglip
            else _extract_layers(model, processor, paths)
        )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "path_count": len(paths),
            "path_fingerprint": path_fingerprint,
            **layers,
        },
        cache_path,
    )
    return layers


def _clean_queries(diagnostics: Path, output_dir: Path, pair_id: str):
    records = read_manifest(diagnostics / "common_clean.jsonl")
    paths = [output_dir / "canonical_images" / record.relative_path for record in records]
    metadata = [
        {
            "pair_id": pair_id,
            "image_id": record.image_id,
            "query_kind": "clean",
        }
        for record in records
    ]
    return paths, metadata


def _query_exclusions(calibration, query_metadata) -> tuple[tuple[int, ...], ...]:
    calibration_indices: dict[str, list[int]] = {}
    for index, record in enumerate(calibration):
        calibration_indices.setdefault(record.image_id, []).append(index)
    return tuple(
        tuple(calibration_indices.get(str(metadata["image_id"]), ())) for metadata in query_metadata
    )


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
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(path, buffer.getvalue())


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CKA validity extraction")
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[3]
    diagnostics = output_dir / "diagnostics" / args.diagnostics_name
    result_dir = output_dir / "diagnostics" / args.result_name
    calibration_manifest = args.calibration_manifest
    if calibration_manifest is None:
        calibration_manifest = Path("evaluation/manifests/calibration.jsonl")
    if not calibration_manifest.is_absolute():
        calibration_manifest = output_dir / calibration_manifest
    calibration = read_manifest(calibration_manifest)
    calibration_paths = [
        output_dir / "canonical_images" / record.relative_path for record in calibration
    ]
    labels = [record.human_label for record in calibration]
    layer_rows = []
    null_rows = []
    local_rows = []
    summaries = []
    for pair_id in args.pairs:
        pair = get_pair(pair_id)
        query_paths, query_metadata = _clean_queries(diagnostics, output_dir, pair_id)
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
                    proxy_subset = proxy[proxy_layer][indices]
                    target_subset = target[target_layer][indices]
                    baseline = cka_permutation_baseline(
                        proxy_subset,
                        target_subset,
                        permutation_count=args.permutations,
                        seed=args.seed,
                    )
                    layer_rows.append(
                        {
                            "pair_id": pair_id,
                            "proxy_layer": proxy_layer,
                            "target_layer": target_layer,
                            "subset": subset,
                            "image_count": len(indices),
                            "cka": float(linear_cka(proxy_subset, target_subset)),
                        }
                    )
                    null_rows.append(
                        {
                            "pair_id": pair_id,
                            "proxy_layer": proxy_layer,
                            "target_layer": target_layer,
                            "subset": subset,
                            "image_count": len(indices),
                            "true_cka": baseline.true_cka,
                            "null_mean": baseline.null_mean,
                            "null_std": baseline.null_std,
                            "z_score": baseline.z_score,
                            "empirical_p_value": baseline.empirical_p_value,
                            "null_exceedance_count": baseline.null_exceedance_count,
                            "permutations_evaluated": baseline.permutations_evaluated,
                            "exact_null": int(baseline.exact),
                        }
                    )
        proxy_projected = proxy["projected"][:calibration_count]
        target_projected = target["projected"]
        query_exclusions = _query_exclusions(calibration, query_metadata)
        similarity = proxy_target_similarity(
            proxy_projected,
            target_projected,
            proxy["projected"][calibration_count:],
            neighbor_count=8,
            excluded_calibration_indices=query_exclusions,
        )
        for metadata, excluded, local_cka, neighbors in zip(
            query_metadata,
            query_exclusions,
            similarity.local_cka,
            similarity.neighbor_indices,
            strict=True,
        ):
            local_baseline = cka_permutation_baseline(
                proxy_projected[list(neighbors)],
                target_projected[list(neighbors)],
                permutation_count=args.permutations,
                seed=args.seed,
            )
            null_headroom = 1.0 - local_baseline.null_mean
            local_rows.append(
                {
                    **metadata,
                    "local_cka": local_cka,
                    "local_null_mean": local_baseline.null_mean,
                    "local_null_std": local_baseline.null_std,
                    "local_cka_excess": local_cka - local_baseline.null_mean,
                    "local_cka_normalized": (
                        (local_cka - local_baseline.null_mean) / null_headroom
                        if null_headroom > 0
                        else float("nan")
                    ),
                    "local_z_score": local_baseline.z_score,
                    "local_empirical_p_value": local_baseline.empirical_p_value,
                    "local_null_exceedance_count": (local_baseline.null_exceedance_count),
                    "local_permutations_evaluated": (local_baseline.permutations_evaluated),
                    "query_in_calibration": int(bool(excluded)),
                    "excluded_calibration_indices": "|".join(map(str, excluded)),
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
        _write_csv(result_dir / "cka_permutation_null.csv", null_rows)
        write_json(result_dir / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
