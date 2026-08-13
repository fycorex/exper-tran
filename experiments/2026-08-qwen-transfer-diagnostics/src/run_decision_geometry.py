import argparse
import csv
import gc
import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.experiment.attack_generation import _cuda_images
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.proxies.generative import GenerativeProxy
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.models.proxies.visual import (
    gemma_proxy_embeddings,
    gemma_visual_inputs,
    internvl_proxy_embeddings,
    internvl_visual_inputs,
    qwen_proxy_embeddings,
    qwen_visual_inputs,
)
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

PAIR_IDS = ("P02", "P06", "P11", "P14", "P16", "P19", "P20", "P21", "P22")
TARGET_INDEX = 6
SOURCE_INDEX = 7


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", choices=PAIR_IDS)
    parser.add_argument("--diagnostics-name", default="objective_split_common48_rho03")
    parser.add_argument("--result-name", default="decision_geometry")
    return parser.parse_args()


def _bf16_target_adapter(model_id: str, hf_home: Path) -> GenerativeProxy:
    snapshot = local_snapshot(hf_home, model_id)
    model = load_target_for_generation(snapshot, torch.device("cuda"))
    model.config.use_cache = False
    processor = load_processor(snapshot)
    if model_id.startswith("Qwen/"):
        visual_inputs = qwen_visual_inputs

        def embeddings(images):
            return qwen_proxy_embeddings(model_id, model, images)

        microbatch_size = 3
    elif model_id.startswith("OpenGVLab/InternVL"):
        visual_inputs = internvl_visual_inputs

        def embeddings(images):
            return internvl_proxy_embeddings(model_id, model, images, microbatch_size=4)

        microbatch_size = 4
    elif model_id.startswith("google/gemma"):

        def visual_inputs(images):
            return gemma_visual_inputs(processor, images)

        def embeddings(images):
            return gemma_proxy_embeddings(model_id, model, processor, images)

        microbatch_size = 1
    else:
        raise ValueError(f"Unsupported generative target: {model_id}")
    return GenerativeProxy(
        model,
        processor,
        visual_inputs,
        embeddings,
        class_margin=2.0,
        margin_weight=1.0,
        margin_temperature=0.5,
        microbatch_size=microbatch_size,
    )


def _logits(adapter: GenerativeProxy, images: torch.Tensor) -> torch.Tensor:
    output = adapter.target_loss(images, 7, CLASSIFICATION_PROMPT)
    if output.class_logits is None:
        raise RuntimeError("Differentiable target did not return closed-set logits")
    return output.class_logits


def _margin(logits: torch.Tensor, kind: str) -> torch.Tensor:
    target = logits[:, TARGET_INDEX]
    if kind == "source_target":
        return target - logits[:, SOURCE_INDEX]
    if kind == "robust":
        mask = torch.arange(logits.shape[1], device=logits.device) != TARGET_INDEX
        return target - logits[:, mask].max(dim=1).values
    raise ValueError(kind)


def _gradient(adapter, images: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    gradients = []
    margins = []
    for image in images.split(1):
        differentiated = image.detach().clone().requires_grad_(True)
        margin = _margin(_logits(adapter, differentiated), kind)
        gradient = torch.autograd.grad(margin.sum(), differentiated, only_inputs=True)[0]
        gradients.append(gradient.detach().float().cpu())
        margins.append(margin.detach().float().cpu())
    return torch.cat(gradients), torch.cat(margins)


def _batched_logits(adapter, images: torch.Tensor) -> torch.Tensor:
    return torch.cat([_logits(adapter, image) for image in images.split(1)])


def _png_batch(directory: Path, suffix: str, count: int) -> torch.Tensor:
    rows = []
    for index in range(count):
        with Image.open(directory / f"{index:02d}_{suffix}.png") as image:
            rows.append(pil_to_tensor(image.convert("RGB")).float().div(255))
    return torch.stack(rows).cuda()


def _trial_states(diagnostics: Path, pair_id: str):
    states = []
    for path in sorted((diagnostics / "trials").glob(f"{pair_id}__*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            states.append(state)
    return states


def _artifact_dir(output_dir: Path, state: dict[str, object]) -> Path:
    objective = str(state["objective"])
    if state.get("rho") is not None:
        objective += f"_rho_{float(state['rho']):g}"
    return (
        output_dir
        / "attacks"
        / str(state["pair_id"])
        / str(state["attack"]["phase"])
        / "batch_00"
        / objective
        / f"lambda_{float(state['lambda_cka']):g}"
    )


def _alignment(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
    flat_left = left.flatten(1)
    flat_right = right.flatten(1)
    sign_left = flat_left.sign()
    sign_right = flat_right.sign()
    return {
        "raw_cosine": functional.cosine_similarity(flat_left, flat_right, dim=1),
        "sign_cosine": functional.cosine_similarity(sign_left, sign_right, dim=1),
        "sign_agreement": sign_left.eq(sign_right).float().mean(dim=1),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(path, buffer.getvalue())


def _gap_closure(clean_margin: float, margin_change: float) -> float:
    """Fraction of a negative clean target-margin gap closed by the attack."""
    if clean_margin >= 0:
        return float("nan")
    return margin_change / -clean_margin


def _run_pair(
    output_dir: Path,
    project_root: Path,
    pair_id: str,
    diagnostics_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pair = get_pair(pair_id)
    diagnostics = output_dir / "diagnostics" / diagnostics_name
    records = read_manifest(diagnostics / "common_clean.jsonl")
    clean = _cuda_images(output_dir / "canonical_images", records, 224)
    os.environ.pop("PRIMARY_ML_CKA_KEEP_VISION_BF16", None)
    proxy = load_proxy(
        pair.proxy_model, project_root / ".hf-cache", torch.device("cuda"), AttackConfig()
    )
    proxy_gradients = {}
    proxy_clean_margins = {}
    for kind in ("source_target", "robust"):
        proxy_gradients[kind], proxy_clean_margins[kind] = _gradient(proxy, clean, kind)
    del proxy
    gc.collect()
    torch.cuda.empty_cache()

    target = _bf16_target_adapter(pair.target_model, project_root / ".hf-cache")
    target_gradients = {}
    target_clean_margins = {}
    for kind in ("source_target", "robust"):
        target_gradients[kind], target_clean_margins[kind] = _gradient(target, clean, kind)
    alignments = {
        kind: _alignment(proxy_gradients[kind], target_gradients[kind])
        for kind in ("source_target", "robust")
    }
    rows = []
    summaries = []
    for state in _trial_states(diagnostics, pair_id):
        directory = _artifact_dir(output_dir, state)
        clean_png = _png_batch(directory, "clean", len(records))
        adversarial = _png_batch(directory, "adv", len(records))
        delta = (adversarial - clean_png).float().cpu()
        with torch.no_grad():
            target_adv_logits = _batched_logits(target, adversarial).detach().float().cpu()
        for kind in ("source_target", "robust"):
            target_adv_margin = _margin(target_adv_logits, kind)
            proxy_derivative = (proxy_gradients[kind] * delta).flatten(1).sum(dim=1)
            target_derivative = (target_gradients[kind] * delta).flatten(1).sum(dim=1)
            actual_change = target_adv_margin - target_clean_margins[kind]
            gap_closures = [
                _gap_closure(float(target_clean_margins[kind][index]), float(actual_change[index]))
                for index in range(len(records))
            ]
            for index, record in enumerate(records):
                rows.append(
                    {
                        "pair_id": pair_id,
                        "objective": state["objective"],
                        "margin_kind": kind,
                        "image_id": record.image_id,
                        "target_hit": int(state["target_hit_mask"][index]),
                        "proxy_target_gradient_cosine": float(
                            alignments[kind]["raw_cosine"][index]
                        ),
                        "proxy_target_sign_cosine": float(alignments[kind]["sign_cosine"][index]),
                        "proxy_target_sign_agreement": float(
                            alignments[kind]["sign_agreement"][index]
                        ),
                        "proxy_directional_derivative": float(proxy_derivative[index]),
                        "target_directional_derivative": float(target_derivative[index]),
                        "target_clean_margin": float(target_clean_margins[kind][index]),
                        "target_actual_margin_change": float(actual_change[index]),
                        "target_adversarial_margin": float(target_adv_margin[index]),
                        "target_gap_closure": gap_closures[index],
                        "decision_score_type": "teacher_forced_closed_set",
                    }
                )
            summaries.append(
                {
                    "pair_id": pair_id,
                    "objective": state["objective"],
                    "margin_kind": kind,
                    "image_count": len(records),
                    "target_hits": int(state["target_hits"]),
                    "tasr_percent": float(state["tasr_percent"]),
                    "untargeted_hits": int(state["untargeted_hits"]),
                    "untargeted_asr_percent": 100.0
                    * int(state["untargeted_hits"])
                    / int(state["target_denominator"]),
                    "mean_clean_margin": float(target_clean_margins[kind].mean()),
                    "mean_margin_change": float(actual_change.mean()),
                    "mean_adversarial_margin": float(target_adv_margin.mean()),
                    "mean_gap_closure": statistics.fmean(gap_closures),
                    "median_gap_closure": statistics.median(gap_closures),
                    "boundary_crossing_count": int((target_adv_margin > 0).sum()),
                    "decision_score_type": "teacher_forced_closed_set",
                }
            )
    del target
    gc.collect()
    torch.cuda.empty_cache()
    return rows, summaries


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decision-geometry diagnosis")
    project_root = Path(__file__).resolve().parents[3]
    output_dir = args.output_dir.resolve()
    pairs = (args.pair_id,) if args.pair_id else PAIR_IDS
    result_dir = output_dir / "diagnostics" / args.result_name
    all_rows = []
    all_summaries = []
    for pair_id in pairs:
        rows, summaries = _run_pair(output_dir, project_root, pair_id, args.diagnostics_name)
        _write_csv(result_dir / f"{pair_id}.csv", rows)
        _write_csv(result_dir / f"{pair_id}_summary.csv", summaries)
        all_rows.extend(rows)
        all_summaries.extend(summaries)
        print(f"complete={pair_id} rows={len(rows)}", flush=True)
    if args.pair_id is None:
        _write_csv(result_dir / "all_pairs.csv", all_rows)
        _write_csv(result_dir / "gap_closure_summary.csv", all_summaries)


if __name__ == "__main__":
    main()
