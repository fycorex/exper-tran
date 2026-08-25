#!/usr/bin/env python3
"""Resumable controlled eight-image semantic-contrastive ablation."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

# Required by deterministic CUDA GEMM when deterministic mode is requested.
# This must be set before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.loader import load_config
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.target_generation import evaluate_local_frozen_batch
from primary_ml_cka.experiment.attack_generation import attack_one_batch
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=())
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attacks are forbidden")
    raw = load_config(args.config)
    pair = get_pair(str(raw["pair_id"]))
    steps = int(args.steps or raw["steps"])
    phase = (
        f"semantic_contrastive_v3_steps_{steps}_"
        f"{raw['representation_type']}_layer_{raw['representation_layer']}"
    )
    manifests = args.output_dir / "evaluation" / "manifests"
    source = read_manifest(manifests / "attack_images.jsonl")
    source_references = read_manifest(manifests / "source_references.jsonl")
    target_references = read_manifest(manifests / "target_references.jsonl")
    if len(source) != 8:
        raise RuntimeError(f"Expected exactly eight attack images, got {len(source)}")
    attacked_ids = {record.image_id for record in source}
    if attacked_ids & {record.image_id for record in source_references}:
        raise RuntimeError("Source reference bank overlaps attacked images")
    if attacked_ids & {record.image_id for record in target_references}:
        raise RuntimeError("Target reference bank overlaps attacked images")

    selected = set(args.arms or ())
    arms = [arm for arm in raw["arms"] if not selected or str(arm["name"]) in selected]
    if not arms:
        raise ValueError("No configured ablation arm was selected")
    attack_config = AttackConfig(
        reference_bank_size=int(raw["reference_count"]),
        cls_loss_mode="ce_margin",
        semantic_temperature=float(raw["semantic_temperature"]),
        semantic_target_logit_weight=float(raw.get("semantic_target_logit_weight", 1.0)),
        semantic_source_logit_weight=float(raw.get("semantic_source_logit_weight", 1.0)),
        representation_type=str(raw["representation_type"]),
        representation_layer=int(raw["representation_layer"]),
        representation_pooling=str(raw["representation_pooling"]),
    )
    data_config = DataConfig(8, 7, 50, 48, 32, 16)

    for index, arm in enumerate(arms, start=1):
        name = str(arm["name"])
        state_path = args.output_dir / "states" / pair.pair_id / phase / f"{name}.json"
        if args.resume and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                print(f"trial={index}/{len(arms)} resume {pair.pair_id} {name}", flush=True)
                continue
        state = {
            "status": "running",
            "pair_id": pair.pair_id,
            "arm": name,
            "steps": steps,
            "config": arm,
        }
        write_json(state_path, state)
        print(f"trial={index}/{len(arms)} start {pair.pair_id} {name}", flush=True)
        try:
            if "rho" in arm and "rho_cls_sem" in arm:
                raise ValueError("Configure only one of rho or rho_cls_sem")
            rho = arm.get("rho_cls_sem", arm.get("rho"))
            if "push_pull_alpha" in arm and (
                "target_logit_weight" in arm or "source_logit_weight" in arm
            ):
                raise ValueError(
                    "push_pull_alpha cannot be combined with legacy target/source weights"
                )
            pull_weight = (
                1.0
                if "push_pull_alpha" in arm
                else float(
                    arm.get(
                        "target_logit_weight",
                        raw.get("semantic_target_logit_weight", 1.0),
                    )
                )
            )
            push_weight = (
                float(arm["push_pull_alpha"])
                if "push_pull_alpha" in arm
                else float(
                    arm.get(
                        "source_logit_weight",
                        raw.get("semantic_source_logit_weight", 1.0),
                    )
                )
            )
            lambda_sem = float(arm["lambda_sem"])
            semantic_weight = 1.0 if lambda_sem > 0 else 0.0
            result = attack_one_batch(
                pair,
                project_root=Path.cwd(),
                output_dir=args.output_dir,
                phase=phase,
                source_records=source,
                reference_records=target_references,
                source_reference_records=source_references,
                source_batch_index=0,
                reference_batch_index=0,
                lambda_cka=lambda_sem,
                seed=int(raw["seed"]),
                steps=steps,
                attack_config=attack_config,
                data_config=data_config,
                reference_bank_size=int(raw["reference_count"]),
                cka_source_weight=0.0,
                cka_target_weight=0.0,
                semantic_target_weight=semantic_weight,
                gradient_ratio=None if rho is None else float(rho),
                objective_tag=name,
                progress_interval=max(1, min(10, steps)),
                prompt=CLASSIFICATION_PROMPT,
                cls_loss_mode=str(arm["cls_loss_mode"]),
                lambda_cls=float(arm["lambda_cls"]),
                semantic_mode=str(arm["semantic_mode"]),
                semantic_temperature=float(
                    arm.get("semantic_temperature", raw["semantic_temperature"])
                ),
                semantic_target_logit_weight=pull_weight,
                semantic_source_logit_weight=push_weight,
                representation_type=str(raw["representation_type"]),
                representation_layer=int(raw["representation_layer"]),
                representation_pooling=str(raw["representation_pooling"]),
                gradient_trace_steps=(
                    tuple(step for step in (0, 25, 50, 99) if step < steps)
                    if name == "cls_plus_contrastive"
                    else ()
                ),
                checkpoint_steps=tuple(args.checkpoint_steps),
            )
            artifact_dir = (
                args.output_dir
                / "attacks"
                / pair.pair_id
                / phase
                / "batch_00"
                / name
                / f"lambda_{lambda_sem:g}"
            )
            evaluation = evaluate_local_frozen_batch(
                model_id=pair.target_model,
                hf_home=Path(".hf-cache"),
                artifact_dir=artifact_dir,
                image_count=8,
                prompt=CLASSIFICATION_PROMPT,
                source_human_label=8,
                target_human_label=7,
            )
            state.update(
                {
                    "status": "complete",
                    "attack": asdict(result),
                    "tasr_hits": evaluation.rates.targeted_hit_count,
                    "tasr_percent": evaluation.rates.tasr_percent,
                    "asr_hits": evaluation.rates.untargeted_hit_count,
                    "asr_percent": evaluation.rates.asr_percent,
                    "clean_valid_count": evaluation.rates.clean_valid_count,
                    "target_outputs": asdict(evaluation),
                }
            )
            print(
                f"trial={index}/{len(arms)} complete {pair.pair_id} {name} "
                f"proxy={result.proxy_target_hit_count}/8 "
                f"TASR={evaluation.rates.targeted_hit_count}/8 "
                f"ASR={evaluation.rates.untargeted_hit_count}/8",
                flush=True,
            )
        except Exception as exc:
            state.update({"status": "error", "error": repr(exc)})
            print(f"trial={index}/{len(arms)} ERROR {type(exc).__name__}: {exc}", flush=True)
            if args.fail_on_error:
                write_json(state_path, state)
                raise
        finally:
            write_json(state_path, state)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
