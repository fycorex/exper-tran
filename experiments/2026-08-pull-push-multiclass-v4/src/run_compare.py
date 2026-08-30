#!/usr/bin/env python3
"""Run resumable pull+push versus multiclass attacks on ten transitions."""

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    classification_prompt,
    load_experiment,
    pair_specs,
    transition_dir,
    transitions,
)

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.loader import load_config
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.target_generation import evaluate_local_frozen_batch
from primary_ml_cka.experiment.attack_generation import attack_one_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--transitions", nargs="+")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument(
        "--arm-config",
        type=Path,
        help="Optional YAML whose arms are merged with the primary arms",
    )
    parser.add_argument("--steps", type=int, help="Smoke-only step override")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attacks are forbidden")
    raw = load_experiment(args.config)
    prompt = classification_prompt(raw)
    specs = pair_specs(raw)
    chosen_pairs = tuple(args.pairs or specs)
    chosen_transitions = tuple(args.transitions or (t.transition_id for t in transitions(raw)))
    arms_by_name = {str(arm["name"]): arm for arm in raw["arms"]}
    if args.arm_config is not None:
        extra = load_config(args.arm_config)
        arms_by_name.update({str(arm["name"]): arm for arm in extra["arms"]})
    chosen_arms = tuple(args.arms or arms_by_name)
    unknown = set(chosen_pairs) - set(specs)
    if unknown:
        raise ValueError(f"Unknown/non-primary pair IDs: {sorted(unknown)}")
    unknown = set(chosen_transitions) - {t.transition_id for t in transitions(raw)}
    if unknown:
        raise ValueError(f"Unknown transition IDs: {sorted(unknown)}")
    unknown = set(chosen_arms) - set(arms_by_name)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    transition_lookup = {t.transition_id: t for t in transitions(raw)}
    class_references = tuple(
        read_manifest(
            args.output_dir
            / "evaluation"
            / "manifests"
            / f"class_references_{label:02d}.jsonl"
        )
        for label in range(1, 11)
    )
    reference_count = int(raw["reference_count"])
    if any(len(records) < reference_count for records in class_references):
        raise RuntimeError("A ten-class reference bank is incomplete")

    trial_count = len(chosen_pairs) * len(chosen_transitions) * len(chosen_arms)
    trial_index = 0
    for pair_id in chosen_pairs:
        pair = get_pair(pair_id)
        spec = specs[pair_id]
        for transition_id in chosen_transitions:
            transition = transition_lookup[transition_id]
            manifests = transition_dir(args.output_dir, transition_id)
            source = read_manifest(manifests / f"{pair_id}_attack_images.jsonl")
            if len(source) != int(raw["attack_count"]):
                raise RuntimeError(
                    f"{pair_id}/{transition_id} requires {raw['attack_count']} clean-valid images"
                )
            attacked_ids = {record.image_id for record in source}
            source_references = class_references[transition.source - 1]
            target_references = class_references[transition.target - 1]
            if attacked_ids & {
                record.image_id for record in source_references + target_references
            }:
                raise RuntimeError(f"{pair_id}/{transition_id} references overlap attacks")
            for arm_name in chosen_arms:
                trial_index += 1
                arm = arms_by_name[arm_name]
                steps = int(args.steps or arm["steps"])
                step_size = float(arm["step_size"])
                rho = float(arm.get("rho", raw["rho"]))
                semantic_temperature = float(
                    arm.get("semantic_temperature", raw["semantic_temperature"])
                )
                target_weight = float(arm.get("target_logit_weight", 1.0))
                source_weight = float(arm.get("source_logit_weight", 1.0))
                gradient_trace_steps = tuple(
                    int(value) for value in arm.get("gradient_trace_steps", ())
                )
                checkpoint_steps = tuple(
                    int(value) for value in arm.get("checkpoint_steps", ())
                )
                trial_seed = int(arm.get("seed", raw["seed"]))
                representation_type = str(
                    arm.get("representation_type", spec["representation_type"])
                )
                representation_layer = int(
                    arm.get("representation_layer", spec["representation_layer"])
                )
                representation_pooling = str(arm.get("pooling", spec["pooling"]))
                early_stop_proxy_gate = bool(arm.get("early_stop_proxy_gate", False))
                if rho <= 0 or semantic_temperature <= 0:
                    raise ValueError("rho and semantic temperature must be positive")
                if target_weight < 0 or source_weight < 0:
                    raise ValueError("Pull/push weights must be non-negative")
                if target_weight == 0 and source_weight == 0:
                    raise ValueError("At least one pull/push weight must be positive")
                phase = f"v4_{transition_id}_{steps}steps"
                state_path = (
                    args.output_dir / "states" / pair_id / transition_id / f"{arm_name}.json"
                )
                if args.resume and state_path.is_file():
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    if state.get("status") == "complete" and int(state["steps"]) == steps:
                        print(
                            f"trial={trial_index}/{trial_count} resume "
                            f"{pair_id}/{transition_id}/{arm_name}",
                            flush=True,
                        )
                        continue
                state = {
                    "status": "running",
                    "pair_id": pair_id,
                    "transition_id": transition_id,
                    "source_human_label": transition.source,
                    "target_human_label": transition.target,
                    "arm": arm_name,
                    "semantic_mode": arm["semantic_mode"],
                    "steps": steps,
                    "step_size": step_size,
                    "rho": rho,
                    "semantic_temperature": semantic_temperature,
                    "target_logit_weight": target_weight,
                    "source_logit_weight": source_weight,
                    "gradient_trace_steps": gradient_trace_steps,
                    "checkpoint_steps": checkpoint_steps,
                    "seed": trial_seed,
                    "representation_type": representation_type,
                    "representation_layer": representation_layer,
                    "representation_pooling": representation_pooling,
                    "early_stop_proxy_gate": early_stop_proxy_gate,
                }
                write_json(state_path, state)
                print(
                    f"trial={trial_index}/{trial_count} start "
                    f"{pair_id}/{transition_id}/{arm_name}",
                    flush=True,
                )
                try:
                    attack_config = AttackConfig(
                        epsilon=16 / 255,
                        step_size=step_size,
                        batch_size=8,
                        steps=steps,
                        momentum=1.0,
                        random_start=True,
                        class_margin=2.0,
                        margin_weight=1.0,
                        margin_temperature=0.5,
                        proxy_probability_threshold=0.9,
                        require_proxy_free_generation=True,
                        reference_bank_size=reference_count,
                        cls_loss_mode="margin_only",
                        semantic_mode=str(arm["semantic_mode"]),
                        semantic_temperature=semantic_temperature,
                        representation_type=representation_type,
                        representation_layer=representation_layer,
                        representation_pooling=representation_pooling,
                    )
                    data_config = DataConfig(
                        transition.source,
                        transition.target,
                        int(raw["candidate_count"]),
                        reference_count,
                        8,
                        0,
                        allow_partial_main_batch=False,
                        source_reference_count=reference_count,
                    )
                    result = attack_one_batch(
                        pair,
                        project_root=Path.cwd(),
                        output_dir=args.output_dir,
                        phase=phase,
                        source_records=source,
                        reference_records=target_references,
                        source_reference_records=source_references,
                        class_reference_records=(
                            class_references
                            if arm["semantic_mode"] == "multiclass_prototype"
                            else None
                        ),
                        source_batch_index=0,
                        reference_batch_index=0,
                        lambda_cka=1.0,
                        seed=trial_seed,
                        steps=steps,
                        attack_config=attack_config,
                        data_config=data_config,
                        reference_bank_size=reference_count,
                        cka_source_weight=0.0,
                        cka_target_weight=0.0,
                        semantic_target_weight=1.0,
                        gradient_ratio=rho,
                        objective_tag=arm_name,
                        early_stop_proxy_gate=early_stop_proxy_gate,
                        progress_interval=max(1, min(10, steps)),
                        prompt=prompt,
                        cls_loss_mode="margin_only",
                        lambda_cls=1.0,
                        semantic_mode=str(arm["semantic_mode"]),
                        semantic_temperature=semantic_temperature,
                        semantic_target_logit_weight=target_weight,
                        semantic_source_logit_weight=source_weight,
                        representation_type=representation_type,
                        representation_layer=representation_layer,
                        representation_pooling=representation_pooling,
                        gradient_trace_steps=gradient_trace_steps,
                        checkpoint_steps=checkpoint_steps,
                    )
                    artifact_dir = (
                        args.output_dir
                        / "attacks"
                        / pair_id
                        / phase
                        / "batch_00"
                        / arm_name
                        / "lambda_1"
                    )
                    evaluation = evaluate_local_frozen_batch(
                        model_id=pair.target_model,
                        hf_home=Path(".hf-cache"),
                        artifact_dir=artifact_dir,
                        image_count=len(source),
                        prompt=prompt,
                        source_human_label=transition.source,
                        target_human_label=transition.target,
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
                        f"complete {pair_id}/{transition_id}/{arm_name} "
                        f"proxy={result.proxy_target_hit_count}/{len(source)} "
                        f"TASR={evaluation.rates.targeted_hit_count}/"
                        f"{evaluation.rates.clean_valid_count} "
                        f"ASR={evaluation.rates.untargeted_hit_count}/"
                        f"{evaluation.rates.clean_valid_count}",
                        flush=True,
                    )
                except Exception as exc:
                    state.update({"status": "error", "error": repr(exc)})
                    print(f"ERROR {pair_id}/{transition_id}/{arm_name}: {exc}", flush=True)
                    if args.fail_on_error:
                        write_json(state_path, state)
                        raise
                finally:
                    write_json(state_path, state)
                    gc.collect()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
