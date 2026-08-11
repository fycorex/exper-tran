import argparse
import csv
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.loader import load_config
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.target_generation import evaluate_local_frozen_batch
from primary_ml_cka.experiment.attack_generation import attack_one_batch, load_phase_records
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_proxy_tap,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.prompts.variants import get_prompt

FIELDNAMES = (
    "pair_id",
    "prompt_id",
    "lambda_cka",
    "alpha",
    "beta",
    "status",
    "proxy_hits",
    "proxy_denominator",
    "free_generation_hits",
    "target_hits",
    "target_denominator",
    "tasr_percent",
    "untargeted_hits",
    "proxy_min_margin",
    "proxy_min_probability",
    "cka_adv_source",
    "cka_adv_reference",
    "reference_cka_gain",
    "source_cka_drop",
    "grad_ml_l1",
    "grad_cka_weighted_l1",
    "grad_component_cosine",
    "elapsed_seconds",
    "error",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/2026-08-qwen-transfer-diagnostics/config/sweep.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _objective(prompt_id: str, alpha: float, beta: float) -> str:
    suffix = "" if beta == 0 else f"_beta_{beta:g}"
    return f"prompt_{prompt_id}_alpha_{alpha:g}{suffix}"


def _state_path(
    diagnostics: Path,
    pair_id: str,
    prompt_id: str,
    lambda_cka: float,
    alpha: float,
    beta: float,
) -> Path:
    beta_suffix = "" if beta == 0 else f"__beta_{beta:g}"
    name = (
        f"{pair_id}__{prompt_id}__lambda_{lambda_cka:g}__alpha_{alpha:g}"
        f"{beta_suffix}.json"
    )
    return diagnostics / "trials" / name


def _write_summary(diagnostics: Path) -> None:
    states = []
    for path in sorted((diagnostics / "trials").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            states.append(payload)
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for state in states:
            writer.writerow({field: state.get(field, "") for field in FIELDNAMES})
    temporary.replace(csv_path)

    eligible = [
        state
        for state in states
        if state.get("proxy_hits") == state.get("proxy_denominator") == 8
        and state.get("free_generation_hits") == 8
        and state.get("target_denominator") == 8
    ]
    selected = {}
    for pair_id in sorted({state["pair_id"] for state in eligible}):
        candidates = [state for state in eligible if state["pair_id"] == pair_id]
        if candidates:
            selected[pair_id] = max(
                candidates,
                key=lambda state: (
                    state.get("target_hits", -1),
                    state.get("target_denominator", 0),
                    state.get("reference_cka_gain", float("-inf")),
                    -state["lambda_cka"],
                ),
            )
    write_json(diagnostics / "selected.json", selected)


def _trial_state(
    pair_id: str, prompt_id: str, lambda_cka: float, alpha: float, beta: float
):
    return {
        "pair_id": pair_id,
        "prompt_id": prompt_id,
        "lambda_cka": lambda_cka,
        "alpha": alpha,
        "beta": beta,
        "status": "running",
    }


def main() -> None:
    args = _arguments()
    project_root = Path.cwd().resolve()
    output_dir = args.output_dir.resolve()
    context = CommandContext(
        project_root,
        output_dir,
        None,
        args.resume,
        False,
        None,
        None,
        8,
    )
    require_real_run_ready(context)
    raw = load_config(args.config)
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    references = read_manifest(
        output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )
    diagnostics_name = str(raw.get("diagnostics_name", "pair_prompt_sweep_v2"))
    diagnostics = output_dir / "diagnostics" / diagnostics_name
    phase = str(raw.get("phase", "tuning"))
    early_stop_proxy_gate = bool(raw.get("early_stop_proxy_gate", True))
    if "trials" in raw:
        trials = [
            (
                str(trial["pair"]),
                str(trial.get("prompt", "original")),
                float(trial["lambda"]),
                float(trial["alpha"]),
                float(trial.get("beta", 0)),
            )
            for trial in raw["trials"]
        ]
    else:
        trials = [
            (pair_id, prompt_id, float(lambda_cka), float(alpha), 0.0)
            for pair_id in raw["pairs"]
            for prompt_id in raw["prompts"]
            for lambda_cka in raw["lambdas"]
            for alpha in raw["alphas"]
            if float(lambda_cka) > 0 or float(alpha) == 1
            if not get_pair(pair_id).proxy_model.startswith(
                ("openai/clip", "google/siglip")
            )
            or prompt_id == "original"
        ]
    print(f"planned_trials={len(trials)}", flush=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 0.75 * total_bytes:
        raise RuntimeError(
            "A4000 is already occupied: "
            f"free={free_bytes / 2**30:.2f} GiB of {total_bytes / 2**30:.2f} GiB. "
            "Stop the existing GPU job before starting this sweep."
        )
    for trial_index, (pair_id, prompt_id, lambda_cka, alpha, beta) in enumerate(
        trials, start=1
    ):
        state_path = _state_path(
            diagnostics, pair_id, prompt_id, lambda_cka, alpha, beta
        )
        if args.resume and state_path.is_file():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                print(f"trial={trial_index}/{len(trials)} resumed {state_path.name}", flush=True)
                continue
        pair = get_pair(pair_id)
        prompt = get_prompt(prompt_id)
        state = _trial_state(pair_id, prompt_id, lambda_cka, alpha, beta)
        state["prompt"] = prompt
        write_json(state_path, state)
        fatal_error = None
        try:
            require_proxy_tap(context, pair.proxy_model)
            source = load_phase_records(output_dir, pair.target_model, "main")[:8]
            if len(source) != 8:
                raise RuntimeError(f"{pair_id} does not have eight clean-valid images")
            objective = _objective(prompt_id, alpha, beta)
            print(
                f"trial={trial_index}/{len(trials)} pair={pair_id} prompt={prompt_id} "
                f"lambda={lambda_cka:g} alpha={alpha:g} beta={beta:g}",
                flush=True,
            )
            result = attack_one_batch(
                pair,
                project_root=project_root,
                output_dir=output_dir,
                phase=phase,
                source_records=source,
                reference_records=references,
                source_batch_index=0,
                lambda_cka=lambda_cka,
                seed=int(raw["seed"]),
                steps=int(raw["steps"]),
                attack_config=attack_config,
                data_config=data_config,
                cka_target_weight=alpha,
                semantic_target_weight=beta,
                objective_tag=objective,
                early_stop_proxy_gate=early_stop_proxy_gate,
                progress_interval=10,
                prompt=prompt,
            )
            rates = None
            if result.proxy_target_all_hit:
                artifact_dir = (
                    output_dir
                    / "attacks"
                    / pair_id
                    / phase
                    / "batch_00"
                    / objective
                    / f"lambda_{lambda_cka:g}"
                )
                evaluation = evaluate_local_frozen_batch(
                    model_id=pair.target_model,
                    hf_home=project_root / ".hf-cache",
                    artifact_dir=artifact_dir,
                    image_count=8,
                    prompt=prompt,
                    source_human_label=data_config.source_human_label,
                    target_human_label=data_config.target_human_label,
                )
                rates = evaluation.rates
                write_json(
                    diagnostics / "target_outputs" / state_path.name,
                    evaluation,
                )
            state.update(
                {
                    "status": "complete",
                    "proxy_hits": result.proxy_target_hit_count,
                    "proxy_denominator": result.proxy_target_hit_denominator,
                    "free_generation_hits": result.proxy_free_target_hit_count,
                    "target_hits": rates.targeted_hit_count if rates else "",
                    "target_denominator": rates.clean_valid_count if rates else "",
                    "tasr_percent": rates.tasr_percent if rates else "",
                    "untargeted_hits": rates.untargeted_hit_count if rates else "",
                    "proxy_min_margin": result.proxy_min_target_logit_margin,
                    "proxy_min_probability": result.proxy_min_target_probability,
                    "cka_adv_source": result.cka_adv_source,
                    "cka_adv_reference": result.cka_adv_reference,
                    "reference_cka_gain": result.reference_cka_gain,
                    "source_cka_drop": result.source_cka_drop,
                    "grad_ml_l1": result.grad_ml_l1,
                    "grad_cka_weighted_l1": result.grad_cka_weighted_l1,
                    "grad_component_cosine": result.grad_component_cosine,
                    "elapsed_seconds": result.elapsed_seconds,
                    "error": "",
                    "attack": asdict(result),
                }
            )
        except Exception as exc:
            state.update({"status": "error", "error": repr(exc)})
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                fatal_error = exc
            print(
                f"trial={trial_index}/{len(trials)} ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            write_json(state_path, state)
            _write_summary(diagnostics)
            gc.collect()
            torch.cuda.empty_cache()
        if fatal_error is not None:
            raise RuntimeError(
                "Stopping sweep after CUDA OOM; completed trials remain resumable"
            ) from fatal_error
    _write_summary(diagnostics)


if __name__ == "__main__":
    main()
