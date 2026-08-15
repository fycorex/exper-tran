import argparse
import csv
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.loader import load_config
from primary_ml_cka.data.manifests import read_manifest, write_manifest
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
    "effective_lambda_cka",
    "rho",
    "objective",
    "source_weight",
    "alpha",
    "beta",
    "target_cka_mode",
    "target_alignment_temperature",
    "reference_count",
    "status",
    "proxy_hits",
    "proxy_denominator",
    "proxy_hit_mask",
    "proxy_gate_type",
    "proxy_free_check_applicable",
    "free_generation_hits",
    "target_hits",
    "target_denominator",
    "target_hit_mask",
    "target_hits_among_proxy_hits",
    "proxy_hit_target_denominator",
    "tasr_percent",
    "untargeted_hits",
    "asr_percent",
    "proxy_min_margin",
    "proxy_min_probability",
    "cka_adv_source",
    "cka_adv_reference",
    "reference_cka_gain",
    "source_cka_drop",
    "source_repulsion_achieved",
    "target_attraction_achieved",
    "grad_ml_l1",
    "grad_cka_weighted_l1",
    "grad_component_cosine",
    "elapsed_seconds",
    "error",
)

CONTRASTIVE_PREFIXES = ("openai/clip", "google/siglip")


def _proxy_gate_reporting(pair_id: str) -> tuple[str, bool]:
    is_contrastive = get_pair(pair_id).proxy_model.startswith(CONTRASTIVE_PREFIXES)
    return (
        "contrastive_closed_set" if is_contrastive else "generative_strict",
        not is_contrastive,
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
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop at the first failed trial and require every planned state to be complete.",
    )
    return parser.parse_args()


def _objective(
    prompt_id: str,
    alpha: float,
    beta: float,
    *,
    objective_id: str | None = None,
    rho: float | None = None,
    target_cka_mode: str = "spatial_index_legacy",
    target_alignment_temperature: float = 0.07,
) -> str:
    target_suffix = (
        ""
        if target_cka_mode == "spatial_index_legacy"
        else f"_{target_cka_mode}_temp_{target_alignment_temperature:g}"
    )
    if objective_id is not None:
        rho_suffix = "" if rho is None else f"_rho_{rho:g}"
        return f"{objective_id}{rho_suffix}{target_suffix}"
    suffix = "" if beta == 0 else f"_beta_{beta:g}"
    return f"prompt_{prompt_id}_alpha_{alpha:g}{suffix}{target_suffix}"


def _state_path(
    diagnostics: Path,
    pair_id: str,
    prompt_id: str,
    lambda_cka: float,
    alpha: float,
    beta: float,
    objective_id: str | None = None,
    rho: float | None = None,
    target_cka_mode: str = "spatial_index_legacy",
    target_alignment_temperature: float = 0.07,
) -> Path:
    beta_suffix = "" if beta == 0 else f"__beta_{beta:g}"
    objective_suffix = "" if objective_id is None else f"__objective_{objective_id}"
    rho_suffix = "" if rho is None else f"__rho_{rho:g}"
    target_mode_suffix = (
        ""
        if target_cka_mode == "spatial_index_legacy"
        else (
            f"__target_{target_cka_mode}"
            f"__temperature_{target_alignment_temperature:g}"
        )
    )
    name = (
        f"{pair_id}__{prompt_id}__lambda_{lambda_cka:g}__alpha_{alpha:g}"
        f"{beta_suffix}{objective_suffix}{rho_suffix}{target_mode_suffix}.json"
    )
    return diagnostics / "trials" / name


def _write_summary(diagnostics: Path, *, materialize_selection: bool = False) -> None:
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

    eligible = []
    for state in states:
        _, free_check_applicable = _proxy_gate_reporting(str(state["pair_id"]))
        proxy_gate_passed = state.get("proxy_hits") == state.get("proxy_denominator") == 8
        free_gate_passed = not free_check_applicable or state.get("free_generation_hits") == 8
        if proxy_gate_passed and free_gate_passed and state.get("target_denominator") == 8:
            eligible.append(state)
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
    write_json(diagnostics / "diagnostic_best.json", selected)
    if materialize_selection:
        write_json(diagnostics / "selected.json", selected)
    else:
        # Diagnostic-only runs must not leave a selection artifact from an
        # older invocation for a scale runner to consume silently.
        selected_path = diagnostics / "selected.json"
        if selected_path.exists():
            selected_path.unlink()


def _trial_state(trial: dict[str, object]):
    return {
        **trial,
        "status": "running",
    }


def _attack_diagnostic_fields(result: object) -> dict[str, object]:
    return {
        "proxy_hits": result.proxy_target_hit_count,
        "proxy_denominator": result.proxy_target_hit_denominator,
        "proxy_hit_mask": list(result.proxy_target_hit_mask),
        "free_generation_hits": result.proxy_free_target_hit_count,
        "effective_lambda_cka": result.effective_lambda_cka,
        "proxy_min_margin": result.proxy_min_target_logit_margin,
        "proxy_min_probability": result.proxy_min_target_probability,
        "cka_adv_source": result.cka_adv_source,
        "cka_adv_reference": result.cka_adv_reference,
        "reference_cka_gain": result.reference_cka_gain,
        "source_cka_drop": result.source_cka_drop,
        "source_repulsion_achieved": result.source_repulsion_achieved,
        "target_attraction_achieved": result.target_attraction_achieved,
        "grad_ml_l1": result.grad_ml_l1,
        "grad_cka_weighted_l1": result.grad_cka_weighted_l1,
        "grad_component_cosine": result.grad_component_cosine,
        "elapsed_seconds": result.elapsed_seconds,
        "attack": asdict(result),
    }


def _common_clean_records(
    output_dir: Path,
    pair_ids: list[str],
    model_ids: list[str] | None,
    count: int,
    source_human_label: int,
    frozen_manifest: Path,
):
    if frozen_manifest.is_file():
        frozen = read_manifest(frozen_manifest)
        if len(frozen) < count:
            raise RuntimeError(
                f"Frozen common-clean manifest has {len(frozen)} images; requested {count}"
            )
        if model_ids is not None:
            frozen_ids = {record.image_id for record in frozen[:count]}
            for model_id in model_ids:
                safe_name = model_id.replace("/", "__")
                screen_path = output_dir / "evaluation" / f"{safe_name}__clean_screen.jsonl"
                valid_ids = {
                    payload["image_id"]
                    for payload in (
                        json.loads(line)
                        for line in screen_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
                    if payload.get("parsed_label") == source_human_label
                }
                invalid = frozen_ids - valid_ids
                if invalid:
                    raise RuntimeError(
                        f"Frozen common-clean manifest is invalid for {model_id}: "
                        f"{sorted(invalid)}"
                    )
        return frozen[:count]

    target_ids = tuple(
        dict.fromkeys(
            model_ids
            if model_ids is not None
            else (get_pair(pair_id).target_model for pair_id in pair_ids)
        )
    )
    candidates = read_manifest(
        output_dir / "evaluation" / "manifests" / "source_validation_candidates.jsonl"
    )
    valid_sets = []
    for model_id in target_ids:
        safe_name = model_id.replace("/", "__")
        screen_path = output_dir / "evaluation" / f"{safe_name}__clean_screen.jsonl"
        valid_ids = set()
        with screen_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if payload.get("parsed_label") == source_human_label:
                        valid_ids.add(payload["image_id"])
        valid_sets.append(valid_ids)
    common_ids = set.intersection(*valid_sets)
    common = tuple(record for record in candidates if record.image_id in common_ids)[:count]
    if len(common) != count:
        raise RuntimeError(
            f"Only {len(common)} images are clean-valid for every configured "
            f"target; requested {count}"
        )
    return common


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
    materialize_selection = bool(raw.get("materialize_selection", False))
    require_auxiliary_progress = bool(raw.get("require_auxiliary_progress", False))
    evaluate_target = bool(raw.get("evaluate_target", True))
    if "trials" in raw:
        trials = [
            {
                "pair_id": str(trial["pair"]),
                "prompt_id": str(trial.get("prompt", "original")),
                "lambda_cka": float(trial["lambda"]),
                "alpha": float(trial.get("alpha", 1)),
                "beta": float(trial.get("beta", 0)),
                "source_weight": float(trial.get("source_weight", 1)),
                "rho": (None if trial.get("rho") is None else float(trial["rho"])),
                "reference_count": int(trial.get("reference_count", raw.get("reference_count", 8))),
                "target_cka_mode": str(
                    trial.get("target_cka_mode", "spatial_index_legacy")
                ),
                "target_alignment_temperature": float(
                    trial.get("target_alignment_temperature", 0.07)
                ),
                "objective": trial.get("objective"),
            }
            for trial in raw["trials"]
        ]
    else:
        trials = [
            {
                "pair_id": pair_id,
                "prompt_id": prompt_id,
                "lambda_cka": float(lambda_cka),
                "alpha": float(alpha),
                "beta": 0.0,
                "source_weight": 1.0,
                "rho": None,
                "reference_count": int(raw.get("reference_count", 8)),
                "target_cka_mode": "spatial_index_legacy",
                "target_alignment_temperature": 0.07,
                "objective": None,
            }
            for pair_id in raw["pairs"]
            for prompt_id in raw["prompts"]
            for lambda_cka in raw["lambdas"]
            for alpha in raw["alphas"]
            if float(lambda_cka) > 0 or float(alpha) == 1
            if not get_pair(pair_id).proxy_model.startswith(("openai/clip", "google/siglip"))
            or prompt_id == "original"
        ]
    for proxy_model in dict.fromkeys(
        get_pair(str(trial["pair_id"])).proxy_model for trial in trials
    ):
        require_proxy_tap(context, proxy_model)
    common_source = None
    if bool(raw.get("common_clean", False)):
        common_source = _common_clean_records(
            output_dir,
            [
                str(pair_id)
                for pair_id in raw.get(
                    "common_clean_pairs",
                    [trial["pair_id"] for trial in trials],
                )
            ],
            (
                None
                if raw.get("common_clean_models") is None
                else [str(model_id) for model_id in raw["common_clean_models"]]
            ),
            int(raw.get("image_count", 8)),
            data_config.source_human_label,
            diagnostics / "common_clean.jsonl",
        )
        write_manifest(diagnostics / "common_clean.jsonl", common_source)
        print(
            "common_clean=" + "|".join(record.image_id for record in common_source),
            flush=True,
        )
    print(f"planned_trials={len(trials)}", flush=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 0.75 * total_bytes:
        raise RuntimeError(
            "A4000 is already occupied: "
            f"free={free_bytes / 2**30:.2f} GiB of {total_bytes / 2**30:.2f} GiB. "
            "Stop the existing GPU job before starting this sweep."
        )
    planned_state_paths = []
    for trial_index, trial in enumerate(trials, start=1):
        pair_id = str(trial["pair_id"])
        prompt_id = str(trial["prompt_id"])
        lambda_cka = float(trial["lambda_cka"])
        alpha = float(trial["alpha"])
        beta = float(trial["beta"])
        source_weight = float(trial["source_weight"])
        rho = trial["rho"]
        reference_count = int(trial["reference_count"])
        target_cka_mode = str(trial["target_cka_mode"])
        target_alignment_temperature = float(trial["target_alignment_temperature"])
        objective_id = trial["objective"]
        state_path = _state_path(
            diagnostics,
            pair_id,
            prompt_id,
            lambda_cka,
            alpha,
            beta,
            None if objective_id is None else str(objective_id),
            None if rho is None else float(rho),
            target_cka_mode,
            target_alignment_temperature,
        )
        planned_state_paths.append(state_path)
        if args.resume and state_path.is_file():
            existing = json.loads(state_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                print(f"trial={trial_index}/{len(trials)} resumed {state_path.name}", flush=True)
                continue
        pair = get_pair(pair_id)
        proxy_gate_type, proxy_free_check_applicable = _proxy_gate_reporting(pair_id)
        prompt = get_prompt(prompt_id)
        state = _trial_state(trial)
        state["prompt"] = prompt
        write_json(state_path, state)
        fatal_error = None
        try:
            source = (
                common_source
                if common_source is not None
                else load_phase_records(output_dir, pair.target_model, "main")[:8]
            )
            if len(source) != 8:
                raise RuntimeError(f"{pair_id} does not have eight clean-valid images")
            objective = _objective(
                prompt_id,
                alpha,
                beta,
                objective_id=None if objective_id is None else str(objective_id),
                rho=None if rho is None else float(rho),
                target_cka_mode=target_cka_mode,
                target_alignment_temperature=target_alignment_temperature,
            )
            print(
                f"trial={trial_index}/{len(trials)} pair={pair_id} prompt={prompt_id} "
                f"objective={objective} lambda={lambda_cka:g} rho={rho} "
                f"source={source_weight:g} alpha={alpha:g} beta={beta:g} "
                f"references={reference_count} target_cka_mode={target_cka_mode}",
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
                reference_bank_size=reference_count,
                cka_source_weight=source_weight,
                cka_target_weight=alpha,
                semantic_target_weight=beta,
                target_cka_mode=target_cka_mode,
                target_alignment_temperature=target_alignment_temperature,
                gradient_ratio=None if rho is None else float(rho),
                objective_tag=objective,
                early_stop_proxy_gate=early_stop_proxy_gate,
                progress_interval=10,
                prompt=prompt,
            )
            if require_auxiliary_progress:
                failed_components = []
                if source_weight > 0 and not result.source_repulsion_achieved:
                    failed_components.append("source_repulsion")
                if alpha > 0 and not result.target_attraction_achieved:
                    failed_components.append("target_attraction")
                if failed_components:
                    raise RuntimeError(
                        "Auxiliary objective did not improve required components: "
                        + ", ".join(failed_components)
                    )
            if not evaluate_target:
                state.update(
                    {
                        "status": "complete",
                        "proxy_gate_type": proxy_gate_type,
                        "proxy_free_check_applicable": proxy_free_check_applicable,
                        "target_hits": "",
                        "target_denominator": "",
                        "target_hit_mask": "",
                        "target_hits_among_proxy_hits": "",
                        "proxy_hit_target_denominator": "",
                        "tasr_percent": "",
                        "untargeted_hits": "",
                        "asr_percent": "",
                        "error": "",
                        **_attack_diagnostic_fields(result),
                    }
                )
                continue
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
                image_count=len(source),
                prompt=prompt,
                source_human_label=data_config.source_human_label,
                target_human_label=data_config.target_human_label,
            )
            rates = evaluation.rates
            write_json(
                diagnostics / "target_outputs" / state_path.name,
                evaluation,
            )
            target_hit_mask = tuple(
                clean.parsed_label == data_config.source_human_label
                and adversarial.parsed_label == data_config.target_human_label
                for clean, adversarial in zip(
                    evaluation.clean_outputs,
                    evaluation.adversarial_outputs,
                    strict=True,
                )
            )
            proxy_hit_target_denominator = sum(result.proxy_target_hit_mask)
            target_hits_among_proxy_hits = sum(
                proxy_hit and target_hit
                for proxy_hit, target_hit in zip(
                    result.proxy_target_hit_mask, target_hit_mask, strict=True
                )
            )
            state.update(
                {
                    "status": "complete",
                    "proxy_gate_type": proxy_gate_type,
                    "proxy_free_check_applicable": proxy_free_check_applicable,
                    "target_hits": rates.targeted_hit_count,
                    "target_denominator": rates.clean_valid_count,
                    "target_hit_mask": list(target_hit_mask),
                    "target_hits_among_proxy_hits": target_hits_among_proxy_hits,
                    "proxy_hit_target_denominator": proxy_hit_target_denominator,
                    "tasr_percent": rates.tasr_percent,
                    "untargeted_hits": rates.untargeted_hit_count,
                    "asr_percent": rates.asr_percent,
                    "error": "",
                    **_attack_diagnostic_fields(result),
                }
            )
        except Exception as exc:
            state.update({"status": "error", "error": repr(exc)})
            if isinstance(exc, torch.cuda.OutOfMemoryError) or args.fail_on_error:
                fatal_error = exc
            print(
                f"trial={trial_index}/{len(trials)} ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            write_json(state_path, state)
            _write_summary(diagnostics, materialize_selection=materialize_selection)
            gc.collect()
            torch.cuda.empty_cache()
        if fatal_error is not None:
            reason = (
                "CUDA OOM"
                if isinstance(fatal_error, torch.cuda.OutOfMemoryError)
                else "trial error"
            )
            raise RuntimeError(
                f"Stopping sweep after {reason}; completed trials remain resumable"
            ) from fatal_error
    _write_summary(diagnostics, materialize_selection=materialize_selection)
    if args.fail_on_error:
        incomplete = []
        for state_path in planned_state_paths:
            status = "missing"
            if state_path.is_file():
                status = json.loads(state_path.read_text(encoding="utf-8")).get("status", "unknown")
            if status != "complete":
                incomplete.append(f"{state_path.name}:{status}")
        if incomplete:
            raise RuntimeError(
                "Sweep did not complete every planned trial: " + ", ".join(incomplete)
            )


if __name__ == "__main__":
    main()
