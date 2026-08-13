import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from primary_ml_cka.artifacts.writers import write_json, write_results_csv
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS, ModelPair
from primary_ml_cka.evaluation.attack_metrics import AttackRates, attack_rates
from primary_ml_cka.experiment.attack_generation import (
    AttackRunResult,
    attack_one_batch,
    blocked_result_row,
    load_phase_records,
    result_row,
)
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_proxy_tap,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from primary_ml_cka.prompts.variants import get_prompt


def _log_path(
    output_dir: Path,
    pair: ModelPair,
    phase: str,
    batch_index: int,
    lambda_cka: float,
) -> Path:
    return output_dir / "logs" / pair.pair_id / phase / f"{batch_index:02d}_{lambda_cka:g}.json"


def _migrate_attack_result_payload(payload: dict[str, object]) -> dict[str, object]:
    migrated = dict(payload)
    # Schema migration for logs written before objective decomposition. Do not
    # fabricate the old per-image proxy mask: aggregate scale resume does not
    # require it, and None preserves that the information is unavailable.
    migrated.setdefault("effective_lambda_cka", migrated["lambda_cka"])
    migrated.setdefault("gradient_ratio", None)
    migrated.setdefault("cka_source_weight", 1.0)
    migrated.setdefault("semantic_target_weight", 0.0)
    migrated.setdefault("proxy_target_hit_mask", None)
    migrated.setdefault("proxy_tap_path", "legacy-unrecorded")
    migrated["source_image_ids"] = tuple(migrated["source_image_ids"])
    migrated["target_reference_ids"] = tuple(migrated["target_reference_ids"])
    if migrated["proxy_target_hit_mask"] is not None:
        migrated["proxy_target_hit_mask"] = tuple(migrated["proxy_target_hit_mask"])
    return migrated


def _load_attack_result(path: Path) -> AttackRunResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AttackRunResult(**_migrate_attack_result_payload(payload))


def _artifact_dir(
    output_dir: Path,
    pair: ModelPair,
    phase: str,
    batch_index: int,
    lambda_cka: float,
) -> Path:
    return (
        output_dir
        / "attacks"
        / pair.pair_id
        / phase
        / f"batch_{batch_index:02d}"
        / f"lambda_{lambda_cka:g}"
    )


def _evaluation_path(output_dir: Path, pair: ModelPair, phase: str, batch_index: int) -> Path:
    return output_dir / "evaluation" / phase / pair.pair_id / f"batch_{batch_index:02d}.json"


def _load_rates(path: Path) -> AttackRates:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AttackRates(**payload["rates"])


def _reference_batch_index(fixed_reference_bank: bool) -> int | None:
    """Freeze the first reference bank when running a controlled scale-up."""
    return 0 if fixed_reference_bank else None


def _evaluate_pending_batches(
    context: CommandContext,
    pair: ModelPair,
    phase: str,
    lambda_cka: float,
    attacks: list[tuple[int, AttackRunResult]],
    source_label: int,
    target_label: int,
    prompt: str,
    *,
    evaluate_all_frozen: bool = False,
) -> dict[int, AttackRates]:
    rates_by_batch: dict[int, AttackRates] = {}
    pending = []
    for batch_index, result in attacks:
        if not evaluate_all_frozen and not result.proxy_target_all_hit:
            continue
        path = _evaluation_path(context.output_dir, pair, phase, batch_index)
        if context.resume and path.is_file():
            rates_by_batch[batch_index] = _load_rates(path)
        else:
            pending.append((batch_index, result, path))
    if not pending:
        return rates_by_batch

    model = None
    try:
        snapshot = local_snapshot(context.project_root / ".hf-cache", pair.target_model)
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"))
        generator = TransformersTargetGenerator(model, processor)
        for batch_index, result, path in pending:
            artifact_dir = _artifact_dir(context.output_dir, pair, phase, batch_index, lambda_cka)
            count = len(result.source_image_ids)
            clean_outputs = tuple(
                generator.generate_label(artifact_dir / f"{index:02d}_clean.png", prompt)
                for index in range(count)
            )
            adversarial_outputs = tuple(
                generator.generate_label(artifact_dir / f"{index:02d}_adv.png", prompt)
                for index in range(count)
            )
            rates = attack_rates(
                tuple(output.parsed_label for output in clean_outputs),
                tuple(output.parsed_label for output in adversarial_outputs),
                source_human_label=source_label,
                target_human_label=target_label,
            )
            write_json(
                path,
                {
                    "clean_outputs": tuple(asdict(output) for output in clean_outputs),
                    "adversarial_outputs": tuple(asdict(output) for output in adversarial_outputs),
                    "rates": asdict(rates),
                },
            )
            rates_by_batch[batch_index] = rates
            print(
                f"{pair.pair_id} {phase} batch={batch_index:02d} "
                f"TASR={rates.targeted_hit_count}/{rates.clean_valid_count} "
                f"ASR={rates.untargeted_hit_count}/{rates.clean_valid_count}",
                flush=True,
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
    return rates_by_batch


def run_scaled(context: CommandContext) -> str:
    require_real_run_ready(context)
    if context.image_count not in {8, 50, 500}:
        raise ValueError("attack scaled requires --image-count 8, 50, or 500")
    attack_config = resolve_attack_config(context, require_canonical_lambda_grid=False)
    if len(attack_config.lambdas) != 1:
        raise ValueError("Scaled runs require exactly one selected lambda")
    lambda_cka = attack_config.lambdas[0]
    data_config = resolve_data_config(context)
    prompt = get_prompt(os.environ.get("PRIMARY_ML_CKA_PROMPT_ID", "original"))
    phase = f"scale_{context.image_count}"
    evaluate_all_frozen = os.environ.get("PRIMARY_ML_CKA_EVALUATE_ALL_FROZEN") == "1"
    fixed_reference_bank = os.environ.get("PRIMARY_ML_CKA_FIXED_REFERENCE_BANK") == "1"
    pairs = tuple(
        pair for pair in MODEL_PAIRS if context.pair_id is None or pair.pair_id == context.pair_id
    )
    if context.dry_run:
        return (
            f"dry-run: {phase} pairs={len(pairs)} lambda={lambda_cka:g} "
            f"alpha={attack_config.cka_target_weight:g}"
        )
    references = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )
    rows = []
    summaries = []
    for pair in pairs:
        attacks: list[tuple[int, AttackRunResult]] = []
        try:
            require_proxy_tap(context, pair.proxy_model)
            records = load_phase_records(context.output_dir, pair.target_model, "main")
            if len(records) < context.image_count:
                raise RuntimeError(
                    f"{pair.target_model} has only {len(records)} clean-valid main images; "
                    f"requested {context.image_count}"
                )
            selected = records[: context.image_count]
            batch_size = attack_config.batch_size
            batches = tuple(
                selected[start : min(start + batch_size, len(selected))]
                for start in range(0, len(selected), batch_size)
            )
            for batch_index, source in enumerate(batches):
                log_path = _log_path(context.output_dir, pair, phase, batch_index, lambda_cka)
                if context.resume and log_path.is_file():
                    result = _load_attack_result(log_path)
                    print(
                        f"{pair.pair_id} {phase} batch={batch_index:02d} resumed",
                        flush=True,
                    )
                else:
                    result = attack_one_batch(
                        pair,
                        project_root=context.project_root,
                        output_dir=context.output_dir,
                        phase=phase,
                        source_records=source,
                        reference_records=references,
                        source_batch_index=batch_index,
                        reference_batch_index=_reference_batch_index(fixed_reference_bank),
                        lambda_cka=lambda_cka,
                        seed=(context.seed if context.seed is not None else 42) + batch_index,
                        steps=attack_config.steps,
                        attack_config=attack_config,
                        data_config=data_config,
                        reference_bank_size=attack_config.reference_bank_size,
                        cka_source_weight=attack_config.cka_source_weight,
                        cka_target_weight=attack_config.cka_target_weight,
                        semantic_target_weight=attack_config.semantic_target_weight,
                        gradient_ratio=attack_config.gradient_ratio,
                        # The 8/8 gate determines transfer eligibility after
                        # optimization; it must not shorten the configured
                        # attack budget, otherwise lambda/alpha trials receive
                        # different numbers of updates.
                        early_stop_proxy_gate=False,
                        progress_interval=10,
                        prompt=prompt,
                    )
                attacks.append((batch_index, result))
            rates = _evaluate_pending_batches(
                context,
                pair,
                phase,
                lambda_cka,
                attacks,
                data_config.source_human_label,
                data_config.target_human_label,
                prompt,
                evaluate_all_frozen=evaluate_all_frozen,
            )
            seed = context.seed if context.seed is not None else attack_config.main_seed
            for batch_index, result in attacks:
                rows.append(
                    result_row(
                        pair,
                        result,
                        seed + batch_index,
                        attack_config.steps,
                        rates.get(batch_index),
                    )
                )
            proxy_eligible = sum(result.proxy_target_all_hit for _, result in attacks)
            target_hits = sum(item.targeted_hit_count for item in rates.values())
            target_denominator = sum(item.clean_valid_count for item in rates.values())
            metric_name = "frozen_TASR" if evaluate_all_frozen else "conditional_TASR"
            summaries.append(
                f"{pair.pair_id}: proxy_eligible_batches={proxy_eligible}/{len(attacks)} "
                f"{metric_name}={target_hits}/{target_denominator}"
            )
        except Exception as exc:
            seed = context.seed if context.seed is not None else attack_config.main_seed
            rows.append(
                blocked_result_row(
                    pair,
                    phase=phase,
                    seed=seed,
                    steps=attack_config.steps,
                    error=exc,
                    lambda_cka=lambda_cka,
                )
            )
            summaries.append(f"{pair.pair_id}: BLOCKED {exc!r}")
        write_results_csv(context.output_dir / "summaries" / f"{phase}_results.csv", rows)
    return "\n".join(summaries)
