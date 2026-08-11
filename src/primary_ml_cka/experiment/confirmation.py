import json
import os

from primary_ml_cka.artifacts.writers import write_json, write_results_csv
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.experiment.attack_generation import (
    attack_one_batch,
    blocked_result_row,
    load_phase_records,
    result_row,
)
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.evaluation.target_generation import evaluate_local_frozen_batch
from primary_ml_cka.prompts.variants import get_prompt


def run_confirmation(context: CommandContext) -> str:
    require_real_run_ready(context)
    selected = context.output_dir / "summaries" / "selected_lambda.json"
    if not context.dry_run and not selected.is_file():
        raise RuntimeError("No selected positive lambda artifact exists")
    if context.dry_run:
        return "dry-run: confirmation uses disjoint manifest and configured confirmation seed"
    selected_values = {
        item["pair_id"]: float(item["lambda_cka"])
        for item in json.loads(selected.read_text(encoding="utf-8"))
    }
    config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    prompt = get_prompt(os.environ.get("PRIMARY_ML_CKA_PROMPT_ID", "original"))
    references = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )
    rows = []
    failures = []
    for pair in MODEL_PAIRS:
        if context.pair_id is not None and pair.pair_id != context.pair_id:
            continue
        if pair.pair_id not in selected_values:
            failures.append(f"{pair.pair_id}: BLOCKED no selected lambda")
            continue
        try:
            records = load_phase_records(context.output_dir, pair.target_model, "confirmation")
            for batch_index in range(len(records) // config.batch_size):
                source = records[
                    batch_index * config.batch_size : (batch_index + 1) * config.batch_size
                ]
                result = attack_one_batch(
                    pair,
                    project_root=context.project_root,
                    output_dir=context.output_dir,
                    phase="confirmation",
                    source_records=source,
                    reference_records=references,
                    source_batch_index=batch_index,
                    reference_batch_index=(
                        data_config.main_max_count // config.batch_size + batch_index
                    ),
                    lambda_cka=selected_values[pair.pair_id],
                    seed=context.seed if context.seed is not None else config.confirmation_seed,
                    steps=config.steps,
                    attack_config=config,
                    data_config=data_config,
                    cka_target_weight=config.cka_target_weight,
                    prompt=prompt,
                )
                rates = None
                if result.proxy_target_all_hit:
                    artifact_dir = (
                        context.output_dir
                        / "attacks"
                        / pair.pair_id
                        / "confirmation"
                        / f"batch_{batch_index:02d}"
                        / f"lambda_{selected_values[pair.pair_id]:g}"
                    )
                    evaluation = evaluate_local_frozen_batch(
                        model_id=pair.target_model,
                        hf_home=context.project_root / ".hf-cache",
                        artifact_dir=artifact_dir,
                        image_count=len(source),
                        prompt=prompt,
                        source_human_label=data_config.source_human_label,
                        target_human_label=data_config.target_human_label,
                    )
                    rates = evaluation.rates
                    write_json(
                        context.output_dir
                        / "evaluation"
                        / "confirmation"
                        / pair.pair_id
                        / f"batch_{batch_index:02d}.json",
                        evaluation,
                    )
                rows.append(
                    result_row(
                        pair,
                        result,
                        context.seed if context.seed is not None else config.confirmation_seed,
                        config.steps,
                        rates,
                    )
                )
        except Exception as exc:
            seed = context.seed if context.seed is not None else config.confirmation_seed
            rows.append(
                blocked_result_row(
                    pair,
                    phase="confirmation",
                    seed=seed,
                    steps=config.steps,
                    error=exc,
                )
            )
            failures.append(f"{pair.pair_id}: BLOCKED {exc!r}")
    write_results_csv(context.output_dir / "summaries" / "confirmation_results.csv", rows)
    return f"completed_rows={len(rows)}\n" + "\n".join(failures)
