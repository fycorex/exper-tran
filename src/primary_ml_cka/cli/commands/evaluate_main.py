import csv
import gc
import json
from dataclasses import asdict

import torch

from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_real_run_ready,
    resolve_data_config,
)
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def run(context: CommandContext) -> str:
    require_real_run_ready(context)
    if context.dry_run:
        return "dry-run: evaluate frozen main PNGs only; attack generation is not called"
    results_path = context.output_dir / "summaries" / "all_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError("Main result rows are missing")
    with results_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    data_config = resolve_data_config(context)
    summaries = []
    pair_ids = tuple(dict.fromkeys(row["pair_id"] for row in rows))
    from primary_ml_cka.domain.identifiers import get_pair

    for pair_id in pair_ids:
        if context.pair_id is not None and pair_id != context.pair_id:
            continue
        pair = get_pair(pair_id)
        model = None
        try:
            snapshot = local_snapshot(context.project_root / ".hf-cache", pair.target_model)
            processor = load_processor(snapshot)
            model = load_target_for_generation(snapshot, torch.device("cuda"))
            generator = TransformersTargetGenerator(model, processor)
            raw_records = []
            for row in rows:
                proxy_gate_passed = row.get("proxy_target_all_hit") == "True" and row.get(
                    "proxy_target_hit_count"
                ) == row.get("proxy_target_hit_denominator")
                if (
                    row["pair_id"] != pair_id
                    or row["phase"] != "main"
                    or row["status"] != "ok"
                    or not proxy_gate_passed
                ):
                    continue
                artifact_dir = (
                    context.output_dir
                    / "attacks"
                    / pair_id
                    / "main"
                    / f"batch_{int(row['batch_id']):02d}"
                    / f"lambda_{float(row['lambda']):g}"
                )
                source_ids = row["source_image_ids"].split("|")
                clean_outputs = tuple(
                    generator.generate_label(
                        artifact_dir / f"{index:02d}_clean.png",
                        CLASSIFICATION_PROMPT,
                    )
                    for index in range(len(source_ids))
                )
                adversarial_outputs = tuple(
                    generator.generate_label(
                        artifact_dir / f"{index:02d}_adv.png",
                        CLASSIFICATION_PROMPT,
                    )
                    for index in range(len(source_ids))
                )
                rates = attack_rates(
                    tuple(item.parsed_label for item in clean_outputs),
                    tuple(item.parsed_label for item in adversarial_outputs),
                    source_human_label=data_config.source_human_label,
                    target_human_label=data_config.target_human_label,
                )
                row["clean_valid_count"] = str(rates.clean_valid_count)
                row["targeted_hit_count"] = str(rates.targeted_hit_count)
                row["tasr_percent"] = str(rates.tasr_percent)
                row["untargeted_hit_count"] = str(rates.untargeted_hit_count)
                row["asr_percent"] = str(rates.asr_percent)
                for image_id, clean_output, adversarial_output in zip(
                    source_ids, clean_outputs, adversarial_outputs, strict=True
                ):
                    raw_records.append(
                        {
                            "pair_id": pair_id,
                            "batch_id": row["batch_id"],
                            "lambda": row["lambda"],
                            "image_id": image_id,
                            "clean": asdict(clean_output),
                            "adversarial": asdict(adversarial_output),
                        }
                    )
            atomic_text_write(
                context.output_dir / "evaluation" / f"{pair_id}__main_outputs.jsonl",
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in raw_records),
            )
            summaries.append(
                f"{pair_id}: evaluated_rows={sum(row['pair_id'] == pair_id for row in rows)}"
            )
        except Exception as exc:
            summaries.append(f"{pair_id}: BLOCKED {exc!r}")
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
    temporary = results_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(results_path)
    return "\n".join(summaries)
