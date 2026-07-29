import csv
import gc
import io
import json
from dataclasses import asdict

import torch

from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.evaluation.target_generation import evaluate_paths
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.targets.contrastive import load_clip_target_generator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

PROXY_MODEL_ID = "openai/clip-vit-large-patch14"
TARGET_MODEL_ID = "openai/clip-vit-base-patch32"


def run(context: CommandContext) -> str:
    require_real_run_ready(context, require_taps=False)
    if context.dry_run:
        return (
            "dry-run: evaluate P06 CLIP-L/14 smoke PNGs with black-box "
            "CLIP-B/32 ten-class target"
        )
    smoke_path = context.output_dir / "summaries" / "smoke_results.csv"
    if not smoke_path.is_file():
        raise FileNotFoundError("P06 smoke results are missing")
    with smoke_path.open(encoding="utf-8", newline="") as handle:
        source_rows = [
            row
            for row in csv.DictReader(handle)
            if row["pair_id"] == "P06" and row["phase"] == "smoke"
        ]
    if not source_rows:
        raise RuntimeError("No P06 smoke rows were found")
    for row in source_rows:
        if (
            row["status"] != "ok"
            or row["proxy_target_all_hit"] != "True"
            or row["proxy_target_hit_count"] != row["proxy_target_hit_denominator"]
        ):
            raise RuntimeError(f"Lambda {row['lambda']} failed the frozen-PNG proxy gate")

    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    generator = load_clip_target_generator(
        TARGET_MODEL_ID,
        context.project_root / ".hf-cache",
        attack_config,
    )
    results = []
    try:
        first = source_rows[0]
        image_count = len(first["source_image_ids"].split("|"))
        first_artifact = (
            context.output_dir
            / "attacks"
            / "P06"
            / "smoke"
            / f"batch_{int(first['batch_id']):02d}"
            / f"lambda_{float(first['lambda']):g}"
        )
        clean_paths = tuple(
            first_artifact / f"{index:02d}_clean.png" for index in range(image_count)
        )
        clean_outputs = evaluate_paths(generator, clean_paths, CLASSIFICATION_PROMPT)
        for row in source_rows:
            artifact_dir = (
                context.output_dir
                / "attacks"
                / "P06"
                / "smoke"
                / f"batch_{int(row['batch_id']):02d}"
                / f"lambda_{float(row['lambda']):g}"
            )
            adversarial_paths = tuple(
                artifact_dir / f"{index:02d}_adv.png" for index in range(image_count)
            )
            adversarial_outputs = evaluate_paths(
                generator,
                adversarial_paths,
                CLASSIFICATION_PROMPT,
            )
            rates = attack_rates(
                tuple(output.parsed_label for output in clean_outputs),
                tuple(output.parsed_label for output in adversarial_outputs),
                source_human_label=data_config.source_human_label,
                target_human_label=data_config.target_human_label,
            )
            results.append(
                {
                    "proxy_model": PROXY_MODEL_ID,
                    "target_model": TARGET_MODEL_ID,
                    "lambda": float(row["lambda"]),
                    **asdict(rates),
                    "clean_outputs": [asdict(output) for output in clean_outputs],
                    "adversarial_outputs": [asdict(output) for output in adversarial_outputs],
                }
            )
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    output_base = context.output_dir / "evaluation" / "clip_intra"
    atomic_text_write(
        output_base / "raw_results.json",
        json.dumps(results, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = (
        "proxy_model",
        "target_model",
        "lambda",
        "clean_valid_count",
        "targeted_hit_count",
        "tasr_percent",
        "untargeted_hit_count",
        "asr_percent",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({key: result[key] for key in fieldnames})
    atomic_text_write(output_base / "results.csv", buffer.getvalue())
    return "\n".join(
        f"lambda={result['lambda']:g}: "
        f"TASR={result['targeted_hit_count']}/{result['clean_valid_count']} "
        f"ASR={result['untargeted_hit_count']}/{result['clean_valid_count']}"
        for result in results
    )
