#!/usr/bin/env python3
"""Audit exact answer-token positions for the v3 zero-based output codes."""

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.domain.output_codes import OUTPUT_CODES
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

MODELS = (
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "OpenGVLab/InternVL3_5-2B-HF",
    "OpenGVLab/InternVL3_5-4B-HF",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real tokenizer/model audit")
    output = args.output_dir / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in MODELS:
        proxy = load_proxy(model_id, Path(".hf-cache"), torch.device("cuda"), AttackConfig())
        dummy = torch.zeros(1, 3, 224, 224, device="cuda")
        with torch.no_grad():
            visual_count = proxy._visual_token_count(proxy.visual_inputs(dummy))
        for output_code in OUTPUT_CODES:
            rendered, _, _, mask = proxy._answer_encoding(
                CLASSIFICATION_PROMPT, output_code, visual_count
            )
            rows.append(
                {
                    "model_id": model_id,
                    "revision": MODEL_REVISIONS[model_id],
                    "output_code": output_code,
                    "token_ids": list(mask.answer_token_ids),
                    "token_count": len(mask.answer_token_ids),
                    "answer_positions": list(mask.label_positions),
                    "rendered_answer": output_code,
                    "rendered_prompt": rendered,
                }
            )
        del proxy
        gc.collect()
        torch.cuda.empty_cache()
    (output / "tokenization_audit.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "tokenization_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = tuple(key for key in rows[0] if key != "rendered_prompt")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(f"wrote {len(rows)} tokenization rows to {output}", flush=True)


if __name__ == "__main__":
    main()
