import argparse
import gc
import json
from pathlib import Path

import torch

from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.experiment.attack_generation import _cuda_images
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

MODEL_IDS = (
    "openai/clip-vit-large-patch14",
    "google/siglip2-so400m-patch14-384",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for contrastive clean preflight")
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[3]
    candidates = read_manifest(
        output_dir / "evaluation" / "manifests" / "source_validation_candidates.jsonl"
    )
    images = _cuda_images(output_dir / "canonical_images", candidates, 224)
    for model_id in MODEL_IDS:
        proxy = load_proxy(model_id, project_root / ".hf-cache", torch.device("cuda"))
        rows = []
        with torch.no_grad():
            for offset in range(0, len(candidates), 8):
                logits = proxy.target_loss(
                    images[offset : offset + 8], 7, CLASSIFICATION_PROMPT
                ).class_logits
                if logits is None:
                    raise RuntimeError(f"{model_id} did not return class logits")
                labels = logits.argmax(dim=1).add(1).tolist()
                for record, label in zip(candidates[offset : offset + 8], labels, strict=True):
                    rows.append(
                        {
                            "image_id": record.image_id,
                            "parsed_label": int(label),
                            "raw_output": str(int(label)),
                            "gate_type": "contrastive_closed_set",
                        }
                    )
        safe_name = model_id.replace("/", "__")
        atomic_text_write(
            output_dir / "evaluation" / f"{safe_name}__clean_screen.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )
        print(
            f"model={model_id} source_correct="
            f"{sum(row['parsed_label'] == 8 for row in rows)}/{len(rows)}",
            flush=True,
        )
        del proxy
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
