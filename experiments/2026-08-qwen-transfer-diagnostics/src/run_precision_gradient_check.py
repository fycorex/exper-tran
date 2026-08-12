import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional

from primary_ml_cka.config.schema import AttackConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.experiment.attack_generation import _cuda_images
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", default="P21")
    return parser.parse_args()


def _gradient(
    model_id: str,
    images: torch.Tensor,
    hf_home: Path,
    *,
    keep_vision_bf16: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if keep_vision_bf16:
        os.environ["PRIMARY_ML_CKA_KEEP_VISION_BF16"] = "1"
    else:
        os.environ.pop("PRIMARY_ML_CKA_KEEP_VISION_BF16", None)
    proxy = load_proxy(model_id, hf_home, torch.device("cuda"), AttackConfig())
    attack_images = images.detach().clone().requires_grad_(True)
    output = proxy.target_loss(attack_images, 7, CLASSIFICATION_PROMPT)
    gradient = torch.autograd.grad(output.loss, attack_images, only_inputs=True)[0]
    logits = output.class_logits.detach().float().cpu()
    gradient = gradient.detach().float().cpu()
    del proxy, attack_images, output
    gc.collect()
    torch.cuda.empty_cache()
    return gradient, logits


def main() -> None:
    args = _arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for precision gradient diagnosis")
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[3]
    pair = get_pair(args.pair_id)
    records = read_manifest(
        output_dir
        / "diagnostics"
        / "objective_split_common48_rho03"
        / "common_clean.jsonl"
    )
    images = _cuda_images(output_dir / "canonical_images", records, 224)
    nf4_gradient, nf4_logits = _gradient(
        pair.proxy_model,
        images,
        project_root / ".hf-cache",
        keep_vision_bf16=False,
    )
    bf16_gradient, bf16_logits = _gradient(
        pair.proxy_model,
        images,
        project_root / ".hf-cache",
        keep_vision_bf16=True,
    )
    per_image_cosine = functional.cosine_similarity(
        nf4_gradient.flatten(1), bf16_gradient.flatten(1), dim=1
    )
    result = {
        "pair_id": pair.pair_id,
        "comparison": "all_nf4_vs_bf16_vision_and_projector_with_nf4_language",
        "image_count": len(records),
        "gradient_cosine_per_image": per_image_cosine.tolist(),
        "gradient_cosine_mean": float(per_image_cosine.mean()),
        "gradient_cosine_min": float(per_image_cosine.min()),
        "nf4_gradient_l1": float(nf4_gradient.abs().mean()),
        "bf16_vision_gradient_l1": float(bf16_gradient.abs().mean()),
        "gradient_l1_ratio_nf4_over_bf16_vision": float(
            nf4_gradient.abs().mean() / bf16_gradient.abs().mean()
        ),
        "closed_set_logit_cosine_mean": float(
            functional.cosine_similarity(nf4_logits, bf16_logits, dim=1).mean()
        ),
        "closed_set_logit_max_abs_difference": float(
            (nf4_logits - bf16_logits).abs().max()
        ),
    }
    path = output_dir / "diagnostics" / "cka_validity" / f"{pair.pair_id}_precision_gradient.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
