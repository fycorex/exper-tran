from dataclasses import dataclass
from enum import StrEnum


class ExperimentType(StrEnum):
    CROSS_FAMILY = "Cross-Family"
    INTRA_FAMILY = "Intra-Family"


@dataclass(frozen=True, slots=True)
class ModelPair:
    pair_id: str
    exp_type: ExperimentType
    proxy_model: str
    target_model: str


MODEL_PAIRS = (
    ModelPair("P02", ExperimentType.CROSS_FAMILY, "Qwen/Qwen3.5-4B", "google/gemma-4-E4B-it"),
    ModelPair(
        "P06",
        ExperimentType.CROSS_FAMILY,
        "openai/clip-vit-large-patch14",
        "OpenGVLab/InternVL3_5-2B-HF",
    ),
    ModelPair(
        "P11",
        ExperimentType.CROSS_FAMILY,
        "google/siglip2-so400m-patch14-384",
        "google/gemma-4-E2B-it",
    ),
    ModelPair("P14", ExperimentType.INTRA_FAMILY, "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"),
    ModelPair("P20", ExperimentType.INTRA_FAMILY, "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B"),
    ModelPair(
        "P16",
        ExperimentType.INTRA_FAMILY,
        "OpenGVLab/InternVL3_5-2B-HF",
        "OpenGVLab/InternVL3_5-4B-HF",
    ),
    ModelPair(
        "P21",
        ExperimentType.INTRA_FAMILY,
        "OpenGVLab/InternVL3_5-4B-HF",
        "OpenGVLab/InternVL3_5-2B-HF",
    ),
    ModelPair(
        "P19",
        ExperimentType.INTRA_FAMILY,
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
    ),
    ModelPair(
        "P22",
        ExperimentType.INTRA_FAMILY,
        "google/gemma-4-E4B-it",
        "google/gemma-4-E2B-it",
    ),
)

MODEL_REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "OpenGVLab/InternVL3_5-2B-HF": "3f301ffcf3dcbb47893afae6650ea3e78d96fb6d",
    "OpenGVLab/InternVL3_5-4B-HF": "6bd4487402110ef9889ba50eb7aefeb302526fed",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "openai/clip-vit-base-patch32": "c7244be81152024ce0e99ac8d2e373a8953d9f9a",
    "google/siglip2-so400m-patch14-384": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
}


def get_pair(pair_id: str) -> ModelPair:
    try:
        return next(pair for pair in MODEL_PAIRS if pair.pair_id == pair_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown pair ID: {pair_id}") from exc
