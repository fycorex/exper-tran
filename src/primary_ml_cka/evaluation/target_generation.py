from pathlib import Path

from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.models.common.protocols import TargetGenerator


def evaluate_paths(
    generator: TargetGenerator, paths: tuple[Path, ...], prompt: str
) -> tuple[GenerationOutput, ...]:
    return tuple(generator.generate_label(path, prompt) for path in paths)
