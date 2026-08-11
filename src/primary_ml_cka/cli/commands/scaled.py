from primary_ml_cka.experiment.orchestration import CommandContext
from primary_ml_cka.experiment.scaled import run_scaled


def run(context: CommandContext) -> str:
    return run_scaled(context)
