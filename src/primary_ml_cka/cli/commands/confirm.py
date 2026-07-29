from primary_ml_cka.experiment.confirmation import run_confirmation
from primary_ml_cka.experiment.orchestration import CommandContext


def run(context: CommandContext) -> str:
    return run_confirmation(context)
