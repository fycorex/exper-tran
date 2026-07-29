from primary_ml_cka.experiment.orchestration import CommandContext
from primary_ml_cka.experiment.smoke import run_smoke


def run(context: CommandContext) -> str:
    return run_smoke(context)
