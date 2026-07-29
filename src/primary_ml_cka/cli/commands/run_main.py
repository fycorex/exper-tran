from primary_ml_cka.experiment.main_ablation import run_main_ablation
from primary_ml_cka.experiment.orchestration import CommandContext


def run(context: CommandContext) -> str:
    return run_main_ablation(context)
