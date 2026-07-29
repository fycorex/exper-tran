from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.experiment.tap_inspection import inspect_proxy_taps


def run(context: CommandContext) -> str:
    config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    lines = inspect_proxy_taps(
        context.project_root / ".hf-cache",
        context.output_dir,
        context.pair_id,
        context.dry_run,
        config,
        data_config,
    )
    return "\n".join(lines)
