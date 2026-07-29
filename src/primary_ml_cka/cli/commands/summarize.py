from primary_ml_cka.experiment.orchestration import CommandContext
from primary_ml_cka.reporting.tables import csv_to_xlsx


def run(context: CommandContext) -> str:
    csv_path = context.output_dir / "summaries" / "all_results.csv"
    xlsx_path = context.output_dir / "summaries" / "paper_matrix.xlsx"
    if context.dry_run:
        return f"dry-run: summarize {csv_path} into {xlsx_path}"
    if not csv_path.is_file():
        raise FileNotFoundError(
            "all_results.csv does not exist; incomplete results are not fabricated"
        )
    csv_to_xlsx(csv_path, xlsx_path)
    return f"wrote {xlsx_path}"
