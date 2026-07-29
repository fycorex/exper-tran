import subprocess
import sys

from primary_ml_cka.experiment.orchestration import CommandContext
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.infrastructure.device import require_cuda


def run(context: CommandContext) -> str:
    if context.dry_run:
        return "dry-run: pytest tests/unit tests/integration (GPU smoke excluded)"
    require_cuda()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit", "tests/integration", "-m", "not gpu"],
        cwd=context.project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    report = result.stdout + result.stderr
    atomic_text_write(context.output_dir / "summaries" / "test_report.txt", report)
    if result.returncode != 0:
        raise RuntimeError(f"Correctness tests failed with exit code {result.returncode}\n{report}")
    return report.rstrip()
