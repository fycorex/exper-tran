import argparse
from collections.abc import Callable
from pathlib import Path

from primary_ml_cka.cli.commands import (
    clip_intra,
    clip_intra_alpha_scan,
    clip_intra_scan,
    confirm,
    evaluate_main,
    inspect_taps,
    model_similarity,
    prepare,
    run_main,
    scaled,
    screen,
    select_lambda,
    smoke,
    summarize,
    tests,
)
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.experiment.environment import write_environment
from primary_ml_cka.experiment.orchestration import CommandContext

COMMANDS: dict[tuple[str, str], Callable[[CommandContext], str]] = {
    ("data", "prepare"): prepare.run,
    ("target", "screen"): screen.run,
    ("models", "inspect-taps"): inspect_taps.run,
    ("tests", "run"): tests.run,
    ("attack", "smoke"): smoke.run,
    ("attack", "main"): run_main.run,
    ("attack", "scaled"): scaled.run,
    ("selection", "lambda"): select_lambda.run,
    ("evaluation", "main"): evaluate_main.run,
    ("analysis", "model-similarity"): model_similarity.run,
    ("attack", "confirm"): confirm.run,
    ("report", "summarize"): summarize.run,
    ("diagnostics", "clip-intra"): clip_intra.run,
    ("diagnostics", "clip-intra-scan"): clip_intra_scan.run,
    ("diagnostics", "clip-intra-alpha-scan"): clip_intra_alpha_scan.run,
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pair-id", choices=tuple(pair.pair_id for pair in MODEL_PAIRS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--image-count", type=int, choices=(8, 50, 500))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/primary_ml_cka_v1"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="primary-ml-cka")
    subparsers = parser.add_subparsers(dest="group", required=True)
    grouped: dict[str, list[str]] = {}
    for group, command in COMMANDS:
        grouped.setdefault(group, []).append(command)
    grouped["run"] = ["all"]
    for group, commands in grouped.items():
        group_parser = subparsers.add_parser(group)
        command_parsers = group_parser.add_subparsers(dest="command", required=True)
        for command in commands:
            command_parser = command_parsers.add_parser(command)
            _add_common(command_parser)
    return parser


def _context(args: argparse.Namespace) -> CommandContext:
    root = Path.cwd().resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    return CommandContext(
        root,
        output,
        args.pair_id,
        args.resume,
        args.dry_run,
        args.seed,
        args.config,
        args.image_count,
    )


def run_all(context: CommandContext) -> tuple[str, ...]:
    order = (
        ("data", "prepare"),
        ("target", "screen"),
        ("models", "inspect-taps"),
        ("tests", "run"),
        ("attack", "smoke"),
        ("attack", "main"),
        ("selection", "lambda"),
        ("evaluation", "main"),
        ("analysis", "model-similarity"),
        ("attack", "confirm"),
        ("report", "summarize"),
    )
    return tuple(
        f"{group} {command}: {COMMANDS[(group, command)](context)}" for group, command in order
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = _context(args)
    context.output_dir.mkdir(parents=True, exist_ok=True)
    write_environment(context.project_root, context.output_dir)
    if (args.group, args.command) == ("run", "all"):
        for line in run_all(context):
            print(line)
    else:
        print(COMMANDS[(args.group, args.command)](context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
