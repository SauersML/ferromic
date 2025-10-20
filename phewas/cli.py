"""Command line entrypoint for configuring and launching the PheWAS pipeline."""

from __future__ import annotations

import argparse
from typing import Sequence

from . import run


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argparse sanitises this path in tests
        raise argparse.ArgumentTypeError("Expected an integer value") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Ferromic PheWAS pipeline with optional configuration overrides.",
    )
    parser.add_argument(
        "--min-cases-controls",
        type=_positive_int,
        help=(
            "Minimum number of cases and controls required to prefilter phenotypes before Stage-1. "
            "Adjusts which phenotypes are enqueued without modifying downstream model thresholds."
        ),
    )
    parser.add_argument(
        "--pop-label",
        type=str,
        help=(
            "Restrict the analysis to participants with the provided population label. "
            "Matches the ancestry labels produced during shared setup."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def apply_cli_configuration(args: argparse.Namespace) -> None:
    if getattr(args, "min_cases_controls", None) is not None:
        threshold = int(args.min_cases_controls)
        run.CLI_MIN_CASES_CONTROLS_OVERRIDE = threshold
    else:
        run.CLI_MIN_CASES_CONTROLS_OVERRIDE = None

    if getattr(args, "pop_label", None) is not None:
        run.POPULATION_FILTER = args.pop_label


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_cli_configuration(args)
    run.supervisor_main()


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
