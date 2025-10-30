"""Command line entrypoint for configuring and launching the PheWAS pipeline."""

from __future__ import annotations

import argparse
import os
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
            "Minimum number of cases and controls required throughout the PheWAS run. "
            "Applies to both prefiltering and downstream model validation."
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
    parser.add_argument(
        "--pheno",
        type=str,
        help=(
            "Filter to analyze only a single phenotype by name. "
            "All other phenotypes will be excluded from the analysis."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def apply_cli_configuration(args: argparse.Namespace) -> None:
    if getattr(args, "min_cases_controls", None) is not None:
        threshold = int(args.min_cases_controls)
        run.CLI_MIN_CASES_CONTROLS_OVERRIDE = threshold
        run.MIN_CASES_FILTER = threshold
        run.MIN_CONTROLS_FILTER = threshold
    else:
        run.CLI_MIN_CASES_CONTROLS_OVERRIDE = None
        run.MIN_CASES_FILTER = run.DEFAULT_MIN_CASES_FILTER
        run.MIN_CONTROLS_FILTER = run.DEFAULT_MIN_CONTROLS_FILTER

    raw_label = getattr(args, "pop_label", None)
    if raw_label is not None:
        label = raw_label.strip()
        normalized = label or "all"
        run.POPULATION_FILTER = normalized
        os.environ["FERROMIC_POPULATION_FILTER"] = normalized
    else:
        run.POPULATION_FILTER = "all"
        os.environ.pop("FERROMIC_POPULATION_FILTER", None)

    pheno_name = getattr(args, "pheno", None)
    if pheno_name is not None:
        run.PHENOTYPE_FILTER = pheno_name.strip()
        os.environ["FERROMIC_PHENOTYPE_FILTER"] = run.PHENOTYPE_FILTER
    else:
        run.PHENOTYPE_FILTER = None
        os.environ.pop("FERROMIC_PHENOTYPE_FILTER", None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    apply_cli_configuration(args)
    run.supervisor_main()


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
