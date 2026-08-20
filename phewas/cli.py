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
    phenotype_selection = parser.add_mutually_exclusive_group()
    phenotype_selection.add_argument(
        "--pheno",
        type=str,
        help=(
            "Filter to analyze only a single phenotype by name. "
            "All other phenotypes will be excluded from the analysis."
        ),
    )
    phenotype_selection.add_argument(
        "--pheno-file",
        type=str,
        help=(
            "Analyze only the phenotype names listed in this text file, one sanitized "
            "phenotype name per line. Blank lines and lines beginning with '#' are ignored."
        ),
    )
    parser.add_argument(
        "--pc-source",
        type=str,
        choices=list(run.VALID_PC_SOURCES),
        default=run.PC_SOURCE_GLOBAL,
        help=(
            "Which genetic principal components to adjust for. 'global' (default) uses the "
            "reference-projected components published with the callset, and is what every "
            "pooled result uses. 'within-ancestry' uses components fit separately inside a "
            "single ancestry group, resolving fine-scale structure that projected global "
            "components cannot; it requires --pop-label because the axes differ by group."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def apply_cli_configuration(args: argparse.Namespace) -> dict[str, object]:
    """Apply CLI arguments to runtime globals and return a pipeline config."""

    pipeline_config: dict[str, object] = {
        "min_cases_controls": None,
        "population_filter": "all",
        "phenotype_filter": None,
        "phenotype_file": None,
    }

    if getattr(args, "min_cases_controls", None) is not None:
        threshold = int(args.min_cases_controls)
        pipeline_config["min_cases_controls"] = threshold
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
        pipeline_config["population_filter"] = normalized
        run.POPULATION_FILTER = normalized
        os.environ["FERROMIC_POPULATION_FILTER"] = normalized
    else:
        run.POPULATION_FILTER = "all"
        os.environ.pop("FERROMIC_POPULATION_FILTER", None)

    pheno_name = getattr(args, "pheno", None)
    if pheno_name is not None:
        pipeline_config["phenotype_filter"] = pheno_name.strip()
        run.PHENOTYPE_FILTER = pheno_name.strip()
        os.environ["FERROMIC_PHENOTYPE_FILTER"] = run.PHENOTYPE_FILTER
    else:
        run.PHENOTYPE_FILTER = None
        os.environ.pop("FERROMIC_PHENOTYPE_FILTER", None)

    phenotype_file = getattr(args, "pheno_file", None)
    if phenotype_file is not None:
        normalized_file = phenotype_file.strip()
        if not normalized_file:
            raise SystemExit("--pheno-file must name a non-empty path")
        pipeline_config["phenotype_file"] = normalized_file

    pc_source = run._normalize_pc_source(
        getattr(args, "pc_source", None) or run.PC_SOURCE_GLOBAL
    )
    if (
        pc_source == run.PC_SOURCE_WITHIN_ANCESTRY
        and pipeline_config["population_filter"] == "all"
    ):
        raise SystemExit(
            "--pc-source within-ancestry requires --pop-label. Ancestry-specific principal "
            "components are fit separately inside each genetic ancestry group, so they are "
            "not comparable across groups and cannot be used in a pooled multi-ancestry run."
        )
    pipeline_config["pc_source"] = pc_source
    run.PC_SOURCE = pc_source
    if pc_source == run.PC_SOURCE_GLOBAL:
        os.environ.pop("FERROMIC_PC_SOURCE", None)
    else:
        os.environ["FERROMIC_PC_SOURCE"] = pc_source

    return pipeline_config


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pipeline_config = apply_cli_configuration(args)
    run.supervisor_main(pipeline_config=pipeline_config)


if __name__ == "__main__":  # pragma: no cover - CLI execution
    main()
