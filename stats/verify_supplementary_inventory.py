#!/usr/bin/env python3
"""Verify and export the final supplementary figure/table inventory.

This checks only repository metadata and the table generator. It never opens,
edits, or generates a manuscript document.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from supplementary_inventory import (
    FINAL_SUPPLEMENTARY_FIGURES,
    FINAL_SUPPLEMENTARY_TABLE_ORDER,
    ORIGINAL_FIGURE_TO_FINAL,
    RESPONSE_ONLY_FIGURE_TITLES,
)


def verify_inventory() -> None:
    figure_numbers = [figure.number for figure in FINAL_SUPPLEMENTARY_FIGURES]
    if figure_numbers != list(range(1, 21)):
        raise RuntimeError(
            f"Figure inventory must be exactly S1-S20; observed {figure_numbers}."
        )
    figure_keys = [figure.key for figure in FINAL_SUPPLEMENTARY_FIGURES]
    if len(figure_keys) != len(set(figure_keys)):
        raise RuntimeError("Supplementary figure keys are not unique.")
    if any(
        response_title in figure.title
        for figure in FINAL_SUPPLEMENTARY_FIGURES
        for response_title in RESPONSE_ONLY_FIGURE_TITLES
    ):
        raise RuntimeError(
            "The response-only 17q21.31 comparison panel was promoted to a "
            "supplementary figure."
        )
    for figure in FINAL_SUPPLEMENTARY_FIGURES:
        if figure.source.startswith("original Figure"):
            if figure.original_number is None or figure.asset or figure.caption:
                raise RuntimeError(
                    f"Original Figure S{figure.number} has revision-asset metadata."
                )
        elif figure.source == "revision figure":
            if figure.original_number is not None or not figure.asset or not figure.caption:
                raise RuntimeError(
                    f"Revision Figure S{figure.number} lacks an asset or caption."
                )
        else:
            raise RuntimeError(
                f"Unknown source class for Figure S{figure.number}: {figure.source!r}"
            )
    if 11 in ORIGINAL_FIGURE_TO_FINAL:
        raise RuntimeError(
            "Original Figure S11 is response-only and must not be carried into "
            "the final supplement."
        )
    expected_old_map = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 11,
        6: 12,
        7: 13,
        8: 14,
        9: 16,
        10: 18,
        12: 19,
        13: 20,
    }
    if ORIGINAL_FIGURE_TO_FINAL != expected_old_map:
        raise RuntimeError(
            "Original-to-final supplementary figure mapping is incorrect: "
            f"{ORIGINAL_FIGURE_TO_FINAL!r}."
        )
    if len(FINAL_SUPPLEMENTARY_TABLE_ORDER) != 21:
        raise RuntimeError("Table inventory must be exactly S1-S21.")


def write_inventory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("type", "number", "title", "source_or_key"))
        for figure in FINAL_SUPPLEMENTARY_FIGURES:
            writer.writerow(
                ("Figure", f"S{figure.number}", figure.title, figure.source)
            )
        for number, title in enumerate(FINAL_SUPPLEMENTARY_TABLE_ORDER, start=1):
            writer.writerow(
                ("Table", f"S{number}", title, "stats/generate_tables.py")
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-tsv", type=Path)
    args = parser.parse_args()
    try:
        verify_inventory()
        if args.write_tsv:
            write_inventory(args.write_tsv)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "Verified final supplementary inventory: 20 figures and 21 tables; "
        "response-only figures excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
