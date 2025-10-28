#!/usr/bin/env python3
"""Determine the most likely figures directory within an extracted artifact."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

ALLOWED_EXTENSIONS = {".png", ".pdf", ".svg"}
DEFAULT_LIMIT = 512


def iter_candidate_files(root: Path, limit: int) -> Iterable[Path]:
    """Yield up to *limit* files beneath *root* with allowed extensions."""
    count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        yield path
        count += 1
        if count >= limit:
            break


def find_named_figures_parent(path: Path) -> Optional[Path]:
    """Return the nearest ancestor directory explicitly named ``figures``."""
    for parent in path.parents:
        if parent.name == "figures":
            return parent
    return None


def choose_common_parent(paths: Iterable[Path]) -> Optional[Path]:
    """Compute the common parent directory of the provided paths."""
    parents = [p.parent for p in paths]
    if not parents:
        return None
    common = Path(os.path.commonpath([str(p) for p in parents]))
    return common


def determine_figures_root(search_root: Path, limit: int) -> Optional[Path]:
    """Inspect *search_root* for figure-like files and return a plausible root."""
    if not search_root.exists():
        return None

    figure_files = list(iter_candidate_files(search_root, limit))
    if not figure_files:
        return None

    for fig in figure_files:
        named_parent = find_named_figures_parent(fig)
        if named_parent is not None:
            return named_parent

    return choose_common_parent(figure_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    figures_root = determine_figures_root(args.search_root, args.limit)
    if figures_root is None:
        return 0

    print(figures_root.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
