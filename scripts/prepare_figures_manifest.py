#!/usr/bin/env python3
"""Prepare the figures manifest and copy figure assets for the Next.js gallery."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ALLOWED_EXTENSIONS = {".png", ".svg", ".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy generated figures into the Next.js public directory and emit a "
            "manifest describing the available assets."
        )
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        required=True,
        help="Directory containing generated figure assets to publish.",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        required=True,
        help="Destination directory inside the Next.js app's public folder.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path where the figures manifest JSON should be written.",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        help=(
            "Optional ISO timestamp to write into the manifest. Defaults to the "
            "current UTC time."
        ),
    )
    return parser.parse_args()


def collect_figure_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return files


def slugify(path_obj: Path) -> str:
    if str(path_obj) in {".", ""}:
        return "top-level-figures"
    slug = re.sub(r"[^a-z0-9]+", "-", path_obj.as_posix().lower()).strip("-")
    return slug or "figures"


def copy_figures(figures: Iterable[Path], source_root: Path, destination_root: Path) -> dict[Path, list[Path]]:
    destination_root = destination_root.resolve()
    shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[Path, list[Path]] = {}
    for figure in figures:
        relative = figure.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure, destination)
        grouped.setdefault(relative.parent, []).append(relative)
    return grouped


def build_manifest(groups: dict[Path, list[Path]], generated_at: str) -> dict:
    manifest_groups = []
    for directory in sorted(groups.keys(), key=lambda p: p.as_posix()):
        dir_str = directory.as_posix()
        title = "Top-level figures" if dir_str in {".", ""} else dir_str
        slug = slugify(directory)
        items = []
        for rel_path in sorted(groups[directory], key=lambda p: p.as_posix()):
            rel_posix = rel_path.as_posix()
            href = f"figures/{rel_posix}"
            suffix = rel_path.suffix.lower()
            file_type = "pdf" if suffix == ".pdf" else "image"
            items.append(
                {
                    "name": rel_path.name,
                    "href": href,
                    "preview": href,
                    "type": file_type,
                }
            )
        manifest_groups.append(
            {
                "title": title,
                "slug": slug,
                "items": items,
            }
        )
    return {"generatedAt": generated_at, "groups": manifest_groups}


def main() -> None:
    args = parse_args()

    figures_root = args.figures_root.resolve()
    public_root = args.public_root.resolve()
    manifest_path = args.manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    figures = collect_figure_files(figures_root)
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()

    if not figures:
        empty_manifest = {"generatedAt": generated_at, "groups": []}
        manifest_path.write_text(json.dumps(empty_manifest, indent=2), encoding="utf-8")
        print("No figures discovered. Wrote empty manifest with generated timestamp.")
        return

    grouped = copy_figures(figures, figures_root, public_root)
    manifest = build_manifest(grouped, generated_at)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "Copied %d figures across %d group(s)." % (
            sum(len(items) for items in grouped.values()),
            len(grouped),
        )
    )


if __name__ == "__main__":
    main()
