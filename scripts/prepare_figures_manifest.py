#!/usr/bin/env python3
"""Prepare the figures manifest and copy figure assets for the Next.js gallery."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ALLOWED_EXTENSIONS = {".png", ".svg", ".pdf"}


def _generate_pop_dosage_plot(figures_root: Path) -> None:
    """Generate the population dosage plot if it's missing from artifacts.

    The build step sometimes downloads figure artifacts that omit the
    ``special/pop_dosage_plot.pdf`` output produced by ``stats/pop_dosage_plot.py``
    in the run-analysis workflow. To keep the gallery build resilient, run the
    script locally and copy the outputs into the expected figures root when the
    required asset is absent.
    """

    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "stats/pop_dosage_plot.py"

    if not script_path.exists():
        print(f"Population dosage script not found at {script_path}; skipping generation.")
        return

    print("special/pop_dosage_plot.pdf missing; generating via stats/pop_dosage_plot.py...")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=project_root)

    for ext in (".pdf", ".png"):
        source = project_root / f"special/pop_dosage_plot{ext}"
        if not source.exists():
            continue

        destination_dir = figures_root / "special"
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)


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
    parser.add_argument(
        "--required-figures-config",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON file(s) describing figures that must be present in the figures "
            "root. The format may be a list of objects with a 'filename' key or a "
            "dict containing groups -> figures -> filename."
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


def build_figure_index(figures: Iterable[Path], source_root: Path) -> dict[Path, Path]:
    """Return a mapping of relative -> source path."""

    index: dict[Path, Path] = {}
    for figure in figures:
        relative = figure.relative_to(source_root)
        index[relative] = figure
    return index


def load_required_figures(config_paths: Iterable[Path]) -> tuple[set[Path], dict[Path, set[Path]]]:
    required: set[Path] = set()

    def ensure_added(filename: str, source: Path, bucket: set[Path]) -> None:
        if not filename:
            return
        path = Path(filename)
        required.add(path)
        bucket.add(path)

    configs_loaded: dict[Path, int] = {}
    per_config: dict[Path, set[Path]] = {}

    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(f"Required figures config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        previous_count = len(required)
        added_paths: set[Path] = set()

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    ensure_added(entry.get("filename", ""), config_path, added_paths)
        elif isinstance(data, dict):
            for group in data.get("groups", []):
                for figure in group.get("figures", []):
                    if isinstance(figure, dict):
                        ensure_added(figure.get("filename", ""), config_path, added_paths)
        else:
            raise ValueError(
                f"Unsupported required figures config format in {config_path}"
            )

        configs_loaded[config_path] = len(required) - previous_count
        per_config[config_path] = added_paths

    for config_path, count in configs_loaded.items():
        if count == 0:
            raise ValueError(
                f"Required figures config {config_path} did not list any figure filenames"
            )

    return required, per_config


def copy_figures(figures: dict[Path, Path], destination_root: Path) -> dict[Path, list[Path]]:
    destination_root = destination_root.resolve()
    shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)

    grouped: dict[Path, list[Path]] = {}
    for relative, source in sorted(figures.items(), key=lambda item: item[0].as_posix()):
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
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

    figure_index = build_figure_index(
        collect_figure_files(figures_root), figures_root
    )

    if args.required_figures_config:
        required_figures, _ = load_required_figures(args.required_figures_config)

        required_dirs = {
            path.parent
            for path in required_figures
            if path.parent.as_posix() not in {".", ""}
        }
        missing_required_dirs = sorted(
            rel_dir.as_posix()
            for rel_dir in required_dirs
            if not (figures_root / rel_dir).exists()
        )
        if missing_required_dirs:
            missing_dir_list = "\n".join(missing_required_dirs)
            raise SystemExit(
                "Required figure directories were missing under the figures root:\n"
                f"{missing_dir_list}"
            )

        missing = {rel for rel in required_figures if rel not in figure_index}

        if Path("special/pop_dosage_plot.pdf") in missing:
            _generate_pop_dosage_plot(args.figures_root)
            figure_index = build_figure_index(
                collect_figure_files(args.figures_root), args.figures_root
            )
            missing = {rel for rel in required_figures if rel not in figure_index}

        if missing:
            missing_list = "\n".join(sorted(rel.as_posix() for rel in missing))
            raise SystemExit(
                "Required figure assets were missing from the figures root:\n"
                f"{missing_list}"
            )

    figures = list(figure_index.keys())
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat()

    if not figure_index:
        empty_manifest = {"generatedAt": generated_at, "groups": []}
        manifest_path.write_text(json.dumps(empty_manifest, indent=2), encoding="utf-8")
        print("No figures discovered. Wrote empty manifest with generated timestamp.")
        return

    grouped = copy_figures(figure_index, public_root)
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
