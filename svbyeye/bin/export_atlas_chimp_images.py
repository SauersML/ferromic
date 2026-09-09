#!/usr/bin/env python3
"""Export the chimpanzee alignment plots as the atlas review images.

The review tool (svbyeye/review_chimp_alignments.py) and the figures site serve
``web/figures-site/public/inversions/img/<inv_id>.chimp.webp`` with a 660 px
thumbnail under ``thumb/``. This turns every rendered ``<inv_id>.png`` into
those two files at the sizes the site already uses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

IMAGE_WIDTH = 1400
THUMB_WIDTH = 660
QUALITY = 82


def export(png: Path, image_dir: Path, thumb_dir: Path) -> None:
    inv_id = png.stem
    with Image.open(png) as source:
        image = source.convert("RGB")
    for width, directory in ((IMAGE_WIDTH, image_dir), (THUMB_WIDTH, thumb_dir)):
        height = round(image.height * width / image.width)
        resized = image.resize((width, height), Image.LANCZOS)
        resized.save(directory / f"{inv_id}.chimp.webp", "WEBP", quality=QUALITY)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path, action="append", required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--thumb-dir", type=Path, required=True)
    args = parser.parse_args()
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.thumb_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for plot_dir in args.plot_dir:
        for png in sorted(plot_dir.glob("*.png")):
            export(png, args.image_dir, args.thumb_dir)
            count += 1
    print(f"Exported {count} plots to {args.image_dir} and {args.thumb_dir}")


if __name__ == "__main__":
    main()
