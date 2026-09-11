#!/usr/bin/env python3
"""Correct the two raw-p axis titles in the pinned Figure 5 artwork.

stats/volcano.py plots -log10(P_LRT_Overall); its BH cutoff only determines
significance. stats/PGS_control_plot.py plots -log10(P_Value) and
-log10(P_Value_NoCustomControls), not BH_FDR_Q. PGS covariate adjustment is
distinct from multiple-testing correction.

Remove only the erroneous "BH " glyphs from each raster title, retaining the
original antialiasing, font, minus sign, subscript and upright p. Recenter the
shortened title inside its original box. Pixel hashes reject changed source
labels, and a pixel comparison ensures all other artwork stays identical.
The separate pinned panel C preserves the letter's original crop and size.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from replace_main_figure_5a import write_pdf

# (box, horizontal cut after rotating clockwise, expected RGB pixel hash)
MAIN_LABELS = (
    ((669, 198, 694, 307), (58, 90),
     "46c31a3e72ea4abcfead6bfacefb4e1d2a3259877884354d055f49aa456fa2e9"),
    ((752, 676, 777, 778), (52, 84),
     "2425fbb693b53eee4212ef6be26b9bcefab9ce299efeaac1f11f88d5989e8792"),
)
LETTER_LABELS = (
    ((12, 125, 38, 225), (47, 72),
     "df9cbb4a916bd7535d671668bb2a3e67a11fcda9f43d395ebb2a3a07b1bb101e"),
)


def correct_labels(source: Image.Image, labels: tuple) -> Image.Image:
    source = source.convert("RGB")
    result = source.copy()
    permitted = np.zeros((source.height, source.width), dtype=bool)
    for box, (start, end), expected_hash in labels:
        original = source.crop(box)
        if hashlib.sha256(original.tobytes()).hexdigest() != expected_hash:
            raise ValueError(f"Axis title at {box} does not match the pinned artwork")
        horizontal = original.transpose(Image.Transpose.ROTATE_270)
        shortened = Image.new("RGB", (horizontal.width - (end - start), horizontal.height), "white")
        shortened.paste(horizontal.crop((0, 0, start, horizontal.height)), (0, 0))
        shortened.paste(horizontal.crop((end, 0, horizontal.width, horizontal.height)), (start, 0))
        centered = Image.new("RGB", horizontal.size, "white")
        centered.paste(shortened, ((end - start) // 2, 0))
        result.paste(centered.transpose(Image.Transpose.ROTATE_90), box)
        x0, y0, x1, y1 = box
        permitted[y0:y1, x0:x1] = True
    changed = np.any(np.asarray(source) != np.asarray(result), axis=2)
    if not changed.any() or np.any(changed & ~permitted):
        raise ValueError("Expected title-only pixel changes")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--letter-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, size, labels, name in (
        (args.base, (1430, 980), MAIN_LABELS, "Figure_5"),
        (args.letter_base, (559, 381), LETTER_LABELS, "Figure_5C"),
    ):
        with Image.open(path) as source:
            if source.size != size:
                raise ValueError(f"Unexpected source dimensions: {path}: {source.size}")
            corrected = correct_labels(source, labels)
        corrected.save(args.output_dir / f"{name}.png")
        write_pdf(corrected, args.output_dir / f"{name}.pdf", 300)
        print(f"Verified title-only correction: {name}, {size[0]} x {size[1]}")


if __name__ == "__main__":
    main()
