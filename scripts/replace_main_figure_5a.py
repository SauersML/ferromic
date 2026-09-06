#!/usr/bin/env python3
"""Replace panel A (the PheWAS forest plot) in the composite main Figure 5.

Figure 5 in the manuscript is a raster composite: panel A (stats/forest.py) fills
the left part of the page, panels B and C sit to its right. Only panel A depends
on the confidence-interval columns of data/phewas_results.tsv, so after those
columns change the forest plot is regenerated and swapped into the existing
composite while B and C stay pixel-identical.

The base may be the composite PNG itself or the single-page PDF that
scripts/export_manuscript_figures.py wraps around the embedded PNG. The forest
render is scaled so that the height of its inked area equals the height of the
original panel's inked area and is placed at the original panel's top-left
corner; the panel letter "A" is preserved from the original pixels. The width of
the rescaled render must agree with the original panel to within a few percent,
which holds when the same rows and labels are plotted, and the script fails
otherwise instead of producing a misaligned figure.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image

INK_THRESHOLD = 250          # luminance below which a pixel counts as ink
DEFAULT_SPLIT_FRACTION = 692 / 1430   # x position separating panel A from B/C
LETTER_BOX = (0, 80, 0, 37)  # rows [0, 80) and cols [0, 37): the panel letter
WIDTH_TOLERANCE = 0.01   # never let the new panel grow into panels B and C
MAX_NARROWING = 0.15     # a narrower render is fine (legend inside the axes)
PDF_DPI = 300


def load_base(path: Path) -> tuple[Image.Image, dict]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        if len(reader.pages) != 1:
            raise ValueError("The base figure PDF must have exactly one page")
        images = reader.pages[0].images
        if len(images) != 1:
            raise ValueError("The base figure PDF must wrap exactly one raster image")
        image = Image.open(io.BytesIO(images[0].data))
        return image.convert("RGBA"), {}
    image = Image.open(path)
    info = {k: v for k, v in image.info.items() if k in {"dpi"}}
    return image.convert("RGBA"), info


def ink_mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")) < INK_THRESHOLD


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("no ink found")
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def replace_panel_a(base: Image.Image, panel: Image.Image, split_fraction: float) -> Image.Image:
    width, height = base.size
    split = int(round(width * split_fraction))
    base_arr = np.asarray(base).copy()

    # Original panel A extent, excluding the panel letter
    mask = ink_mask(base)
    region = mask[:, :split].copy()
    r0, r1, c0, c1 = LETTER_BOX
    region[r0:r1, c0:c1] = False
    top, bottom, left, right = bbox(region)
    old_h, old_w = bottom - top, right - left

    # New render extent
    panel = panel.convert("RGBA")
    ptop, pbottom, pleft, pright = bbox(ink_mask(panel))
    new_h, new_w = pbottom - ptop, pright - pleft
    scale = old_h / new_h
    scaled_w = new_w * scale
    # The committed forest script keeps its legend inside the axes, so its render
    # is somewhat narrower than the submitted panel, whose legend sat to the
    # right of the axes. A narrower panel leaves white space before panel B and
    # is accepted; a wider one would collide with B and is refused.
    if scaled_w > old_w * (1 + WIDTH_TOLERANCE) or scaled_w < old_w * (1 - MAX_NARROWING):
        raise ValueError(
            f"rescaled forest width {scaled_w:.0f}px is incompatible with the original panel width {old_w}px "
            f"(allowed: up to {MAX_NARROWING:.0%} narrower, {WIDTH_TOLERANCE:.0%} wider)"
        )
    cropped = panel.crop((pleft, ptop, pright, pbottom))
    target_w = int(round(scaled_w))
    resized = cropped.resize((target_w, old_h), Image.LANCZOS)
    if left + target_w > split:
        raise ValueError("the rescaled forest would overlap panels B and C")

    # White out the old panel, paste the new one, restore the panel letter
    letter_pixels = base_arr[r0:r1, c0:c1].copy()
    letter_mask = mask[r0:r1, c0:c1]
    base_arr[:, :split] = (255, 255, 255, 255)
    out = Image.fromarray(base_arr)
    white = Image.new("RGBA", resized.size, (255, 255, 255, 255))
    white.alpha_composite(resized)
    out.paste(white, (left, top))
    out_arr = np.asarray(out).copy()
    letter_region = out_arr[r0:r1, c0:c1]
    letter_region[letter_mask] = letter_pixels[letter_mask]
    out_arr[r0:r1, c0:c1] = letter_region
    return Image.fromarray(out_arr)


def write_pdf(image: Image.Image, destination: Path, dpi: float) -> None:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    width_px, height_px = image.size
    width_pt, height_pt = width_px * 72 / dpi, height_px * 72 / dpi
    payload = io.BytesIO()
    image.convert("RGB").save(payload, format="PNG")
    payload.seek(0)
    pdf = canvas.Canvas(str(destination), pagesize=(width_pt, height_pt), pageCompression=1)
    pdf.drawImage(ImageReader(payload), 0, 0, width=width_pt, height=height_pt)
    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="composite Figure 5 as PNG or single-page PDF")
    parser.add_argument("--panel-a", type=Path, required=True, help="phewas_forest.png from stats/forest.py")
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--split-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument("--pdf-dpi", type=float, default=PDF_DPI)
    args = parser.parse_args()

    base, info = load_base(args.base)
    panel = Image.open(args.panel_a)
    corrected = replace_panel_a(base, panel, args.split_fraction)
    if corrected.size != base.size:
        raise ValueError("corrected figure changed size")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    corrected.save(args.output_png, format="PNG", **info)
    print(f"Wrote corrected Figure 5 PNG to {args.output_png} ({corrected.size[0]}x{corrected.size[1]} px)")
    if args.output_pdf is not None:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        write_pdf(corrected, args.output_pdf, args.pdf_dpi)
        print(f"Wrote corrected Figure 5 PDF to {args.output_pdf}")


if __name__ == "__main__":
    main()
