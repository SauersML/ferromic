#!/usr/bin/env python3
"""Replace panel B in the existing four-panel main Figure 3 PDF."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas


def white_page(width: float, height: float):
    payload = io.BytesIO()
    pdf = canvas.Canvas(payload, pagesize=(width, height), pageCompression=1)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.showPage()
    pdf.save()
    payload.seek(0)
    return PdfReader(payload).pages[0]


def replace_panel(base_path: Path, panel_path: Path, output_path: Path) -> None:
    base_reader = PdfReader(base_path)
    panel_reader = PdfReader(panel_path)
    if len(base_reader.pages) != 1 or len(panel_reader.pages) != 1:
        raise ValueError("The base figure and replacement panel must each be one page")

    base = base_reader.pages[0]
    panel = panel_reader.pages[0]
    page_width = float(base.mediabox.width)
    page_height = float(base.mediabox.height)

    # The original Figure 3 is a regular 2 x 2 layout. Keep A, C, and D exactly
    # as supplied by the manuscript, and replace the complete top-right quadrant.
    region_x = page_width * 0.475
    region_y = page_height * 0.51
    region_width = page_width - region_x
    region_height = page_height - region_y

    replacement = white_page(region_width, region_height)
    panel_width = float(panel.mediabox.width)
    panel_height = float(panel.mediabox.height)
    scale = min(region_width / panel_width, region_height / panel_height)
    x_offset = (region_width - panel_width * scale) / 2
    y_offset = (region_height - panel_height * scale) / 2
    replacement.merge_transformed_page(
        panel,
        Transformation().scale(scale).translate(x_offset, y_offset),
        over=True,
    )
    base.merge_transformed_page(
        replacement,
        Transformation().translate(region_x, region_y),
        over=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(base)
    with output_path.open("wb") as handle:
        writer.write(handle)

    verified = PdfReader(output_path)
    if len(verified.pages) != 1:
        raise ValueError(f"Corrected Figure 3 is not one page: {output_path}")
    print(f"Wrote corrected main Figure 3 to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replace_panel(args.base, args.panel_b, args.output)


if __name__ == "__main__":
    main()
