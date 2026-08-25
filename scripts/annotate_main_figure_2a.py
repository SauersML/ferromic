#!/usr/bin/env python3
"""Add the two reported orientation-comparison p-values to Main Figure 2A."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


def draw_centered_scientific_pvalue(
    pdf: canvas.Canvas,
    center_x: float,
    baseline_y: float,
    coefficient: str,
) -> None:
    prefix = f"p = {coefficient} × 10"
    suffix = "-5"
    main_size = 5.6
    exponent_size = 4.2
    width = stringWidth(prefix, "Helvetica", main_size)
    width += stringWidth(suffix, "Helvetica", exponent_size)
    text = pdf.beginText(center_x - width / 2, baseline_y)
    text.setFont("Helvetica", main_size)
    text.textOut(prefix)
    text.setRise(2.1)
    text.setFont("Helvetica", exponent_size)
    text.textOut(suffix)
    pdf.drawText(text)


def draw_bracket(
    pdf: canvas.Canvas,
    left: float,
    right: float,
    y: float,
    label: str,
    *,
    scientific_coefficient: str | None = None,
    label_below: bool = False,
) -> None:
    pdf.setStrokeColorRGB(0.12, 0.12, 0.12)
    pdf.setLineWidth(0.65)
    pdf.line(left, y, right, y)
    tip = 3.0
    pdf.line(left, y, left, y - tip)
    pdf.line(right, y, right, y - tip)
    center = (left + right) / 2
    baseline = y - 8.1 if label_below else y + 1.7
    pdf.setFillColorRGB(0.08, 0.08, 0.08)
    if scientific_coefficient is not None:
        draw_centered_scientific_pvalue(
            pdf, center, baseline, scientific_coefficient
        )
    else:
        pdf.setFont("Helvetica", 5.6)
        pdf.drawCentredString(center, baseline, label)


def annotate(base_path: Path, output_path: Path) -> None:
    reader = PdfReader(base_path)
    if len(reader.pages) != 1:
        raise ValueError(f"Main Figure 2 must be one page: {base_path}")
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)

    overlay_bytes = io.BytesIO()
    overlay_pdf = canvas.Canvas(overlay_bytes, pagesize=(width, height))
    bracket_y = height * 0.912
    draw_bracket(
        overlay_pdf,
        width * 0.154,
        width * 0.253,
        bracket_y,
        "",
        scientific_coefficient="7.4",
    )
    draw_bracket(
        overlay_pdf,
        width * 0.412,
        width * 0.517,
        bracket_y,
        "p = 0.372",
        label_below=True,
    )
    overlay_pdf.showPage()
    overlay_pdf.save()
    overlay_bytes.seek(0)
    overlay = PdfReader(overlay_bytes).pages[0]
    page.merge_page(overlay, over=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(page)
    with output_path.open("wb") as handle:
        writer.write(handle)
    if len(PdfReader(output_path).pages) != 1:
        raise ValueError(f"Annotated Figure 2 is not one page: {output_path}")
    print(f"Wrote annotated Main Figure 2 to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotate(args.base, args.output)


if __name__ == "__main__":
    main()
