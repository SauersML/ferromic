"""Build publication-quality SVbyEye supplementary PDFs.

The four redrawn examples are retained as vector graphics. For every other
locus, the complete native-resolution chimpanzee-versus-GRCh38 review image is
embedded without resizing or JPEG recompression.
"""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
from pathlib import Path

import pymupdf
from PIL import Image
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
IMAGE_DIR = REPO / "web" / "figures-site" / "public" / "inversions" / "img"
VECTOR_SOURCE_DIR = REPO / ".svbyeye-preview"
OUTPUT_DIR = REPO / "output" / "pdf" / "svbyeye"
PER_LOCUS_DIR = OUTPUT_DIR / "per_locus"
VECTOR_DIR = OUTPUT_DIR / "vector_examples"

PAGE_WIDTH = 1080.0
PAGE_HEIGHT = 540.0
MARGIN = 32.0
PLOT_TOP = 468.0
PLOT_BOTTOM = 44.0
INDEX_ROWS_PER_PAGE = 31

CALL_LABELS = {
    "direct": "GRCh38 orientation ancestral",
    "inverted": "GRCh38 orientation derived",
    "na": "Not callable",
}

VECTOR_SOURCES = {
    "chr8-7301025-INV-5297356": "chr8_8p23.1-v6.pdf",
    "chr12-46897663-INV-16289": "chr12_12q13.11-v6.pdf",
    "chr15-23345460-INV-5044410": "chr15_15q11.2-v6.pdf",
    "chr17-45585160-INV-706887": "chr17_17q21.31-v6.pdf",
}

EXAMPLE_IDS = (
    "chr8-7301025-INV-5297356",
    "chr15-23345460-INV-5044410",
)


def chromosome_key(record: dict) -> tuple[int, int]:
    chrom = record["chrom"].removeprefix("chr")
    rank = {"X": 23, "Y": 24}.get(chrom, int(chrom) if chrom.isdigit() else 99)
    return rank, int(record["start"])


def load_records() -> list[dict]:
    payload = json.loads((DATA / "chimp_alignment_responses.json").read_text())
    records = sorted(payload["responses"], key=chromosome_key)
    if len(records) != 292:
        raise RuntimeError(f"Expected 292 reviewed loci, found {len(records)}")

    counts = {call: 0 for call in CALL_LABELS}
    for record in records:
        call = record["classification"]
        if call not in counts:
            raise RuntimeError(f"Unexpected orientation call: {call}")
        counts[call] += 1
        image_path = IMAGE_DIR / record["image_file"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    expected = {"direct": 121, "inverted": 12, "na": 159}
    if counts != expected:
        raise RuntimeError(f"Unexpected orientation counts: {counts}; expected {expected}")
    return records


def load_properties() -> dict[str, dict[str, str]]:
    properties: dict[str, dict[str, str]] = {}
    with (DATA / "inv_properties.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            consensus = row["0_single_1_recur_consensus"].strip()
            recurrence = {
                "0": "Single-event",
                "1": "Recurrent",
            }.get(consensus, "No consensus call")
            af = row["Inverted_AF"].strip()
            properties[row["OrigID"]] = {
                "recurrence": recurrence,
                "inverted_af": af if af else "NA",
            }
    return properties


def make_base_page(record: dict, metadata: dict, source_type: str) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1, invariant=1)
    pdf.setTitle(record["inv_id"])
    pdf.setFillColor(HexColor("#15324A"))
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(MARGIN, 510, record["inv_id"])
    pdf.setFillColor(HexColor("#405363"))
    pdf.setFont("Helvetica", 10)
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 510, record["region"])

    call = CALL_LABELS[record["classification"]]
    details = (
        f"{metadata['recurrence']}   |   inverted AF: {metadata['inverted_af']}"
        f"   |   chimpanzee call: {call}"
    )
    pdf.setFont("Helvetica", 10.5)
    pdf.drawString(MARGIN, 491, details)
    pdf.setStrokeColor(HexColor("#D8E0E6"))
    pdf.setLineWidth(0.7)
    pdf.line(MARGIN, 480, PAGE_WIDTH - MARGIN, 480)
    pdf.setFillColor(HexColor("#657580"))
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(
        MARGIN,
        20,
        "Chimpanzee (panTro6) versus GRCh38 orientation-review image.",
    )
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 20, source_type)
    pdf.save()
    return buffer.getvalue()


def add_native_image(page, image_path: Path) -> None:
    with Image.open(image_path) as image:
        pixel_width, pixel_height = image.size
    max_width = PAGE_WIDTH - 2 * MARGIN
    max_height = PLOT_TOP - PLOT_BOTTOM
    scale = min(max_width / pixel_width, max_height / pixel_height)
    width = pixel_width * scale
    height = pixel_height * scale
    x = (PAGE_WIDTH - width) / 2
    y = PLOT_BOTTOM + (max_height - height) / 2

    overlay = io.BytesIO()
    pdf = canvas.Canvas(overlay, pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1, invariant=1)
    pdf.drawImage(
        ImageReader(str(image_path)),
        x,
        y,
        width=width,
        height=height,
        preserveAspectRatio=True,
        mask="auto",
    )
    pdf.save()
    page.merge_page(PdfReader(io.BytesIO(overlay.getvalue())).pages[0])


def add_vector_plot(page, source_path: Path) -> None:
    source = PdfReader(str(source_path)).pages[0]
    source_width = float(source.mediabox.width)
    source_height = float(source.mediabox.height)
    max_width = PAGE_WIDTH - 2 * MARGIN
    max_height = PLOT_TOP - PLOT_BOTTOM
    scale = min(max_width / source_width, max_height / source_height)
    x = (PAGE_WIDTH - source_width * scale) / 2
    y = PLOT_BOTTOM + (max_height - source_height * scale) / 2
    page.merge_transformed_page(
        source,
        Transformation().scale(scale).translate(x, y),
        over=True,
    )


def write_locus_pdf(record: dict, metadata: dict) -> str:
    vector_name = VECTOR_SOURCES.get(record["inv_id"])
    source_type = "True vector source" if vector_name else "Native 1400-pixel review image"
    page = PdfReader(io.BytesIO(make_base_page(record, metadata, source_type))).pages[0]
    if vector_name:
        add_vector_plot(page, VECTOR_SOURCE_DIR / vector_name)
    else:
        add_native_image(page, IMAGE_DIR / record["image_file"])

    output_path = PER_LOCUS_DIR / f"{record['inv_id']}.pdf"
    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({
        "/Title": record["inv_id"],
        "/Subject": "SVbyEye chimpanzee-versus-GRCh38 orientation review",
    })
    with output_path.open("wb") as handle:
        writer.write(handle)
    return source_type


def write_vector_copies() -> None:
    for filename in VECTOR_SOURCES.values():
        source = VECTOR_SOURCE_DIR / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, VECTOR_DIR / filename)


def merge_source_on_page(page, source_path: Path, x: float, y: float, width: float) -> None:
    source = PdfReader(str(source_path)).pages[0]
    source_width = float(source.mediabox.width)
    scale = width / source_width
    page.merge_transformed_page(
        source,
        Transformation().scale(scale).translate(x, y),
        over=True,
    )


def write_example_figure() -> Path:
    output_path = OUTPUT_DIR / "Supplemental_Figure_SVbyEye_orientation_examples.pdf"
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, 950), pageCompression=1, invariant=1)
    pdf.setFillColor(white)
    pdf.rect(0, 0, PAGE_WIDTH, 950, fill=1, stroke=0)
    pdf.save()
    page = PdfReader(io.BytesIO(buffer.getvalue())).pages[0]
    merge_source_on_page(page, VECTOR_SOURCE_DIR / VECTOR_SOURCES[EXAMPLE_IDS[0]], 35, 480, 1010)
    merge_source_on_page(page, VECTOR_SOURCE_DIR / VECTOR_SOURCES[EXAMPLE_IDS[1]], 35, 18, 1010)

    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata({
        "/Title": "Representative SVbyEye orientation-polarization alignments",
        "/Subject": "8p23.1 direct and 15q11.2 inverted examples",
    })
    with output_path.open("wb") as handle:
        writer.write(handle)
    document = pymupdf.open(output_path)
    pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), alpha=False)
    pixmap.save(OUTPUT_DIR / "Supplemental_Figure_SVbyEye_orientation_examples.png")
    document.close()
    return output_path


def write_legend() -> Path:
    path = OUTPUT_DIR / "Supplemental_Figure_SVbyEye_orientation_examples_legend.txt"
    path.write_text(
        "Figure S[X]. Representative chimpanzee-versus-GRCh38 alignments used for "
        "orientation polarization. Dashed red boxes mark the human inversion boundaries. "
        "(A) At 8p23.1, the chimpanzee alignment has the same orientation inside the "
        "interval and in the flanks, supporting the GRCh38 orientation as ancestral. "
        "(B) At 15q11.2, alignment orientation reverses within the inversion while the "
        "flanks retain the same orientation, supporting the GRCh38 orientation as "
        "derived. Ribbon color denotes alignment strand.\n"
    )
    return path


def write_audit_legend() -> Path:
    path = OUTPUT_DIR / "Supplemental_File_SVbyEye_all_292_loci_legend.txt"
    path.write_text(
        "Supplemental File S[X]. Chimpanzee-versus-GRCh38 alignments used for "
        "orientation polarization at all 292 reviewed inversion loci. Loci are ordered "
        "by genomic position. Each page reports the inversion coordinates, recurrence "
        "classification, inverted allele frequency, and manual orientation call. Calls "
        "were based on alignment orientation within the inversion relative to both "
        "flanks: ancestral indicates that the GRCh38 orientation matches chimpanzee, "
        "derived indicates that it is reversed relative to chimpanzee, and not callable "
        "indicates insufficient or ambiguous alignment. A uniformly reversed chimpanzee "
        "contig across both the locus and its flanks was not interpreted as evidence that "
        "the human orientation is derived. Contents pages and the accompanying TSV provide "
        "the complete locus-to-page audit trail. Four redrawn examples are retained as "
        "vector graphics; all other pages retain the complete native resolution of the "
        "original review image.\n"
    )
    return path


def draw_cover(pdf: canvas.Canvas, records: list[dict]) -> None:
    pdf.setFillColor(HexColor("#15324A"))
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 30)
    pdf.drawString(62, 416, "SVbyEye orientation-polarization audit")
    pdf.setFont("Helvetica", 16)
    pdf.drawString(62, 382, "Chimpanzee (panTro6) versus GRCh38 alignments for 292 inversion loci")

    counts = {call: sum(r["classification"] == call for r in records) for call in CALL_LABELS}
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawString(62, 304, f"{counts['direct']} ancestral   |   {counts['inverted']} derived   |   {counts['na']} not callable")
    pdf.setFont("Helvetica", 12)
    text = pdf.beginText(62, 254)
    text.setLeading(19)
    text.textLine("Each locus page records the inversion coordinates, recurrence classification,")
    text.textLine("inverted allele frequency, and manual chimpanzee-orientation call.")
    text.textLine("The contents pages provide the complete locus-to-page audit trail.")
    text.textLine("Four redrawn examples are vector graphics; all remaining plots retain the")
    text.textLine("complete native resolution of the original review image.")
    pdf.drawText(text)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(62, 48, "Generated from the reviewed callset and source plots in the ferromic analysis repository.")


def draw_index_page(pdf: canvas.Canvas, rows: list[dict], index_number: int) -> None:
    pdf.setFillColor(HexColor("#15324A"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(MARGIN, 508, f"Contents ({index_number})")
    headers = (("Locus", 32), ("Coordinates", 310), ("Recurrence", 548), ("Chimp call", 700), ("Page", 990))
    pdf.setFont("Helvetica-Bold", 9)
    for label, x in headers:
        pdf.drawString(x, 482, label)
    pdf.setStrokeColor(HexColor("#AAB7C0"))
    pdf.line(MARGIN, 475, PAGE_WIDTH - MARGIN, 475)

    y = 458
    pdf.setFont("Helvetica", 8.1)
    for row in rows:
        if row["classification"] == "direct":
            call = "Ancestral"
        elif row["classification"] == "inverted":
            call = "Derived"
        else:
            call = "Not callable"
        pdf.setFillColor(HexColor("#263842"))
        pdf.drawString(32, y, row["inv_id"])
        pdf.drawString(310, y, row["region"])
        pdf.drawString(548, y, row["recurrence"])
        pdf.drawString(700, y, call)
        pdf.drawRightString(1038, y, str(row["combined_page"]))
        y -= 14.1


def write_front_matter(records: list[dict], index_rows: list[dict]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1, invariant=1)
    draw_cover(pdf, records)
    pdf.showPage()
    for page_number in range(math.ceil(len(index_rows) / INDEX_ROWS_PER_PAGE)):
        start = page_number * INDEX_ROWS_PER_PAGE
        draw_index_page(pdf, index_rows[start:start + INDEX_ROWS_PER_PAGE], page_number + 1)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def write_index(index_rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "svbyeye_locus_index.tsv"
    fields = (
        "inversion_id",
        "coordinates",
        "size_bp",
        "recurrence_class",
        "inverted_allele_frequency",
        "chimpanzee_orientation_call",
        "combined_pdf_page",
        "source_type",
        "source_file",
        "exclusion_reason",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in index_rows:
            writer.writerow({
                "inversion_id": row["inv_id"],
                "coordinates": row["region"],
                "size_bp": row["size_bp"],
                "recurrence_class": row["recurrence"],
                "inverted_allele_frequency": row["inverted_af"],
                "chimpanzee_orientation_call": row["classification"],
                "combined_pdf_page": row["combined_page"],
                "source_type": row["source_type"],
                "source_file": row["source_file"],
                "exclusion_reason": (
                    "Not callable; specific reason was not recorded in the manual-review dataset"
                    if row["classification"] == "na" else ""
                ),
            })
    return path


def write_audit_pdf(records: list[dict], index_rows: list[dict]) -> Path:
    output_path = OUTPUT_DIR / "Supplemental_File_SVbyEye_all_292_loci.pdf"
    writer = PdfWriter()
    for page in PdfReader(io.BytesIO(write_front_matter(records, index_rows))).pages:
        writer.add_page(page)
    for record in records:
        locus_path = PER_LOCUS_DIR / f"{record['inv_id']}.pdf"
        writer.add_page(PdfReader(str(locus_path)).pages[0])
    writer.add_metadata({
        "/Title": "SVbyEye orientation-polarization audit for 292 inversion loci",
        "/Subject": "Complete chimpanzee-versus-GRCh38 alignment audit",
    })
    with output_path.open("wb") as handle:
        writer.write(handle)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_LOCUS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    properties = load_properties()
    missing_properties = [record["inv_id"] for record in records if record["inv_id"] not in properties]
    if missing_properties:
        raise RuntimeError(f"Missing inversion properties: {missing_properties[:5]}")

    write_vector_copies()
    index_page_count = math.ceil(len(records) / INDEX_ROWS_PER_PAGE)
    first_locus_page = 2 + index_page_count
    index_rows = []
    for offset, record in enumerate(records):
        metadata = properties[record["inv_id"]]
        source_type = write_locus_pdf(record, metadata)
        if (offset + 1) % 25 == 0 or offset + 1 == len(records):
            print(f"Built {offset + 1}/{len(records)} per-locus PDFs")
        index_rows.append({
            **record,
            **metadata,
            "combined_page": first_locus_page + offset,
            "source_type": source_type,
            "source_file": (
                VECTOR_SOURCES[record["inv_id"]]
                if record["inv_id"] in VECTOR_SOURCES else record["image_file"]
            ),
        })

    example_path = write_example_figure()
    legend_path = write_legend()
    audit_legend_path = write_audit_legend()
    index_path = write_index(index_rows)
    audit_path = write_audit_pdf(records, index_rows)
    print(f"Example figure: {example_path}")
    print(f"Legend: {legend_path}")
    print(f"Audit legend: {audit_legend_path}")
    print(f"Index: {index_path}")
    print(f"Audit supplement: {audit_path}")


if __name__ == "__main__":
    main()
