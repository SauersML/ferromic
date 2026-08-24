#!/usr/bin/env python3
"""Assemble the canonical boxed SVbyEye plots for the 93 consensus loci."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import pymupdf
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import white
from reportlab.pdfgen import canvas


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO / ".svbyeye-consensus-93"
OUTPUT_DIR = REPO / "output" / "pdf" / "svbyeye"
PLOT_SCRIPT = REPO / "svbyeye" / "bin" / "plot_chimp_hires.R"

EXAMPLE_IDS = (
    "chr8-7301025-INV-5297356",
    "chr15-23345460-INV-5044410",
)
PLOTS_PER_PAGE = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chromosome_key(row: dict[str, str]) -> tuple[int, int]:
    chromosome = row["chrom"].removeprefix("chr")
    rank = 23 if chromosome == "X" else int(chromosome)
    return rank, int(row["inv_start"])


def load_source_manifest(source_dir: Path) -> list[dict[str, str]]:
    manifest = source_dir / "manifest.tsv"
    provenance_path = source_dir / "provenance.json"
    if not manifest.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"Incomplete SVbyEye source bundle: {source_dir}")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 93 or len({row["inv_id"] for row in rows}) != 93:
        raise RuntimeError("SVbyEye source bundle must contain exactly 93 unique loci")
    if rows != sorted(rows, key=chromosome_key):
        raise RuntimeError("SVbyEye source manifest is not in genomic order")

    provenance = json.loads(provenance_path.read_text())
    expected = {
        "schema_version": 1,
        "locus_count": 93,
        "generator": "svbyeye/bin/plot_chimp_hires.R",
        "generator_sha256": sha256(PLOT_SCRIPT),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise RuntimeError(
                f"SVbyEye source provenance mismatch for {key}: "
                f"expected {value!r}, observed {provenance.get(key)!r}"
            )
    return rows


def load_consensus_properties() -> dict[str, str]:
    properties: dict[str, str] = {}
    with (REPO / "data" / "inv_properties.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            consensus = row["0_single_1_recur_consensus"].strip()
            if consensus in {"0", "1"}:
                properties[row["OrigID"]] = "Single-event" if consensus == "0" else "Recurrent"
    if len(properties) != 93:
        raise RuntimeError(f"Expected 93 consensus properties, found {len(properties)}")
    return properties


def load_orientation_calls() -> dict[str, str]:
    payload = json.loads((REPO / "data" / "chimp_alignment_responses.json").read_text())
    calls = {row["inv_id"]: row["classification"] for row in payload["responses"]}
    if set(calls.values()) - {"direct", "inverted", "na"}:
        raise RuntimeError("Unexpected chimpanzee-orientation call")
    return calls


def source_pages(
    source_dir: Path, rows: list[dict[str, str]]
) -> tuple[list[tuple[dict[str, str], Path]], tuple[float, float]]:
    result = []
    expected_size: tuple[float, float] | None = None
    for row in rows:
        path = source_dir / "plots" / f"{row['inv_id']}.pdf"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        reader = PdfReader(str(path))
        if len(reader.pages) != 1:
            raise RuntimeError(f"Expected one-page source plot: {path}")
        page = reader.pages[0]
        size = (float(page.mediabox.width), float(page.mediabox.height))
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise RuntimeError(
                f"Mixed SVbyEye source page sizes: expected {expected_size}, found {size} in {path}"
            )
        result.append((row, path))
    assert expected_size is not None
    return result, expected_size


def write_consensus_pdf(
    pages: list[tuple[dict[str, str], Path]], source_size: tuple[float, float]
) -> Path:
    output = OUTPUT_DIR / "Supplemental_File_SVbyEye_consensus_93_loci.pdf"
    source_width, source_height = source_size
    writer = PdfWriter()
    for offset in range(0, len(pages), PLOTS_PER_PAGE):
        combined = writer.add_blank_page(
            width=source_width,
            height=source_height * PLOTS_PER_PAGE,
        )
        for position, (_, path) in enumerate(pages[offset : offset + PLOTS_PER_PAGE]):
            plot = PdfReader(str(path)).pages[0]
            y = source_height if position == 0 else 0
            combined.merge_transformed_page(
                plot,
                Transformation().translate(0, y),
                over=True,
            )
    writer.add_metadata(
        {
            "/Title": "SVbyEye alignments for 93 consensus-classified inversions",
            "/Subject": (
                "Canonical panTro6-versus-GRCh38 boxed inversion plots, "
                "two full-width plots per page"
            ),
        }
    )
    with output.open("wb") as handle:
        writer.write(handle)
    return output


def merge_plot(page, source: Path, x: float, y: float, width: float) -> None:
    plot = PdfReader(str(source)).pages[0]
    scale = width / float(plot.mediabox.width)
    page.merge_transformed_page(
        plot,
        Transformation().scale(scale).translate(x, y),
        over=True,
    )


def write_example_figure(by_id: dict[str, Path]) -> Path:
    output = OUTPUT_DIR / "Supplemental_Figure_SVbyEye_orientation_examples.pdf"
    buffer = io.BytesIO()
    background = canvas.Canvas(buffer, pagesize=(1080, 950), pageCompression=1, invariant=1)
    background.setFillColor(white)
    background.rect(0, 0, 1080, 950, fill=1, stroke=0)
    background.save()
    page = PdfReader(io.BytesIO(buffer.getvalue())).pages[0]
    merge_plot(page, by_id[EXAMPLE_IDS[0]], 35, 480, 1010)
    merge_plot(page, by_id[EXAMPLE_IDS[1]], 35, 18, 1010)

    writer = PdfWriter()
    writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": "Examples of inversion alignments to chimpanzee used for polarization",
            "/Subject": "8p23.1 and 15q11.2 SVbyEye alignments",
        }
    )
    with output.open("wb") as handle:
        writer.write(handle)

    document = pymupdf.open(output)
    pixmap = document[0].get_pixmap(matrix=pymupdf.Matrix(2.5, 2.5), alpha=False)
    pixmap.save(OUTPUT_DIR / "Supplemental_Figure_SVbyEye_orientation_examples.png")
    document.close()
    return output


def write_legends() -> None:
    (OUTPUT_DIR / "Supplemental_Figure_SVbyEye_orientation_examples_legend.txt").write_text(
        "Figure S[X]. Examples of inversion alignments to chimpanzee used for polarization. "
        "SVbyEye shows alignments between GRCh38 (top) and panTro6 (bottom) across "
        "(A) the recurrent 8p23.1 inversion and (B) the single-event 15q11.2 inversion. "
        "Green and blue indicate forward and reverse alignments, respectively. Red dashed "
        "boxes indicate the inversion coordinates in GRCh38. The alignment in (A) indicates "
        "that the inversion is ancestral, whereas the alignment in (B) indicates that the "
        "inversion is derived.\n"
    )
    (OUTPUT_DIR / "Supplemental_File_SVbyEye_consensus_93_loci_legend.txt").write_text(
        "SVbyEye alignments between GRCh38 (top) and panTro6 (bottom) across the 93 "
        "consensus-classified inversions. Green and blue indicate forward and reverse "
        "alignments, respectively. Red dashed boxes indicate the inversion coordinates "
        "in GRCh38. Two full-width loci are shown per page, ordered top-to-bottom "
        "and then by genomic position.\n"
    )


def write_index(
    rows: list[dict[str, str]], properties: dict[str, str], calls: dict[str, str]
) -> Path:
    output = OUTPUT_DIR / "svbyeye_locus_index.tsv"
    fields = (
        "page",
        "position_on_page",
        "inversion_id",
        "chromosome",
        "start",
        "end",
        "cytoband",
        "recurrence_class",
        "chimpanzee_orientation_call",
        "source_pdf",
    )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "page": index // PLOTS_PER_PAGE + 1,
                    "position_on_page": "top" if index % PLOTS_PER_PAGE == 0 else "bottom",
                    "inversion_id": row["inv_id"],
                    "chromosome": row["chrom"],
                    "start": row["inv_start"],
                    "end": row["inv_end"],
                    "cytoband": row["label"],
                    "recurrence_class": properties[row["inv_id"]],
                    "chimpanzee_orientation_call": calls[row["inv_id"]],
                    "source_pdf": f"plots/{row['inv_id']}.pdf",
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_source_manifest(args.source_dir)
    properties = load_consensus_properties()
    calls = load_orientation_calls()
    if {row["inv_id"] for row in rows} != set(properties):
        raise RuntimeError("Source bundle does not match the 93 consensus-classified loci")
    pages, page_size = source_pages(args.source_dir, rows)
    by_id = {row["inv_id"]: path for row, path in pages}
    write_example_figure(by_id)
    consensus = write_consensus_pdf(pages, page_size)
    write_legends()
    write_index(rows, properties, calls)
    output_pages = len(PdfReader(str(consensus)).pages)
    output_size = (page_size[0], page_size[1] * PLOTS_PER_PAGE)
    print(
        f"Built 93 canonical boxed SVbyEye plots on {output_pages} pages "
        f"at fixed size {output_size} (two full-width plots per page)"
    )


if __name__ == "__main__":
    main()
