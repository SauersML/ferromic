#!/usr/bin/env python3
"""Export every numbered main and supplementary figure as a standalone PDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

from PIL import Image
from pypdf import PdfReader
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAWING = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
OFFICE_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PACKAGE_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PDF_DPI = 300


@dataclass(frozen=True)
class FigureSource:
    label: str
    document: Path
    media_target: str
    payload: bytes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_images(document_path: Path) -> list[tuple[str, bytes]]:
    with ZipFile(document_path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relationships = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        targets = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships.findall(f"{PACKAGE_REL}Relationship")
        }

        body = document.find(f"{WORD}body")
        if body is None:
            raise ValueError(f"DOCX has no document body: {document_path}")

        ordered_targets: list[str] = []
        for blip in body.iter(f"{DRAWING}blip"):
            relationship_id = blip.attrib.get(f"{OFFICE_REL}embed")
            target = targets.get(relationship_id or "")
            if target and target.startswith("media/"):
                ordered_targets.append(target)

        return [
            (target, archive.read(f"word/{target}")) for target in ordered_targets
        ]


def numbered_sources(
    document_path: Path, *, prefix: str, expected_count: int
) -> list[FigureSource]:
    images = document_images(document_path)
    if len(images) < expected_count:
        raise ValueError(
            f"Expected at least {expected_count} body images in {document_path}, "
            f"found {len(images)}"
        )

    selected = images[:expected_count]
    labels = [f"{prefix}{number}" for number in range(1, expected_count + 1)]
    sources = [
        FigureSource(label, document_path, target, payload)
        for label, (target, payload) in zip(labels, selected, strict=True)
    ]
    if len({source.media_target for source in sources}) != expected_count:
        raise ValueError(f"Numbered figures reuse embedded media in {document_path}")
    return sources


def write_pdf(source: FigureSource, destination: Path) -> tuple[int, int]:
    suffix = Path(source.media_target).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported embedded image type for Figure {source.label}: {suffix}"
        )

    with Image.open(io.BytesIO(source.payload)) as image:
        image.verify()
    with Image.open(io.BytesIO(source.payload)) as image:
        width_px, height_px = image.size

    width_points = width_px * 72 / PDF_DPI
    height_points = height_px * 72 / PDF_DPI
    pdf = canvas.Canvas(
        str(destination),
        pagesize=(width_points, height_points),
        pageCompression=1,
    )
    pdf.drawImage(
        ImageReader(io.BytesIO(source.payload)),
        0,
        0,
        width=width_points,
        height=height_points,
        preserveAspectRatio=True,
        anchor="c",
        mask="auto",
    )
    pdf.showPage()
    pdf.save()

    pages = PdfReader(destination).pages
    if len(pages) != 1:
        raise ValueError(f"Figure PDF is not one page: {destination}")
    return width_px, height_px


def export(
    main_document: Path,
    supplementary_document: Path,
    output_dir: Path,
    supplementary_count: int,
    expected_main_figure_3_sha256: str,
) -> None:
    sources = numbered_sources(main_document, prefix="", expected_count=4)
    observed_figure_3_sha256 = hashlib.sha256(sources[2].payload).hexdigest()
    if observed_figure_3_sha256 != expected_main_figure_3_sha256:
        raise ValueError(
            "Main Figure 3 does not match the revised CDS figure: "
            f"expected {expected_main_figure_3_sha256}, "
            f"observed {observed_figure_3_sha256}"
        )
    sources.extend(
        numbered_sources(
            supplementary_document,
            prefix="S",
            expected_count=supplementary_count,
        )
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest_path = output_dir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "figure",
                "pdf",
                "source_document",
                "source_document_sha256",
                "source_media",
                "width_px",
                "height_px",
            )
        )
        document_hashes = {
            main_document: sha256(main_document),
            supplementary_document: sha256(supplementary_document),
        }
        for source in sources:
            pdf_path = output_dir / f"Figure_{source.label}.pdf"
            width_px, height_px = write_pdf(source, pdf_path)
            writer.writerow(
                (
                    source.label,
                    pdf_path.name,
                    source.document.name,
                    document_hashes[source.document],
                    source.media_target,
                    width_px,
                    height_px,
                )
            )

    expected_names = {
        *(f"Figure_{number}.pdf" for number in range(1, 5)),
        *(f"Figure_S{number}.pdf" for number in range(1, supplementary_count + 1)),
    }
    actual_names = {path.name for path in output_dir.glob("*.pdf")}
    if actual_names != expected_names:
        raise ValueError(
            f"Figure PDF set mismatch: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    print(
        f"Exported {len(actual_names)} standalone figure PDFs "
        f"(4 main, {supplementary_count} supplementary) to {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supplementary-count", type=int, default=21)
    parser.add_argument("--expected-main-figure-3-sha256", required=True)
    args = parser.parse_args()
    export(
        args.main,
        args.supplement,
        args.output_dir,
        args.supplementary_count,
        args.expected_main_figure_3_sha256,
    )


if __name__ == "__main__":
    main()
