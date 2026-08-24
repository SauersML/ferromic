#!/usr/bin/env python3
"""Verify the exported supplementary workbook without modifying it.

This intentionally uses only the Python standard library. It independently
checks the XLSX container, worksheet order, table titles, header uniqueness,
formula absence, and the fixed revision row inventory after ``generate_tables``
has finished.
"""

from __future__ import annotations

import argparse
import posixpath
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from generate_tables import (
    EXPECTED_SUPPLEMENTARY_DATA_ROWS,
    FINAL_SUPPLEMENTARY_TABLE_ORDER,
)


MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for item in root.findall(f"{{{MAIN}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{MAIN}}}t")))
    return values


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{MAIN}}}t"))
    value = cell.find(f"{{{MAIN}}}v")
    if value is None or value.text is None:
        return ""
    if kind == "s":
        return shared[int(value.text)]
    return value.text


def verify(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Workbook is missing or empty: {path}")

    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"Corrupt XLSX member: {bad_member}")

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            rel.get("Id"): rel.get("Target")
            for rel in relationships.findall(f"{{{PKG_REL}}}Relationship")
        }
        sheets = workbook.find(f"{{{MAIN}}}sheets")
        if sheets is None:
            raise RuntimeError("Workbook has no worksheets.")
        sheet_nodes = list(sheets)
        expected_names = ["Information"] + [
            f"Table S{i}" for i in range(1, len(FINAL_SUPPLEMENTARY_TABLE_ORDER) + 1)
        ]
        observed_names = [node.get("name") for node in sheet_nodes]
        if observed_names != expected_names:
            raise RuntimeError(
                f"Wrong worksheet order: expected {expected_names}, observed {observed_names}."
            )

        shared = _shared_strings(archive)
        for index, (node, expected_title, expected_rows) in enumerate(
            zip(
                sheet_nodes[1:],
                FINAL_SUPPLEMENTARY_TABLE_ORDER,
                EXPECTED_SUPPLEMENTARY_DATA_ROWS,
            ),
            start=1,
        ):
            rel_id = node.get(f"{{{REL}}}id")
            target = targets.get(rel_id)
            if not target:
                raise RuntimeError(f"Table S{index} has no worksheet relationship.")
            member = target.lstrip("/")
            if not member.startswith("xl/"):
                member = posixpath.normpath(posixpath.join("xl", member))
            root = ET.fromstring(archive.read(member))
            rows = root.findall(f".//{{{MAIN}}}sheetData/{{{MAIN}}}row")
            by_number = {int(row.get("r")): row for row in rows}
            observed_data_rows = sum(number >= 3 for number in by_number)
            if observed_data_rows != expected_rows:
                raise RuntimeError(
                    f"Table S{index} has {observed_data_rows:,} data rows; "
                    f"expected {expected_rows:,}."
                )

            title_cells = (
                by_number[1].findall(f"{{{MAIN}}}c") if 1 in by_number else []
            )
            title = _cell_text(title_cells[0], shared) if title_cells else ""
            expected_prefix = f"Table S{index}. {expected_title}."
            if not title.startswith(expected_prefix):
                raise RuntimeError(
                    f"Table S{index} title mismatch: expected prefix {expected_prefix!r}, "
                    f"got {title!r}."
                )

            header_cells = (
                by_number[2].findall(f"{{{MAIN}}}c") if 2 in by_number else []
            )
            headers = [_cell_text(cell, shared).strip() for cell in header_cells]
            if not headers or any(not header for header in headers):
                raise RuntimeError(f"Table S{index} contains a blank header.")
            if len(headers) != len(set(headers)):
                raise RuntimeError(f"Table S{index} contains duplicate printed headers.")
            if root.findall(f".//{{{MAIN}}}f"):
                raise RuntimeError(f"Table S{index} unexpectedly contains formulas.")

    print(
        f"Verified {path}: Information plus {len(FINAL_SUPPLEMENTARY_TABLE_ORDER)} "
        "ordered tables, fixed row inventory, unique headers, and no formulas."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workbook", type=Path)
    args = parser.parse_args()
    try:
        verify(args.workbook)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
