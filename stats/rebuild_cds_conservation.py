#!/usr/bin/env python3
"""Rebuild CDS-conservation tables from a pinned PHYLIP archive."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stats import cds_differences, per_gene_cds_differences_jackknife  # noqa: E402


REQUIRED_OUTPUTS = (
    "cds_identical_proportions.tsv",
    "gene_inversion_direct_inverted.tsv",
    "region_identical_proportions.tsv",
    "skipped_details.tsv",
)


def extract_alignments(archive_path: Path, destination: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.endswith((".phy", ".phy.gz")):
                continue
            filename = Path(member.filename).name
            if filename != member.filename:
                raise ValueError(f"Unexpected nested archive member: {member.filename}")
            output_name = filename.removesuffix(".gz")
            output_path = destination / output_name
            with archive.open(member) as source:
                if filename.endswith(".gz"):
                    with gzip.open(source) as decompressed, output_path.open("wb") as output:
                        shutil.copyfileobj(decompressed, output)
                else:
                    with output_path.open("wb") as output:
                        shutil.copyfileobj(source, output)
            count += 1
    if count == 0:
        raise ValueError(f"No PHYLIP alignments found in {archive_path}")
    return count


def rebuild(archive_path: Path, inv_properties: Path, output_dir: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not inv_properties.is_file():
        raise FileNotFoundError(inv_properties)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ferromic-cds-") as temporary_name:
        temporary = Path(temporary_name)
        count = extract_alignments(archive_path, temporary)
        shutil.copy2(inv_properties, temporary / "inv_properties.tsv")
        previous = Path.cwd()
        try:
            os.chdir(temporary)
            cds_differences.main()
            per_gene_cds_differences_jackknife.main()
        finally:
            os.chdir(previous)

        missing = [name for name in REQUIRED_OUTPUTS if not (temporary / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "CDS pipeline did not generate required outputs: " + ", ".join(missing)
            )
        for name in REQUIRED_OUTPUTS:
            shutil.copy2(temporary / name, output_dir / name)
        print(f"Rebuilt {len(REQUIRED_OUTPUTS)} CDS tables from {count} PHYLIP alignments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inv-properties", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rebuild(args.archive, args.inv_properties, args.output_dir)


if __name__ == "__main__":
    main()
