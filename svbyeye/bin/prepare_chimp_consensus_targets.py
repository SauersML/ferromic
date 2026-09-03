#!/usr/bin/env python3
"""Prepare the 93 consensus inversion windows for one panTro6 alignment."""

from __future__ import annotations

import argparse
import csv
import gzip
import subprocess
from pathlib import Path


FIELDS = (
    "inv_id",
    "chrom",
    "inv_start",
    "inv_end",
    "window_start",
    "window_end",
    "recurrence",
    "label",
)


def chromosome_rank(chromosome: str) -> int:
    return 23 if chromosome == "chrX" else int(chromosome.removeprefix("chr"))


def read_consensus_loci(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["0_single_1_recur_consensus"].strip() in {"0", "1"}
        ]
    rows.sort(key=lambda row: (chromosome_rank(row["Chromosome"]), int(row["Start"])))
    if len(rows) != 93:
        raise RuntimeError(f"Expected 93 consensus loci, found {len(rows)}")
    return rows


def read_reference_lengths(path: Path) -> dict[str, int]:
    index = Path(f"{path}.fai")
    if not index.is_file():
        raise FileNotFoundError(index)
    with index.open() as handle:
        return {line.split("\t", 2)[0]: int(line.split("\t", 2)[1]) for line in handle}


def read_cytobands(path: Path) -> dict[str, list[tuple[int, int, str]]]:
    bands: dict[str, list[tuple[int, int, str]]] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            chromosome, start, end, name, *_ = line.rstrip().split("\t")
            bands.setdefault(chromosome, []).append((int(start), int(end), name))
    return bands


def span_band(
    bands: dict[str, list[tuple[int, int, str]]], chromosome: str, start: int, end: int
) -> str:
    """Cytoband label covering the locus.

    A midpoint band mislabels anything that crosses a boundary: the 23 Mb chr2
    inversion spans p11.2 through q13 and its midpoint names it q11.2, putting a
    p-arm region on the q arm.
    """

    def band_at(position: int) -> str:
        matches = [
            name
            for left, right, name in bands.get(chromosome, [])
            if left <= position < right
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one cytoband for {chromosome}:{position}; found {matches}"
            )
        return matches[0]

    first = band_at(start)
    last = band_at(end - 1)
    return first if first == last else f"{first}-{last}"


def extract_region(reference: Path, region: str) -> bytes:
    completed = subprocess.run(
        ["samtools", "faidx", str(reference), region],
        check=True,
        stdout=subprocess.PIPE,
    )
    header, newline, sequence = completed.stdout.partition(b"\n")
    if not newline or not header.startswith(b">") or not sequence:
        raise RuntimeError(f"samtools faidx returned an invalid FASTA for {region}")
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cytobands", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    loci = read_consensus_loci(args.properties)
    reference_lengths = read_reference_lengths(args.reference)
    cytobands = read_cytobands(args.cytobands)
    manifest: list[dict[str, object]] = []

    args.fasta.parent.mkdir(parents=True, exist_ok=True)
    with args.fasta.open("wb") as output:
        for row in loci:
            chromosome = row["Chromosome"]
            start = int(row["Start"])
            end = int(row["End"])
            inversion_length = end - start
            if inversion_length <= 0:
                raise RuntimeError(f"Invalid inversion interval: {row['OrigID']}")
            window_start = max(0, start - inversion_length)
            window_end = min(reference_lengths[chromosome], end + inversion_length)
            region = f"{chromosome}:{window_start + 1}-{window_end}"
            sequence = extract_region(args.reference, region)
            output.write(f">{row['OrigID']}\n".encode())
            output.write(sequence)
            manifest.append(
                {
                    "inv_id": row["OrigID"],
                    "chrom": chromosome,
                    "inv_start": start,
                    "inv_end": end,
                    "window_start": window_start,
                    "window_end": window_end,
                    "recurrence": (
                        "Recurrent"
                        if row["0_single_1_recur_consensus"].strip() == "1"
                        else "Single-event"
                    ),
                    "label": span_band(cytobands, chromosome, start, end),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Prepared {len(manifest)} consensus loci in {args.fasta}")


if __name__ == "__main__":
    main()
