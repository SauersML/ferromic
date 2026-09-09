#!/usr/bin/env python3
"""Write data/chimp_alignment_responses.json from reviewed orientation calls.

Input is the table from build_orientation_ledger.py plus an optional overrides
table (``inv_id``, ``classification``, ``note``) holding the calls changed on
inspection of the plots. Locus coordinates, region strings and image names are
carried over from the previous ledger so downstream readers see the same
record shape; the top-level metadata records which assembly the calls are for.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = {"direct", "inverted", "na"}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--previous-ledger", type=Path, required=True)
    parser.add_argument("--assembly", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    previous = json.loads(args.previous_ledger.read_text())
    by_id = {record["inv_id"]: record for record in previous["responses"]}
    calls = {row["inv_id"]: row for row in read_tsv(args.calls)}
    overrides: dict[str, dict[str, str]] = {}
    if args.overrides is not None:
        for row in read_tsv(args.overrides):
            if row["classification"] not in ALLOWED:
                raise ValueError(f"Bad override: {row!r}")
            overrides[row["inv_id"]] = row
    unknown = set(overrides) - set(calls)
    if unknown:
        raise ValueError(f"Overrides for loci without calls: {sorted(unknown)}")

    timestamp = datetime.now(timezone.utc).isoformat()
    responses = []
    counts: dict[str, int] = {}
    for inv_id, record in by_id.items():
        call = calls[inv_id]["auto_call"]
        source = "alignment call"
        if inv_id in overrides:
            call = overrides[inv_id]["classification"]
            source = "manual review: " + overrides[inv_id].get("note", "")
        if call not in ALLOWED:
            raise ValueError(f"Bad call for {inv_id}: {call}")
        counts[call] = counts.get(call, 0) + 1
        responses.append(
            {
                "inv_id": inv_id,
                "chrom": record["chrom"],
                "start": record["start"],
                "end": record["end"],
                "region": record["region"],
                "size_bp": record["size_bp"],
                "image_file": record["image_file"],
                "classification": call,
                "call_source": source,
                "updated_at": timestamp,
            }
        )
    document = {
        "schema_version": 1,
        "dataset": "chimp_vs_hg38_inversion_alignments",
        "assembly": args.assembly,
        "method": (
            "minimap2 asm20 alignment of the chimpanzee assembly to GRCh38 windows "
            "with secondary alignments kept; calls from svbyeye/bin/"
            "build_orientation_ledger.py: synteny anchors on both flanks of the same "
            "chimpanzee chromosome, interior strand read between them; every plot "
            "inspected"
        ),
        "updated_at": timestamp,
        "responses": responses,
    }
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Wrote {len(responses)} calls to {args.out}: {counts}")


if __name__ == "__main__":
    main()
