#!/usr/bin/env python3
"""Write the GRCh38 alignment windows for every locus in the orientation ledger.

The 93 consensus loci keep the windows recorded in the consensus manifest. Every
other ledger locus gets the same rule the manifest was built with: a flank on
each side equal to the inversion length, or 300 kb where that is larger. The
window end is clipped to the chromosome length by the caller, which has the
reference index at hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MIN_FLANK = 300_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        manifest = {
            row["inv_id"]: row for row in csv.DictReader(handle, delimiter="\t")
        }
    ledger = json.loads(args.ledger.read_text())["responses"]

    fields = [
        "inv_id",
        "chrom",
        "inv_start",
        "inv_end",
        "window_start",
        "window_end",
        "label",
        "source",
    ]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in ledger:
            inv_id = record["inv_id"]
            row = manifest.get(inv_id)
            if row is not None:
                writer.writerow(
                    {
                        "inv_id": inv_id,
                        "chrom": row["chrom"],
                        "inv_start": row["inv_start"],
                        "inv_end": row["inv_end"],
                        "window_start": row["window_start"],
                        "window_end": row["window_end"],
                        "label": row["label"],
                        "source": "consensus93",
                    }
                )
                continue
            start = int(record["start"])
            end = int(record["end"])
            flank = max(MIN_FLANK, end - start)
            writer.writerow(
                {
                    "inv_id": inv_id,
                    "chrom": record["chrom"],
                    "inv_start": start,
                    "inv_end": end,
                    "window_start": max(0, start - flank),
                    "window_end": end + flank,
                    "label": inv_id,
                    "source": "ledger",
                }
            )
    print(f"Wrote {len(ledger)} windows to {args.out}")


if __name__ == "__main__":
    main()
