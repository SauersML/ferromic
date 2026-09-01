#!/usr/bin/env python
"""Strictly validate a sharded reference-simulation grid and record provenance."""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re

import run_grid


IDENTITY = (
    "scenario", "depth", "rho", "m_flux", "inv_freq", "sample_size", "seed",
)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row: dict) -> tuple:
    return (
        row["scenario"],
        row["depth"],
        float(row["rho"]),
        float(row["m_flux"]),
        float(row["inv_freq"]),
        int(row["sample_size"]),
        int(row["seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--upstream-repository", required=True)
    parser.add_argument("--upstream-ref", required=True)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{40}", args.upstream_commit):
        raise SystemExit("--upstream-commit must be a full lowercase Git SHA")

    paths = sorted({path for pattern in args.inputs for path in glob.glob(pattern)})
    if not paths:
        raise SystemExit("no simulation shards matched")

    observed: list[tuple] = []
    failures = []
    for path in paths:
        with open(path, newline="") as handle:
            for line_number, row in enumerate(csv.DictReader(handle), start=2):
                if row.get("error"):
                    failures.append((path, line_number, row["error"]))
                else:
                    observed.append(identity(row))
    if failures:
        path, line_number, message = failures[0]
        raise SystemExit(
            f"{len(failures)} failed simulations; first at "
            f"{path}:{line_number}: {message}"
        )
    if len(observed) != len(set(observed)):
        raise SystemExit("duplicate simulation identities found across shards")

    expected = [identity(row) for row in run_grid.build_grid(args.task)]
    missing = set(expected) - set(observed)
    extra = set(observed) - set(expected)
    if missing or extra:
        raise SystemExit(
            f"grid differs from specification: missing={len(missing)}, "
            f"extra={len(extra)}; examples missing={list(missing)[:2]}, "
            f"extra={list(extra)[:2]}"
        )
    if len(observed) != len(expected):
        raise SystemExit(
            f"row count differs: observed={len(observed)}, expected={len(expected)}"
        )

    payload = {
        "task": args.task,
        "rows": len(observed),
        "identity_fields": list(IDENTITY),
        "scenarios": sorted({row[0] for row in observed}),
        "depths": sorted({row[1] for row in observed}),
        "recombination_rates": sorted({row[2] for row in observed}),
        "gene_flux_rates": sorted({row[3] for row in observed}),
        "inversion_frequencies": sorted({row[4] for row in observed}),
        "sample_sizes": sorted({row[5] for row in observed}),
        "upstream": {
            "repository": args.upstream_repository,
            "ref": args.upstream_ref,
            "resolved_commit": args.upstream_commit,
        },
        "input_shards": [
            {
                "path": os.path.basename(path),
                "sha256": sha256(path),
            }
            for path in paths
        ],
    }
    with open(args.provenance, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"validated {len(observed):,} rows for {args.task} across "
        f"{len(paths)} shards -> {args.provenance}"
    )


if __name__ == "__main__":
    main()
