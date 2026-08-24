#!/usr/bin/env python3
"""Split a combined panTro6 PAF and render all 93 canonical SVbyEye plots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 93 or len({row["inv_id"] for row in rows}) != 93:
        raise RuntimeError("Manifest must contain exactly 93 unique loci")
    return rows


def split_paf(path: Path, loci: set[str], output_dir: Path) -> dict[str, Path]:
    handles = {}
    counts: dict[str, int] = defaultdict(int)
    paths = {locus: output_dir / "paf" / f"{locus}.paf" for locus in loci}
    (output_dir / "paf").mkdir(parents=True, exist_ok=True)
    try:
        with path.open() as source:
            for line in source:
                fields = line.split("\t", 6)
                if len(fields) < 6:
                    raise RuntimeError("Malformed PAF record")
                target = fields[5]
                if target not in loci:
                    raise RuntimeError(f"Unexpected PAF target: {target}")
                handle = handles.get(target)
                if handle is None:
                    handle = paths[target].open("w")
                    handles[target] = handle
                handle.write(line)
                counts[target] += 1
    finally:
        for handle in handles.values():
            handle.close()
    missing = sorted(loci - counts.keys())
    if missing:
        raise RuntimeError(f"No panTro6 alignments for {len(missing)} loci: {missing}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--paf", type=Path, required=True)
    parser.add_argument("--plot-script", type=Path, required=True)
    parser.add_argument("--rscript", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pafs = split_paf(args.paf, {row["inv_id"] for row in rows}, args.output_dir)
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    def render(row: dict[str, str]) -> str:
        prefix = plot_dir / row["inv_id"]
        subprocess.run(
            [
                str(args.rscript),
                str(args.plot_script),
                str(pafs[row["inv_id"]]),
                str(prefix),
                row["inv_id"],
                row["chrom"],
                row["inv_start"],
                row["inv_end"],
                row["window_start"],
                row["label"],
            ],
            check=True,
        )
        for suffix in (".pdf", ".png", ".orientation.tsv"):
            output = Path(f"{prefix}{suffix}")
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"Missing renderer output: {output}")
        return row["inv_id"]

    completed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(render, row): row["inv_id"] for row in rows}
        for future in as_completed(futures):
            inv_id = future.result()
            completed += 1
            print(f"Rendered {completed}/93: {inv_id}", flush=True)

    (args.output_dir / "manifest.tsv").write_bytes(args.manifest.read_bytes())
    provenance = {
        "schema_version": 1,
        "locus_count": 93,
        "generator": "svbyeye/bin/plot_chimp_hires.R",
        "generator_sha256": sha256(args.plot_script),
        "combined_paf_sha256": sha256(args.paf),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
