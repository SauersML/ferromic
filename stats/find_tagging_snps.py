#!/usr/bin/env python3
"""
Identify tagging SNPs between inversion orientation groups using PHYLIP alignments.

The script expects PHYLIP files produced by the pipeline in `src/process.rs`, which
writes one alignment per inversion group via `write_phylip_file`. Files follow the
pattern:

    inversion_group{group}_{chrom}_start{start}_end{end}.phy.gz

where `group` is 0 (direct orientation) or 1 (inverted orientation), `start` and
`end` are 1-based inclusive positions, and the alignment contains haplotypes
labelled with `_L` or `_R` suffixes (left/right haplotypes from the originating
VCF). Sample names are written exactly as emitted by the Rust writer, which sorts
the names and separates them from the sequence with two spaces. The script pairs
files that share the same chromosome and coordinates (one direct, one inverted),
then scans the combined alignment for SNPs whose allele frequencies differ between
the two groups.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

INVERSION_PHY_RE = re.compile(
    r"^inversion_group(?P<group>[01])_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy(?:\.gz)?$"
)

DIRECT_GROUP = 0
INVERTED_GROUP = 1

CHAIN_CACHE = Path(__file__).resolve().parent / "liftover_chains"


@dataclass(frozen=True)
class InversionKey:
    chrom: str
    start: int
    end: int

    @property
    def label(self) -> str:
        return f"{self.chrom}:{self.start}-{self.end}"


@dataclass
class Alignment:
    sequences: np.ndarray  # shape (n_samples, n_sites)
    sample_names: List[str]

    @property
    def n_samples(self) -> int:
        return self.sequences.shape[0]

    @property
    def n_sites(self) -> int:
        return self.sequences.shape[1]


class PhylipError(Exception):
    pass


def open_text_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_phylip(path: str) -> Alignment:
    """Parse a PHYLIP file written by ``write_phylip_file``.

    The writer emits a header ``"{n} {m}"`` followed by one line per sequence in the
    form ``"{sample_name}  {sequence}"``. Sample names can include underscores and the
    sequences are uppercase DNA characters.
    """

    try:
        with open_text_maybe_gzip(path) as handle:
            lines = [line.rstrip("\n") for line in handle if line.strip()]
    except OSError as exc:
        raise PhylipError(f"Failed to read {path}: {exc}") from exc

    if not lines:
        raise PhylipError(f"{path} is empty")

    try:
        header_parts = lines[0].split()
        expected_samples, expected_sites = int(header_parts[0]), int(header_parts[1])
    except (IndexError, ValueError) as exc:
        raise PhylipError(f"Invalid PHYLIP header in {path!r}: {lines[0]!r}") from exc

    sample_names: List[str] = []
    sequences: List[List[str]] = []

    for line in lines[1:]:
        try:
            sample, sequence = line.split(None, 1)
        except ValueError as exc:
            raise PhylipError(f"Malformed sequence line in {path!r}: {line!r}") from exc

        bases = list(sequence.strip())
        if expected_sites and len(bases) != expected_sites:
            raise PhylipError(
                f"Sequence length mismatch in {path}: got {len(bases)}, expected {expected_sites}"
            )

        sample_names.append(sample)
        sequences.append(bases)

    if expected_samples and len(sample_names) != expected_samples:
        raise PhylipError(
            f"Sample count mismatch in {path}: got {len(sample_names)}, expected {expected_samples}"
        )

    if not sequences:
        raise PhylipError(f"No sequences found in {path}")

    stacked = np.array(sequences, dtype="U1")
    return Alignment(sequences=stacked, sample_names=sample_names)


def discover_inversion_files(base_dir: str) -> Dict[InversionKey, Dict[int, str]]:
    """Locate inversion PHYLIP files and bucket them by region key and group."""

    grouped: Dict[InversionKey, Dict[int, str]] = defaultdict(dict)

    for root, _dirs, files in os.walk(base_dir):
        for filename in files:
            match = INVERSION_PHY_RE.match(filename)
            if not match:
                continue

            group = int(match.group("group"))
            key = InversionKey(
                chrom=match.group("chrom"),
                start=int(match.group("start")),
                end=int(match.group("end")),
            )

            if group in grouped[key]:
                raise PhylipError(
                    f"Duplicate inversion group {group} for {key.label}: {grouped[key][group]} and {os.path.join(root, filename)}"
                )

            grouped[key][group] = os.path.join(root, filename)

    return grouped


def site_allele_frequencies(encoded: np.ndarray, cutoff_missing: Iterable[str] = ("N", "-")) -> Tuple[str, Dict[str, float]]:
    """Return the major allele and frequencies of all alleles at a site."""

    values, counts = np.unique(encoded, return_counts=True)
    freq = {allele: count / encoded.size for allele, count in zip(values, counts)}

    informative = [(allele, count) for allele, count in zip(values, counts) if allele not in cutoff_missing]
    if not informative:
        raise ValueError("Site has only missing/placeholder bases")

    major = max(informative, key=lambda pair: pair[1])[0]
    return major, freq


def ensure_liftover() -> str:
    """Ensure the UCSC liftOver binary is available and return its path."""

    existing = shutil.which("liftOver")
    if existing:
        return existing

    import platform

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine == "x86_64":
        url = "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
    elif system == "darwin" and machine == "x86_64":
        url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver"
    elif system == "darwin" and machine == "arm64":
        url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver"
    else:
        raise RuntimeError(f"Unsupported platform for automatic liftOver install: {system} {machine}")

    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    liftover_path = local_bin / "liftOver"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, liftover_path.open("wb") as f:
        f.write(r.read())

    liftover_path.chmod(0o755)

    current = os.environ.get("PATH", "")
    if str(local_bin) not in current.split(":"):
        os.environ["PATH"] = f"{local_bin}:{current}"

    return str(liftover_path)


def ensure_chain_file(from_build: str, to_build: str) -> Path:
    CHAIN_CACHE.mkdir(parents=True, exist_ok=True)
    chain_name = f"{from_build}To{to_build.capitalize()}.over.chain.gz"
    chain_path = CHAIN_CACHE / chain_name
    if chain_path.exists():
        return chain_path

    url = f"https://hgdownload.soe.ucsc.edu/goldenPath/{from_build}/liftOver/{chain_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, chain_path.open("wb") as f:
        f.write(r.read())

    return chain_path


def liftover_sites(regions: List[dict], from_build: str, to_build: str) -> Dict[int, dict]:
    """Liftover 0-based regions and return a mapping of row index to lifted coords."""

    if not regions:
        return {}

    liftover_bin = ensure_liftover()
    chain_path = ensure_chain_file(from_build, to_build)

    tmpdir = Path(tempfile.gettempdir())
    pid = os.getpid()
    bed_in = tmpdir / f"liftover_in_{pid}.bed"
    bed_out = tmpdir / f"liftover_out_{pid}.bed"
    bed_unmapped = tmpdir / f"liftover_unmapped_{pid}.bed"

    region_map: Dict[str, dict] = {}
    prefix = "r_"

    with bed_in.open("w") as f:
        for i, r in enumerate(regions):
            chrom = str(r["chrom"])
            chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
            start = int(r["start"])
            end = int(r["end"])
            rid = f"{prefix}{i}"
            region_map[rid] = dict(r)
            f.write(f"{chrom}\t{start}\t{end}\t{rid}\n")

    cmd = [liftover_bin, str(bed_in), str(chain_path), str(bed_out), str(bed_unmapped)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    lifted: Dict[int, dict] = {}

    try:
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())

        if bed_out.exists():
            with bed_out.open() as f:
                for line in f:
                    chrom_out, start_out, end_out, rid = line.rstrip("\n").split("\t")[:4]
                    base = region_map.get(rid)
                    if base is None:
                        continue
                    idx = base.get("idx")
                    if idx is None or idx in lifted:
                        continue
                    chrom_no_chr = chrom_out[3:] if chrom_out.startswith("chr") else chrom_out
                    lifted[idx] = {
                        "chrom": chrom_no_chr,
                        "start": int(start_out),
                        "end": int(end_out),
                    }
    finally:
        bed_in.unlink(missing_ok=True)
        bed_out.unlink(missing_ok=True)
        bed_unmapped.unlink(missing_ok=True)

    return lifted


def analyze_inversion_pair(key: InversionKey, files: Dict[int, str]) -> List[dict]:
    """Compute tagging SNP stats for a paired inversion alignment."""

    if DIRECT_GROUP not in files or INVERTED_GROUP not in files:
        missing = [
            str(group)
            for group in (DIRECT_GROUP, INVERTED_GROUP)
            if group not in files
        ]
        raise PhylipError(
            f"Missing inversion group(s) {', '.join(missing)} for {key.label}"
        )

    alignment_direct = parse_phylip(files[DIRECT_GROUP])
    alignment_inverted = parse_phylip(files[INVERTED_GROUP])

    if alignment_direct.n_sites != alignment_inverted.n_sites:
        raise PhylipError(
            "Site count mismatch for {}: direct has {}, inverted has {}".format(
                key.label, alignment_direct.n_sites, alignment_inverted.n_sites
            )
        )

    combined_sequences = np.vstack([alignment_direct.sequences, alignment_inverted.sequences])
    group_labels = np.array(
        [DIRECT_GROUP] * alignment_direct.n_samples
        + [INVERTED_GROUP] * alignment_inverted.n_samples
    )

    results: List[dict] = []
    for site_index in range(combined_sequences.shape[1]):
        column = combined_sequences[:, site_index]
        try:
            major_allele, _freqs = site_allele_frequencies(column)
        except ValueError:
            continue

        encoded_major = (column == major_allele).astype(int)
        if encoded_major.sum() == 0 or encoded_major.sum() == encoded_major.size:
            # Monomorphic relative to the major allele; skip
            continue

        correlation, _ = stats.pearsonr(group_labels, encoded_major)
        if np.isnan(correlation):
            continue

        freq_direct = encoded_major[: alignment_direct.n_samples].mean()
        freq_inverted = encoded_major[alignment_direct.n_samples :].mean()

        results.append(
            {
                "inversion_region": key.label,
                "chromosome": key.chrom,
                "region_start": key.start,
                "region_end": key.end,
                "site_index": site_index,
                "position": key.start + site_index,
                "position_hg38": key.start + site_index,
                "chromosome_hg38": key.chrom,
                "direct_group_size": alignment_direct.n_samples,
                "inverted_group_size": alignment_inverted.n_samples,
                "allele_freq_direct": freq_direct,
                "allele_freq_inverted": freq_inverted,
                "allele_freq_difference": abs(freq_direct - freq_inverted),
                "correlation": correlation,
            }
        )

    return results


def find_tagging_snps(phy_dir: str, output_file: str) -> None:
    grouped = discover_inversion_files(phy_dir)

    if not grouped:
        print(f"No inversion PHYLIP files found in '{phy_dir}'.")
        sys.exit(1)

    has_errors = False
    aggregated: List[dict] = []

    for key, files in grouped.items():
        try:
            aggregated.extend(analyze_inversion_pair(key, files))
        except Exception as exc:  # noqa: BLE001 - user-facing script
            has_errors = True
            print(f"Error processing {key.label}: {exc}", file=sys.stderr)

    if not aggregated:
        print("No variable SNPs found across all inversion regions.")
        if has_errors:
            sys.exit(1)
        return

    df = pd.DataFrame(aggregated)
    df["chromosome_hg37"] = pd.NA
    df["position_hg37"] = pd.NA

    liftover_regions = [
        {
            "chrom": row["chromosome"],
            "start": int(row["position_hg38"]) - 1,
            "end": int(row["position_hg38"]),
            "idx": idx,
        }
        for idx, row in df.iterrows()
    ]

    try:
        lifted = liftover_sites(liftover_regions, "hg38", "hg19")
        for idx, coords in lifted.items():
            df.at[idx, "chromosome_hg37"] = coords["chrom"]
            df.at[idx, "position_hg37"] = coords["start"] + 1
    except Exception as exc:  # noqa: BLE001 - best-effort liftover
        print(f"Warning: failed to liftover hg38→hg19: {exc}", file=sys.stderr)

    df["abs_correlation"] = df["correlation"].abs()
    df = df.sort_values(["inversion_region", "abs_correlation"], ascending=[True, False])
    df = df.drop(columns=["abs_correlation"])

    df.to_csv(output_file, sep="\t", index=False, float_format="%.6f")
    print(f"Successfully wrote tagging SNP report to {output_file}")

    if has_errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find tagging SNPs for inversion groups and write a TSV report.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("phy_dir", help="Directory containing inversion .phy or .phy.gz files.")
    parser.add_argument(
        "--output",
        default="tagging_snps.tsv",
        help="Path to the output TSV file (default: tagging_snps.tsv).",
    )
    args = parser.parse_args()

    find_tagging_snps(args.phy_dir, args.output)


if __name__ == "__main__":
    main()
