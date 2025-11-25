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
from concurrent.futures import ProcessPoolExecutor, as_completed
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

# Limit base handling to the common alignment characters to keep memory overhead small.
BASE_CODES = np.array([ord("A"), ord("C"), ord("G"), ord("T"), ord("N"), ord("-")], dtype=np.uint8)
MISSING_BASE_INDICES = {4, 5}  # indices within BASE_CODES corresponding to N and '-'
BASE_TO_INDEX = np.full(256, -1, dtype=np.int8)
for i, code in enumerate(BASE_CODES):
    BASE_TO_INDEX[code] = i

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
    sequences: np.ndarray  # shape (n_samples, n_sites), dtype=np.uint8
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
    sequences are uppercase DNA characters. Sequences are stored as a dense uint8
    matrix to minimize memory overhead while enabling fast vectorized operations.
    """

    try:
        with open_text_maybe_gzip(path) as handle:
            header = None
            for line in handle:
                if line.strip():
                    header = line.rstrip("\n")
                    break
            if header is None:
                raise PhylipError(f"{path} is empty")

            try:
                header_parts = header.split()
                expected_samples, expected_sites = int(header_parts[0]), int(header_parts[1])
            except (IndexError, ValueError) as exc:
                raise PhylipError(f"Invalid PHYLIP header in {path!r}: {header!r}") from exc

            if expected_samples <= 0 or expected_sites <= 0:
                raise PhylipError(f"Invalid dimensions in {path!r}: {expected_samples}x{expected_sites}")

            sequences = np.empty((expected_samples, expected_sites), dtype=np.uint8)
            sample_names: List[str] = []
            row = 0

            for line in handle:
                if not line.strip():
                    continue
                try:
                    sample, sequence = line.split(None, 1)
                except ValueError as exc:
                    raise PhylipError(f"Malformed sequence line in {path!r}: {line!r}") from exc

                # Accept lowercase bases by normalizing to uppercase before encoding.
                seq_str = sequence.strip().upper()
                if len(seq_str) != expected_sites:
                    raise PhylipError(
                        f"Sequence length mismatch in {path}: got {len(seq_str)}, expected {expected_sites}"
                    )

                seq_arr = np.frombuffer(seq_str.encode("ascii"), dtype=np.uint8)
                if np.any(BASE_TO_INDEX[seq_arr] < 0):
                    raise PhylipError(
                        f"Unexpected character in sequence from {path!r}: {seq_str}"
                    )

                sequences[row] = seq_arr
                sample_names.append(sample)
                row += 1

            if row != expected_samples:
                raise PhylipError(
                    f"Sample count mismatch in {path}: got {row}, expected {expected_samples}"
                )
    except OSError as exc:
        raise PhylipError(f"Failed to read {path}: {exc}") from exc

    return Alignment(sequences=sequences, sample_names=sample_names)


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

    n_sites = alignment_direct.n_sites
    n_direct = alignment_direct.n_samples
    n_inverted = alignment_inverted.n_samples
    n_total = n_direct + n_inverted

    base_counts_direct = np.empty((len(BASE_CODES), n_sites), dtype=np.int32)
    base_counts_inverted = np.empty_like(base_counts_direct)
    for i, base_code in enumerate(BASE_CODES):
        base_counts_direct[i] = (alignment_direct.sequences == base_code).sum(axis=0)
        base_counts_inverted[i] = (alignment_inverted.sequences == base_code).sum(axis=0)

    combined_counts = base_counts_direct + base_counts_inverted

    informative_indices = np.array([i for i in range(len(BASE_CODES)) if i not in MISSING_BASE_INDICES])
    informative_counts = combined_counts[informative_indices]

    informative_totals = informative_counts.sum(axis=0)
    if not np.any(informative_totals):
        return []

    major_rel_indices = np.argmax(informative_counts, axis=0)
    major_base_indices = informative_indices[major_rel_indices]
    sites = np.arange(n_sites)

    major_counts_total = combined_counts[major_base_indices, sites]
    major_counts_direct = base_counts_direct[major_base_indices, sites]
    major_counts_inverted = base_counts_inverted[major_base_indices, sites]

    # Skip monomorphic or wholly missing sites.
    valid = (
        informative_totals > 0
        & (major_counts_total > 0)
        & (major_counts_total < n_total)
    )

    if not np.any(valid):
        return []

    sum_x = major_counts_total[valid].astype(np.float64)
    sum_xg = major_counts_inverted[valid].astype(np.float64)

    numerator = n_total * sum_xg - sum_x * n_inverted
    denom_constant = float(n_direct * n_inverted)
    denom = np.sqrt((n_total * sum_x - sum_x * sum_x) * denom_constant)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = numerator / denom

    freq_direct = major_counts_direct[valid] / n_direct
    freq_inverted = major_counts_inverted[valid] / n_inverted

    valid_sites = sites[valid]
    results: List[dict] = []

    for site_index, corr, f_dir, f_inv in zip(valid_sites, correlations, freq_direct, freq_inverted):
        if np.isnan(corr):
            continue

        results.append(
            {
                "inversion_region": key.label,
                "chromosome": key.chrom,
                "region_start": key.start,
                "region_end": key.end,
                "site_index": int(site_index),
                "position": key.start + int(site_index),
                "position_hg38": key.start + int(site_index),
                "chromosome_hg38": key.chrom,
                "direct_group_size": n_direct,
                "inverted_group_size": n_inverted,
                "allele_freq_direct": float(f_dir),
                "allele_freq_inverted": float(f_inv),
                "allele_freq_difference": float(abs(f_dir - f_inv)),
                "correlation": float(corr),
            }
        )

    return results


def process_inversion_pair(item: Tuple[InversionKey, Dict[int, str]]):
    """Wrapper suitable for running ``analyze_inversion_pair`` in a worker."""

    key, files = item
    try:
        return True, key.label, analyze_inversion_pair(key, files)
    except Exception as exc:  # noqa: BLE001 - user-facing script
        return False, key.label, str(exc)


def find_tagging_snps(
    phy_dir: str,
    output_file: str,
    workers: int | None = None,
    chromosomes: Iterable[str] | None = None,
) -> None:
    if workers is None:
        workers = os.cpu_count() or 1
    elif workers < 1:
        raise ValueError("workers must be at least 1")

    print(f"Using {workers} worker(s) for inversion processing.", flush=True)

    grouped = discover_inversion_files(phy_dir)

    if chromosomes:
        allowed = {chrom.strip() for chrom in chromosomes if chrom.strip()}
        if not allowed:
            raise ValueError("No valid chromosomes provided for filtering")

        grouped = {key: files for key, files in grouped.items() if key.chrom in allowed}

        print(
            f"Filtering to {len(grouped)} inversion region(s) on chromosomes: {', '.join(sorted(allowed))}.",
            flush=True,
        )

    if not grouped:
        print(f"No inversion PHYLIP files found in '{phy_dir}'.")
        sys.exit(1)

    print(
        f"Discovered {len(grouped)} inversion region(s) with paired alignments under {phy_dir}.",
        flush=True,
    )

    has_errors = False
    aggregated: List[dict] = []
    total_regions = len(grouped)
    grouped_items = list(grouped.items())

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_key = {
            executor.submit(process_inversion_pair, item): item[0] for item in grouped_items
        }

        for processed, future in enumerate(as_completed(future_to_key), start=1):
            key = future_to_key[future]
            try:
                success, label, result = future.result()
            except Exception as exc:  # noqa: BLE001 - user-facing script
                has_errors = True
                print(
                    f"[{processed}/{total_regions}] Unexpected failure processing {key.label}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if success:
                aggregated.extend(result)
                print(
                    f"[{processed}/{total_regions}] Completed {label} with {len(result)} SNP(s).",
                    flush=True,
                )
            else:
                has_errors = True
                print(
                    f"[{processed}/{total_regions}] Error processing {label}: {result}",
                    file=sys.stderr,
                    flush=True,
                )

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
        print(
            f"Lifting over {len(liftover_regions)} site(s) from hg38 to hg19 (best effort)...",
            flush=True,
        )
        lifted = liftover_sites(liftover_regions, "hg38", "hg19")
        for idx, coords in lifted.items():
            df.at[idx, "chromosome_hg37"] = coords["chrom"]
            df.at[idx, "position_hg37"] = coords["start"] + 1
        print(
            f"Liftover completed for {len(lifted)} site(s); {len(liftover_regions) - len(lifted)} remained unmapped.",
            flush=True,
        )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use (default: system CPU count).",
    )
    parser.add_argument(
        "--chromosomes",
        type=str,
        default=None,
        help=(
            "Comma-separated list of chromosomes to include; filters alignments before processing."
        ),
    )
    args = parser.parse_args()

    chrom_list = None
    if args.chromosomes:
        chrom_list = [c.strip() for c in args.chromosomes.split(",") if c.strip()]

    find_tagging_snps(args.phy_dir, args.output, workers=args.workers, chromosomes=chrom_list)


if __name__ == "__main__":
    main()
