"""
Utility for downloading PheWeb phenotype results.

New CLI options:
- Provide one or more phenocodes via --phenocode or a newline-delimited file via
  --phenocode-file to restrict downloads.
- --plan to split work into shards for CI matrix jobs.
- --aggregate to merge per-shard outputs into a consolidated TSV.
"""
import argparse
import csv
import gzip
import json
import os
import signal
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event
from typing import Dict, Iterable, List, Sequence, Tuple

import bgzip
import pandas as pd
import requests
from tqdm import tqdm

# Example URL: https://pheweb.org/UKB-TOPMed/download/208

# how to know which number corresponds to which phenotype?

# 290.11 e.g. for Alzheimer's disease, so URL is https://pheweb.org/UKB-TOPMed/pheno/290.11

# We can use the file phenotypes.tsv to get the mapping from name to number. The file has this format:
# chrom   pos     ref     alt     rsids   nearest_genes   pval    num_cases  num_controls     num_samples     phenostring     num_peaks
#        phenocode  gc_lambda_hundred        category
# 6       25616225        A       G       rs78912080      CARMIL1 0       665405082   405747  Disorders of iron metabolism    8
#       275.1   1       hematopoietic
# 6       32418606        A       G       rs2187819       BTNL2   0       1846333378  335224  Celiac disease  7       557.1   1.0283
#        digestive

# this file should be downloaded from github at: https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phenotypes.tsv

# DOWNLOAD all of the pheno results for every phenotype in list below and combine into a single file

# list of phenotypes to download: every value in the column "All_Matches" at this URL:
# https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/mappings_final.tsv
# it has this format:
# Source_Phenotype      Has_Good_Match  Best_Match      All_Matches     Reasoning       Source_ICD10
# Mild cognitive impairment     True    Mild cognitive impairment       Mild cognitive impairment; Memory loss; Delirium dementia and amnestic and
# other cognitive disorders    The source phenotype 'Mild cognitive impairment' (ICD-9 331.83, ICD-10 G31.84) has an exact name match in the target list.
# While 'Memory loss' and 'Delirium dementia and amnestic and other cognitive disorders' are related concepts, the verbatim match is the most precise and
# appropriate mapping. G31.84
# Benign mammary dysplasias     True    Benign mammary dysplasias       Benign mammary dysplasias; Fibroadenosis of breast; Cystic mastopathy;
# Fibrosclerosis of breast; Other specified benign mammary dysplasias; Other nonmalignant breast conditions       The source phenotype 'Benign mammary
# dysplasias' (ICD-9 610 series; ICD-10 N60 series) has an exact string match in the UK Biobank candidate list. While several subtypes of this condition
# (e.g., Fibroadenosis of breast, Cystic mastopathy, Fibrosclerosis of breast) appear in the target list, the parent category 'Benign mammary dysplasias'
# is the precise semantic and hierarchical equivalent. N60;N60.0;N60.01;N60.02;N60.09;N60.1;N60.11;N60.12;N60.19;N60.2;N60.21;N60.22;N60.29;N60.3;N60.31
# ;N60.32;N60.39;N60.4;N60.41;N60.42;N60.49;N60.8;N60.81;N60.82;N60.89;N60.9;N60.91;N60.92;N60.99
# Other diseases of stomach and duodenum        True    Other disorders of stomach and duodenum Other disorders of stomach and duodenum; Dyspepsia and
# other specified disorders of function of stomach; Functional digestive disorders The source phenotype 'Other diseases of stomach and duodenum' includes
# ICD-9 codes 536 (Disorders of function of stomach) and 537 (Other disorders of stomach and duodenum) and ICD-10 K31 (Other diseases of stomach and
# duodenum). The target list contains 'Other disorders of stomach and duodenum', which is a nearly exact string and semantic match for the source name and
# the K31/537 category. While 'Dyspepsia and other specified disorders of function of stomach' is relevant for the 536 codes, the target 'Other disorders of
# stomach and duodenum' is the most encompassing and direct match for the source label.     K31;K31.0;K31.1;K31.2;K31.3;K31.6;K31.8;K31.83;K31.84;K31.89;K31.9;
# K31.A;K31.A0;K31.A1;K31.A11;K31.A12;K31.A13;K31.A14;K31.A15;K31.A19;K31.A2;K31.A21;K31.A22;K31.A29
# Abnormal cytological findings in specimens from genital organs        True    Abnormal Papanicolaou smear of cervix and cervical HPV  Abnormal Papanicolaou
# smear of cervix and cervical HPV; Dysplasia of female genital organs; Cervical intraepithelial neoplasia [CIN] [Cervical dysplasia]; Symptoms involving
# female genital tract      The source phenotype 'Abnormal cytological findings in specimens from genital organs' is defined by ICD codes (e.g., ICD-9 795.0,
# ICD-10 R87.61) that specifically refer to abnormal findings on Papanicolaou (Pap) smears and the presence of cervical HPV. The target phenotype 'Abnormal
# Papanicolaou smear of cervix and cervical HPV' precisely describes this clinical finding and is the most specific and accurate match for the provided
# definition.   R85.61;R85.610;R85.611;R85.612;R85.613;R85.614;R85.615;R85.616;R85.618;R85.619;R85.81;R85.82;R87.6;R87.610;R87.611;R87.612;R87.613;R87.614;
# R87.615;R87.616;R87.618;R87.619;R87.62;R87.620;R87.621;R87.622;R87.623;R87.624;R87.625;R87.628;R87.629;R87.69;R87.8;R87.81;R87.810;R87.811;R87.82;R87.820;
# R87.821
# etc.

# Note, All_Matches has MULTIPLE values in each cell / row. We MUST get ALL PHEWAS results for EACH ONE, not just the best.

# For EACH phenotype file we download, it will have this format:

# chrom   pos     ref     alt     rsids   nearest_genes   consequence     pvalbeta    sebeta  af      case_af control_af      tstat
# 1       10612   A       C       rs1441784962    OR4F5   upstream_gene_variant       0.85    -1.1    5.8     5.2e-05 1.8e-05 5.2e-05 -0.033
# 1       10894   G       A               OR4F5   upstream_gene_variant   0.46-1.6    2.2     0.00066 0.00029 0.00066 -0.34
# 1       10915   G       A               OR4F5   upstream_gene_variant   0.48-1.6    2.2     0.00049 0.00015 0.00049 -0.32
# 1       10930   G       A               OR4F5   upstream_gene_variant   0.35-1.6    1.7     0.0013  0.00067 0.0013  -0.54
# etc.

# we'll download in parallel (capped) then combine all into a single file.

PHENOTYPES_META_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/phenotypes.tsv"
MAPPINGS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/mappings_final.tsv"
BASE_PHEWEB_URL = "https://pheweb.org/UKB-TOPMed/download/"
OUTPUT_FILENAME = "merged_phenotype_results.tsv"
LOG_FILENAME = "completed_phenotypes.log"
MAX_WORKERS = 4  # Conservative number to prevent timeouts on large files
DEFAULT_PLAN_FILE = "pheweb_plan.json"
DEFAULT_COORD_CHUNK = 1_000_000
BEST_TAGGING_SNPS = Path(__file__).resolve().parent.parent / "data" / "best_tagging_snps_qvalues.tsv"

RANGE_CHUNK_SIZE = 512 * 1024  # 512 KiB chunks for ranged BGZF reads
HAS_BGZIP_READER = hasattr(bgzip, "BGZipFile")

# Global event to handle interruption cleanly
stop_event = Event()


def signal_handler(signum, frame):
    print("\nInterrupt received! Stopping new downloads. Please wait for current writes to finish...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)


def load_completed_phenocodes(log_path: str) -> set:
    """Reads the log file to find which phenocodes are already done."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def append_to_log(phenocode: str, log_path: str) -> None:
    """Marks a phenocode as complete."""
    with open(log_path, 'a') as f:
        f.write(f"{phenocode}\n")


def get_phenotype_mapping():
    print(f"Downloading metadata from {PHENOTYPES_META_URL}...")
    try:
        df = pd.read_csv(PHENOTYPES_META_URL, sep='\t')
        # Create dictionary: {"Phenotype Name": "Phenocode"}
        # Note: Depending on file format, phenocode might be float or string. Force string.
        mapping = dict(zip(df['phenostring'].str.strip(), df['phenocode'].astype(str)))
        return mapping
    except Exception as e:
        print(f"CRITICAL ERROR: Could not download metadata: {e}")
        sys.exit(1)


def resolve_requested_targets(mapping_dict, requested_identifiers: Sequence[str]):
    """
    Resolve a list of user-specified identifiers (phenocodes or phenotype names)
    into the standard target format used by download_task.
    """
    if not requested_identifiers:
        return []

    reverse_mapping = {}
    for name, code in mapping_dict.items():
        reverse_mapping.setdefault(str(code), name)

    requested_targets = []
    missing = []

    for identifier in requested_identifiers:
        identifier = identifier.strip()
        if not identifier:
            continue
        if identifier in mapping_dict:
            requested_targets.append({
                'phenostring': identifier,
                'phenocode': mapping_dict[identifier]
            })
        elif identifier in reverse_mapping:
            requested_targets.append({
                'phenostring': reverse_mapping[identifier],
                'phenocode': identifier
            })
        else:
            missing.append(identifier)

    if missing:
        print("Warning: the following requested identifiers could not be resolved:")
        for identifier in missing:
            print(f"  - {identifier}")

    # Deduplicate and preserve insertion order
    seen = set()
    unique_targets = []
    for target in requested_targets:
        phenocode = str(target['phenocode'])
        if phenocode in seen:
            continue
        seen.add(phenocode)
        unique_targets.append(target)

    return unique_targets


def get_target_list(mapping_dict):
    print(f"Downloading target list from {MAPPINGS_URL}...")
    try:
        df = pd.read_csv(MAPPINGS_URL, sep='\t')

        target_phenocodes = []
        missing_phenotypes = set()

        # Iterate over the 'All_Matches' column
        for entry in df['All_Matches'].dropna():
            matches = [m.strip() for m in entry.split(';') if m.strip()]

            for phenotype_name in matches:
                if phenotype_name in mapping_dict:
                    target_phenocodes.append({
                        'phenostring': phenotype_name,
                        'phenocode': mapping_dict[phenotype_name]
                    })
                else:
                    missing_phenotypes.add(phenotype_name)

        # Deduplicate based on phenocode, keep unique objects
        unique_targets = {v['phenocode']: v for v in target_phenocodes}.values()

        if missing_phenotypes:
            print(f"\nWarning: {len(missing_phenotypes)} phenotypes from target list were not found in PheWeb metadata:")
            for mp in sorted(list(missing_phenotypes)):
                print(f"  - {mp}")
            print("-" * 30)

        return list(unique_targets)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not process target list: {e}")
        sys.exit(1)


def _describe_remote(url: str, session: requests.Session) -> Dict[str, str]:
    head = session.head(url, allow_redirects=True, timeout=30)
    info = {
        'status': str(head.status_code),
        'content_type': head.headers.get('content-type', 'unknown'),
        'content_length': head.headers.get('content-length', 'unknown'),
        'content_encoding': head.headers.get('content-encoding', 'unknown'),
        'accept_ranges': head.headers.get('accept-ranges', 'none'),
    }
    print(
        "Remote file info - status: {status}, type: {content_type}, encoding: {content_encoding}, "
        "length: {content_length}, accept-ranges: {accept_ranges}".format(**info)
    )
    return info


def _sniff_compression(url: str, session: requests.Session, info: Dict[str, str]) -> Dict[str, str | bool | int]:
    """Detect gzip vs BGZF and whether the server honors range requests."""

    headers = {}
    if info.get('accept_ranges') == 'bytes':
        headers['Range'] = 'bytes=0-255'

    resp = session.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    sample = resp.content[:256]
    is_gzip = sample.startswith(b'\x1f\x8b')
    is_bgzf = False

    # BGZF has an extra subfield with ID "BC" and length 2
    if is_gzip and len(sample) >= 18:
        xlen = int.from_bytes(sample[10:12], 'little')
        extra = sample[12:12 + xlen]
        idx = 0
        while idx + 4 <= len(extra):
            sub_id = extra[idx:idx + 2]
            sub_len = int.from_bytes(extra[idx + 2:idx + 4], 'little')
            if sub_id == b'BC' and sub_len == 2:
                is_bgzf = True
                break
            idx += 4 + sub_len

    status_for_range = resp.status_code
    if headers and status_for_range != 206:
        print(
            "Warning: Range request returned status {status_for_range}; server may ignore range requests."
            .format(status_for_range=status_for_range)
        )

    print(
        "Compression sniff: gzip={is_gzip}, bgzf={is_bgzf}, range_status={status_for_range}, "
        "bytes_sampled={sampled}".format(
            is_gzip=is_gzip,
            is_bgzf=is_bgzf,
            status_for_range=status_for_range,
            sampled=len(sample),
        )
    )

    return {
        'is_gzip': is_gzip,
        'is_bgzf': is_bgzf,
        'range_status': status_for_range,
    }


class HTTPRangeReader:
    """Minimal random-access HTTP reader that caches byte ranges."""

    def __init__(self, url: str, session: requests.Session, size: int, chunk_size: int = RANGE_CHUNK_SIZE):
        self.url = url
        self.session = session
        self.size = size
        self.chunk_size = chunk_size
        self._cache: Dict[Tuple[int, int], bytes] = {}
        self._pos = 0
        self._lock = threading.Lock()

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self._pos + offset
        elif whence == 2:
            new_pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")

        self._pos = max(0, min(self.size, new_pos))
        return self._pos

    def _fetch(self, start: int, end: int) -> bytes:
        key = (start, end)
        if key in self._cache:
            return self._cache[key]

        headers = {"Range": f"bytes={start}-{end - 1}"}
        resp = self.session.get(self.url, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.content
        self._cache[key] = data
        return data

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            size = self.size - self._pos

        with self._lock:
            start = self._pos
            end = min(self.size, start + size)
            data = bytearray()
            cursor = start
            while cursor < end:
                chunk_end = min(end, cursor + self.chunk_size)
                data.extend(self._fetch(cursor, chunk_end))
                cursor = chunk_end
            self._pos = end
        return bytes(data)

    def close(self):
        self._cache.clear()


def _parse_positions(raw_positions: Sequence[str]) -> Dict[str, set]:
    parsed: Dict[str, set] = defaultdict(set)
    for raw in raw_positions:
        cleaned = raw.strip()
        if not cleaned:
            continue
        chrom = None
        pos = None
        if ':' in cleaned:
            chrom, pos = cleaned.split(':', 1)
        elif '\t' in cleaned:
            chrom, pos = cleaned.split('\t', 1)
        else:
            parts = cleaned.replace(',', ':').split(':')
            if len(parts) == 2:
                chrom, pos = parts
        if chrom is None or pos is None:
            print(f"Skipping malformed coordinate: {cleaned}")
            continue
        try:
            if '-' in pos:
                for piece in pos.split('-'):
                    if piece:
                        parsed[chrom].add(int(float(piece)))
            else:
                parsed[chrom].add(int(float(pos)))
        except ValueError:
            print(f"Skipping coordinate with non-integer position: {cleaned}")
    return parsed


def _streaming_gzip_filter(
    url: str,
    coords_by_chrom: Dict[str, set],
    session: requests.Session,
    info: Dict[str, str] | None = None,
    compression_info: Dict[str, str] | None = None,
) -> Tuple[List[str], List[str]]:
    """Stream-decompress sequentially when random access is unsafe or unavailable."""

    header: List[str] = []
    filtered_rows: List[List[str]] = []
    if not coords_by_chrom:
        return header, filtered_rows

    info = info or _describe_remote(url, session)
    compression_info = compression_info or {}

    total_targets = sum(len(v) for v in coords_by_chrom.values())
    supports_range = info.get('accept_ranges') == 'bytes'

    def _iter_lines(reader_obj):
        for raw_line in reader_obj:
            try:
                yield raw_line.decode('utf-8').rstrip('\n')
            except AttributeError:
                if isinstance(raw_line, bytes):
                    yield raw_line.decode('utf-8', errors='replace').rstrip('\n')
                else:
                    yield str(raw_line).replace('\n', '')

    # Prefer ranged BGZF reads so we can stop early without downloading the world
    if compression_info.get('is_bgzf') and supports_range and HAS_BGZIP_READER:
        try:
            content_length = int(info.get('content_length', 0))
        except ValueError:
            content_length = 0

        if content_length > 0:
            print("Using ranged BGZF reads with cached byte windows for targeted lookup...")
            reader = HTTPRangeReader(url, session, size=content_length)
            gz_reader = bgzip.BGZipFile(fileobj=reader)
            line_iter = _iter_lines(gz_reader)
        else:
            line_iter = None
    else:
        line_iter = None

    if compression_info.get('is_bgzf') and not HAS_BGZIP_READER:
        print("BGZF support unavailable in bgzip module; falling back to streaming gzip reader.")

    if line_iter is None:
        print("Streaming from start because range/BGZF support is insufficient for targeted seeks...")
        resp = session.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        compressor = resp.headers.get('content-encoding', info.get('content_encoding', ''))
        reader = resp.raw
        if compression_info.get('is_bgzf') and HAS_BGZIP_READER:
            reader = bgzip.BGZipFile(fileobj=resp.raw)
        elif 'gzip' in compressor or 'gzip' in info.get('content_type', ''):
            reader = gzip.GzipFile(fileobj=resp.raw)
        line_iter = _iter_lines(reader)

    processed = 0
    for line in line_iter:
        if not line:
            continue
        processed += 1
        if processed % 100000 == 0:
            remaining = sum(len(v) for v in coords_by_chrom.values())
            print(f"Scanned {processed:,} rows; remaining targets: {remaining}/{total_targets}")

        parts = line.split('\t')
        if not header:
            header = parts
            continue
        chrom = parts[0]
        try:
            pos = int(parts[1])
        except (IndexError, ValueError):
            continue
        if pos in coords_by_chrom.get(chrom, set()):
            filtered_rows.append(parts)
            coords_by_chrom[chrom].discard(pos)
            if not coords_by_chrom[chrom]:
                coords_by_chrom.pop(chrom)
            remaining = sum(len(v) for v in coords_by_chrom.values())
            print(f"Found {chrom}:{pos}. Remaining targets: {remaining}")
            if remaining == 0:
                break

    return header, filtered_rows


def download_task(target, coords_by_chrom: Dict[str, set] | None = None):
    """
    Worker function.
    1. Checks stop_event.
    2. Attempts download.
    3. Handles 404 retry logic (290.0 -> 290).
    4. Returns Dataframe and size stats.
    """
    if stop_event.is_set():
        return None

    phenocode = str(target['phenocode'])
    phenostring = target['phenostring']

    # Logic for URL construction and retries
    urls_to_try = []

    def _add_url(url: str):
        if url not in urls_to_try:
            urls_to_try.append(url)

    zero_prefixed_codes: list[str] = []

    def _maybe_add_zero_variant(code: str):
        if not code:
            return

        # Queue "0{code}" and, if necessary, "00{code}" for sequential retries.
        # This follows the "prefix with a zero, and if that fails prefix with another" rule.
        if not code.startswith("0"):
            zero_prefixed_codes.extend([f"0{code}", f"00{code}"])
        elif not code.startswith("00"):
            zero_prefixed_codes.append(f"0{code}")

    _add_url(f"{BASE_PHEWEB_URL}{phenocode}")
    _maybe_add_zero_variant(phenocode)

    # If code looks like "290.0", add "290" as a fallback
    if phenocode.endswith(".0"):
        _add_url(f"{BASE_PHEWEB_URL}{phenocode[:-2]}")
        _maybe_add_zero_variant(phenocode[:-2])

    df = None
    bytes_downloaded = 0
    error_msg = ""

    for url in urls_to_try:
        try:
            with requests.Session() as session:
                info = _describe_remote(url, session)
                compression_info = _sniff_compression(url, session, info)

                coords_summary = "no coordinate filter"
                if coords_by_chrom:
                    remaining = sum(len(v) for v in coords_by_chrom.values())
                    coords_summary = f"{remaining} target position(s) across {len(coords_by_chrom)} chromosome(s)"
                print(
                    f"Preparing download for {phenocode} ({phenostring}); {coords_summary}. "
                    f"Range support: {info.get('accept_ranges')}, gzip: {compression_info.get('is_gzip')}, "
                    f"bgzf: {compression_info.get('is_bgzf')}"
                )

                if info.get('status') == '404':
                    error_msg = f"404 Not Found at {url}"
                    if zero_prefixed_codes:
                        zero_code = zero_prefixed_codes.pop(0)
                        _add_url(f"{BASE_PHEWEB_URL}{zero_code}")
                    continue

                try:
                    bytes_downloaded = int(info.get('content_length'))
                except (TypeError, ValueError):
                    bytes_downloaded = 0

                if coords_by_chrom:
                    coords_copy = {chrom: set(values) for chrom, values in coords_by_chrom.items()}
                    header, rows = _streaming_gzip_filter(
                        url,
                        coords_copy,
                        session,
                        info=info,
                        compression_info=compression_info,
                    )

                    if header:
                        df = pd.DataFrame(rows, columns=header)
                    else:
                        with session.get(url, stream=True, timeout=60) as resp:
                            resp.raise_for_status()
                            df = pd.read_csv(resp.raw, sep='\t', compression='gzip')
                else:
                    with session.get(url, stream=True, timeout=60) as resp:
                        if resp.status_code == 404:
                            error_msg = f"404 Not Found at {url}"
                            continue
                        resp.raise_for_status()

                        total_length = resp.headers.get('content-length')
                        if total_length:
                            try:
                                bytes_downloaded = int(total_length)
                            except ValueError:
                                bytes_downloaded = 0

                        df = pd.read_csv(resp.raw, sep='\t', compression='gzip')
                break
        except Exception as e:
            error_msg = str(e)
            continue

    if df is not None:
        # Add identifying columns
        df.insert(0, 'phenostring', phenostring)
        df.insert(1, 'phenocode', phenocode)
        return {'status': 'success', 'data': df, 'bytes': bytes_downloaded, 'phenocode': phenocode}
    else:
        return {'status': 'failed', 'error': error_msg, 'phenostring': phenostring, 'phenocode': phenocode}


def download_targets(targets_to_process, output_path: str, log_path: str, max_workers: int, coords_by_chrom: Dict[str, set] | None = None):
    print(f"Starting downloads (Output: {output_path})...")

    # Check if we need to write header (if output file doesn't exist)
    write_header = not os.path.exists(output_path)

    total_bytes_so_far = 0
    files_sized_so_far = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_task, t, coords_by_chrom): t for t in targets_to_process}

        pbar = tqdm(as_completed(futures), total=len(targets_to_process), unit="file")

        for future in pbar:
            if stop_event.is_set():
                break

            result = future.result()

            if not result:
                continue

            if result['status'] == 'success':
                df = result['data']
                phenocode = result['phenocode']
                file_bytes = result['bytes']

                if file_bytes > 0:
                    total_bytes_so_far += file_bytes
                    files_sized_so_far += 1

                    avg_size = total_bytes_so_far / files_sized_so_far
                    est_remaining_bytes = avg_size * (len(targets_to_process) - pbar.n)
                    est_gb = est_remaining_bytes / (1024 ** 3)

                    pbar.set_description(f"Est. Remaining: {est_gb:.2f} GB")

                try:
                    if not df.empty:
                        df.to_csv(output_path, sep='\t', mode='a', header=write_header, index=False)
                        write_header = False
                    append_to_log(phenocode, log_path)
                except Exception as e:
                    pbar.write(f"Error writing data for {phenocode}: {e}")

            else:
                pbar.write(f"Failed to download {result['phenostring']} ({result['phenocode']}): {result['error']}")

    print("\nProcessing finished.")


def chunk_targets(targets: Sequence[dict], shard_count: int) -> List[List[dict]]:
    shard_count = max(1, shard_count)
    if not targets:
        return [[] for _ in range(shard_count)]

    base_size = len(targets) // shard_count
    remainder = len(targets) % shard_count

    chunks = []
    start = 0
    for i in range(shard_count):
        extra = 1 if i < remainder else 0
        end = start + base_size + extra
        chunks.append(list(targets[start:end]))
        start = end
    return chunks


def plan_targets(targets: Sequence[dict], shard_count: int, plan_output: str) -> None:
    chunks = chunk_targets(targets, shard_count)
    plan_payload = {
        "total_targets": len(targets),
        "shards": [
            {
                "id": idx,
                "phenocodes": [str(target['phenocode']) for target in shard]
            }
            for idx, shard in enumerate(chunks)
        ],
    }

    Path(plan_output).write_text(json.dumps(plan_payload, indent=2))
    print(f"Wrote plan for {len(targets)} targets across {shard_count} shards to {plan_output}")


def aggregate_outputs(input_dir: Path, output_path: Path) -> None:
    tsv_files = sorted(input_dir.glob('**/*.tsv'))
    if not tsv_files:
        print(f"No TSV files found in {input_dir}; nothing to aggregate.")
        return

    frames = []
    for tsv_file in tsv_files:
        try:
            frames.append(pd.read_csv(tsv_file, sep='\t'))
        except Exception as exc:
            print(f"Skipping {tsv_file} due to read error: {exc}")

    if not frames:
        print("No readable TSV files found; aggregation aborted.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, sep='\t', index=False)
    print(f"Aggregated {len(frames)} file(s) into {output_path}")


def _split_identifiers(raw: str) -> List[str]:
    # Accept comma- or whitespace-delimited input in a single argument
    parts = []
    for chunk in raw.replace(",", " ").split():
        cleaned = chunk.strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def parse_requested_phenocodes(args: argparse.Namespace) -> List[str]:
    requested: List[str] = []

    for group in args.phenocode:
        for value in group:
            requested.extend(_split_identifiers(value))

    for value in args.phenocode_arg:
        requested.extend(_split_identifiers(value))

    if args.phenocode_file:
        requested.extend([line.strip() for line in Path(args.phenocode_file).read_text().splitlines() if line.strip()])

    return requested


def parse_requested_coords(args: argparse.Namespace) -> Dict[str, set]:
    requested: List[str] = []

    for value in args.coord:
        requested.extend(_split_identifiers(value))

    if args.coord_file:
        requested.extend([line.strip() for line in Path(args.coord_file).read_text().splitlines() if line.strip()])

    return _parse_positions(requested)


def load_best_tagging_coords(path: Path = BEST_TAGGING_SNPS) -> Dict[str, set]:
    coords: Dict[str, set] = defaultdict(set)
    if not path.exists():
        print(f"Best-tagging SNP list missing at {path}; downloading latest copy...")
        resp = requests.get(
            "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/best_tagging_snps_qvalues.tsv",
            timeout=60,
        )
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        print(f"Saved best-tagging SNP list to {path}")

    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            chrom = row.get('chromosome_hg37') or row.get('chromosome') or ''
            hg38_coord = row.get('hg38') or row.get('position_hg38') or ''
            hg38_coord = hg38_coord.strip()
            if not hg38_coord:
                continue

            if ':' in hg38_coord:
                chrom_part, pos_part = hg38_coord.split(':', 1)
                chrom = chrom_part.replace('chr', '') or chrom
            else:
                pos_part = hg38_coord

            if '-' in pos_part:
                pieces = [p for p in pos_part.split('-') if p]
            else:
                pieces = [pos_part]

            for piece in pieces:
                try:
                    coords[str(chrom).replace('chr', '')].add(int(float(piece)))
                except ValueError:
                    continue

    total = sum(len(v) for v in coords.values())
    print(f"Loaded {total} unique coordinate(s) from {path}")
    return coords


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PheWeb phenotype results")
    parser.add_argument('--phenocode', action='append', nargs='+', default=[], help='Phenocode or phenotype name to download (accepts multiple values per flag).')
    parser.add_argument('--phenocode-file', help='Path to a newline-delimited list of phenocodes or phenotype names.')
    parser.add_argument('phenocode_arg', nargs='*', default=[], help='Phenocode or phenotype name provided as positional arguments.')
    parser.add_argument('--output', default=OUTPUT_FILENAME, help='Output TSV path.')
    parser.add_argument('--log', default=LOG_FILENAME, help='Log file used to track completed phenocodes.')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS, help='Max concurrent download workers.')
    parser.add_argument('--coord', action='append', default=[], help='Chrom:pos coordinate to retain (comma or whitespace delimited).')
    parser.add_argument('--coord-file', help='Path to newline-delimited coordinates (chrom\tpos or chrom:pos).')

    parser.add_argument('--plan', action='store_true', help='Generate a plan JSON instead of downloading.')
    parser.add_argument('--plan-output', default=DEFAULT_PLAN_FILE, help='Where to write the JSON plan.')
    parser.add_argument('--shards', type=int, default=20, help='Number of shards to split downloads across when planning.')

    parser.add_argument('--aggregate', action='store_true', help='Aggregate TSV artifacts from a directory instead of downloading.')
    parser.add_argument('--input-dir', default='artifacts', help='Directory containing TSV files to aggregate.')

    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)

    if args.plan and args.aggregate:
        print("Cannot use --plan and --aggregate together.")
        sys.exit(1)

    if args.aggregate:
        aggregate_outputs(Path(args.input_dir), Path(args.output))
        return

    name_to_code = get_phenotype_mapping()

    requested_identifiers = parse_requested_phenocodes(args)
    coords_by_chrom = parse_requested_coords(args)
    if not coords_by_chrom:
        coords_by_chrom = load_best_tagging_coords()
    if not coords_by_chrom:
        print("Warning: no coordinate filter loaded; downloads will not be restricted to target SNPs.")
    user_provided_targets = bool(args.phenocode or args.phenocode_file)
    targets = resolve_requested_targets(name_to_code, requested_identifiers)

    if not targets and user_provided_targets:
        print("No valid phenocodes were provided; exiting.")
        return

    if not targets:
        targets = get_target_list(name_to_code)

    if args.plan:
        plan_targets(targets, args.shards, args.plan_output)
        return

    # Download mode
    completed_codes = load_completed_phenocodes(args.log)
    targets_to_process = [t for t in targets if str(t['phenocode']) not in completed_codes]

    print(f"\nTotal targets: {len(targets)}")
    print(f"Already completed: {len(completed_codes)}")
    print(f"Remaining to download: {len(targets_to_process)}")

    if not targets_to_process:
        print("Nothing left to do.")
        return

    download_targets(targets_to_process, args.output, args.log, args.max_workers, coords_by_chrom if coords_by_chrom else None)


if __name__ == "__main__":
    main()
