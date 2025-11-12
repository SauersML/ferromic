import os
import re
import sys
import io
import gzip
import glob
import time
import math
import shutil
import resource
import traceback
import logging
import requests
import multiprocessing
from collections import defaultdict
from contextlib import contextmanager

# =========================
# --- Configuration -----
# =========================

METADATA_FILE = 'phy_metadata.tsv'

# UCSC hg38 vs PanTro6 net AXT
AXT_URL = 'http://hgdownload.soe.ucsc.edu/goldenpath/hg38/vsPanTro6/hg38.panTro6.net.axt.gz'
AXT_GZ_FILENAME = 'hg38.panTro6.net.axt.gz'
AXT_FILENAME = 'hg38.PanTro6.net.axt'

# Divergence QC threshold (%)
DIVERGENCE_THRESHOLD = 10.0

# Debug: set to ENST id or to region key to print sequence snippet
DEBUG_TRANSCRIPT = None   # e.g., 'ENST00000367770.8'
DEBUG_REGION = None       # e.g., 'inv_7_60911891_61578023'

# Bin size (bp) for interval indexing over the genome (faster than per-base maps)
BIN_SIZE = int(os.environ.get("BIN_SIZE", "1000"))
VALID_BASES = {"A", "C", "G", "T"}

# Cache containers populated at runtime
_PHY_CACHE = {}
_BIN_INDEX = None

_AXT_HEADER_RE = re.compile(
    r"^(-?\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+([+-])\s+(\d+)$"
)

# Verbosity knobs
DEBUG_VERBOSE = os.environ.get("DEBUG_VERBOSE", "0") == "1"
DEBUG_CHUNK_SAMPLE = int(os.environ.get("DEBUG_CHUNK_SAMPLE", "0"))  # e.g., 1000

# =========================
# --- Logging Helpers ------
# =========================

LOG_DIR = os.environ.get("AXT_LOG_DIR", ".")
LOG_FILE_BASENAME = None
LOG_FILE_PATH = None
DETAIL_LOGGER = None


def setup_detail_logger():
    """Initialise the detailed run logger that writes to a .log file."""
    global LOG_FILE_BASENAME, LOG_FILE_PATH, DETAIL_LOGGER

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    LOG_FILE_BASENAME = f"axt_to_phy_{timestamp}.log"
    LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_BASENAME)

    os.makedirs(os.path.dirname(LOG_FILE_PATH) or ".", exist_ok=True)

    DETAIL_LOGGER = logging.getLogger("axt_to_phy.detail")
    DETAIL_LOGGER.setLevel(logging.INFO)
    DETAIL_LOGGER.handlers = []

    handler = logging.FileHandler(LOG_FILE_PATH, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    DETAIL_LOGGER.addHandler(handler)
    DETAIL_LOGGER.propagate = False

    return LOG_FILE_PATH


def log_detail(entity_type, identifier, status, message, **metrics):
    """Write a structured line to the detailed log."""
    if DETAIL_LOGGER is None:
        return

    identifier = identifier or "<unknown>"
    metric_parts = []
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            metric_parts.append(f"{key}={value:.4f}")
        else:
            metric_parts.append(f"{key}={value}")

    suffix = f" | {' '.join(metric_parts)}" if metric_parts else ""
    DETAIL_LOGGER.info("[%s] %s | %s | %s%s", entity_type, identifier, status, message, suffix)


def summarise_sequence(seq):
    """Return coverage statistics for a reconstructed sequence."""
    length = len(seq)
    if length == 0:
        return {
            "length": 0,
            "covered": 0,
            "coverage_pct": 0.0,
            "block_count": 0,
            "longest_block": 0,
            "longest_gap": 0,
            "first_covered": None,
            "last_covered": None,
        }

    covered = 0
    block_count = 0
    longest_block = 0
    longest_gap = 0
    current_block = 0
    current_gap = 0
    first_covered = None
    last_covered = None

    for idx, base in enumerate(seq):
        if base != '-':
            covered += 1
            last_covered = idx + 1  # 1-based for readability
            if first_covered is None:
                first_covered = idx + 1
            current_block += 1
            if current_gap:
                if current_gap > longest_gap:
                    longest_gap = current_gap
                current_gap = 0
            if current_block == 1:
                block_count += 1
        else:
            current_gap += 1
            if current_block:
                if current_block > longest_block:
                    longest_block = current_block
                current_block = 0

    if current_block and current_block > longest_block:
        longest_block = current_block
    if current_gap and current_gap > longest_gap:
        longest_gap = current_gap

    coverage_pct = (covered / length * 100.0) if length else 0.0

    return {
        "length": length,
        "covered": covered,
        "coverage_pct": coverage_pct,
        "block_count": block_count,
        "longest_block": longest_block,
        "longest_gap": longest_gap,
        "first_covered": first_covered,
        "last_covered": last_covered,
    }


def compute_alignment_metrics(human_seq, chimp_seq):
    """Return alignment statistics between two sequences."""

    aligned_letters = 0
    misaligned_letters = 0
    unaligned_letters = 0

    for h_raw, c_raw in zip(human_seq, chimp_seq):
        h = h_raw.upper()
        c = c_raw.upper()

        human_valid = h in VALID_BASES
        chimp_valid = c in VALID_BASES

        if human_valid and chimp_valid:
            aligned_letters += 1
            if h != c:
                misaligned_letters += 1
        elif human_valid or chimp_valid:
            unaligned_letters += 1

    misaligned_fraction = (misaligned_letters / aligned_letters) if aligned_letters else 0.0
    total_evaluable = aligned_letters + unaligned_letters
    unaligned_fraction = (unaligned_letters / total_evaluable) if total_evaluable else 0.0

    return {
        "aligned_letters": aligned_letters,
        "misaligned_letters": misaligned_letters,
        "unaligned_letters": unaligned_letters,
        "misaligned_fraction": misaligned_fraction,
        "unaligned_fraction": unaligned_fraction,
    }


EMPTY_ALIGNMENT_METRICS = {
    "aligned_letters": 0,
    "misaligned_letters": 0,
    "unaligned_letters": 0,
    "misaligned_fraction": 0.0,
    "unaligned_fraction": 0.0,
}


# =========================
# --- Simple Debug Utils ---
# =========================

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def print_dbg(msg):
    if DEBUG_VERBOSE:
        print(f"[{ts()}] [DEBUG] {msg}", flush=True)

def print_always(msg):
    print(f"[{ts()}] {msg}", flush=True)

def get_rss_kb():
    """Return RSS in kB (Linux), else None."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None

def get_fd_count():
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return None

def human_bytes(n):
    if n is None:
        return "unknown"
    units = ["B","KB","MB","GB","TB"]
    s = 0
    v = float(n)
    while v >= 1024 and s < len(units)-1:
        v /= 1024.0
        s += 1
    return f"{v:.1f} {units[s]}"

CHR_PREFIX = "chr"

def normalize_chromosome(raw):
    """Normalize chromosome identifiers to canonical 'chr*' form."""
    if raw is None:
        return None

    chrom = str(raw).strip()
    if not chrom:
        return None

    # Strip any repeated 'chr' prefixes (case-insensitive)
    while chrom.lower().startswith(CHR_PREFIX):
        chrom = chrom[3:]
        chrom = chrom.strip()
        if not chrom:
            return None

    chrom_lower = chrom.lower()

    if chrom_lower in {"m", "mt", "mtdna", "mito", "mitochondrial"}:
        core = "M"
    elif chrom_lower == "x":
        core = "X"
    elif chrom_lower == "y":
        core = "Y"
    elif chrom_lower.isdigit():
        try:
            core = str(int(chrom_lower))
        except ValueError:
            core = chrom_lower
    else:
        core = chrom.upper()

    return f"{CHR_PREFIX}{core}"

def chromosome_to_bare(chrom):
    """Return the non-'chr' portion of a chromosome label."""
    normalized = normalize_chromosome(chrom)
    if not normalized:
        return None
    return normalized[len(CHR_PREFIX):]

def progress_bar(label, done, total, width=40):
    if total <= 0:
        bar = "-" * width
        pct = 0
    else:
        filled = int(width * done // total)
        bar = "█" * filled + "-" * (width - filled)
        pct = int(done * 100 // total)
    print(f"\r{label} |{bar}| {done}/{total} ({pct}%)", end='', flush=True)

@contextmanager
def time_block(name):
    t0 = time.time()
    print_always(f"BEGIN: {name}")
    try:
        yield
    finally:
        dt = time.time() - t0
        print_always(f"END  : {name} [{dt:.2f}s]")

# =========================
# --- Logger --------------
# =========================

class Logger:
    """Collects warnings/notes and prints summary at end."""
    def __init__(self, max_prints=500):
        self.warnings = defaultdict(list)
        self.max_prints = max_prints

    def add(self, category, message):
        self.warnings[category].append(message)

    def report(self):
        print("\n--- Validation & Processing Summary ---")
        if not self.warnings:
            print("All checks passed without warnings.")
            return
        for category, messages in self.warnings.items():
            print(f"\nCategory '{category}': {len(messages)} total warnings/notifications.")
            for msg in sorted(messages)[:self.max_prints]:
                print(f"  - {msg}")
            if len(messages) > self.max_prints:
                print(f"  ... and {len(messages) - self.max_prints} more.")
        print("-" * 35)

logger = Logger()

# =========================
# --- Utilities -----------
# =========================

def download_axt_file():
    if os.path.exists(AXT_FILENAME) or os.path.exists(AXT_GZ_FILENAME):
        print_always("AXT file already present; skipping download.")
        return
    print_always(f"Downloading '{AXT_GZ_FILENAME}' from UCSC...")
    try:
        with requests.get(AXT_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('Content-Length', '0'))
            got = 0
            chunk = 8192 * 8
            with open(AXT_GZ_FILENAME, 'wb') as f:
                last_print = time.time()
                for block in r.iter_content(chunk_size=chunk):
                    if not block:
                        continue
                    f.write(block)
                    got += len(block)
                    if time.time() - last_print > 0.25:
                        if total:
                            progress_bar("[Download AXT]", got, total)
                        else:
                            print(f"\r[Download AXT] {human_bytes(got)}", end='', flush=True)
                        last_print = time.time()
            if total:
                progress_bar("[Download AXT]", total, total)
                print()
            else:
                print()
    except requests.exceptions.RequestException as e:
        print(f"\nFATAL: Error downloading file: {e}", flush=True)
        sys.exit(1)

def ungzip_file():
    if not os.path.exists(AXT_GZ_FILENAME):
        if not os.path.exists(AXT_FILENAME):
            print_always("FATAL: AXT file not found (neither .gz nor plain).")
            sys.exit(1)
        print_always("AXT .gz not found but plain exists; skipping decompression.")
        return
    if os.path.exists(AXT_FILENAME):
        print_always("AXT plain file exists; skipping decompression.")
        return

    print_always(f"Decompressing '{AXT_GZ_FILENAME}' -> '{AXT_FILENAME}' ...")
    try:
        size_in = os.path.getsize(AXT_GZ_FILENAME)
        done = 0
        chunk = 16 * 1024 * 1024
        with gzip.open(AXT_GZ_FILENAME, 'rb') as f_in, open(AXT_FILENAME, 'wb', buffering=chunk) as f_out:
            while True:
                buf = f_in.read(chunk)
                if not buf:
                    break
                f_out.write(buf)
                done += len(buf)
                progress_bar("[Ungzip AXT]", done, size_in if size_in else 1)
        progress_bar("[Ungzip AXT]", 1, 1)
        print()
    except Exception as e:
        print(f"\nFATAL: Error decompressing file: {e}", flush=True)
        sys.exit(1)

def read_phy_sequences(filename):
    """Reads all sequences from a simple PHYLIP file. Returns list[str]."""
    cached = _PHY_CACHE.get(filename)
    if cached is not None:
        return cached

    sequences = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                _PHY_CACHE[filename] = []
                return []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                m = re.search(r'[ACGTN-]+$', line, re.IGNORECASE)
                if m:
                    sequences.append(m.group(0).upper())
    except Exception:
        _PHY_CACHE[filename] = []
        return []

    _PHY_CACHE[filename] = sequences
    return sequences

# =========================
# --- Input: Transcripts --
# =========================

def parse_transcript_metadata():
    """
    Parses METADATA_FILE and validates group0/group1 .phy lengths for each transcript.
    Returns list of dicts:
        {'info': {...}, 'segments': [(start,end), ...]}
    """
    if not os.path.exists(METADATA_FILE):
        print_always(f"FATAL: Metadata file '{METADATA_FILE}' not found.")
        sys.exit(1)

    print_always("Validating transcript inputs against metadata...")
    # First count lines for progress bar
    with open(METADATA_FILE, 'r') as f:
        total_lines = sum(1 for _ in f) - 1
    total_lines = max(total_lines, 0)

    validated = []
    seen = set()
    processed = 0

    with open(METADATA_FILE, 'r') as f:
        next(f, None)  # skip header
        for line_num, line in enumerate(f, 2):
            processed += 1
            if processed % 50 == 0 or processed == total_lines:
                progress_bar("[Metadata]", processed, total_lines if total_lines else 1)
            parts = [p.strip() for p in line.strip().split('\t')]
            if len(parts) < 9:
                log_detail("CDS", f"line_{line_num}", "SKIP_INCOMPLETE", "Metadata line missing required columns.", raw=line.strip())
                continue

            phy_fname, t_id, gene, chrom, _, start, end, _, coords_str = parts[:9]
            cds_key = (t_id, coords_str)
            if cds_key in seen:
                log_detail("CDS", t_id, "SKIP_DUPLICATE", "Duplicate transcript/coordinate entry encountered; keeping first instance.",
                           coords=coords_str, line=line_num)
                continue
            seen.add(cds_key)

            chrom_norm = normalize_chromosome(chrom)
            if not chrom_norm:
                logger.add("Metadata Parsing Error", f"L{line_num}: Invalid chromosome '{chrom}' for {t_id}.")
                log_detail("CDS", t_id, "SKIP_INVALID_CHROM", f"Invalid chromosome '{chrom}'.", line=line_num)
                continue

            # Parse exon segments and expected length
            try:
                segments = [(int(s), int(e)) for s, e in (p.split('-') for p in coords_str.split(';'))]
                expected_len = sum(e - s + 1 for s, e in segments)
                if expected_len <= 0:
                    log_detail("CDS", t_id, "SKIP_ZERO_LENGTH", "Calculated expected length was <= 0.", coords=coords_str, line=line_num)
                    continue
            except (ValueError, IndexError):
                logger.add("Metadata Parsing Error", f"L{line_num}: Could not parse coordinate chunks for {t_id}.")
                log_detail("CDS", t_id, "SKIP_PARSE_COORDS", "Failed to parse coordinate chunks.", raw_coords=coords_str, line=line_num)
                continue

            # Find group0 and group1 filenames
            if "group0_" in phy_fname:
                g0_fname = phy_fname
                g1_fname = phy_fname.replace("group0_", "group1_")
            elif "group1_" in phy_fname:
                g1_fname = phy_fname
                g0_fname = phy_fname.replace("group1_", "group0_")
            else:
                base = os.path.basename(phy_fname)
                logger.add("Missing Input File", f"L{line_num}: Cannot infer group0/group1 for {t_id} from '{base}'.")
                log_detail("CDS", t_id, "SKIP_MISSING_GROUP_PAIR", f"Unable to infer group0/group1 partner from '{base}'.", line=line_num)
                continue

            g0_seqs = read_phy_sequences(g0_fname)
            g1_seqs = read_phy_sequences(g1_fname)

            if not g0_seqs or not g1_seqs:
                logger.add("Missing Input File", f"{t_id}: group0 or group1 .phy not found or empty.")
                log_detail("CDS", t_id, "SKIP_MISSING_PHY", "Either group0 or group1 .phy file missing/empty.",
                           group0=g0_fname, group1=g1_fname)
                continue

            if not all(len(s) == expected_len for s in g0_seqs):
                g0_lengths = set(len(s) for s in g0_seqs)
                logger.add("Input Length Mismatch", f"{t_id} (group0): lengths {g0_lengths} != expected ({expected_len}).")
                log_detail("CDS", t_id, "SKIP_LENGTH_MISMATCH", "Group0 lengths do not match expected length.",
                           observed=list(g0_lengths), expected=expected_len)
                continue

            if not all(len(s) == expected_len for s in g1_seqs):
                g1_lengths = set(len(s) for s in g1_seqs)
                logger.add("Input Length Mismatch", f"{t_id} (group1): lengths {g1_lengths} != expected ({expected_len}).")
                log_detail("CDS", t_id, "SKIP_LENGTH_MISMATCH", "Group1 lengths do not match expected length.",
                           observed=list(g1_lengths), expected=expected_len)
                continue

            cds_info = {
                'gene_name': gene,
                'transcript_id': t_id,
                'chromosome': chrom_norm,
                'expected_len': expected_len,
                'start': start,
                'end': end,
                'g0_fname': g0_fname,
                'g1_fname': g1_fname
            }
            validated.append({'info': cds_info, 'segments': segments})
            log_detail(
                "CDS",
                t_id,
                "VALIDATED",
                "Transcript metadata validated.",
                chromosome=chrom_norm,
                expected_len=expected_len,
                exons=len(segments),
                coords=coords_str,
            )

    progress_bar("[Metadata]", total_lines if total_lines else 1, total_lines if total_lines else 1)
    print()
    print_dbg(f"Parsed metadata entries: {len(validated)}")
    log_detail("SUMMARY", "transcripts_metadata", "DONE", "Transcript metadata parsed.", validated=len(validated), processed=processed)
    return validated

# =========================
# --- Input: Regions ------
# =========================

REGION_REGEX = re.compile(
    r'^inversion_(group(?P<grp>[01]))_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$'
)

def find_region_sets():
    """
    Scans inversion region PHYLIPs and does header-only length check.
    Returns list of dicts similar to transcripts.
    """
    print_always("Scanning for inversion region PHYLIP files...")
    files = glob.glob('inversion_group[01]_*_start*_end*.phy')
    groups = defaultdict(dict)  # key: (chrom, start, end) -> {'group0': path, 'group1': path}

    for path in files:
        name = os.path.basename(path)
        m = REGION_REGEX.match(name)
        if not m:
            continue
        chrom = normalize_chromosome(m.group('chrom'))
        if not chrom:
            logger.add("Region Parsing Error", f"Invalid chromosome '{m.group('chrom')}' in {name}; skipping group.")
            log_detail("INVERSION", name, "SKIP_INVALID_CHROM", f"Invalid chromosome '{m.group('chrom')}' in filename.")
            continue
        start = int(m.group('start'))
        end = int(m.group('end'))
        grp = m.group('grp')
        key = (chrom, start, end)
        groups[key][f'group{grp}'] = path

    validated = []
    total = len(groups)
    processed = 0
    bar_width = 40

    print_dbg(f"Region candidate groups: {total}")
    for (chrom, start, end), d in groups.items():
        expected_len = end - start + 1
        chrom_bare = chromosome_to_bare(chrom)
        region_id = f"inv_{chrom_bare}_{start}_{end}" if chrom_bare else f"inv_{chrom}_{start}_{end}"
        info = {
            'region_id': region_id,
            'chromosome': chrom,
            'expected_len': expected_len,
            'start': str(start),
            'end': str(end),
            'g0_fname': d.get('group0'),
            'g1_fname': d.get('group1'),
        }

        log_detail(
            "INVERSION",
            region_id,
            "VALIDATED",
            "Region metadata parsed.",
            chromosome=chrom,
            expected_len=expected_len,
            group0=bool(info['g0_fname']),
            group1=bool(info['g1_fname']),
        )

        qc_fname = info['g0_fname'] or info['g1_fname']
        if not qc_fname:
            logger.add("Region Missing File", f"{region_id}: neither group0 nor group1 file present; skipping QC.")
            log_detail("INVERSION", region_id, "QC_WARNING", "Neither group0 nor group1 file present for QC header check.")
        else:
            try:
                with open(qc_fname, 'r') as f:
                    first = f.readline().strip()
                mlen = re.match(r'\s*\d+\s+(\d+)\s*$', first)
                if not mlen:
                    logger.add("Region QC Warning", f"{region_id}: could not parse header length in {os.path.basename(qc_fname)}.")
                    log_detail("INVERSION", region_id, "QC_WARNING", "Unable to parse PHYLIP header length.", file=os.path.basename(qc_fname))
                else:
                    header_len = int(mlen.group(1))
                    if header_len != expected_len:
                        logger.add("Region Input Length Mismatch", f"{region_id}: header length {header_len} != expected ({expected_len}).")
                        log_detail(
                            "INVERSION",
                            region_id,
                            "QC_LENGTH_MISMATCH",
                            "Header length did not match expected region length.",
                            header_len=header_len,
                            expected_len=expected_len,
                            file=os.path.basename(qc_fname),
                        )
            except Exception:
                logger.add("Region QC Warning", f"{region_id}: failed to read header from {os.path.basename(qc_fname)}.")
                log_detail(
                    "INVERSION",
                    region_id,
                    "QC_WARNING",
                    "Failed to read header for QC.",
                    file=os.path.basename(qc_fname) if qc_fname else None,
                )

        validated.append({'info': info, 'segments': [(start, end)]})

        processed += 1
        progress_bar("[Region QC]", processed, total if total else 1)

    if total > 0:
        progress_bar("[Region QC]", total, total)
    print()
    print_always(f"Found {len(validated)} candidate regions.")
    log_detail("SUMMARY", "regions_metadata", "DONE", "Region metadata parsed.", validated=len(validated), scanned=len(files))
    return validated

# =========================
# --- Interval Index ------
# =========================

def _bin_range(start, end, bin_size):
    """Yield bin ids covered by [start, end] inclusive (1-based coords)."""
    a = max(0, start - 1)
    b = end
    first = a // bin_size
    last = (b - 1) // bin_size
    for k in range(first, last + 1):
        yield k

def build_bin_index(transcripts, regions):
    """
    Builds per-chromosome bin index: index[chrom][bin_id] -> list(records)
    record = (id, seg_start, seg_end, offset)
    """
    print_always("Building bin index (overlap-aware) for transcripts and regions...")
    t0 = time.time()
    index = {}  # chrom -> bin -> [records]
    tx_info_map = {}
    rg_info_map = {}

    # Transcripts
    total_tx = sum(len(t['segments']) for t in transcripts)
    done_tx = 0
    for t in transcripts:
        info = t['info']
        chrom = info['chromosome']
        t_id = info['transcript_id']
        tx_info_map[t_id] = info
        offset = 0
        for s, e in t['segments']:
            chrom_bins = index.setdefault(chrom, {})
            for b in _bin_range(s, e, BIN_SIZE):
                chrom_bins.setdefault(b, []).append((t_id, s, e, offset))
            offset += (e - s + 1)
            done_tx += 1
            if done_tx % 50 == 0 or done_tx == total_tx:
                progress_bar("[BinIndex TX]", done_tx, total_tx if total_tx else 1)
    if total_tx:
        progress_bar("[BinIndex TX]", total_tx, total_tx)
        print()

    # Regions
    total_rg = len(regions)
    done_rg = 0
    for r in regions:
        info = r['info']
        chrom = info['chromosome']
        r_id = info['region_id']
        rg_info_map[r_id] = info
        (s, e) = r['segments'][0]
        chrom_bins = index.setdefault(chrom, {})
        for b in _bin_range(s, e, BIN_SIZE):
            chrom_bins.setdefault(b, []).append((r_id, s, e, 0))
        done_rg += 1
        if done_rg % 20 == 0 or done_rg == total_rg:
            progress_bar("[BinIndex RG]", done_rg, total_rg if total_rg else 1)
    if total_rg:
        progress_bar("[BinIndex RG]", total_rg, total_rg)
        print()

    dt = time.time() - t0
    # Quick size stats
    chrom_stats = {c: len(bins) for c, bins in index.items()}
    print_dbg(f"Bin index built in {dt:.2f}s; chrom bins: {chrom_stats}")
    rss = get_rss_kb()
    print_always(f"Bin index memory snapshot: RSS ~ {rss} KB" if rss else "Bin index memory snapshot: RSS unknown")
    return index, tx_info_map, rg_info_map

# =========================
# --- AXT Processing -------
# =========================

def _parse_axt_header_line(line):
    match = _AXT_HEADER_RE.match(line)
    if not match:
        return None
    try:
        score, t_name, t_start, t_end, q_name, q_start, q_end, strand, q_size = match.groups()
        return (
            int(score),
            normalize_chromosome(t_name),
            int(t_start),
            int(t_end),
            q_name,
            int(q_start),
            int(q_end),
            strand,
            int(q_size),
        )
    except ValueError:
        return None


def process_axt_chunk(chunk_start, chunk_end, bin_index=None):
    """
    Worker to parse a slice of the AXT file and collect chimp bases.
    Returns dict: id -> {target_idx: base}
    """
    if bin_index is None:
        bin_index = _BIN_INDEX
    if bin_index is None:
        raise RuntimeError("Bin index not initialized for worker process.")

    bin_index_get = bin_index.get
    results = defaultdict(dict)  # id -> {pos_idx: base}
    parsed_headers = 0
    try:
        with open(AXT_FILENAME, 'r', buffering=1024*1024) as f:
            f.seek(chunk_start)
            if chunk_start != 0:
                f.readline()  # align to line boundary

            while True:
                line_start = f.tell()
                if chunk_end is not None and line_start >= chunk_end:
                    break

                header_line = f.readline()
                if not header_line:
                    break
                header_line = header_line.strip()
                if not header_line:
                    continue

                parsed = _parse_axt_header_line(header_line)
                if not parsed:
                    continue

                (
                    _score,
                    axt_chr,
                    human_start,
                    _human_end,
                    _q_name,
                    _q_start,
                    _q_end,
                    _strand,
                    _q_size,
                ) = parsed

                human_pos = human_start

                human_seq = f.readline()
                chimp_seq = f.readline()
                if not human_seq or not chimp_seq:
                    break

                human_seq = human_seq.strip().upper()
                chimp_seq = chimp_seq.strip().upper()

                parsed_headers += 1
                if not axt_chr:
                    continue
                if DEBUG_CHUNK_SAMPLE and (parsed_headers % DEBUG_CHUNK_SAMPLE == 0):
                    print_dbg(
                        f"Worker chunk[{chunk_start}:{chunk_end}] parsed {parsed_headers} blocks (tell={f.tell()})"
                    )

                chrom_bins = bin_index_get(axt_chr)
                if not chrom_bins:
                    continue

                for h_char, c_char in zip(human_seq, chimp_seq):
                    if h_char != '-':
                        bin_id = (human_pos - 1) // BIN_SIZE
                        records = chrom_bins.get(bin_id)
                        if records:
                            for ident, seg_start, seg_end, offset in records:
                                if seg_start <= human_pos <= seg_end:
                                    target_idx = offset + (human_pos - seg_start)
                                    if target_idx not in results[ident]:
                                        results[ident][target_idx] = c_char
                        human_pos += 1

    except Exception as e:
        # Return an error sentinel
        return {"__error__": f"{e.__class__.__name__}: {e}", "__trace__": traceback.format_exc(),
                "__chunk__": (chunk_start, chunk_end), "__parsed__": parsed_headers}

    return dict(results)


def _init_worker(bin_index):
    """Initializer for worker processes to set the global bin index."""
    global _BIN_INDEX
    _BIN_INDEX = bin_index


def process_axt_chunk_worker(args):
    """Wrapper to unpack chunk arguments for pool workers."""
    chunk_start, chunk_end = args
    return process_axt_chunk(chunk_start, chunk_end)

def _workers_cap():
    try:
        cpu = len(os.sched_getaffinity(0))
    except Exception:
        cpu = multiprocessing.cpu_count()
    env = os.environ.get("AXT_WORKERS")
    if env:
        try:
            want = max(1, int(env))
            return min(want, cpu)
        except ValueError:
            pass
    return cpu


def _merge_partial_results(res, tx_scaffolds, rg_scaffolds):
    for ident, posmap in res.items():
        if ident in tx_scaffolds:
            sc = tx_scaffolds[ident]
        elif ident in rg_scaffolds:
            sc = rg_scaffolds[ident]
        else:
            continue
        for pos_idx, base in posmap.items():
            if 0 <= pos_idx < len(sc) and sc[pos_idx] == '-':
                sc[pos_idx] = base

def _print_system_limits():
    pid = os.getpid()
    rss = get_rss_kb()
    fds = get_fd_count()
    nproc = resource.getrlimit(resource.RLIMIT_NPROC)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    print_always(f"Process PID={pid} | RSS={rss} KB | FDs={fds} | RLIMIT_NPROC={nproc} | RLIMIT_NOFILE={nofile}")

def _chunk_plan(file_size, n_workers):
    if n_workers <= 1 or file_size == 0:
        return [(0, None)]

    approx_size = max(1, file_size // n_workers)
    targets = [approx_size * i for i in range(1, n_workers)]
    boundaries = [0]

    with open(AXT_FILENAME, 'r', buffering=1024 * 1024) as f:
        for target in targets:
            f.seek(target)
            f.readline()  # finish current line

            while True:
                pos = f.tell()
                if pos >= file_size:
                    boundaries.append(file_size)
                    break
                line = f.readline()
                if not line:
                    boundaries.append(file_size)
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if _parse_axt_header_line(stripped):
                    boundaries.append(pos)
                    break

            if boundaries[-1] >= file_size:
                break

    if boundaries[-1] != file_size:
        boundaries.append(file_size)

    deduped = []
    last = None
    for b in boundaries:
        if last is None or b > last:
            deduped.append(b)
            last = b

    ranges = []
    for i in range(len(deduped) - 1):
        start = deduped[i]
        end = deduped[i + 1]
        if start < end:
            ranges.append((start, end))

    if not ranges:
        return [(0, None)]

    last_start, last_end = ranges[-1]
    ranges[-1] = (last_start, None)
    return ranges

def build_outgroups_and_filter(transcripts, regions):
    """
    Build chimp sequences for transcripts (CDS) and regions (inversions) using AXT.
    Apply divergence QC and write .phy outgroups for both sets.
    """
    if not transcripts and not regions:
        print_always("No transcript or region entries to process.")
        log_detail("SYSTEM", "build_outgroups", "SKIP", "No transcripts or regions available after validation.")
        return

    # Build bin index
    bin_index, tx_info_map, rg_info_map = build_bin_index(transcripts, regions)

    # Create empty scaffolds
    tx_scaffolds = {t['info']['transcript_id']: ['-'] * t['info']['expected_len'] for t in transcripts}
    rg_scaffolds = {r['info']['region_id']: ['-'] * r['info']['expected_len'] for r in regions}

    print_always(f"Processing '{AXT_FILENAME}' in parallel (process pool)...")
    if not os.path.exists(AXT_FILENAME):
        print_always("FATAL: AXT plain file missing.")
        sys.exit(1)

    file_size = os.path.getsize(AXT_FILENAME)
    if file_size == 0:
        print_always("FATAL: AXT file is empty.")
        sys.exit(1)

    # Worker count + system info
    workers = _workers_cap()
    try:
        cpu_all = len(os.sched_getaffinity(0))
    except Exception:
        cpu_all = multiprocessing.cpu_count()
    print_always(f"CPU detected: {cpu_all} | Planned workers: {workers} (override with AXT_WORKERS)")
    _print_system_limits()

    # Chunking plan
    chunk_ranges = _chunk_plan(file_size, workers)
    print_dbg(f"AXT file size: {human_bytes(file_size)}; chunk ranges (first 5): {chunk_ranges[:5]}")

    # Create pool of worker processes
    print_always(f"Creating process pool with {workers} workers ...")
    t_pool_create0 = time.time()
    pool = multiprocessing.Pool(processes=workers, initializer=_init_worker, initargs=(bin_index,))
    t_pool_create1 = time.time()
    print_always(f"Process pool ready ({workers} workers) in {t_pool_create1 - t_pool_create0:.2f}s.")

    # Kick off work
    t0 = time.time()
    print_always(f"[AXT parse] START — scheduling {len(chunk_ranges)} chunks")
    progress_bar("[AXT parse]", 0, len(chunk_ranges))

    try:
        completed = 0
        for res in pool.imap_unordered(process_axt_chunk_worker, chunk_ranges, chunksize=1):
            completed += 1
            progress_bar("[AXT parse]", completed, len(chunk_ranges))
            # Handle worker error sentinel
            if isinstance(res, dict) and "__error__" in res:
                print("\n[AXT parse][WORKER ERROR]")
                print(res["__error__"])
                print(res.get("__trace__", ""))
                print(f"Chunk: {res.get('__chunk__')}, parsed headers before error: {res.get('__parsed__')}")
                continue
            _merge_partial_results(res, tx_scaffolds, rg_scaffolds)
    finally:
        pool.close()
        pool.join()

    progress_bar("[AXT parse]", len(chunk_ranges), len(chunk_ranges))
    print()
    print_always(f"Finished parallel AXT processing in {time.time() - t0:.2f} seconds.")

    # --- Write transcripts ---
    print_always("Writing transcript outgroups (after divergence QC)...")
    with time_block("Write transcript outgroups"):
        tx_written = 0
        total_tx = len(transcripts)
        for i, t in enumerate(transcripts, 1):
            info = t['info']
            t_id = info['transcript_id']
            gene = info['gene_name']
            chrom = info['chromosome']
            chrom_label = chromosome_to_bare(chrom) or chrom
            start = info['start']
            end = info['end']
            g0_fname = info['g0_fname']

            seq_list = tx_scaffolds.get(t_id)
            if seq_list is None:
                logger.add("No Alignment Found", f"No chimp alignment found for {t_id}.")
                print_always(f"[TX][{t_id}] ERROR: chimp scaffold missing.")
                log_detail(
                    "CDS",
                    t_id,
                    "NO_SCAFFOLD",
                    "Chimp scaffold missing after AXT processing.",
                    expected_len=info['expected_len'],
                    aligned_letters=0,
                    misaligned_fraction=0.0,
                    unaligned_fraction=0.0,
                )
                progress_bar("[TX write]", i, total_tx)
                continue
            final_seq = "".join(seq_list)
            seq_stats = summarise_sequence(final_seq)
            span_label = (
                f"{seq_stats['first_covered']}-{seq_stats['last_covered']}"
                if seq_stats['first_covered'] is not None
                else "none"
            )
            print_always(
                f"[TX][{t_id}] coverage={seq_stats['coverage_pct']:.2f}% "
                f"({seq_stats['covered']}/{seq_stats['length']} bp) blocks={seq_stats['block_count']} "
                f"longest_block={seq_stats['longest_block']} longest_gap={seq_stats['longest_gap']} span={span_label}"
            )

            if seq_stats['covered'] == 0:
                logger.add("No Alignment Found", f"No chimp alignment found for {t_id}.")
                log_detail(
                    "CDS",
                    t_id,
                    "NO_ALIGNMENT",
                    "No chimp overlap detected across transcript.",
                    coverage_pct=seq_stats['coverage_pct'],
                    expected_len=seq_stats['length'],
                    aligned_letters=0,
                    misaligned_fraction=0.0,
                    unaligned_fraction=0.0,
                )
                progress_bar("[TX write]", i, total_tx)
                continue

            # Divergence QC vs group0 reference (first sequence)
            human_seqs = read_phy_sequences(g0_fname)
            if not human_seqs:
                logger.add("Human File Missing for QC", f"Could not read human seqs from {g0_fname} for divergence check on {t_id}.")
                print_always(f"[TX][{t_id}] QC SKIP: missing human reference ({g0_fname}).")
                log_detail(
                    "CDS",
                    t_id,
                    "QC_HUMAN_MISSING",
                    "Human reference sequence missing for divergence QC.",
                    coverage_pct=seq_stats['coverage_pct'],
                    expected_len=seq_stats['length'],
                    span=span_label,
                    aligned_letters=0,
                    misaligned_fraction=0.0,
                    unaligned_fraction=0.0,
                )
                progress_bar("[TX write]", i, total_tx)
                continue
            human_ref = human_seqs[0]
            metrics = compute_alignment_metrics(human_ref, final_seq)
            divergence = metrics["misaligned_fraction"] * 100

            outname = f"outgroup_{gene}_{t_id}_{chrom_label}_start{start}_end{end}.phy"
            if divergence > DIVERGENCE_THRESHOLD:
                logger.add("QC Filter: High Divergence", f"'{gene} ({t_id})' removed. Divergence vs chimp: {divergence:.2f}% (> {DIVERGENCE_THRESHOLD}%).")
                if os.path.exists(outname):
                    try:
                        os.remove(outname)
                    except Exception:
                        pass
                print_always(
                    f"[TX][{t_id}] FAIL divergence {divergence:.2f}% (threshold {DIVERGENCE_THRESHOLD:.2f}%) — file removed."
                )
                log_detail(
                    "CDS",
                    t_id,
                    "FILTER_HIGH_DIVERGENCE",
                    "Failed divergence QC; output removed.",
                    coverage_pct=seq_stats['coverage_pct'],
                    divergence=divergence,
                    threshold=DIVERGENCE_THRESHOLD,
                    span=span_label,
                    filled=seq_stats['covered'],
                    aligned_letters=metrics["aligned_letters"],
                    misaligned_fraction=metrics["misaligned_fraction"],
                    unaligned_fraction=metrics["unaligned_fraction"],
                )
                progress_bar("[TX write]", i, total_tx)
                continue

            if DEBUG_TRANSCRIPT == t_id:
                print_always(f"\n--- DEBUG TX {t_id} --- len={len(final_seq)}\n{final_seq[:120]}...\n")

            with open(outname, 'w') as f_out:
                f_out.write(f" 1 {len(final_seq)}\n")
                f_out.write(f"{'PanTro6':<10}{final_seq}\n")
            tx_written += 1
            print_always(f"[TX][{t_id}] PASS divergence {divergence:.2f}% — wrote {outname}.")
            log_detail(
                "CDS",
                t_id,
                "PASS",
                "Outgroup written after passing divergence QC.",
                coverage_pct=seq_stats['coverage_pct'],
                divergence=divergence,
                span=span_label,
                filled=seq_stats['covered'],
                longest_block=seq_stats['longest_block'],
                longest_gap=seq_stats['longest_gap'],
                out_file=outname,
                aligned_letters=metrics["aligned_letters"],
                misaligned_fraction=metrics["misaligned_fraction"],
                unaligned_fraction=metrics["unaligned_fraction"],
            )
            progress_bar("[TX write]", i, total_tx)
        if total_tx:
            progress_bar("[TX write]", total_tx, total_tx)
            print()
        print_always(f"Wrote {tx_written} transcript outgroup PHYLIPs (passed QC).")
        log_detail("SUMMARY", "transcripts", "DONE", "Transcript processing finished.", passed=tx_written, total=total_tx)

    # --- Write regions ---
    print_always("Writing region outgroups (after divergence QC)...")
    with time_block("Write region outgroups"):
        rg_written = 0
        total_rg = len(regions)
        for i, r in enumerate(regions, 1):
            info = r['info']
            r_id = info['region_id']              # inv_<chrom>_<start>_<end>
            chrom_label = chromosome_to_bare(info['chromosome']) or info['chromosome']
            start = info['start']
            end = info['end']
            g0_fname = info['g0_fname'] or info['g1_fname']

            seq_list = rg_scaffolds.get(r_id)
            if seq_list is None:
                logger.add("No Alignment Found (Region)", f"No chimp alignment found for {r_id}.")
                print_always(f"[RG][{r_id}] ERROR: chimp scaffold missing.")
                log_detail(
                    "INVERSION",
                    r_id,
                    "NO_SCAFFOLD",
                    "Chimp scaffold missing after AXT processing.",
                    expected_len=info['expected_len'],
                    aligned_letters=0,
                    misaligned_fraction=0.0,
                    unaligned_fraction=0.0,
                )
                progress_bar("[RG write]", i, total_rg)
                continue
            final_seq = "".join(seq_list)
            seq_stats = summarise_sequence(final_seq)
            span_label = (
                f"{seq_stats['first_covered']}-{seq_stats['last_covered']}"
                if seq_stats['first_covered'] is not None
                else "none"
            )
            print_always(
                f"[RG][{r_id}] coverage={seq_stats['coverage_pct']:.2f}% "
                f"({seq_stats['covered']}/{seq_stats['length']} bp) blocks={seq_stats['block_count']} "
                f"longest_block={seq_stats['longest_block']} longest_gap={seq_stats['longest_gap']} span={span_label}"
            )

            if seq_stats['covered'] == 0:
                logger.add("No Alignment Found (Region)", f"No chimp alignment found for {r_id}.")
                log_detail(
                    "INVERSION",
                    r_id,
                    "NO_ALIGNMENT",
                    "No chimp overlap detected across region.",
                    coverage_pct=seq_stats['coverage_pct'],
                    expected_len=seq_stats['length'],
                    aligned_letters=0,
                    misaligned_fraction=0.0,
                    unaligned_fraction=0.0,
                )
                progress_bar("[RG write]", i, total_rg)
                continue

            # Divergence QC vs human reference (group0 preferred)
            metrics = EMPTY_ALIGNMENT_METRICS.copy()
            if not g0_fname:
                logger.add("Region File Missing for QC", f"{r_id}: no group file for divergence check; skipping QC.")
                divergence = 0.0
                log_detail(
                    "INVERSION",
                    r_id,
                    "QC_HUMAN_MISSING",
                    "No human reference file available for divergence QC.",
                    coverage_pct=seq_stats['coverage_pct'],
                    expected_len=seq_stats['length'],
                    span=span_label,
                    aligned_letters=metrics["aligned_letters"],
                    misaligned_fraction=metrics["misaligned_fraction"],
                    unaligned_fraction=metrics["unaligned_fraction"],
                )
            else:
                human_seqs = read_phy_sequences(g0_fname)
                if not human_seqs:
                    logger.add("Region File Missing for QC", f"{r_id}: cannot read {os.path.basename(g0_fname)}; skipping QC.")
                    divergence = 0.0
                    log_detail(
                        "INVERSION",
                        r_id,
                        "QC_HUMAN_MISSING",
                        "Human reference sequence missing for divergence QC.",
                        coverage_pct=seq_stats['coverage_pct'],
                        expected_len=seq_stats['length'],
                        span=span_label,
                        aligned_letters=metrics["aligned_letters"],
                        misaligned_fraction=metrics["misaligned_fraction"],
                        unaligned_fraction=metrics["unaligned_fraction"],
                    )
                else:
                    human_ref = human_seqs[0]
                    metrics = compute_alignment_metrics(human_ref, final_seq)
                    divergence = metrics["misaligned_fraction"] * 100

            outname = f"outgroup_inversion_{chrom_label}_start{start}_end{end}.phy"
            if divergence > DIVERGENCE_THRESHOLD:
                logger.add("QC Filter: High Divergence (Region)", f"{r_id} removed. Divergence vs chimp: {divergence:.2f}% (> {DIVERGENCE_THRESHOLD}%).")
                if os.path.exists(outname):
                    try:
                        os.remove(outname)
                    except Exception:
                        pass
                print_always(
                    f"[RG][{r_id}] FAIL divergence {divergence:.2f}% (threshold {DIVERGENCE_THRESHOLD:.2f}%) — file removed."
                )
                log_detail(
                    "INVERSION",
                    r_id,
                    "FILTER_HIGH_DIVERGENCE",
                    "Failed divergence QC; output removed.",
                    coverage_pct=seq_stats['coverage_pct'],
                    divergence=divergence,
                    threshold=DIVERGENCE_THRESHOLD,
                    span=span_label,
                    filled=seq_stats['covered'],
                    aligned_letters=metrics["aligned_letters"],
                    misaligned_fraction=metrics["misaligned_fraction"],
                    unaligned_fraction=metrics["unaligned_fraction"],
                )
                progress_bar("[RG write]", i, total_rg)
                continue

            if DEBUG_REGION == r_id:
                print_always(f"\n--- DEBUG RG {r_id} --- len={len(final_seq)}\n{final_seq[:120]}...\n")

            with open(outname, 'w') as f_out:
                f_out.write(f" 1 {len(final_seq)}\n")
                f_out.write(f"{'PanTro6':<10}{final_seq}\n")
            rg_written += 1
            print_always(f"[RG][{r_id}] PASS divergence {divergence:.2f}% — wrote {outname}.")
            log_detail(
                "INVERSION",
                r_id,
                "PASS",
                "Outgroup written after passing divergence QC.",
                coverage_pct=seq_stats['coverage_pct'],
                divergence=divergence,
                span=span_label,
                filled=seq_stats['covered'],
                longest_block=seq_stats['longest_block'],
                longest_gap=seq_stats['longest_gap'],
                out_file=outname,
                aligned_letters=metrics["aligned_letters"],
                misaligned_fraction=metrics["misaligned_fraction"],
                unaligned_fraction=metrics["unaligned_fraction"],
            )
            progress_bar("[RG write]", i, total_rg)
        if total_rg:
            progress_bar("[RG write]", total_rg, total_rg)
            print()
        print_always(f"Wrote {rg_written} region outgroup PHYLIPs (passed QC).")
        log_detail("SUMMARY", "regions", "DONE", "Region processing finished.", passed=rg_written, total=total_rg)

# =========================
# --- Fixed-diff stats ----
# =========================

def calculate_and_print_differences_transcripts():
    print_always("--- Final Difference Calculation & Statistics (Transcripts) ---")
    key_regex = re.compile(r"(ENST[0-9]+\.[0-9]+)_([^_]+)_start([0-9]+)_end([0-9]+)")
    cds_groups = defaultdict(dict)
    all_phys = glob.glob('*.phy')

    # Scan files with progress
    total_files = len(all_phys)
    for i, fpath in enumerate(all_phys, 1):
        base = os.path.basename(fpath)
        m = key_regex.search(base)
        if m:
            t_id, chrom_token, start_token, end_token = m.groups()
            chrom_norm = normalize_chromosome(chrom_token)
            if not chrom_norm:
                logger.add("Stats Parsing Error", f"Skipped file '{base}' due to invalid chromosome '{chrom_token}'.")
                continue
            key = (t_id, chrom_norm, start_token, end_token)
            cds_groups[key][base.split('_')[0]] = fpath
        if i % 25 == 0 or i == total_files:
            progress_bar("[TX stats: scan]", i, total_files if total_files else 1)
    if total_files:
        progress_bar("[TX stats: scan]", total_files, total_files)
        print()

    total_fixed_diffs = 0
    g0_matches = 0
    g1_matches = 0
    per_tx_g0 = {}
    per_tx_g1 = {}
    comparable_sets = 0

    keys = list(cds_groups.items())
    total_keys = len(keys)
    print_dbg(f"Comparable TX groups detected (pre-filter): {total_keys}")

    print_always("Analyzing each comparable transcript set (passed QC)...")
    for idx, (identifier, files) in enumerate(keys, 1):
        if {'group0', 'group1', 'outgroup'}.issubset(files.keys()):
            g0_seqs = read_phy_sequences(files['group0'])
            g1_seqs = read_phy_sequences(files['group1'])
            out_seq_list = read_phy_sequences(files['outgroup'])
            if not out_seq_list:
                progress_bar("[TX stats]", idx, total_keys if total_keys else 1)
                continue
            out_seq = out_seq_list[0]

            g0_len = set(len(s) for s in g0_seqs)
            g1_len = set(len(s) for s in g1_seqs)
            if not (len(g0_len) == 1 and len(g1_len) == 1):
                logger.add("Intra-file Length Mismatch", f"Not all sequences in a .phy have same length for {identifier[0]}.")
                progress_bar("[TX stats]", idx, total_keys if total_keys else 1)
                continue

            L0 = g0_len.pop()
            L1 = g1_len.pop()
            if L0 != L1 or L0 != len(out_seq):
                logger.add("Final Comparison Error", f"Length mismatch between groups for {identifier[0]}.")
                progress_bar("[TX stats]", idx, total_keys if total_keys else 1)
                continue

            comparable_sets += 1
            n = L0
            t_id = identifier[0]
            # Extract gene name from filename of group0 (2nd token)
            gene_name = os.path.basename(files['group0']).split('_')[1]

            local_fd = 0
            local_g0 = 0
            local_g1 = 0

            for i in range(n):
                g0_alleles = {s[i] for s in g0_seqs if s[i] != '-'}
                g1_alleles = {s[i] for s in g1_seqs if s[i] != '-'}
                if len(g0_alleles) == 1 and len(g1_alleles) == 1 and g0_alleles != g1_alleles:
                    local_fd += 1
                    total_fixed_diffs += 1
                    g0_a = next(iter(g0_alleles))
                    g1_a = next(iter(g1_alleles))
                    chimp_a = out_seq[i]
                    if chimp_a == g0_a:
                        g0_matches += 1
                        local_g0 += 1
                    elif chimp_a == g1_a:
                        g1_matches += 1
                        local_g1 += 1

            if local_fd > 0:
                key = f"{gene_name} ({t_id})"
                per_tx_g0[key] = (local_g0 / local_fd) * 100.0
                per_tx_g1[key] = (local_g1 / local_fd) * 100.0

        progress_bar("[TX stats]", idx, total_keys if total_keys else 1)

    if total_keys:
        progress_bar("[TX stats]", total_keys, total_keys)
        print()

    if comparable_sets == 0:
        print_always("CRITICAL: No complete transcript sets found to compare after filtering.")
        return

    print_always(f"Successfully analyzed {comparable_sets} complete transcript CDS sets.")
    print("\n" + "="*50)
    print(f" TRANSCRIPTS REPORT (QC < {DIVERGENCE_THRESHOLD:.1f}%)")
    print("="*50)

    if total_fixed_diffs > 0:
        g0_perc = (g0_matches / total_fixed_diffs) * 100.0
        g1_perc = (g1_matches / total_fixed_diffs) * 100.0
        print(f"Total fixed differences: {total_fixed_diffs}")
        print(f"  - Group 0 allele matched Chimp: {g0_perc:.2f}%")
        print(f"  - Group 1 allele matched Chimp: {g1_perc:.2f}%")

        sorted_g0 = sorted(per_tx_g0.items(), key=lambda x: x[1])
        print("\nTop 5 CDS where Group 0 allele LEAST resembles Chimp (fixed diffs):")
        for name, score in sorted_g0[:5]:
            print(f"  - {name:<40}: {score:.2f}% match")

        sorted_g1 = sorted(per_tx_g1.items(), key=lambda x: x[1])
        print("\nTop 5 CDS where Group 1 allele LEAST resembles Chimp (fixed diffs):")
        for name, score in sorted_g1[:5]:
            print(f"  - {name:<40}: {score:.2f}% match")
    else:
        print("No fixed differences were found among the filtered transcript genes.")
    print("="*50 + "\n")

def calculate_and_print_differences_regions():
    print_always("--- Final Difference Calculation & Statistics (Regions) ---")
    # Match inversion group files
    inv_regex = re.compile(r"^inversion_group(?P<grp>[01])_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$")
    # Outgroup for region files
    out_regex = re.compile(r"^outgroup_inversion_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$")

    groups = defaultdict(dict)  # key: (chrom,start,end) -> dict of role->file
    all_phys = glob.glob('*.phy')

    # Scan files with progress
    total_files = len(all_phys)
    for i, fpath in enumerate(all_phys, 1):
        base = os.path.basename(fpath)
        m = inv_regex.match(base)
        if m:
            chrom_norm = normalize_chromosome(m.group('chrom'))
            if not chrom_norm:
                logger.add("Region Stats Parsing Error", f"Skipped '{base}' due to invalid chromosome '{m.group('chrom')}'.")
            else:
                start = int(m.group('start'))
                end = int(m.group('end'))
                key = (chrom_norm, start, end)
                role = f"group{m.group('grp')}"
                groups[key][role] = fpath
        m2 = out_regex.match(base)
        if m2:
            chrom_norm = normalize_chromosome(m2.group('chrom'))
            if not chrom_norm:
                logger.add("Region Stats Parsing Error", f"Skipped '{base}' due to invalid chromosome '{m2.group('chrom')}'.")
            else:
                start = int(m2.group('start'))
                end = int(m2.group('end'))
                key2 = (chrom_norm, start, end)
                groups[key2]['outgroup'] = fpath
        if i % 25 == 0 or i == total_files:
            progress_bar("[RG stats: scan]", i, total_files if total_files else 1)
    if total_files:
        progress_bar("[RG stats: scan]", total_files, total_files)
        print()

    total_fixed_diffs = 0
    g0_matches = 0
    g1_matches = 0
    per_region_g0 = {}
    per_region_g1 = {}
    comparable_sets = 0

    keys = list(groups.items())
    total_keys = len(keys)
    print_dbg(f"Comparable REGION groups detected (pre-filter): {total_keys}")

    print_always("Analyzing each comparable REGION set (passed QC)...")
    for idx, (key, files) in enumerate(keys, 1):
        if {'group0', 'group1', 'outgroup'}.issubset(files.keys()):
            g0_seqs = read_phy_sequences(files['group0'])
            g1_seqs = read_phy_sequences(files['group1'])
            out_seq_list = read_phy_sequences(files['outgroup'])
            if not out_seq_list:
                progress_bar("[RG stats]", idx, total_keys if total_keys else 1)
                continue
            out_seq = out_seq_list[0]

            g0_len = set(len(s) for s in g0_seqs)
            g1_len = set(len(s) for s in g1_seqs)
            if not (len(g0_len) == 1 and len(g1_len) == 1):
                logger.add("Intra-file Length Mismatch (Region)", f"Not all sequences same length for region {key}.")
                progress_bar("[RG stats]", idx, total_keys if total_keys else 1)
                continue

            L0 = g0_len.pop()
            L1 = g1_len.pop()
            if L0 != L1 or L0 != len(out_seq):
                logger.add("Final Comparison Error (Region)", f"Length mismatch between groups for region {key}.")
                progress_bar("[RG stats]", idx, total_keys if total_keys else 1)
                continue

            comparable_sets += 1
            n = L0
            region_label = f"{key[0]}:{key[1]}-{key[2]}"

            local_fd = 0
            local_g0 = 0
            local_g1 = 0

            for i in range(n):
                g0_alleles = {s[i] for s in g0_seqs if s[i] != '-'}
                g1_alleles = {s[i] for s in g1_seqs if s[i] != '-'}
                if len(g0_alleles) == 1 and len(g1_alleles) == 1 and g0_alleles != g1_alleles:
                    local_fd += 1
                    total_fixed_diffs += 1
                    g0_a = next(iter(g0_alleles))
                    g1_a = next(iter(g1_alleles))
                    chimp_a = out_seq[i]
                    if chimp_a == g0_a:
                        g0_matches += 1
                        local_g0 += 1
                    elif chimp_a == g1_a:
                        g1_matches += 1
                        local_g1 += 1

            if local_fd > 0:
                per_region_g0[region_label] = (local_g0 / local_fd) * 100.0
                per_region_g1[region_label] = (local_g1 / local_fd) * 100.0

        progress_bar("[RG stats]", idx, total_keys if total_keys else 1)

    if total_keys:
        progress_bar("[RG stats]", total_keys, total_keys)
        print()

    if comparable_sets == 0:
        print_always("CRITICAL: No complete REGION sets found to compare after filtering.")
        return

    print_always(f"Successfully analyzed {comparable_sets} complete REGION sets.")
    print("\n" + "="*50)
    print(f" REGIONS REPORT (QC < {DIVERGENCE_THRESHOLD:.1f}%)")
    print("="*50)

    if total_fixed_diffs > 0:
        g0_perc = (g0_matches / total_fixed_diffs) * 100.0
        g1_perc = (g1_matches / total_fixed_diffs) * 100.0
        print(f"Total fixed differences: {total_fixed_diffs}")
        print(f"  - Group 0 allele matched Chimp: {g0_perc:.2f}%")
        print(f"  - Group 1 allele matched Chimp: {g1_perc:.2f}%")

        sorted_g0 = sorted(per_region_g0.items(), key=lambda x: x[1])
        print("\nTop 5 REGIONS where Group 0 allele LEAST resembles Chimp (fixed diffs):")
        for name, score in sorted_g0[:5]:
            print(f"  - {name:<30}: {score:.2f}% match")

        sorted_g1 = sorted(per_region_g1.items(), key=lambda x: x[1])
        print("\nTop 5 REGIONS where Group 1 allele LEAST resembles Chimp (fixed diffs):")
        for name, score in sorted_g1[:5]:
            print(f"  - {name:<30}: {score:.2f}% match")
    else:
        print("No fixed differences were found among the filtered regions.")
    print("="*50 + "\n")

# =========================
# --- Main ----------------
# =========================

def main():
    print_always("--- Starting Chimp Outgroup Generation for Transcripts + Regions ---")
    print_dbg(f"Using BIN_SIZE={BIN_SIZE}, DEBUG_VERBOSE={DEBUG_VERBOSE}, DEBUG_CHUNK_SAMPLE={DEBUG_CHUNK_SAMPLE}")

    log_path = setup_detail_logger()
    print_always(f"Detailed log will be written to: {log_path}")
    log_detail("SYSTEM", "startup", "BEGIN", "Run initialised.", bin_size=BIN_SIZE, divergence_threshold=DIVERGENCE_THRESHOLD)

    with time_block("Download + prepare AXT"):
        download_axt_file()
        ungzip_file()

    # Parse inputs
    with time_block("Parse transcript metadata"):
        transcripts = parse_transcript_metadata()

    with time_block("Scan region PHYLIPs"):
        regions = find_region_sets()

    if not transcripts and not regions:
        print_always("No valid transcripts or regions found after initial validation.")
    else:
        with time_block("Build outgroups + filter + write"):
            build_outgroups_and_filter(transcripts, regions)
        # Stats for each domain
        with time_block("Compute TX stats"):
            calculate_and_print_differences_transcripts()
        with time_block("Compute REG stats"):
            calculate_and_print_differences_regions()

    logger.report()
    log_detail("SYSTEM", "shutdown", "END", "Run completed.")
    print_always("--- Script finished. ---")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        sys.exit(130)
