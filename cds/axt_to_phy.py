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
import requests
import multiprocessing
from collections import defaultdict
from contextlib import contextmanager

# =========================
# --- Configuration -----
# =========================

METADATA_FILE = 'phy_metadata.tsv'

# UCSC hg38 vs panTro5 net AXT
AXT_URL = 'http://hgdownload.soe.ucsc.edu/goldenpath/hg38/vsPanTro5/hg38.panTro5.net.axt.gz'
AXT_GZ_FILENAME = 'hg38.panTro5.net.axt.gz'
AXT_FILENAME = 'hg38.panTro5.net.axt'

# Divergence QC threshold (%)
DIVERGENCE_THRESHOLD = 10.0

# Debug: set to ENST id or to region key to print sequence snippet
DEBUG_TRANSCRIPT = None   # e.g., 'ENST00000367770.8'
DEBUG_REGION = None       # e.g., 'inv_7_60911891_61578023'

# Bin size (bp) for interval indexing over the genome (faster than per-base maps)
BIN_SIZE = int(os.environ.get("BIN_SIZE", "1000"))

# Verbosity knobs
DEBUG_VERBOSE = os.environ.get("DEBUG_VERBOSE", "0") == "1"
DEBUG_CHUNK_SAMPLE = int(os.environ.get("DEBUG_CHUNK_SAMPLE", "0"))  # e.g., 1000

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
    sequences = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                return []
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                m = re.search(r'[ACGTN-]+$', line, re.IGNORECASE)
                if m:
                    sequences.append(m.group(0).upper())
    except Exception:
        pass
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
                continue

            phy_fname, t_id, gene, chrom, _, start, end, _, coords_str = parts[:9]
            cds_key = (t_id, coords_str)
            if cds_key in seen:
                continue
            seen.add(cds_key)

            # Parse exon segments and expected length
            try:
                segments = [(int(s), int(e)) for s, e in (p.split('-') for p in coords_str.split(';'))]
                expected_len = sum(e - s + 1 for s, e in segments)
                if expected_len <= 0:
                    continue
            except (ValueError, IndexError):
                logger.add("Metadata Parsing Error", f"L{line_num}: Could not parse coordinate chunks for {t_id}.")
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
                continue

            g0_seqs = read_phy_sequences(g0_fname)
            g1_seqs = read_phy_sequences(g1_fname)

            if not g0_seqs or not g1_seqs:
                logger.add("Missing Input File", f"{t_id}: group0 or group1 .phy not found or empty.")
                continue

            if not all(len(s) == expected_len for s in g0_seqs):
                g0_lengths = set(len(s) for s in g0_seqs)
                logger.add("Input Length Mismatch", f"{t_id} (group0): lengths {g0_lengths} != expected ({expected_len}).")
                continue

            if not all(len(s) == expected_len for s in g1_seqs):
                g1_lengths = set(len(s) for s in g1_seqs)
                logger.add("Input Length Mismatch", f"{t_id} (group1): lengths {g1_lengths} != expected ({expected_len}).")
                continue

            cds_info = {
                'gene_name': gene,
                'transcript_id': t_id,
                'chromosome': 'chr' + chrom,
                'expected_len': expected_len,
                'start': start,
                'end': end,
                'g0_fname': g0_fname,
                'g1_fname': g1_fname
            }
            validated.append({'info': cds_info, 'segments': segments})

    progress_bar("[Metadata]", total_lines if total_lines else 1, total_lines if total_lines else 1)
    print()
    print_dbg(f"Parsed metadata entries: {len(validated)}")
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
        chrom = m.group('chrom')
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
        region_id = f"inv_{chrom}_{start}_{end}"
        info = {
            'region_id': region_id,
            'chromosome': 'chr' + str(chrom),
            'expected_len': expected_len,
            'start': str(start),
            'end': str(end),
            'g0_fname': d.get('group0'),
            'g1_fname': d.get('group1'),
        }

        qc_fname = info['g0_fname'] or info['g1_fname']
        if not qc_fname:
            logger.add("Region Missing File", f"{region_id}: neither group0 nor group1 file present; skipping QC.")
        else:
            try:
                with open(qc_fname, 'r') as f:
                    first = f.readline().strip()
                mlen = re.match(r'\s*\d+\s+(\d+)\s*$', first)
                if not mlen:
                    logger.add("Region QC Warning", f"{region_id}: could not parse header length in {os.path.basename(qc_fname)}.")
                else:
                    header_len = int(mlen.group(1))
                    if header_len != expected_len:
                        logger.add("Region Input Length Mismatch", f"{region_id}: header length {header_len} != expected ({expected_len}).")
            except Exception:
                logger.add("Region QC Warning", f"{region_id}: failed to read header from {os.path.basename(qc_fname)}.")

        validated.append({'info': info, 'segments': [(start, end)]})

        processed += 1
        progress_bar("[Region QC]", processed, total if total else 1)

    if total > 0:
        progress_bar("[Region QC]", total, total)
    print()
    print_always(f"Found {len(validated)} candidate regions.")
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
    record = ('TX'|'RG', id, seg_start, seg_end, offset)
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
                chrom_bins.setdefault(b, []).append(('TX', t_id, s, e, offset))
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
            chrom_bins.setdefault(b, []).append(('RG', r_id, s, e, 0))
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

def process_axt_chunk(chunk_start, chunk_end, bin_index):
    """
    Worker to parse a slice of the AXT file and collect chimp bases.
    Returns dict: id -> {target_idx: base}
    """
    results = defaultdict(dict)  # id -> {pos_idx: base}
    parsed_headers = 0
    try:
        with open(AXT_FILENAME, 'r', buffering=1024*1024) as f:
            f.seek(chunk_start)
            if chunk_start != 0:
                f.readline()  # align to line boundary

            while f.tell() < chunk_end:
                header = f.readline()
                if not header:
                    break
                header = header.strip()
                if not header:
                    continue

                parts = header.split()
                if len(parts) != 9:
                    # Skip non-AXT lines (shouldn't occur in .net.axt)
                    # Also skip two seq lines to stay aligned if this was a header-ish line.
                    _ = f.readline()
                    _ = f.readline()
                    continue

                axt_chr = parts[1]  # e.g., 'chr7'
                try:
                    human_pos = int(parts[2])  # tStart
                except ValueError:
                    # Malformed; skip 2 sequence lines to stay aligned
                    f.readline(); f.readline()
                    continue

                human_seq = f.readline()
                chimp_seq = f.readline()
                if not human_seq or not chimp_seq:
                    break

                human_seq = human_seq.strip().upper()
                chimp_seq = chimp_seq.strip().upper()

                parsed_headers += 1
                if DEBUG_CHUNK_SAMPLE and (parsed_headers % DEBUG_CHUNK_SAMPLE == 0):
                    # Light periodic debug from worker
                    print_dbg(f"Worker chunk[{chunk_start}:{chunk_end}] parsed {parsed_headers} blocks (tell={f.tell()})")

                # If chromosome not indexed at all, skip fast
                chrom_bins = bin_index.get(axt_chr)
                if not chrom_bins:
                    continue

                # Iterate alignment columns
                for h_char, c_char in zip(human_seq, chimp_seq):
                    if h_char != '-':
                        # Query bin
                        bin_id = (human_pos - 1) // BIN_SIZE
                        records = chrom_bins.get(bin_id)
                        if records:
                            for kind, ident, seg_start, seg_end, offset in records:
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

def _safe_pool_create(desired):
    """Create a ThreadPool with fallback reductions if creation is slow/fails."""
    from multiprocessing.dummy import Pool as ThreadPool
    attempts = []
    plan = [desired]
    if desired > 32:
        plan.append(32)
    if desired > 16:
        plan.append(16)
    if desired > 8:
        plan.append(8)
    if desired > 4:
        plan.append(4)
    plan = list(dict.fromkeys(plan))  # uniq, preserve order

    last_exc = None
    for n in plan:
        print_always(f"Creating thread pool with {n} workers ...")
        t0 = time.time()
        try:
            pool = ThreadPool(processes=n)
            dt = time.time() - t0
            print_always(f"Thread pool ready ({n} workers) in {dt:.2f}s.")
            return pool, n
        except Exception as e:
            last_exc = e
            attempts.append((n, f"{e}"))
            print_always(f"Pool creation failed for {n}: {e}. Trying fewer ...")
    raise RuntimeError(f"Unable to create pool. Attempts: {attempts}") from last_exc

def _workers_cap():
    try:
        cpu = len(os.sched_getaffinity(0))
    except Exception:
        cpu = multiprocessing.cpu_count()
    # Default cap to prevent spawn stalls on big nodes
    default = min(cpu, 32)
    env = os.environ.get("AXT_WORKERS")
    if env:
        try:
            want = max(1, int(env))
            return min(want, cpu)
        except ValueError:
            pass
    return default

def _print_system_limits():
    pid = os.getpid()
    rss = get_rss_kb()
    fds = get_fd_count()
    nproc = resource.getrlimit(resource.RLIMIT_NPROC)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    print_always(f"Process PID={pid} | RSS={rss} KB | FDs={fds} | RLIMIT_NPROC={nproc} | RLIMIT_NOFILE={nofile}")

def _chunk_plan(file_size, n_workers):
    # guard tiny chunks
    base = max(1, file_size // n_workers)
    offsets = []
    start = 0
    for i in range(n_workers):
        end = file_size if i == n_workers - 1 else min(file_size, start + base)
        offsets.append((start, end))
        start = end
    return offsets

def build_outgroups_and_filter(transcripts, regions):
    """
    Build chimp sequences for transcripts (CDS) and regions (inversions) using AXT.
    Apply divergence QC and write .phy outgroups for both sets.
    """
    if not transcripts and not regions:
        print_always("No transcript or region entries to process.")
        return

    # Build bin index
    bin_index, tx_info_map, rg_info_map = build_bin_index(transcripts, regions)

    # Create empty scaffolds
    tx_scaffolds = {t['info']['transcript_id']: ['-'] * t['info']['expected_len'] for t in transcripts}
    rg_scaffolds = {r['info']['region_id']: ['-'] * r['info']['expected_len'] for r in regions}

    print_always(f"Processing '{AXT_FILENAME}' in parallel (threaded)...")
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

    # Create pool (with fallbacks)
    t_pool_create0 = time.time()
    pool, actual_workers = _safe_pool_create(workers)
    t_pool_create1 = time.time()
    print_dbg(f"Pool creation took {t_pool_create1 - t_pool_create0:.2f}s; actual workers={actual_workers}")

    # Kick off work
    t0 = time.time()
    print_always(f"[AXT parse] START — scheduling {len(chunk_ranges)} chunks")
    progress_bar("[AXT parse]", 0, len(chunk_ranges))
    parts = []

    # Use a wrapper to include bin_index by reference without copying
    def _runner(args):
        cs, ce = args
        return process_axt_chunk(cs, ce, bin_index)

    # imap_unordered returns results as ready; keep UI responsive
    try:
        completed = 0
        for res in pool.imap_unordered(_runner, chunk_ranges):
            completed += 1
            progress_bar("[AXT parse]", completed, len(chunk_ranges))
            # Handle worker error sentinel
            if isinstance(res, dict) and "__error__" in res:
                print("\n[AXT parse][WORKER ERROR]")
                print(res["__error__"])
                print(res.get("__trace__", ""))
                print(f"Chunk: {res.get('__chunk__')}, parsed headers before error: {res.get('__parsed__')}")
                # Continue rather than die; we keep partial results
                continue
            parts.append(res)
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass
    progress_bar("[AXT parse]", len(chunk_ranges), len(chunk_ranges))
    print()
    print_always(f"Finished parallel AXT processing in {time.time() - t0:.2f} seconds.")
    print_always(f"Collected {len(parts)} partial result maps.")

    # Merge results
    print_always("Merging results into scaffolds (with divergence QC later)...")
    with time_block("Merge results"):
        total_parts = len(parts)
        merged = 0
        last_print = time.time()
        for res in parts:
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
            merged += 1
            if time.time() - last_print > 0.1 or merged == total_parts:
                progress_bar("[Merge]", merged, total_parts)
                last_print = time.time()
        if total_parts:
            progress_bar("[Merge]", total_parts, total_parts)
            print()

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
            start = info['start']
            end = info['end']
            g0_fname = info['g0_fname']

            seq_list = tx_scaffolds.get(t_id)
            if not seq_list:
                logger.add("No Alignment Found", f"No chimp alignment found for {t_id}.")
                progress_bar("[TX write]", i, total_tx)
                continue
            final_seq = "".join(seq_list)

            if final_seq.count('-') == len(final_seq):
                logger.add("No Alignment Found", f"No chimp alignment found for {t_id}.")
                progress_bar("[TX write]", i, total_tx)
                continue

            # Divergence QC vs group0 reference (first sequence)
            human_seqs = read_phy_sequences(g0_fname)
            if not human_seqs:
                logger.add("Human File Missing for QC", f"Could not read human seqs from {g0_fname} for divergence check on {t_id}.")
                progress_bar("[TX write]", i, total_tx)
                continue
            human_ref = human_seqs[0]

            diff = 0
            comp = 0
            for h, c in zip(human_ref, final_seq):
                if h != '-' and c != '-':
                    comp += 1
                    if h != c:
                        diff += 1
            divergence = (diff / comp) * 100 if comp else 0.0

            outname = f"outgroup_{gene}_{t_id}_{chrom}_start{start}_end{end}.phy"
            if divergence > DIVERGENCE_THRESHOLD:
                logger.add("QC Filter: High Divergence", f"'{gene} ({t_id})' removed. Divergence vs chimp: {divergence:.2f}% (> {DIVERGENCE_THRESHOLD}%).")
                if os.path.exists(outname):
                    try:
                        os.remove(outname)
                    except Exception:
                        pass
                progress_bar("[TX write]", i, total_tx)
                continue

            if DEBUG_TRANSCRIPT == t_id:
                print_always(f"\n--- DEBUG TX {t_id} --- len={len(final_seq)}\n{final_seq[:120]}...\n")

            with open(outname, 'w') as f_out:
                f_out.write(f" 1 {len(final_seq)}\n")
                f_out.write(f"{'panTro5':<10}{final_seq}\n")
            tx_written += 1
            progress_bar("[TX write]", i, total_tx)
        if total_tx:
            progress_bar("[TX write]", total_tx, total_tx)
            print()
        print_always(f"Wrote {tx_written} transcript outgroup PHYLIPs (passed QC).")

    # --- Write regions ---
    print_always("Writing region outgroups (after divergence QC)...")
    with time_block("Write region outgroups"):
        rg_written = 0
        total_rg = len(regions)
        for i, r in enumerate(regions, 1):
            info = r['info']
            r_id = info['region_id']              # inv_<chrom>_<start>_<end>
            chrom_label = info['chromosome'][3:]  # strip 'chr'
            start = info['start']
            end = info['end']
            g0_fname = info['g0_fname'] or info['g1_fname']

            seq_list = rg_scaffolds.get(r_id)
            if not seq_list:
                logger.add("No Alignment Found (Region)", f"No chimp alignment found for {r_id}.")
                progress_bar("[RG write]", i, total_rg)
                continue
            final_seq = "".join(seq_list)

            if final_seq.count('-') == len(final_seq):
                logger.add("No Alignment Found (Region)", f"No chimp alignment found for {r_id}.")
                progress_bar("[RG write]", i, total_rg)
                continue

            # Divergence QC vs human reference (group0 preferred)
            if not g0_fname:
                logger.add("Region File Missing for QC", f"{r_id}: no group file for divergence check; skipping QC.")
                divergence = 0.0
            else:
                human_seqs = read_phy_sequences(g0_fname)
                if not human_seqs:
                    logger.add("Region File Missing for QC", f"{r_id}: cannot read {os.path.basename(g0_fname)}; skipping QC.")
                    divergence = 0.0
                else:
                    human_ref = human_seqs[0]
                    diff = 0
                    comp = 0
                    for h, c in zip(human_ref, final_seq):
                        if h != '-' and c != '-':
                            comp += 1
                            if h != c:
                                diff += 1
                    divergence = (diff / comp) * 100 if comp else 0.0

            outname = f"outgroup_inversion_{chrom_label}_start{start}_end{end}.phy"
            if divergence > DIVERGENCE_THRESHOLD:
                logger.add("QC Filter: High Divergence (Region)", f"{r_id} removed. Divergence vs chimp: {divergence:.2f}% (> {DIVERGENCE_THRESHOLD}%).")
                if os.path.exists(outname):
                    try:
                        os.remove(outname)
                    except Exception:
                        pass
                progress_bar("[RG write]", i, total_rg)
                continue

            if DEBUG_REGION == r_id:
                print_always(f"\n--- DEBUG RG {r_id} --- len={len(final_seq)}\n{final_seq[:120]}...\n")

            with open(outname, 'w') as f_out:
                f_out.write(f" 1 {len(final_seq)}\n")
                f_out.write(f"{'panTro5':<10}{final_seq}\n")
            rg_written += 1
            progress_bar("[RG write]", i, total_rg)
        if total_rg:
            progress_bar("[RG write]", total_rg, total_rg)
            print()
        print_always(f"Wrote {rg_written} region outgroup PHYLIPs (passed QC).")

# =========================
# --- Fixed-diff stats ----
# =========================

def calculate_and_print_differences_transcripts():
    print_always("--- Final Difference Calculation & Statistics (Transcripts) ---")
    key_regex = re.compile(r"(ENST[0-9]+\.[0-9]+)_(chr[^_]+)_start([0-9]+)_end([0-9]+)")
    cds_groups = defaultdict(dict)
    all_phys = glob.glob('*.phy')

    # Scan files with progress
    total_files = len(all_phys)
    for i, fpath in enumerate(all_phys, 1):
        base = os.path.basename(fpath)
        m = key_regex.search(base)
        if m:
            cds_groups[m.groups()][base.split('_')[0]] = fpath
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
            key = (m.group('chrom'), m.group('start'), m.group('end'))
            role = f"group{m.group('grp')}"
            groups[key][role] = fpath
        m2 = out_regex.match(base)
        if m2:
            key2 = (m2.group('chrom'), m2.group('start'), m2.group('end'))
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
            region_label = f"chr{key[0]}:{key[1]}-{key[2]}"

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
    print_always("--- Script finished. ---")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        sys.exit(130)
