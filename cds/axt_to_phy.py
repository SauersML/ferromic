import os
import re
import gzip
import glob
import time
import shutil
import gzip
import requests
import multiprocessing
from collections import defaultdict

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
BIN_SIZE = 1000

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
        return
    print(f"Downloading '{AXT_GZ_FILENAME}' from UCSC...")
    try:
        with requests.get(AXT_URL, stream=True) as r:
            r.raise_for_status()
            with open(AXT_GZ_FILENAME, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 4):
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Error downloading file: {e}")
        exit(1)

def ungzip_file():
    if not os.path.exists(AXT_GZ_FILENAME):
        if not os.path.exists(AXT_FILENAME):
            print(f"FATAL: AXT file not found.")
            exit(1)
        return
    if os.path.exists(AXT_FILENAME):
        return
    print(f"Decompressing '{AXT_GZ_FILENAME}'...")
    try:
        with gzip.open(AXT_GZ_FILENAME, 'rb') as f_in, open(AXT_FILENAME, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
    except Exception as e:
        print(f"FATAL: Error decompressing file: {e}")
        exit(1)

def read_phy_sequences(filename):
    """
    Reads all sequences from a simple PHYLIP file.
    Returns list of uppercase strings (ACGTN-), ignoring names.
    """
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
    Returns:
      - t_entries: list of dicts with keys:
           info: {gene_name, transcript_id, chromosome, expected_len, start, end, g0_fname, g1_fname}
           segments: [(start,end), ...] (1-based inclusive hg38 genomic coords)
    """
    if not os.path.exists(METADATA_FILE):
        print(f"FATAL: Metadata file '{METADATA_FILE}' not found.")
        exit(1)

    print("Validating transcript inputs against metadata...")
    validated = []
    seen = set()

    with open(METADATA_FILE, 'r') as f:
        next(f, None)  # skip header
        for line_num, line in enumerate(f, 2):
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
                # Can't infer sibling; try both patterns
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

    return validated

# =========================
# --- Input: Regions ------
# =========================

REGION_REGEX = re.compile(
    r'^inversion_(group(?P<grp>[01]))_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$'
)

def find_region_sets():
    """
    Finds available inversion region PHYLIPs by scanning current directory.
    Groups group0/group1 by (chrom, start, end).
    Performs header-only length validation for speed and prints a live progress bar.
    Returns list of dicts:
      info: {region_id, chromosome, expected_len, start, end, g0_fname?, g1_fname?}
      segments: [(start,end)]
    """
    print("Scanning for inversion region PHYLIP files...")
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
        if total > 0:
            filled = int(bar_width * processed // total)
            bar = "█" * filled + "-" * (bar_width - filled)
            pct = int((processed * 100) // total)
            print(f"\r[Region QC] |{bar}| {processed}/{total} ({pct}%)", end='', flush=True)

    if total > 0:
        print()

    print(f"Found {len(validated)} candidate regions.")
    return validated

# =========================
# --- Interval Index ------
# =========================

def _bin_range(start, end, bin_size):
    """Yield bin ids covered by [start, end] inclusive (1-based coords)."""
    # Convert to 0-based half-open for binning
    a = max(0, start - 1)
    b = end
    first = a // bin_size
    last = (b - 1) // bin_size
    for k in range(first, last + 1):
        yield k

def build_bin_index(transcripts, regions):
    """
    Builds a per-chromosome bin index mapping BIN -> list of segments.
    Each record: (kind, id, seg_start, seg_end, offset)
      kind: 'TX' or 'RG'
      id: transcript_id or region_id
      offset: where this segment starts in the scaffold (0-based)
    Returns dict: index[chrom][bin_id] -> list of records
    Also returns: info_maps for lookups
    """
    index = {}  # chrom -> bin -> [records]; plain dicts ensure picklability with multiprocessing
    tx_info_map = {}
    rg_info_map = {}

    # Transcripts
    for t in transcripts:
        info = t['info']
        chrom = info['chromosome']
        t_id = info['transcript_id']
        tx_info_map[t_id] = info
        # precompute offsets across exon segments
        offset = 0
        for s, e in t['segments']:
            chrom_bins = index.setdefault(chrom, {})
            for b in _bin_range(s, e, BIN_SIZE):
                chrom_bins.setdefault(b, []).append(('TX', t_id, s, e, offset))
            offset += (e - s + 1)

    # Regions
    for r in regions:
        info = r['info']
        chrom = info['chromosome']
        r_id = info['region_id']
        rg_info_map[r_id] = info
        (s, e) = r['segments'][0]
        chrom_bins = index.setdefault(chrom, {})
        for b in _bin_range(s, e, BIN_SIZE):
            chrom_bins.setdefault(b, []).append(('RG', r_id, s, e, 0))

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
    with open(AXT_FILENAME, 'r') as f:
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
                continue

            axt_chr = parts[1]                  # e.g., 'chr7'
            try:
                human_pos = int(parts[2])       # tStart
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
                                # 0-based position within this scaffold segment
                                target_idx = offset + (human_pos - seg_start)
                                # first-write wins
                                if target_idx not in results[ident]:
                                    results[ident][target_idx] = c_char
                    human_pos += 1

    return dict(results)

def build_outgroups_and_filter(transcripts, regions):
    """
    Build chimp sequences for transcripts (CDS) and regions (inversions) using AXT.
    Apply divergence QC and write .phy outgroups for both sets.
    """
    if not transcripts and not regions:
        print("No transcript or region entries to process.")
        return

    print("Building bin index (overlap-aware) for transcripts and regions...")
    bin_index, tx_info_map, rg_info_map = build_bin_index(transcripts, regions)

    # Create empty scaffolds
    tx_scaffolds = {t['info']['transcript_id']: ['-'] * t['info']['expected_len'] for t in transcripts}
    rg_scaffolds = {r['info']['region_id']: ['-'] * r['info']['expected_len'] for r in regions}

    print(f"Processing '{AXT_FILENAME}' in parallel...")
    file_size = os.path.getsize(AXT_FILENAME)
    try:
        num_procs = len(os.sched_getaffinity(0))
    except AttributeError:
        num_procs = multiprocessing.cpu_count()
    num_procs = max(1, num_procs)
    print(f"Using {num_procs} available CPU cores.")

    chunk_size = file_size // num_procs
    chunks = [(i * chunk_size, (i + 1) * chunk_size, bin_index) for i in range(num_procs)]
    chunks[-1] = (chunks[-1][0], file_size, bin_index)

    t0 = time.time()
    from multiprocessing.dummy import Pool as ThreadPool  # threading-based pool avoids pickling overhead for task arguments
    with ThreadPool(processes=num_procs) as pool:
        parts = pool.starmap(process_axt_chunk, chunks)
    print(f"Finished parallel AXT processing in {time.time() - t0:.2f} seconds.")

    print("Merging results and writing outgroups (with divergence QC)...")
    # Merge into scaffolds
    for res in parts:
        for ident, posmap in res.items():
            if ident in tx_scaffolds:
                sc = tx_scaffolds[ident]
                for pos_idx, base in posmap.items():
                    if 0 <= pos_idx < len(sc) and sc[pos_idx] == '-':
                        sc[pos_idx] = base
            elif ident in rg_scaffolds:
                sc = rg_scaffolds[ident]
                for pos_idx, base in posmap.items():
                    if 0 <= pos_idx < len(sc) and sc[pos_idx] == '-':
                        sc[pos_idx] = base

    # --- Write transcripts ---
    tx_written = 0
    for t in transcripts:
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
            continue
        final_seq = "".join(seq_list)

        if final_seq.count('-') == len(final_seq):
            logger.add("No Alignment Found", f"No chimp alignment found for {t_id}.")
            continue

        # Divergence QC vs group0 reference (first sequence)
        human_seqs = read_phy_sequences(g0_fname)
        if not human_seqs:
            logger.add("Human File Missing for QC", f"Could not read human seqs from {g0_fname} for divergence check on {t_id}.")
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
                os.remove(outname)
            continue

        if DEBUG_TRANSCRIPT == t_id:
            print(f"\n--- DEBUG TX {t_id} --- len={len(final_seq)}\n{final_seq[:120]}...\n")

        with open(outname, 'w') as f_out:
            f_out.write(f" 1 {len(final_seq)}\n")
            f_out.write(f"{'panTro5':<10}{final_seq}\n")
        tx_written += 1

    # --- Write regions ---
    rg_written = 0
    for r in regions:
        info = r['info']
        r_id = info['region_id']              # inv_<chrom>_<start>_<end>
        chrom_label = info['chromosome'][3:]  # strip 'chr'
        start = info['start']
        end = info['end']
        g0_fname = info['g0_fname'] or info['g1_fname']

        seq_list = rg_scaffolds.get(r_id)
        if not seq_list:
            logger.add("No Alignment Found (Region)", f"No chimp alignment found for {r_id}.")
            continue
        final_seq = "".join(seq_list)

        if final_seq.count('-') == len(final_seq):
            logger.add("No Alignment Found (Region)", f"No chimp alignment found for {r_id}.")
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
                os.remove(outname)
            continue

        if DEBUG_REGION == r_id:
            print(f"\n--- DEBUG RG {r_id} --- len={len(final_seq)}\n{final_seq[:120]}...\n")

        with open(outname, 'w') as f_out:
            f_out.write(f" 1 {len(final_seq)}\n")
            f_out.write(f"{'panTro5':<10}{final_seq}\n")
        rg_written += 1

    print(f"Wrote {tx_written} transcript outgroup PHYLIPs and {rg_written} region outgroup PHYLIPs (passed QC).")

# =========================
# --- Fixed-diff stats ----
# =========================

def calculate_and_print_differences_transcripts():
    print("\n--- Final Difference Calculation & Statistics (Transcripts) ---")
    key_regex = re.compile(r"(ENST[0-9]+\.[0-9]+)_(chr[^_]+)_start([0-9]+)_end([0-9]+)")
    cds_groups = defaultdict(dict)
    for f in glob.glob('*.phy'):
        base = os.path.basename(f)
        m = key_regex.search(base)
        if m:
            cds_groups[m.groups()][base.split('_')[0]] = f  # leading token: group0/group1/outgroup

    total_fixed_diffs = 0
    g0_matches = 0
    g1_matches = 0
    per_tx_g0 = {}
    per_tx_g1 = {}
    comparable_sets = 0

    print("Analyzing each comparable transcript set (passed QC)...")
    for identifier, files in cds_groups.items():
        if {'group0', 'group1', 'outgroup'}.issubset(files.keys()):
            g0_seqs = read_phy_sequences(files['group0'])
            g1_seqs = read_phy_sequences(files['group1'])
            out_seq_list = read_phy_sequences(files['outgroup'])
            if not out_seq_list:
                continue
            out_seq = out_seq_list[0]

            g0_len = set(len(s) for s in g0_seqs)
            g1_len = set(len(s) for s in g1_seqs)
            if not (len(g0_len) == 1 and len(g1_len) == 1):
                logger.add("Intra-file Length Mismatch", f"Not all sequences in a .phy have same length for {identifier[0]}.")
                continue

            L0 = g0_len.pop()
            L1 = g1_len.pop()
            if L0 != L1 or L0 != len(out_seq):
                logger.add("Final Comparison Error", f"Length mismatch between groups for {identifier[0]}.")
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

    if comparable_sets == 0:
        print("CRITICAL: No complete transcript sets found to compare after filtering.")
        return

    print(f"Successfully analyzed {comparable_sets} complete transcript CDS sets.")
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
    print("\n--- Final Difference Calculation & Statistics (Regions) ---")
    # Match inversion group files
    inv_regex = re.compile(r"^inversion_group(?P<grp>[01])_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$")
    # Outgroup for region files
    out_regex = re.compile(r"^outgroup_inversion_(?P<chrom>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$")

    groups = defaultdict(dict)  # key: (chrom,start,end) -> dict of role->file

    for f in glob.glob('*.phy'):
        base = os.path.basename(f)
        m = inv_regex.match(base)
        if m:
            key = (m.group('chrom'), m.group('start'), m.group('end'))
            role = f"group{m.group('grp')}"
            groups[key][role] = f
            continue
        m2 = out_regex.match(base)
        if m2:
            key = (m2.group('chrom'), m2.group('start'), m2.group('end'))
            groups[key]['outgroup'] = f

    total_fixed_diffs = 0
    g0_matches = 0
    g1_matches = 0
    per_region_g0 = {}
    per_region_g1 = {}
    comparable_sets = 0

    print("Analyzing each comparable REGION set (passed QC)...")
    for key, files in groups.items():
        if {'group0', 'group1', 'outgroup'}.issubset(files.keys()):
            g0_seqs = read_phy_sequences(files['group0'])
            g1_seqs = read_phy_sequences(files['group1'])
            out_seq_list = read_phy_sequences(files['outgroup'])
            if not out_seq_list:
                continue
            out_seq = out_seq_list[0]

            g0_len = set(len(s) for s in g0_seqs)
            g1_len = set(len(s) for s in g1_seqs)
            if not (len(g0_len) == 1 and len(g1_len) == 1):
                logger.add("Intra-file Length Mismatch (Region)", f"Not all sequences same length for region {key}.")
                continue

            L0 = g0_len.pop()
            L1 = g1_len.pop()
            if L0 != L1 or L0 != len(out_seq):
                logger.add("Final Comparison Error (Region)", f"Length mismatch between groups for region {key}.")
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

    if comparable_sets == 0:
        print("CRITICAL: No complete REGION sets found to compare after filtering.")
        return

    print(f"Successfully analyzed {comparable_sets} complete REGION sets.")
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
    print("--- Starting Chimp Outgroup Generation for Transcripts + Regions ---")
    download_axt_file()
    ungzip_file()

    # Parse inputs
    transcripts = parse_transcript_metadata()
    regions = find_region_sets()

    if not transcripts and not regions:
        print("No valid transcripts or regions found after initial validation.")
    else:
        build_outgroups_and_filter(transcripts, regions)
        # Stats for each domain
        calculate_and_print_differences_transcripts()
        calculate_and_print_differences_regions()

    logger.report()
    print("--- Script finished. ---")

if __name__ == '__main__':
    main()
