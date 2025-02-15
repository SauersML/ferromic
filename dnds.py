"""
dN/dS Analysis Script using PAML's CODEML

This script calculates pairwise dN/dS values using PAML's CODEML program.

1. Input Files:
    Each input file is a PHYLIP-like file with lines consisting of:
    SAMPLE_NAME_L/R + SEQUENCE (no spaces). For example:
        ABC_XYZ_HG01352_LACGGAGTAC...
    Where each sample name ends with "_L" or "_R" before the sequence.

    The input file names follow a pattern including a transcript ID and chromosome info:
        group_0_ENST00000706755.1_chr_19_combined.phy
        group_1_ENST00000704003.1_chr_7_combined.phy
        ...
    Each file corresponds to a single CDS/transcript variant (one full alignment).

2. Sequence Validation:
    - Sequences must be non-empty, length divisible by 3.
    - Only valid bases: A,T,C,G,N,-
    - No in-frame stop codons (TAA,TAG,TGA).
    - If invalid, the sequence is discarded.

Usage:
    python3 dnds.py --phy_dir PATH_TO_PHY_FILES --output_dir OUTPUT_DIRECTORY --codeml_path PATH_TO_CODEML
"""

import os
import sys
import glob
import subprocess
import multiprocessing
import psutil
from itertools import combinations
import pandas as pd
import numpy as np
import shutil
import re
import argparse
import time
import logging
import pickle
import sqlite3
from functools import partial
import hashlib
from tqdm import tqdm

COMPARE_BETWEEN_GROUPS = False

VALIDATION_CACHE = {}
VALIDATION_CACHE_FILE = 'validation_results.pkl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dnds_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Global counters
GLOBAL_INVALID_SEQS = 0
GLOBAL_DUPLICATES = 0
GLOBAL_TOTAL_SEQS = 0
GLOBAL_TOTAL_CDS = 0
GLOBAL_TOTAL_COMPARISONS = 0

ETA_DATA = {
    'start_time': None,
    'completed': 0,
    'rate_smoothed': None,  # Exponential moving average for rate
    'alpha': 0.2,  # smoothing factor
}

manager = multiprocessing.Manager()
GLOBAL_COUNTERS = manager.dict({
    'invalid_seqs': 0,
    'duplicates': 0,
    'total_seqs': 0,
    'total_cds': 0,
    'total_comparisons': 0,
    'stop_codons': 0
})

CACHE = None

NUM_PARALLEL = 128

def print_eta(completed, total, start_time, eta_data):
    print("Calculating ETA for progress...")
    sys.stdout.flush()
    if total <= 0 or completed <= 0:
        logging.info(f"Progress: {completed}/{total}, ETA:N/A")
        return

    elapsed = time.time() - start_time
    current_rate = completed / elapsed

    if eta_data['rate_smoothed'] is None:
        eta_data['rate_smoothed'] = current_rate
    else:
        alpha = eta_data['alpha']
        eta_data['rate_smoothed'] = alpha * current_rate + (1 - alpha) * eta_data['rate_smoothed']

    if eta_data['rate_smoothed'] <= 0:
        logging.info(f"Progress: {completed}/{total}, ETA:N/A")
        return

    remain = total - completed
    eta_sec = remain / eta_data['rate_smoothed']
    hrs = eta_sec / 3600
    logging.info(f"Progress: {completed}/{total} comps. ETA: {hrs:.2f}h")
    print(f"Completed {completed} of {total}, ETA: {hrs:.2f} hours")
    sys.stdout.flush()

def validate_sequence(seq, filepath, sample_name, full_line):
    print(f"Validating sequence for sample {sample_name} from file {filepath}")
    if not seq:
        print("Invalid sequence: empty")
        return None, False

    seq = seq.upper()
    line_len = len(full_line)
    seq_len = len(seq)
    MAX_CDS_LENGTH = 150000

    if seq_len > MAX_CDS_LENGTH:
        print("Invalid sequence: exceeds max length")
        return None, False

    if seq_len > line_len:
        print("Invalid sequence: sequence length greater than line length?")
        return None, False

    if seq_len % 3 != 0:
        print("Invalid sequence: length not divisible by 3")
        return None, False

    valid_bases = set('ATCGN-')
    invalid_chars = set(seq) - valid_bases
    if invalid_chars:
        print(f"Invalid sequence: contains invalid characters {invalid_chars}")
        return None, False

    stop_codons = {'TAA', 'TAG', 'TGA'}
    for i in range(0, seq_len - 2, 3):
        codon = seq[i:i+3]
        if codon in stop_codons:
            print(f"Invalid sequence: in-frame stop codon {codon}")
            return None, True

    print("Sequence validated successfully.")
    return seq, False

def create_paml_ctl(seqfile, outfile, working_dir):
    print(f"Creating PAML control file in {working_dir}")
    sys.stdout.flush()
    seqfile = os.path.abspath(seqfile)
    treefile = os.path.abspath(os.path.join(working_dir, "tree.txt"))
    outfile = os.path.abspath(os.path.join(working_dir, outfile))
    ctl_content = f"""
      seqfile = {seqfile}
      treefile = {treefile}
      outfile = {outfile}
      noisy = 0
      verbose = 0
      runmode = -2
      seqtype = 1
      CodonFreq = 2
      model = 0
      NSsites = 0
      icode = 0
      fix_kappa = 0
      kappa = 2.0
      fix_omega = 0
      omega = 1.0
      fix_alpha = 1
      alpha = 0.0
      getSE = 0
      RateAncestor = 0
      cleandata = 1
    """
    ctl_path = os.path.join(working_dir, 'codeml.ctl')
    with open(ctl_path, 'w') as f:
        f.write(ctl_content)
    print(f"PAML control file created at {ctl_path}")
    sys.stdout.flush()
    return ctl_path

def run_codeml(ctl_path, working_dir, codeml_path):
    import shutil
    import subprocess

    ctl_path = os.path.abspath(ctl_path)
    working_dir = os.path.abspath(working_dir)
    codeml_path = os.path.abspath(codeml_path)

    dst_ctl = os.path.join(working_dir, 'codeml.ctl')
    if not os.path.samefile(ctl_path, dst_ctl):
        shutil.copy2(ctl_path, dst_ctl)

    cmd = [codeml_path]
    cmdstr = " ".join(cmd)
    print(f"\nRunning command (with absolute paths) in working_dir: {cmdstr}")

    try:
        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=b'\n', timeout=30)

        if process.returncode != 0:
            print(f"codeml exited with error. Stderr:\n{stderr.decode().strip()}")
            return False

        results_file = os.path.join(working_dir, 'results.txt')
        if os.path.exists(results_file):
            print("Success: results.txt file created")
            return True
        else:
            print("Failed: no results.txt file created")
            return False

    except subprocess.TimeoutExpired:
        print("Timed out after 30 seconds")
        process.kill()
        return False
    except Exception as e:
        print(f"Error running codeml: {e}")
        return False

def parse_codeml_output(outfile_dir):
    print(f"Parsing CODEML output in {outfile_dir}")
    sys.stdout.flush()
    rst_file = os.path.join(outfile_dir, 'rst')
    if not os.path.exists(rst_file):
        print("No rst file found, returning None.")
        sys.stdout.flush()
        return None, None, None
    try:
        with open(rst_file, 'r') as f:
            lines = f.readlines()

        last_lines = lines[-5:] if len(lines) >= 5 else lines
        print("Last 5 lines of rst file:")
        for line in last_lines:
            print(line.rstrip())

        for i, line in enumerate(lines):
            if line.strip().startswith("seq seq"):
                values_line = lines[i + 1]
                values = values_line.strip().split()
                dN = float(values[4])
                dS = float(values[5])
                omega = float(values[6])
                print(f"Parsed dN={dN}, dS={dS}, omega={omega}")
                sys.stdout.flush()
                return dN, dS, omega

        print("Could not find values line in rst file.")
        sys.stdout.flush()
        return None, None, None
    except Exception as e:
        print(f"Error parsing CODEML output: {e}")
        sys.stdout.flush()
        return None, None, None

def parse_phy_file(filepath):
    print(f"Parsing phy file: {filepath}")
    sequences = {}
    duplicates_found = False
    line_pattern = re.compile(r'^([A-Za-z0-9_]+_[LR01])([ATCGNatcgn-]+)$')

    local_invalid_seqs = 0
    local_duplicates = 0
    local_total_seqs = 0
    local_stop_codons = 0

    if not os.path.isfile(filepath):
        print("File does not exist.")
        return sequences, duplicates_found

    with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.read().strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            match = line_pattern.match(line)
            if not match:
                continue

            orig_name = match.group(1)
            sequence = match.group(2)
            name_parts = orig_name.split('_')
            if len(name_parts) < 4:
                continue

            first = name_parts[0][:3]
            second = name_parts[1][:3]
            hg_part = name_parts[-2]
            group = name_parts[-1]

            md5_val = hashlib.md5(hg_part.encode('utf-8')).hexdigest()
            hash_str = md5_val[:2]
            sample_name = f"{first}{second}{hash_str}_{group}"

            seq_hash = hashlib.md5(sequence.encode('utf-8')).hexdigest()
            cache_key = (os.path.basename(filepath), sample_name, seq_hash)

            if cache_key in VALIDATION_CACHE:
                validated_seq, stop_codon_found = VALIDATION_CACHE[cache_key]
                print(f"Skipping validation for {sample_name}, found in cache.")
            else:
                validated_seq, stop_codon_found = validate_sequence(sequence, filepath, sample_name, line)
                VALIDATION_CACHE[cache_key] = (validated_seq, stop_codon_found)

            if validated_seq is None:
                local_invalid_seqs += 1
                if stop_codon_found:
                    local_stop_codons += 1
            else:
                local_total_seqs += 1
                if sample_name in sequences:
                    duplicates_found = True
                    local_duplicates += 1
                    base_name = sample_name[:2] + sample_name[3:]
                    dup_count = sum(1 for s in sequences if s[:2] + s[3:] == base_name)
                    new_name = sample_name[:2] + str(dup_count) + sample_name[3:]
                    sequences[new_name] = validated_seq
                else:
                    sequences[sample_name] = validated_seq

    GLOBAL_COUNTERS['invalid_seqs'] = GLOBAL_COUNTERS.get('invalid_seqs', 0) + local_invalid_seqs
    GLOBAL_COUNTERS['stop_codons'] = GLOBAL_COUNTERS.get('stop_codons', 0) + local_stop_codons
    GLOBAL_COUNTERS['total_seqs']   = GLOBAL_COUNTERS.get('total_seqs',   0) + local_total_seqs
    GLOBAL_COUNTERS['duplicates']   = GLOBAL_COUNTERS.get('duplicates',   0) + local_duplicates

    print(f"Finished parsing {filepath}, found {len(sequences)} sequences (duplicates_found={duplicates_found})")
    return sequences, duplicates_found

def load_cache(cache_file):
    """
    We only use this to load global counters from a pickle, not pairwise results.
    """
    print(f"Loading cache from {cache_file}")
    sys.stdout.flush()
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            if data:
                for k, v in data.items():
                    GLOBAL_COUNTERS[k] = v
        print(f"Cache loaded with {len(data)} entries (only counters).")
        sys.stdout.flush()
    else:
        print("No cache file found, starting fresh.")
        sys.stdout.flush()

def save_cache(cache_file, cache_data):
    """
    Save only counters to a pickle, not pairwise results.
    """
    print(f"Saving cache to {cache_file} with {len(cache_data)} entries (only counters).")
    sys.stdout.flush()
    with open(cache_file, 'wb') as f:
        normal_dict = dict(cache_data)
        pickle.dump(normal_dict, f)
    print("Cache saved.")
    sys.stdout.flush()

def init_sqlite_db(db_path):
    """
    Initialize (or reuse) a SQLite database to store pairwise results,
    with a table 'pairwise_cache' that has a unique primary key on 'cache_key'.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pairwise_cache (
            cache_key TEXT PRIMARY KEY,
            seq1 TEXT,
            seq2 TEXT,
            group1 INTEGER,
            group2 INTEGER,
            dN REAL,
            dS REAL,
            omega REAL,
            cds TEXT
        )
    """)
    conn.commit()
    return conn

def db_insert_or_ignore(conn, cache_key, record_tuple):
    """
    Insert a record into pairwise_cache if 'cache_key' doesn't already exist.
    record_tuple is (seq1, seq2, group1, group2, dN, dS, omega, cds).
    """
    try:
        conn.execute(
            "INSERT OR IGNORE INTO pairwise_cache "
            "(cache_key, seq1, seq2, group1, group2, dN, dS, omega, cds) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (cache_key,) + record_tuple
        )
        conn.commit()
    except Exception as e:
        print(f"Error inserting into SQLite: {e}")

def db_has_key(conn, cache_key):
    """
    Check if a cache_key exists in pairwise_cache.
    """
    cursor = conn.execute(
        "SELECT cache_key FROM pairwise_cache WHERE cache_key=? LIMIT 1",
        (cache_key,)
    )
    row = cursor.fetchone()
    return row is not None

def db_count_keys(conn):
    """
    Return how many total records exist in pairwise_cache.
    """
    cursor = conn.execute("SELECT COUNT(*) FROM pairwise_cache")
    row = cursor.fetchone()
    return row[0] if row else 0

def _read_csv_rows(csv_path):
    print(f"[CSV Loader] Starting to read {csv_path}")
    rows_to_insert = []
    try:
        df_existing = pd.read_csv(csv_path)
        print(f"[CSV Loader] Successfully read {len(df_existing)} rows from {csv_path}")
        required_cols = {'Seq1', 'Seq2', 'Group1', 'Group2', 'dN', 'dS', 'omega', 'CDS'}
        if required_cols.issubset(df_existing.columns):
            for _, row in df_existing.iterrows():
                seq1_val = str(row['Seq1'])
                seq2_val = str(row['Seq2'])
                group1_val = row['Group1']
                group2_val = row['Group2']
                dn_val = row['dN']
                ds_val = row['dS']
                omega_val = row['omega']
                cid_val = str(row['CDS'])
                cache_key = f"{cid_val}::{seq1_val}::{seq2_val}::{COMPARE_BETWEEN_GROUPS}"
                record_tuple = (
                    seq1_val,
                    seq2_val,
                    group1_val,
                    group2_val,
                    dn_val,
                    ds_val,
                    omega_val,
                    cid_val
                )
                rows_to_insert.append((cache_key, record_tuple))
    except Exception as e:
        print(f"Failed to load CSV {csv_path}: {str(e)}")
    return rows_to_insert

def process_pair(args):
    """
    The worker function for pairwise dN/dS. 
    Returns a tuple or None if the pair is invalid or skipped.
    """
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, _ = args
    seq1_name, seq2_name = pair
    if seq1_name not in sequences or seq2_name not in sequences:
        print("One of the sequences is missing, returning None.")
        sys.stdout.flush()
        return None

    group1 = sample_groups.get(seq1_name)
    group2 = sample_groups.get(seq2_name)
    if not COMPARE_BETWEEN_GROUPS and group1 != group2:
        print("Not comparing between groups, and groups differ. Skipping.")
        sys.stdout.flush()
        return None

    if sequences[seq1_name] == sequences[seq2_name]:
        sys.stdout.flush()
        return (seq1_name, seq2_name, group1, group2, 0.0, 0.0, -1.0, cds_id)

    working_dir = os.path.join(temp_dir, f'{seq1_name}_{seq2_name}')
    os.makedirs(working_dir, exist_ok=True)

    seqfile = os.path.join(working_dir, 'seqfile.phy')
    with open(seqfile, 'w') as f:
        f.write(f" 2 {len(sequences[seq1_name])}\n")
        f.write(f"{seq1_name}  {sequences[seq1_name]}\n")
        f.write(f"{seq2_name}  {sequences[seq2_name]}\n")
    print("Created seqfile.phy for codeml input.")
    sys.stdout.flush()

    treefile = os.path.join(working_dir, 'tree.txt')
    with open(treefile, 'w') as f:
        f.write(f"({seq1_name},{seq2_name});\n")
    print("Created tree.txt for codeml input.")
    sys.stdout.flush()

    ctl_path = create_paml_ctl(seqfile, 'results.txt', working_dir)
    success = run_codeml(ctl_path, working_dir, codeml_path)
    if not success:
        print("Codeml run failed, returning NaN values.")
        sys.stdout.flush()
        return (seq1_name, seq2_name, group1, group2, np.nan, np.nan, np.nan, cds_id)

    dn, ds, omega = parse_codeml_output(working_dir)
    if omega is None:
        omega = np.nan
    print(f"Pair processed: dn={dn}, ds={ds}, omega={omega}")
    sys.stdout.flush()
    return (seq1_name, seq2_name, group1, group2, dn, ds, omega, cds_id)

def estimate_one_file(phy_file, group_num):
    print(f"Estimating comparisons for file {phy_file}")
    sys.stdout.flush()
    sequences, duplicates = parse_phy_file(phy_file)
    if not sequences:
        return (0, 0)

    sample_groups = {}
    skip_file = False
    for s in sequences.keys():
        sample_groups[s] = int(group_num)
    if skip_file:
        return (0, 0)

    if COMPARE_BETWEEN_GROUPS:
        pairs = list(combinations(sequences.keys(), 2))
    else:
        g0 = [s for s, g in sample_groups.items() if g == 0]
        g1 = [s for s, g in sample_groups.items() if g == 1]
        pairs = list(combinations(g0, 2)) + list(combinations(g1, 2))

    cds_count = 1 if pairs else 0
    pair_count = len(pairs)
    print(f"File {phy_file} estimation: {cds_count} CDS, {pair_count} comparisons")
    sys.stdout.flush()
    return (cds_count, pair_count)

TRANSCRIPT_COORDS = {}

def get_transcript_coordinates(transcript_id):
    global TRANSCRIPT_COORDS
    print(f"Getting transcript coordinates for {transcript_id}")
    if transcript_id not in TRANSCRIPT_COORDS:
        print(f"No coordinates found for {transcript_id}")
        return (None, None, None)

    chrom, min_start, max_end = TRANSCRIPT_COORDS[transcript_id]
    print(f"Coordinates for {transcript_id}: {chrom}, {min_start}, {max_end}")
    return (chrom, min_start, max_end)

def overlaps(a_start, a_end, b_start, b_end):
    return not (b_end < a_start or a_end < b_start)

def cluster_by_coordinates(cds_meta):
    print("Clustering CDS by coordinate overlaps...")
    sys.stdout.flush()
    chr_map = {}
    for cd in cds_meta:
        cds_id, tid, chrom, st, en, seqs = cd
        if chrom not in chr_map:
            chr_map[chrom] = []
        chr_map[chrom].append(cd)

    clusters = []
    for chrom in chr_map:
        print(f"Processing chromosome: {chrom}")
        sys.stdout.flush()
        cds_list = chr_map[chrom]
        adjacency = {c[0]: set() for c in cds_list}
        for i in range(len(cds_list)):
            for j in range(i + 1, len(cds_list)):
                idA, tidA, chA, stA, enA, seqA = cds_list[i]
                idB, tidB, chB, stB, enB, seqB = cds_list[j]
                if overlaps(stA, enA, stB, enB):
                    adjacency[idA].add(idB)
                    adjacency[idB].add(idA)

        visited = set()
        for node in adjacency.keys():
            if node not in visited:
                stack = [node]
                comp = []
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        comp.append(n)
                        for neigh in adjacency[n]:
                            if neigh not in visited:
                                stack.append(neigh)
                clusters.append(comp)
    print(f"Clustering complete. Found {len(clusters)} clusters.")
    sys.stdout.flush()
    return clusters

def load_gtf_into_dict(gtf_file):
    transcript_dict = {}
    with open(gtf_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'CDS':
                continue

            chr_ = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            attrs = fields[8].strip()
            tid = None
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('transcript_id "'):
                    tid = attr.split('"')[1]
                    break
            if tid is None:
                continue

            if tid not in transcript_dict:
                transcript_dict[tid] = [chr_, start, end]
            else:
                existing_chr, existing_start, existing_end = transcript_dict[tid]
                if chr_ != existing_chr:
                    pass
                if start < existing_start:
                    transcript_dict[tid][1] = start
                if end > existing_end:
                    transcript_dict[tid][2] = end

    for k in transcript_dict:
        c, s, e = transcript_dict[k]
        transcript_dict[k] = (c, s, e)
    return transcript_dict

def preload_transcript_coords(gtf_file):
    global TRANSCRIPT_COORDS
    if not TRANSCRIPT_COORDS:
        TRANSCRIPT_COORDS = load_gtf_into_dict(gtf_file)
        print(f"[INFO] GTF loaded into memory: {len(TRANSCRIPT_COORDS)} transcripts.")

def main():
    print("Starting main process...")
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    codeml_inferred = os.path.abspath(os.path.join(script_dir, '..', 'paml', 'bin', 'codeml'))
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
    parser.add_argument('--codeml_path', type=str, default=codeml_inferred, help='Path to codeml executable.')
    args = parser.parse_args()

    # Attempt to load existing validation cache
    if os.path.exists(VALIDATION_CACHE_FILE):
        try:
            global VALIDATION_CACHE
            with open(VALIDATION_CACHE_FILE, 'rb') as f:
                VALIDATION_CACHE = pickle.load(f)
            print(f"Loaded validation cache with {len(VALIDATION_CACHE)} entries.")
        except Exception as e:
            print(f"Could not load validation cache: {e}")

    print(f"PHY_DIR: {args.phy_dir}")
    print(f"OUTPUT_DIR: {args.output_dir}")
    print(f"CODEML_PATH: {args.codeml_path}")
    sys.stdout.flush()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load counters from pickle
    cache_file = os.path.join(args.output_dir, 'results_cache.pkl')
    load_cache(cache_file)

    # Initialize SQLite DB
    db_path = os.path.join(args.output_dir, 'pairwise_results.sqlite')
    db_conn = init_sqlite_db(db_path)

    # Pre-load existing CSV data into the SQLite cache
    csv_files_to_load = [
        f for f in glob.glob(os.path.join(args.output_dir, '*.csv'))
        if not f.endswith('_haplotype_stats.csv')
    ]

    print(f"Found {len(csv_files_to_load)} CSV files to scan for existing results.")
    sys.stdout.flush()

    # Force concurrency to 32
    parallel_csv = min(NUM_PARALLEL, len(csv_files_to_load))
    count_loaded_from_csv = 0

    if csv_files_to_load:
        with multiprocessing.Pool(processes=parallel_csv) as pool:
            all_csv_data = []
            for result in tqdm(
                pool.imap_unordered(_read_csv_rows, csv_files_to_load, chunksize=50),
                total=len(csv_files_to_load),
                desc="Loading CSVs"
            ):
                all_csv_data.append(result)
        for result_list in all_csv_data:
            for cache_key, record_tuple in result_list:
                if not db_has_key(db_conn, cache_key):
                    db_insert_or_ignore(db_conn, cache_key, record_tuple)
                    count_loaded_from_csv += 1

    print(f"Preloaded {count_loaded_from_csv} comparison records from CSV into the SQLite DB.")

    preload_transcript_coords('../hg38.knownGene.gtf')

    print("Gathering all existing CSVs to skip up front.")
    csv_files = glob.glob(os.path.join(args.output_dir, '*.csv'))
    completed_cds_ids = set()
    for csv_file in csv_files:
        if csv_file.endswith('_haplotype_stats.csv'):
            continue
        base_name = os.path.basename(csv_file).replace('.csv', '')
        completed_cds_ids.add(base_name)

    print(f"Found {len(completed_cds_ids)} completed CSV sets. Now gathering .phy files...")
    sys.stdout.flush()
    phy_files = glob.glob(os.path.join(args.phy_dir, '*.phy'))
    total_files = len(phy_files)
    print(f"Found {total_files} phy files.")
    sys.stdout.flush()

    filename_pattern = re.compile(r'^group_(\d+)_([A-Za-z0-9\.]+)_chr_([A-Za-z0-9]+)_combined\.phy$')

    cds_meta_all = []

    print("Parsing all phy files for metadata...")
    sys.stdout.flush()
    for phy_file in phy_files:
        basename = os.path.basename(phy_file)
        m = filename_pattern.match(basename)
        if not m:
            continue
        group_num = m.group(1)
        transcript_id = m.group(2)

        chr_, st, en = get_transcript_coordinates(transcript_id)
        if chr_ is None or st is None or en is None:
            continue

        sequences, duplicates = parse_phy_file(phy_file)
        valid_seq_count = len(sequences)
        if valid_seq_count == 0:
            continue

        cds_id = basename.replace('.phy', '')
        cds_meta_all.append((cds_id, transcript_id, chr_, st, en, sequences))

    print(f"Metadata parsing complete. {len(cds_meta_all)} CDS entries collected.")
    sys.stdout.flush()

    print("Clustering CDS by coordinates to select best representative...")
    sys.stdout.flush()
    clusters = cluster_by_coordinates(cds_meta_all)

    cds_id_to_data = {c[0]: c for c in cds_meta_all}
    allowed_cds_ids = set()
    for cluster in clusters:
        best_id = None
        best_len = -1
        for cid in cluster:
            _, _, _, _, _, seqs = cds_id_to_data[cid]
            if seqs:
                length = len(next(iter(seqs.values())))
                if length > best_len:
                    best_len = length
                    best_id = cid
        if best_id:
            allowed_cds_ids.add(best_id)

    all_phy_filtered = [f for f in phy_files if os.path.basename(f).replace('.phy', '') in allowed_cds_ids]
    print(f"Filtering out {len(all_phy_filtered)} .phy files by coordinate clustering...")

    final_phy_files = []
    for pf in all_phy_filtered:
        cds_id = os.path.basename(pf).replace('.phy', '')
        if cds_id in completed_cds_ids:
            print(f"Skipping {pf}, CSV/haplotype CSV are already complete.")
        else:
            final_phy_files.append(pf)

    print(f"After skipping completed CSVs up front, we have {len(final_phy_files)} files to process.")

    print("Estimating total comparisons...")
    sys.stdout.flush()
    def quick_estimate(phy):
        base = os.path.basename(phy)
        m = filename_pattern.match(base)
        group_num = m.group(1)
        return estimate_one_file(phy, group_num)

    results = [quick_estimate(p) for p in final_phy_files]
    total_cds = sum(r[0] for r in results)
    total_comps = sum(r[1] for r in results)
    GLOBAL_COUNTERS['total_cds'] = total_cds
    GLOBAL_COUNTERS['total_comparisons'] = total_comps

    # Extract counters
    GLOBAL_TOTAL_SEQS = GLOBAL_COUNTERS['total_seqs']
    GLOBAL_INVALID_SEQS = GLOBAL_COUNTERS['invalid_seqs']
    GLOBAL_DUPLICATES = GLOBAL_COUNTERS['duplicates']
    GLOBAL_TOTAL_CDS = GLOBAL_COUNTERS['total_cds']
    GLOBAL_TOTAL_COMPARISONS = GLOBAL_COUNTERS['total_comparisons']

    valid_sequences = GLOBAL_TOTAL_SEQS - GLOBAL_INVALID_SEQS
    valid_percentage = (valid_sequences / GLOBAL_TOTAL_SEQS * 100) if GLOBAL_TOTAL_SEQS > 0 else 0

    logging.info("=== START OF RUN SUMMARY ===")
    logging.info(f"Total PHYLIP files: {total_files}")
    logging.info(f"Total sequences encountered: {GLOBAL_TOTAL_SEQS}")
    logging.info(f"Invalid sequences: {GLOBAL_INVALID_SEQS}")
    logging.info(f"Duplicates: {GLOBAL_DUPLICATES}")
    logging.info(f"Stop codons: {GLOBAL_COUNTERS['stop_codons']}")
    logging.info(f"Valid sequences: {valid_sequences} ({valid_percentage:.2f}%)")
    logging.info(f"Total CDS after clustering: {GLOBAL_TOTAL_CDS}")
    logging.info(f"Expected comparisons: {GLOBAL_TOTAL_COMPARISONS}")

    # Count how many are already in the DB
    cached_results_count = db_count_keys(db_conn)
    remaining = GLOBAL_TOTAL_COMPARISONS - cached_results_count
    logging.info(f"Cache: {cached_results_count} results. {remaining} remain.")

    if GLOBAL_TOTAL_COMPARISONS > 0:
        ETA_DATA['start_time'] = time.time()

    start_time = time.time()
    completed_comparisons = cached_results_count

    def run_cds_file(phy_file, group_num, output_dir, codeml_path, db_conn):
        print(f"Running CDS file: {phy_file}")
        sys.stdout.flush()
        cds_id = os.path.basename(phy_file).replace('.phy', '')
        mode_suffix = "_all" if COMPARE_BETWEEN_GROUPS else ""
        output_csv = os.path.join(output_dir, f'{cds_id}{mode_suffix}.csv')
        haplotype_output_csv = os.path.join(output_dir, f'{cds_id}{mode_suffix}_haplotype_stats.csv')

        if os.path.exists(output_csv):
            print("Output already exists, skipping file.")
            sys.stdout.flush()
            return

        sequences, has_duplicates = parse_phy_file(phy_file)
        if not sequences:
            print("No valid sequences, skipping file.")
            sys.stdout.flush()
            return

        sample_groups = {}
        for s in sequences.keys():
            sample_groups[s] = int(group_num)

        if COMPARE_BETWEEN_GROUPS:
            all_samples = list(sample_groups.keys())
            all_pairs = list(combinations(all_samples, 2))
        else:
            g0 = [x for x, y in sample_groups.items() if y == 0]
            g1 = [x for x, y in sample_groups.items() if y == 1]
            all_pairs = list(combinations(g0, 2)) + list(combinations(g1, 2))

        if not all_pairs:
            print("No pairs to compare, skipping.")
            sys.stdout.flush()
            return

        temp_dir = os.path.join(output_dir, 'temp', cds_id)
        os.makedirs(temp_dir, exist_ok=True)

        to_compute = []
        for pair in all_pairs:
            check_key = f"{cds_id}::{pair[0]}::{pair[1]}::{COMPARE_BETWEEN_GROUPS}"
            if not db_has_key(db_conn, check_key):
                to_compute.append(pair)

        if not to_compute:
            print(f"All {len(all_pairs)} pairs for {cds_id} are already in DB, skipping codeml.")
            sys.stdout.flush()
            return

        print(f"Computing {len(to_compute)} new pairs out of {len(all_pairs)} total.")
        sys.stdout.flush()

        pool_args = [
            (pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, None)
            for pair in to_compute
        ]
        # Force concurrency to 32 for pairwise computations
        num_processes = min(NUM_PARALLEL, len(to_compute))
        print(f"Running codeml on {len(to_compute)} pairs with {num_processes} processes.")
        sys.stdout.flush()

        results_accum = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            for r in pool.imap_unordered(process_pair, pool_args, chunksize=10):
                if r is not None:
                    results_accum.append(r)

        for r in results_accum:
            seq1, seq2, grp1, grp2, dn, ds, omega, cid = r
            newkey = f"{cid}::{seq1}::{seq2}::{COMPARE_BETWEEN_GROUPS}"
            record_tuple = (seq1, seq2, grp1, grp2, dn, ds, omega, cid)
            db_insert_or_ignore(db_conn, newkey, record_tuple)

        # Gather final results from the DB for this CDS
        relevant_keys = []
        for pair in all_pairs:
            final_key = f"{cds_id}::{pair[0]}::{pair[1]}::{COMPARE_BETWEEN_GROUPS}"
            relevant_keys.append(final_key)

        # We load them back out of the DB to produce the final CSV
        results_for_csv = []
        query_placeholders = ",".join(["?"] * len(relevant_keys))
        # We'll chunk if relevant_keys is huge
        CHUNK_SIZE = 10000
        start_index = 0
        while start_index < len(relevant_keys):
            chunk_keys = relevant_keys[start_index : start_index + CHUNK_SIZE]
            in_clause = ",".join(["?"] * len(chunk_keys))
            sql = f"""
                SELECT cache_key, seq1, seq2, group1, group2, dN, dS, omega, cds
                FROM pairwise_cache
                WHERE cache_key IN ({in_clause})
            """
            rows = db_conn.execute(sql, chunk_keys).fetchall()
            for row in rows:
                # row = (cache_key, seq1, seq2, group1, group2, dN, dS, omega, cds)
                results_for_csv.append(row[1:])  # skipping cache_key index 0
            start_index += CHUNK_SIZE

        # results_for_csv items are (seq1, seq2, group1, group2, dN, dS, omega, cds)
        df = pd.DataFrame(
            results_for_csv,
            columns=['Seq1','Seq2','Group1','Group2','dN','dS','omega','CDS']
        )
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        sys.stdout.flush()

        # Build haplotype stats
        hap_stats = []
        for sample in sequences.keys():
            sample_df = df[(df['Seq1'] == sample) | (df['Seq2'] == sample)]
            omega_vals = sample_df['omega'].dropna()
            omega_vals = omega_vals[~omega_vals.isin([-1, 99])]
            if not omega_vals.empty:
                mean_omega = omega_vals.mean()
                median_omega = omega_vals.median()
            else:
                mean_omega = np.nan
                median_omega = np.nan
            hap_stats.append({
                'Haplotype': sample,
                'Group': sample_groups[sample],
                'CDS': cds_id,
                'Mean_dNdS': mean_omega,
                'Median_dNdS': median_omega,
                'Num_Comparisons': len(omega_vals)
            })
        hap_df = pd.DataFrame(hap_stats)
        hap_df.to_csv(haplotype_output_csv, index=False)
        print(f"Haplotype stats saved to {haplotype_output_csv}")
        sys.stdout.flush()

    print("Processing each allowed CDS file...")
    sys.stdout.flush()
    completed_comparisons = db_count_keys(db_conn)

    for idx, phy_file in enumerate(final_phy_files, 1):
        basename = os.path.basename(phy_file)
        m = filename_pattern.match(basename)
        group_num = m.group(1)
        logging.info(f"Processing file {idx}/{len(final_phy_files)}: {phy_file}")
        print(f"Processing file {idx}/{len(final_phy_files)}: {phy_file}")
        sys.stdout.flush()

        old_size = db_count_keys(db_conn)
        run_cds_file(phy_file, group_num, args.output_dir, args.codeml_path, db_conn)
        new_size = db_count_keys(db_conn)

        newly_done = new_size - old_size
        completed_comparisons += newly_done
        percent = (
            (completed_comparisons / GLOBAL_TOTAL_COMPARISONS) * 100
            if GLOBAL_TOTAL_COMPARISONS > 0
            else 0
        )
        logging.info(f"Overall: {completed_comparisons}/{GLOBAL_TOTAL_COMPARISONS} comps ({percent:.2f}%)")
        print(f"Overall progress: {completed_comparisons}/{GLOBAL_TOTAL_COMPARISONS} comps ({percent:.2f}%)")
        sys.stdout.flush()
        print_eta(completed_comparisons, GLOBAL_TOTAL_COMPARISONS, ETA_DATA['start_time'], ETA_DATA)

    end_time = time.time()
    final_invalid_pct = (
        (GLOBAL_INVALID_SEQS / GLOBAL_TOTAL_SEQS * 100) if GLOBAL_TOTAL_SEQS > 0 else 0
    )

    logging.info("=== END OF RUN SUMMARY ===")
    logging.info(f"Total PHYLIP: {total_files}")
    logging.info(f"Total seq: {GLOBAL_TOTAL_SEQS}")
    logging.info(f"Invalid seq: {GLOBAL_INVALID_SEQS} ({final_invalid_pct:.2f}%)")
    logging.info(f"Sequences with stop codons: {GLOBAL_COUNTERS['stop_codons']}")
    logging.info(f"Duplicates: {GLOBAL_DUPLICATES}")
    logging.info(f"Total CDS (final): {GLOBAL_TOTAL_CDS}")
    logging.info(f"Expected comps: {GLOBAL_TOTAL_COMPARISONS}")
    logging.info(f"Completed comps: {completed_comparisons}")
    logging.info(f"Total time: {(end_time - ETA_DATA['start_time'])/60:.2f} min"
                 if ETA_DATA['start_time'] else f"Total time: {((end_time - start_time)/60):.2f} min")

    # Save updated counters
    save_cache(cache_file, GLOBAL_COUNTERS)

    # Save the updated validation cache
    try:
        with open(VALIDATION_CACHE_FILE, 'wb') as f:
            pickle.dump(VALIDATION_CACHE, f)
        print(f"Validation cache saved with {len(VALIDATION_CACHE)} entries.")
    except Exception as e:
        print(f"Could not save validation cache: {e}")

    def consolidate_all_csvs(csv_dir, final_csv='all_pairwise_results.csv'):
        all_dfs = []
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        for cf in csv_files:
            if cf.endswith('_haplotype_stats.csv'):
                continue
            df = pd.read_csv(cf)
            all_dfs.append(df)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            outpath = os.path.join(csv_dir, final_csv)
            combined.to_csv(outpath, index=False)
            print(f"Final combined CSV with {len(combined)} rows saved to {outpath}")
        else:
            print("No CSV files found to combine at the end.")

    consolidate_all_csvs(args.output_dir, 'all_pairwise_results.csv')

    db_conn.close()
    logging.info("dN/dS analysis done.")
    print("dN/dS analysis done.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
