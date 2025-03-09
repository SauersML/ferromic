"""
dN/dS Analysis Script using PAML's CODEML

This script calculates pairwise dN/dS values using PAML's CODEML program.

1. Input Files:
    Each input file is a PHYLIP-like file with lines consisting of:
        SAMPLE_NAME_L/R + SEQUENCE (no spaces). For example:
            ABC_XYZ_HG01352_LACGGAGTAC...
    Where each sample name ends with "_L" or "_R" before the sequence.

    The input file names follow a pattern including a transcript ID and chromosome info:
        group_0_ENST00000706755.1_chr_19_start_..._combined.phy
        group_1_ENST00000704003.1_chr_7_start_..._combined.phy
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
import hashlib
from tqdm import tqdm

# --------------------------------------------------------------------------------
# USER CONFIG
# --------------------------------------------------------------------------------

COMPARE_BETWEEN_GROUPS = False  # If False, only compare within the same group.
VALIDATION_CACHE_FILE = 'validation_results.pkl'
NUM_PARALLEL = 96  # Maximum concurrency for pairwise codeml runs.

# --------------------------------------------------------------------------------
# LOGGING CONFIG
# --------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dnds_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --------------------------------------------------------------------------------
# GLOBAL DATA
# --------------------------------------------------------------------------------

manager = multiprocessing.Manager()

# We keep counters in a manager dict so they can be shared safely across processes if needed.
GLOBAL_COUNTERS = manager.dict({
    'invalid_seqs': 0,      # Total invalid sequences across all processed .phy
    'duplicates': 0,        # Duplicate sample names found
    'total_seqs': 0,        # Total valid sequences across all processed .phy
    'total_cds': 0,         # How many final CDS sets are processed
    'total_comparisons': 0, # How many pairwise comparisons are planned
    'stop_codons': 0        # Number of sequences that had in-frame stop codons
})

# These are for ETA updates (time/rate).
ETA_DATA = {
    'start_time': None,
    'completed': 0,
    'rate_smoothed': None,  # Exponential moving average for rate
    'alpha': 0.2,
}

# Holds the in-memory cache of sequences previously validated (keyed by (phy_filename, sample_name, seq_hash)).
# Also stored in a pickle file (VALIDATION_CACHE_FILE) for next run.
VALIDATION_CACHE = {}

# Dictionary to store the fully parsed data from each .phy file exactly once.
# Key: absolute path of the phy file; Value: dict with:
#   {
#       'sequences': {sample_name -> validated_seq, ...},
#       'duplicates_found': bool,
#       'local_invalid': int,
#       'local_stop_codons': int,
#       'local_total_seqs': int,
#       'local_duplicates': int
#   }
PARSED_PHY = {}

# Maps transcript_id -> (chrom, start, end) from the GTF
TRANSCRIPT_COORDS = {}


# --------------------------------------------------------------------------------
# HELPER FUNCTIONS: TIME/ETA
# --------------------------------------------------------------------------------

def print_eta(completed, total, start_time, eta_data):
    """
    Compute and log an estimated time of completion for the pairwise comparisons.
    """
    print("Calculating ETA for progress...")
    sys.stdout.flush()
    if total <= 0 or completed <= 0:
        logging.info(f"Progress: {completed}/{total}, ETA:N/A")
        return

    elapsed = time.time() - start_time
    current_rate = completed / elapsed  # comps/sec

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


# --------------------------------------------------------------------------------
# SEQUENCE VALIDATION
# --------------------------------------------------------------------------------

def validate_sequence(seq, filepath, sample_name, full_line):
    """
    Check the usual constraints:
      - non-empty
      - length divisible by 3
      - valid bases
      - no in-frame stop codons
    Return (validated_seq, stop_codon_found).
    If invalid, validated_seq=None and stop_codon_found is True if we specifically saw a TAA/TAG/TGA.
    """
    print(f"Validating sequence for sample {sample_name} from file {filepath}")
    if not seq:
        print("Invalid sequence: empty")
        return None, False

    seq = seq.upper()
    line_len = len(full_line)
    seq_len = len(seq)
    MAX_CDS_LENGTH = 3000000

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


# --------------------------------------------------------------------------------
# CODEML SETUP & EXECUTION
# --------------------------------------------------------------------------------

def create_paml_ctl(seqfile, outfile, working_dir):
    """
    Create a PAML codeml control file with the appropriate settings.
    """
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
    """
    Run the codeml process in working_dir with the given control file.
    """

    print(f"===== Starting run_codeml for ctl_path: {ctl_path} =====")
    print(f"Converting paths to absolute: ctl_path={ctl_path}, working_dir={working_dir}, codeml_path={codeml_path}")
    ctl_path = os.path.abspath(ctl_path)
    working_dir = os.path.abspath(working_dir)
    codeml_path = os.path.abspath(codeml_path)
    print(f"Absolute paths set: ctl_path={ctl_path}, working_dir={working_dir}, codeml_path={codeml_path}")

    dst_ctl = os.path.join(working_dir, 'codeml.ctl')
    print(f"Destination control file path: {dst_ctl}")
    print(f"Checking if ctl_path and dst_ctl are the same file: {os.path.samefile(ctl_path, dst_ctl) if os.path.exists(dst_ctl) else 'dst_ctl does not exist yet'}")
    if not os.path.samefile(ctl_path, dst_ctl) if os.path.exists(dst_ctl) else True:
        print(f"Copying control file from {ctl_path} to {dst_ctl}")
        try:
            shutil.copy2(ctl_path, dst_ctl)
            print(f"Successfully copied control file to {dst_ctl}")
        except Exception as e:
            print(f"ERROR: Failed to copy control file to {dst_ctl}: {e}")
            return False
    else:
        print(f"No copy needed, ctl_path and dst_ctl are the same")

    print(f"Verifying control file exists at {dst_ctl}: {os.path.exists(dst_ctl)}")
    if not os.path.exists(dst_ctl):
        print(f"ERROR: Control file {dst_ctl} does not exist after copy attempt")
        return False

    print(f"Checking if codeml executable exists at {codeml_path}: {os.path.exists(codeml_path)}")
    if not os.path.exists(codeml_path):
        print(f"ERROR: codeml executable not found at {codeml_path}")
        return False
    print(f"Checking if codeml is executable: {os.access(codeml_path, os.X_OK)}")
    if not os.access(codeml_path, os.X_OK):
        print(f"ERROR: codeml at {codeml_path} is not executable")
        return False

    cmd = [codeml_path]
    cmdstr = " ".join(cmd)
    print(f"\nPreparing to run command: {cmdstr} in working_dir: {working_dir}")

    print(f"Listing current contents of working_dir {working_dir}: {os.listdir(working_dir) if os.path.exists(working_dir) else 'Directory does not exist'}")
    print(f"Starting subprocess.Popen with timeout=30s")
    try:
        print(f"Launching CODEML process with cmd={cmd}, cwd={working_dir}")
        process = subprocess.Popen(
            cmd,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        print(f"Process started with PID: {process.pid}")
        print("Communicating with process (sending newline input)")
        stdout, stderr = process.communicate(input=b'\n', timeout=30)
        print(f"Process completed. Return code: {process.returncode}")
        print(f"CODEML stdout:\n{stdout.decode().strip()}")
        print(f"CODEML stderr:\n{stderr.decode().strip()}")

        if process.returncode != 0:
            print(f"ERROR: codeml exited with non-zero return code {process.returncode}. Stderr:\n{stderr.decode().strip()}")
            return False

        results_file = os.path.join(working_dir, 'results.txt')
        rst_file = os.path.join(working_dir, 'rst')  # Added for extra logging
        print(f"Checking for results file at {results_file}: {os.path.exists(results_file)}")
        print(f"Checking for rst file at {rst_file}: {os.path.exists(rst_file)}")
        print(f"Updated contents of working_dir {working_dir}: {os.listdir(working_dir) if os.path.exists(working_dir) else 'Directory does not exist'}")

        if os.path.exists(results_file):
            print(f"SUCCESS: results.txt file created at {results_file}")
            with open(results_file, 'r') as f:
                print(f"Contents of results.txt:\n{f.read().strip()}")
            if os.path.exists(rst_file):
                print(f"rst file also found at {rst_file}")
                with open(rst_file, 'r') as f:
                    print(f"Contents of rst (first 5 lines):\n''.join(f.readlines()[:5])")
            else:
                print(f"WARNING: rst file not found at {rst_file}, but results.txt exists")
            print(f"CODEML run successful, returning True")
            return True
        else:
            print(f"FAILURE: no results.txt file created at {results_file}")
            if os.path.exists(rst_file):
                print(f"NOTE: rst file exists at {rst_file} despite no results.txt")
                with open(rst_file, 'r') as f:
                    print(f"Contents of rst:\n{f.read().strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"ERROR: CODML timed out after 30 seconds")
        process.kill()
        stdout, stderr = process.communicate()
        print(f"Timeout stdout:\n{stdout.decode().strip()}")
        print(f"Timeout stderr:\n{stderr.decode().strip()}")
        print(f"Process killed, returning False")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected exception running codeml: {e}")
        import traceback
        print(f"Stack trace:\n{traceback.format_exc()}")
        print(f"Returning False due to exception")
        return False
    finally:
        print(f"===== Finished run_codeml for working_dir: {working_dir} =====")


def parse_codeml_output(outfile_dir):
    """
    Parse the 'rst' file from codeml output to extract the dN, dS, and omega values.
    Must remain unchanged per instructions.
    """
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


def load_temp_dir_results(temp_dir, db_conn):
    """
    Scan /dev/shm/paml_output/temp subdirectories for existing CODEML results. If valid, insert them
    into the DB so we can skip re-running those pairs in future steps. If invalid or incomplete,
    do not insert, so they will be re-run.
    """
    if not os.path.exists(temp_dir):
        return
    for cds_dir in os.listdir(temp_dir):
        full_cds_path = os.path.join(temp_dir, cds_dir)
        if not os.path.isdir(full_cds_path):
            continue
        cds_id = cds_dir
        subdirs = os.listdir(full_cds_path)
        for sd in subdirs:
            pair_dir = os.path.join(full_cds_path, sd)
            if not os.path.isdir(pair_dir):
                continue
            pair_name = sd
            underscore_parts = pair_name.split('_')
            if len(underscore_parts) < 2:
                continue
            seq1_name = underscore_parts[0]
            seq2_name = '_'.join(underscore_parts[1:])
            cache_key = f"{cds_id}::{seq1_name}::{seq2_name}::{COMPARE_BETWEEN_GROUPS}"
            if db_has_key(db_conn, cache_key):
                continue
            dn, ds, omega = parse_codeml_output(pair_dir)
            if dn is None or ds is None or omega is None:
                continue
            group1 = 0
            group2 = 0
            record_tuple = (seq1_name, seq2_name, group1, group2, dn, ds, omega, cds_id)
            db_insert_or_ignore(db_conn, cache_key, record_tuple)


# --------------------------------------------------------------------------------
# PARSING PHY FILES EXACTLY ONCE
# --------------------------------------------------------------------------------

def parse_phy_file_once(filepath):
    """
    Parse a .phy file with a header line 'N M', where N is the number of sequences
    and M is the length of each sequence. The next N lines each contain a sample
    name ending with '_L' or '_R', immediately followed by a sequence of M characters
    (no whitespace). Returns a dictionary with sequence data and parsing statistics.
    """
    print(f"Parsing phy file (once): {filepath}")
    data = {
        'sequences': {},           # {sample_name: sequence}
        'duplicates_found': False, # Whether duplicates were encountered
        'local_invalid': 0,        # Count of invalid sequences
        'local_stop_codons': 0,    # Count of sequences with stop codons
        'local_total_seqs': 0,     # Count of valid sequences
        'local_duplicates': 0      # Count of duplicate sequences renamed
    }

    if not os.path.exists(filepath):
        print(f"File does not exist: {filepath}")
        return data

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            # Parse the header
            header_line = f.readline().strip()
            if not header_line:
                print("Empty file or missing header line.")
                return data
            parts = header_line.split()
            if len(parts) != 2:
                print("Invalid PHYLIP header format (expected 'N M').")
                return data
            try:
                num_seqs = int(parts[0])
                seq_len = int(parts[1])
            except ValueError:
                print("PHYLIP header does not contain valid integers.")
                return data

            # Parse each sequence line
            for i in range(num_seqs):
                line = f.readline().strip()
                if not line:
                    print("File ended before reading all sequences.")
                    break
                if len(line) < seq_len:
                    print(f"Skipping line with insufficient length: {len(line)} < {seq_len}")
                    data['local_invalid'] += 1
                    continue

                # Extract sequence (last seq_len characters)
                sequence = line[-seq_len:]
                # Extract sample name (everything before the sequence)
                orig_name = line[:-seq_len]

                # Check if name ends with _L or _R
                if not (orig_name.endswith('_L') or orig_name.endswith('_R')):
                    print(f"Skipping line with invalid sample name format: {orig_name}")
                    data['local_invalid'] += 1
                    continue

                # Construct short sample name (consistent with prior logic)
                name_parts = orig_name.split('_')
                if len(name_parts) < 4:
                    print(f"Skipping unexpected sample naming: {orig_name}")
                    data['local_invalid'] += 1
                    continue
                first = name_parts[0][:3]
                second = name_parts[1][:3]
                hg_part = name_parts[-2]
                group = name_parts[-1]
                md5_val = hashlib.md5(hg_part.encode('utf-8')).hexdigest()
                hash_str = md5_val[:2]
                sample_name = f"{first}{second}{hash_str}_{group}"

                # Validate sequence
                seq_hash = hashlib.md5(sequence.encode('utf-8')).hexdigest()
                cache_key = (os.path.basename(filepath), sample_name, seq_hash)
                if cache_key in VALIDATION_CACHE:
                    validated_seq, stop_codon_found = VALIDATION_CACHE[cache_key]
                    print(f"Skipping validation for {sample_name}, found in cache.")
                else:
                    # Placeholder for validate_sequence; adjust as needed
                    validated_seq, stop_codon_found = validate_sequence(
                        sequence, filepath, sample_name, line
                    )
                    VALIDATION_CACHE[cache_key] = (validated_seq, stop_codon_found)

                if validated_seq is None:
                    data['local_invalid'] += 1
                    if stop_codon_found:
                        data['local_stop_codons'] += 1
                else:
                    data['local_total_seqs'] += 1
                    if sample_name in data['sequences']:
                        data['duplicates_found'] = True
                        data['local_duplicates'] += 1
                        # Rename duplicate
                        base_name = sample_name[:2] + sample_name[3:]
                        dup_count = sum(1 for s in data['sequences']
                                        if s[:2] + s[3:] == base_name)
                        new_name = sample_name[:2] + str(dup_count) + sample_name[3:]
                        data['sequences'][new_name] = validated_seq
                    else:
                        data['sequences'][sample_name] = validated_seq

    except Exception as e:
        print(f"Error reading or parsing file {filepath}: {e}")

    # Print summary
    print(f"Finished parsing {filepath}. Valid: {data['local_total_seqs']}, "
          f"Invalid: {data['local_invalid']}, Stop codons: {data['local_stop_codons']}, "
          f"Duplicates: {data['local_duplicates']} (found: {data['duplicates_found']})")

    return data

# --------------------------------------------------------------------------------
# CACHING GLOBAL COUNTERS
# --------------------------------------------------------------------------------

def load_cache(cache_file):
    """
    Load the global counters from a pickle, if available.
    Does NOT load the sequence cache (that's handled separately).
    """
    print(f"Loading counters cache from {cache_file}")
    sys.stdout.flush()
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            if data:
                for k, v in data.items():
                    GLOBAL_COUNTERS[k] = v
        print(f"Cache loaded with {len(data)} entries (counters).")
        sys.stdout.flush()
    else:
        print("No counters cache file found, starting fresh.")
        sys.stdout.flush()


def save_cache(cache_file, cache_data):
    """
    Save only counters to a pickle, not pairwise results.
    """
    print(f"Saving counters cache to {cache_file} with {len(cache_data)} entries.")
    sys.stdout.flush()
    with open(cache_file, 'wb') as f:
        normal_dict = dict(cache_data)
        pickle.dump(normal_dict, f)
    print("Counters cache saved.")
    sys.stdout.flush()


# --------------------------------------------------------------------------------
# SQLITE PAIRWISE DB
# --------------------------------------------------------------------------------

def init_sqlite_db(db_path):
    """
    Initialize (or reuse) a SQLite database to store pairwise results,
    with table 'pairwise_cache' keyed by 'cache_key'.
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
    Insert a record into pairwise_cache if 'cache_key' doesn't exist.
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
    Return True if 'cache_key' exists in pairwise_cache.
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
    """
    A helper function to read existing CSV results, returning
    a list of (cache_key, record_tuple) to preload into the DB.
    """
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


# --------------------------------------------------------------------------------
# PAIRWISE COMPARISON LOGIC
# --------------------------------------------------------------------------------

def process_pair(args):
    """
    Worker function for pairwise dN/dS. 
    - pair: (seq1_name, seq2_name)
    - sequences: dict of {name->seq}
    - sample_groups: dict of {name->group}
    - cds_id: unique identifier for the alignment
    - codeml_path, temp_dir, ...
    Returns (seq1_name, seq2_name, group1, group2, dN, dS, omega, cds_id) or None if skipped.
    """
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, _ = args
    seq1_name, seq2_name = pair

    print(f"===== STARTING process_pair for pair: {seq1_name} vs {seq2_name} in CDS: {cds_id} =====")
    logging.debug(f"process_pair started: seq1={seq1_name}, seq2={seq2_name}, cds_id={cds_id}")
    print(f"Unpacking args: pair={pair}, cds_id={cds_id}, codeml_path={codeml_path}, temp_dir={temp_dir}")
    print(f"Sequences dict has {len(sequences)} entries")
    print(f"Sample_groups dict has {len(sample_groups)} entries")

    # Check for missing sequences
    print(f"Checking if sequences exist: {seq1_name} in sequences: {seq1_name in sequences}, {seq2_name} in sequences: {seq2_name in sequences}")
    if seq1_name not in sequences or seq2_name not in sequences:
        print(f"EXIT: One of the sequences is missing ({seq1_name} or {seq2_name}), returning None.")
        logging.warning(f"process_pair exiting early: missing sequence for {seq1_name} or {seq2_name} in {cds_id}")
        print(f"NOT CALLING run_codeml due to missing sequence")
        sys.stdout.flush()
        return None
    print(f"Both sequences found: {seq1_name} and {seq2_name}")

    # Fetch group info
    group1 = sample_groups.get(seq1_name)
    group2 = sample_groups.get(seq2_name)
    print(f"Group1 for {seq1_name}: {group1}, Group2 for {seq2_name}: {group2}")
    logging.debug(f"Groups assigned: group1={group1}, group2={group2}")

    # Check for cross-group comparison
    print(f"COMPARE_BETWEEN_GROUPS: {COMPARE_BETWEEN_GROUPS}, group1 == group2: {group1 == group2}")
    if not COMPARE_BETWEEN_GROUPS and group1 != group2:
        print(f"EXIT: Not comparing between groups, and groups differ (group1={group1}, group2={group2}). Skipping.")
        logging.info(f"process_pair skipping: cross-group comparison not allowed for {seq1_name} vs {seq2_name}")
        print(f"NOT CALLING run_codeml due to group mismatch")
        sys.stdout.flush()
        return None
    print(f"Group check passed, proceeding")

    # Check for identical sequences
    seq1 = sequences[seq1_name]
    seq2 = sequences[seq2_name]
    print(f"Sequence lengths: {seq1_name}={len(seq1)}, {seq2_name}={len(seq2)}")
    print(f"Comparing sequences for identity: {seq1_name} vs {seq2_name}")
    are_identical = seq1 == seq2
    print(f"Sequences identical: {are_identical}")
    if are_identical:
        print(f"EXIT: Sequences are identical, returning special marker (omega=-1)")
        logging.info(f"process_pair found identical sequences for {seq1_name} vs {seq2_name}, omega=-1")
        print(f"NOT CALLING run_codeml due to identical sequences")
        result = (seq1_name, seq2_name, group1, group2, 0.0, 0.0, -1.0, cds_id)
        print(f"Returning: {result}")
        sys.stdout.flush()
        return result
    print(f"Sequences differ, proceeding to CODEML execution")
    logging.debug(f"Sequences differ: {seq1_name} and {seq2_name}, length diff={abs(len(seq1) - len(seq2))}")

    # Prepare temporary directory
    working_dir = os.path.join(temp_dir, f'{seq1_name}_{seq2_name}')
    print(f"Preparing working directory: {working_dir}")
    logging.debug(f"Creating working_dir: {working_dir}")
    try:
        os.makedirs(working_dir, exist_ok=True)
        print(f"Successfully created/ensured working_dir: {working_dir}")
    except Exception as e:
        print(f"ERROR: Failed to create working_dir {working_dir}: {e}")
        logging.error(f"Failed to create working_dir: {e}")
        return None
    print(f"Directory contents before files: {os.listdir(working_dir) if os.path.exists(working_dir) else 'empty'}")

    # Create seqfile.phy
    seqfile = os.path.join(working_dir, 'seqfile.phy')
    print(f"Creating seqfile.phy at: {seqfile}")
    logging.debug(f"Writing seqfile: {seqfile}")
    try:
        with open(seqfile, 'w') as f:
            seq_content = f" 2 {len(seq1)}\n{seq1_name}  {seq1}\n{seq2_name}  {seq2}\n"
            f.write(seq_content)
        print(f"Created seqfile.phy for codeml input: {seqfile}")
        print(f"Seqfile content preview: {seq_content[:100]}...")
        logging.info(f"seqfile.phy written for {seq1_name} vs {seq2_name}")
    except Exception as e:
        print(f"ERROR: Failed to write seqfile {seqfile}: {e}")
        logging.error(f"seqfile write failed: {e}")
        return None
    print(f"Confirming seqfile exists: {os.path.exists(seqfile)}")

    # Create tree.txt
    treefile = os.path.join(working_dir, 'tree.txt')
    print(f"Creating tree.txt at: {treefile}")
    logging.debug(f"Writing treefile: {treefile}")
    try:
        with open(treefile, 'w') as f:
            tree_content = f"({seq1_name},{seq2_name});\n"
            f.write(tree_content)
        print(f"Created tree.txt for codeml input: {treefile}")
        print(f"Treefile content: {tree_content.strip()}")
        logging.info(f"tree.txt written for {seq1_name} vs {seq2_name}")
    except Exception as e:
        print(f"ERROR: Failed to write treefile {treefile}: {e}")
        logging.error(f"treefile write failed: {e}")
        return None
    print(f"Confirming treefile exists: {os.path.exists(treefile)}")

    # Prepare to call run_codeml
    print(f"Calling create_paml_ctl with seqfile={seqfile}, outfile='results.txt', working_dir={working_dir}")
    ctl_path = create_paml_ctl(seqfile, 'results.txt', working_dir)
    print(f"create_paml_ctl returned ctl_path: {ctl_path}")
    logging.debug(f"ctl_path generated: {ctl_path}")

    print(f"===== PREPARING TO CALL run_codeml for {seq1_name} vs {seq2_name} =====")
    print(f"run_codeml args: ctl_path={ctl_path}, working_dir={working_dir}, codeml_path={codeml_path}")
    logging.info(f"Calling run_codeml for pair {seq1_name} vs {seq2_name} in {cds_id}")
    success = run_codeml(ctl_path, working_dir, codeml_path)
    print(f"run_codeml returned success: {success}")
    logging.debug(f"run_codeml result: success={success}")

    if not success:
        print(f"EXIT: Codeml run failed, returning None so it can be retried next time")
        logging.warning(f"run_codeml failed for {seq1_name} vs {seq2_name}")
        sys.stdout.flush()
        return None

    print(f"run_codeml succeeded, proceeding to parse output")

    # Parse CODEML output
    print(f"Calling parse_codeml_output with outfile_dir={working_dir}")
    dn, ds, omega = parse_codeml_output(working_dir)
    print(f"parse_codeml_output returned: dn={dn}, ds={ds}, omega={omega}")
    logging.debug(f"Parsed values: dn={dn}, ds={ds}, omega={omega}")
    if omega is None:
        print(f"Omega is None, setting to NaN")
        omega = np.nan
        logging.warning(f"parse_codeml_output returned None for omega, using NaN")

    # Final result
    result = (seq1_name, seq2_name, group1, group2, dn, ds, omega, cds_id)
    print(f"Pair processed successfully: dn={dn}, ds={ds}, omega={omega}")
    print(f"Returning result: {result}")
    logging.info(f"process_pair completed: {seq1_name} vs {seq2_name}, omega={omega}")
    print(f"run_codeml WAS CALLED and completed")
    print(f"===== FINISHED process_pair for pair: {seq1_name} vs {seq2_name} =====")
    sys.stdout.flush()
    return result


# --------------------------------------------------------------------------------
# GTF / TRANSCRIPT COORDS
# --------------------------------------------------------------------------------

def load_gtf_into_dict(gtf_file):
    """
    Parse a GTF for CDS lines only, building a dict of transcript_id->(chr, minStart, maxEnd).
    """
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
                # We won't attempt to unify different chromosomes; 
                # we just keep the first if there's mismatch.
                if chr_ == existing_chr:
                    if start < existing_start:
                        transcript_dict[tid][1] = start
                    if end > existing_end:
                        transcript_dict[tid][2] = end
    # Convert to tuple
    for k in transcript_dict:
        c, s, e = transcript_dict[k]
        transcript_dict[k] = (c, s, e)
    return transcript_dict


def preload_transcript_coords(gtf_file):
    """
    Load once into TRANSCRIPT_COORDS if not already loaded.
    """
    global TRANSCRIPT_COORDS
    if not TRANSCRIPT_COORDS:
        TRANSCRIPT_COORDS = load_gtf_into_dict(gtf_file)
        print(f"[INFO] GTF loaded into memory: {len(TRANSCRIPT_COORDS)} transcripts.")


def get_transcript_coordinates(transcript_id):
    """
    Return (chrom, start, end) for transcript_id, or (None, None, None) if not found.
    """
    print(f"Getting transcript coordinates for {transcript_id}")
    if transcript_id not in TRANSCRIPT_COORDS:
        print(f"No coordinates found for {transcript_id}")
        return (None, None, None)

    chrom, min_start, max_end = TRANSCRIPT_COORDS[transcript_id]
    print(f"Coordinates for {transcript_id}: {chrom}, {min_start}, {max_end}")
    return (chrom, min_start, max_end)


def parallel_handle_file(phy_file):
    global GLOBAL_COUNTERS, PARSED_PHY, db_conn, args
    logging.info(f"Processing file: {phy_file}")
    print(f"Processing file: {phy_file}")
    sys.stdout.flush()

    cds_id = os.path.basename(phy_file).replace('.phy', '')
    mode_suffix = "_all" if COMPARE_BETWEEN_GROUPS else ""
    output_csv = os.path.join(args.output_dir, f'{cds_id}{mode_suffix}.csv')
    haplotype_output_csv = os.path.join(args.output_dir, f'{cds_id}{mode_suffix}_haplotype_stats.csv')

    if os.path.exists(output_csv):
        print(f"Output {output_csv} already exists, skipping.")
        sys.stdout.flush()
        return None, []

    parsed_data = PARSED_PHY.get(phy_file, None)
    if not parsed_data or not parsed_data['sequences']:
        print(f"No valid sequences for {phy_file}, skipping.")
        sys.stdout.flush()
        return None, []

    GLOBAL_COUNTERS['invalid_seqs'] += parsed_data['local_invalid']
    GLOBAL_COUNTERS['stop_codons'] += parsed_data['local_stop_codons']
    GLOBAL_COUNTERS['total_seqs'] += parsed_data['local_total_seqs']
    GLOBAL_COUNTERS['duplicates'] += parsed_data['local_duplicates']

    sequences = parsed_data['sequences']
    group_num_match = re.search(r'^group_(\d+)_', os.path.basename(phy_file))
    if not group_num_match:
        print("Could not parse group number from filename. Defaulting to group=0.")
        group_num = 0
    else:
        group_num = int(group_num_match.group(1))

    sample_groups = {sname: group_num for sname in sequences.keys()}

    if COMPARE_BETWEEN_GROUPS:
        all_samples = list(sample_groups.keys())
        all_pairs = list(combinations(all_samples, 2))
    else:
        group_samples = list(sample_groups.keys())
        all_pairs = list(combinations(group_samples, 2))

    if not all_pairs:
        print(f"No pairs to compare for {phy_file}, skipping.")
        sys.stdout.flush()
        return None, []

    temp_dir = os.path.join(args.output_dir, 'temp', cds_id)
    os.makedirs(temp_dir, exist_ok=True)

    to_compute = []
    for pair in all_pairs:
        check_key = f"{cds_id}::{pair[0]}::{pair[1]}::{COMPARE_BETWEEN_GROUPS}"
        if not db_has_key(db_conn, check_key):
            to_compute.append(pair)

    print(f"File {phy_file} has {len(all_pairs)} total pairs, {len(to_compute)} remain to compute.")
    sys.stdout.flush()

    info_entry = {
        'phy_file': phy_file,
        'cds_id': cds_id,
        'output_csv': output_csv,
        'haplotype_csv': haplotype_output_csv,
        'all_pairs': all_pairs,
        'sequences': sequences,
        'sample_groups': sample_groups
    }

    tasks = []
    for pair in to_compute:
        task_args = (pair, sequences, sample_groups, cds_id, args.codeml_path, temp_dir, None)
        tasks.append(task_args)

    return info_entry, tasks

# --------------------------------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------------------------------

def main():
    print("Starting main process...")
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # We do NOT change how codeml is discovered, to keep environment consistent
    codeml_inferred = os.path.abspath(os.path.join(script_dir, '..', 'paml', 'bin', 'codeml'))
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='/dev/shm/paml_output', help='Directory to store output files.')
    parser.add_argument('--codeml_path', type=str, default=codeml_inferred, help='Path to codeml executable.')
    global args
    args = parser.parse_args()

    # Attempt to load existing validation cache (sequence-level checks)
    global VALIDATION_CACHE
    if os.path.exists(VALIDATION_CACHE_FILE):
        try:
            with open(VALIDATION_CACHE_FILE, 'rb') as f:
                VALIDATION_CACHE = pickle.load(f)
            print(f"Loaded validation cache with {len(VALIDATION_CACHE)} entries.")
        except Exception as e:
            print(f"Could not load validation cache: {e}")
    else:
        print("No prior validation cache found.")

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
    global db_conn
    db_conn = init_sqlite_db(db_path)
    load_temp_dir_results(os.path.join(args.output_dir, 'temp'), db_conn)

    # Pre-load any existing CSV data into the DB
    csv_files_to_load = [
        f for f in glob.glob(os.path.join(args.output_dir, '*.csv'))
        if not f.endswith('_haplotype_stats.csv')
    ]
    print(f"Found {len(csv_files_to_load)} CSV files to scan for existing results.")
    sys.stdout.flush()

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
            # Get the current number of rows in the pairwise_cache table before insertion
            initial_row_count = db_count_keys(db_conn)
            
            # Collect all records from CSV data to insert into the database
            records_to_insert = []
            for result_list in all_csv_data:
                for cache_key, record_tuple in result_list:
                    records_to_insert.append((cache_key,) + record_tuple)
            
            # Perform batch insertion of all records with a single commit
            if records_to_insert:
                print(f"Starting batch insertion of {len(records_to_insert)} records into the database.")
                sys.stdout.flush()
                try:
                    db_conn.executemany(
                        "INSERT OR IGNORE INTO pairwise_cache "
                        "(cache_key, seq1, seq2, group1, group2, dN, dS, omega, cds) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        records_to_insert
                    )
                    db_conn.commit()
                    # Calculate the number of new records inserted by comparing row counts
                    final_row_count = db_count_keys(db_conn)
                    count_loaded_from_csv = final_row_count - initial_row_count
                    print(f"Batch insertion completed: {count_loaded_from_csv} new records added to the database.")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"Batch insertion failed with error: {e}")
                    sys.stdout.flush()
                    count_loaded_from_csv = 0
            else:
                print("No records found to insert from CSV data.")
                sys.stdout.flush()
                count_loaded_from_csv = 0

    print(f"Preloaded {count_loaded_from_csv} comparison records from CSV into the SQLite DB.")

    # Load GTF coords once
    preload_transcript_coords('../hg38.knownGene.gtf')

    # Identify which .phy files already have final CSV (so we skip them)
    existing_csv_files = glob.glob(os.path.join(args.output_dir, '*.csv'))
    completed_cds_ids = set()
    for csv_file in existing_csv_files:
        if csv_file.endswith('_haplotype_stats.csv'):
            continue
        base_name = os.path.basename(csv_file).replace('.csv', '')
        completed_cds_ids.add(base_name)

    # Gather .phy files
    phy_files = glob.glob(os.path.join(args.phy_dir, '*.phy'))
    total_files = len(phy_files)
    print(f"Found {total_files} phy files in {args.phy_dir}")
    sys.stdout.flush()

    # Regex pattern to extract group_num, transcript_id, chromosome, start, end
    filename_pattern = re.compile(
        r'^group(\d+)_([^_]+)_([^_]+)_([^_]+)_chr([^_]+)_start(\d+)_end(\d+)\.phy$'
    )

    # Create a mapping of gene_id to lists of PHY files for each group
    gene_to_files = {}
    
    print("Parsing all .phy files and grouping by gene_id...")
    sys.stdout.flush()
    
    for phy_file in phy_files:
        basename = os.path.basename(phy_file)
        match = filename_pattern.match(basename)
        if not match:
            continue  # skip files that don't match pattern
            
        # Extract components from new filename format
        group_num = match.group(1)
        gene_name = match.group(2)
        gene_id = match.group(3)
        transcript_id = match.group(4)
        chromosome = match.group(5)
        start_pos = match.group(6)
        end_pos = match.group(7)
        
        # Parse the file once, store in PARSED_PHY
        parsed_data = parse_phy_file_once(phy_file)
        valid_seq_count = len(parsed_data['sequences'])
        
        if valid_seq_count == 0:
            # If no valid sequences remain, skip
            PARSED_PHY[phy_file] = parsed_data
            continue
            
        # Store the PHY file path by gene_id and group
        if gene_id not in gene_to_files:
            gene_to_files[gene_id] = {0: [], 1: []}
            
        group_num_int = int(group_num)
        if group_num_int in gene_to_files[gene_id]:
            gene_to_files[gene_id][group_num_int].append(phy_file)
            
        # Store parsed data
        PARSED_PHY[phy_file] = parsed_data
    
    # Find genes that have files in both group 0 and group 1
    valid_genes = {}
    for gene_id, group_files in gene_to_files.items():
        if 0 in group_files and 1 in group_files and group_files[0] and group_files[1]:
            valid_genes[gene_id] = group_files
    
    # Collect all files from valid genes (those present in both groups)
    all_phy_filtered = []
    for gene_id, group_files in valid_genes.items():
        all_phy_filtered.extend(group_files[0])
        all_phy_filtered.extend(group_files[1])
    
    print(f"Found {len(valid_genes)} genes with files in both groups.")
    print(f"Total PHY files to process: {len(all_phy_filtered)}")
    sys.stdout.flush()

    # Next, skip any that already have final CSV
    final_phy_files = []
    for pf in all_phy_filtered:
        cds_id = os.path.basename(pf).replace('.phy', '')
        if cds_id in completed_cds_ids:
            print(f"Skipping {pf}, final CSV/haplotype CSV present.")
            continue
        final_phy_files.append(pf)

    print(f"After skipping existing CSVs, we have {len(final_phy_files)} files to process for pairwise dN/dS.")

    # --------------------------------------------------------------------------------
    # ESTIMATE TOTAL COMPARISONS
    # --------------------------------------------------------------------------------

    # We'll do a quick pass to see how many total comparisons we expect.
    # We do NOT parse files again. Instead, we reuse PARSED_PHY data.
    def quick_estimate(phy_path):
        # Extract group_num from the filename
        base = os.path.basename(phy_path)
        m = filename_pattern.match(base)
        if not m:
            return (0, 0)
        group_num = m.group(1)

        pdict = PARSED_PHY.get(phy_path, None)
        if not pdict or not pdict['sequences']:
            return (0, 0)

        # Build pairs
        group_num_int = int(group_num)
        seq_names = list(pdict['sequences'].keys())
        sample_groups = {name: group_num_int for name in seq_names}

        if COMPARE_BETWEEN_GROUPS:
            pairs = list(combinations(seq_names, 2))
        else:
            # If group_num == 0, we only do within group0? Actually this is for 2-group scenario,
            # but we only have the single group from the file. So just do combos within that group.
            pairs = list(combinations(seq_names, 2))

        if len(pairs) > 0:
            return (1, len(pairs))
        else:
            return (0, 0)

    # Summarize
    totals = [quick_estimate(x) for x in final_phy_files]
    total_cds = sum(t[0] for t in totals)
    total_comps = sum(t[1] for t in totals)
    GLOBAL_COUNTERS['total_cds'] = total_cds
    GLOBAL_COUNTERS['total_comparisons'] = total_comps

    # --------------------------------------------------------------------------------
    # LOG START SUMMARY
    # --------------------------------------------------------------------------------
    total_seq_so_far = GLOBAL_COUNTERS['total_seqs']
    invalid_seq_so_far = GLOBAL_COUNTERS['invalid_seqs']
    duplicates_so_far = GLOBAL_COUNTERS['duplicates']
    stop_codons_so_far = GLOBAL_COUNTERS['stop_codons']

    # These are from any previous runs loaded via results_cache.pkl
    valid_sequences = total_seq_so_far - invalid_seq_so_far
    valid_percentage = (valid_sequences / total_seq_so_far * 100) if total_seq_so_far > 0 else 0

    logging.info("=== START OF RUN SUMMARY ===")
    logging.info(f"Total PHYLIP files found: {total_files}")
    logging.info(f"Total sequences encountered (prev sessions): {total_seq_so_far}")
    logging.info(f"Invalid sequences (prev sessions): {invalid_seq_so_far}")
    logging.info(f"Duplicates (prev sessions): {duplicates_so_far}")
    logging.info(f"Stop codons (prev sessions): {stop_codons_so_far}")
    logging.info(f"Valid sequences so far: {valid_sequences} ({valid_percentage:.2f}%)")
    logging.info(f"Total CDS after gene association: {GLOBAL_COUNTERS['total_cds']}")
    logging.info(f"Expected new comparisons: {GLOBAL_COUNTERS['total_comparisons']}")

    cached_results_count = db_count_keys(db_conn)
    remaining = GLOBAL_COUNTERS['total_comparisons'] - cached_results_count
    logging.info(f"Cache already has {cached_results_count} results. {remaining} remain to run.")
    sys.stdout.flush()

    if GLOBAL_COUNTERS['total_comparisons'] > 0:
        ETA_DATA['start_time'] = time.time()

    start_time = time.time()
    completed_comparisons = cached_results_count

    # --------------------------------------------------------------------------------
    # FUNCTION TO RUN A SINGLE CDS FILE (final pairwise)
    # --------------------------------------------------------------------------------

    def run_cds_file(phy_file, output_dir, codeml_path, db_conn):
        """
        Perform the actual pairwise analysis on one .phy file that we
        have confirmed is allowed and not already completed. 
        Increments global counters (exactly once) for the valid/invalid 
        data from this file if it has not yet been used in a prior run.
        """
        print(f"Running CDS file: {phy_file}")
        sys.stdout.flush()

        cds_id = os.path.basename(phy_file).replace('.phy', '')
        mode_suffix = "_all" if COMPARE_BETWEEN_GROUPS else ""
        output_csv = os.path.join(output_dir, f'{cds_id}{mode_suffix}.csv')
        haplotype_output_csv = os.path.join(output_dir, f'{cds_id}{mode_suffix}_haplotype_stats.csv')

        if os.path.exists(output_csv):
            print(f"Output {output_csv} already exists, skipping.")
            sys.stdout.flush()
            return 0  # no new comparisons

        parsed_data = PARSED_PHY.get(phy_file, None)
        if not parsed_data or not parsed_data['sequences']:
            print(f"No valid sequences for {phy_file}, skipping.")
            sys.stdout.flush()
            return 0

        # Before we do anything, increment global counters exactly once 
        # for this file's local stats (so they appear in final summary).
        GLOBAL_COUNTERS['invalid_seqs'] += parsed_data['local_invalid']
        GLOBAL_COUNTERS['stop_codons']  += parsed_data['local_stop_codons']
        GLOBAL_COUNTERS['total_seqs']   += parsed_data['local_total_seqs']
        GLOBAL_COUNTERS['duplicates']   += parsed_data['local_duplicates']

        sequences = parsed_data['sequences']
        group_num_match = re.search(r'^group_(\d+)_', os.path.basename(phy_file))
        if not group_num_match:
            print("Could not parse group number from filename. Defaulting to group=0.")
            group_num = 0
        else:
            group_num = int(group_num_match.group(1))

        # Build a sample_groups map
        sample_groups = {sname: group_num for sname in sequences.keys()}

        # Generate all pairs
        if COMPARE_BETWEEN_GROUPS:
            all_samples = list(sample_groups.keys())
            all_pairs = list(combinations(all_samples, 2))
        else:
            # For within-group only
            group_samples = list(sample_groups.keys())  # single group
            all_pairs = list(combinations(group_samples, 2))

        if not all_pairs:
            print(f"No pairs to compare for {phy_file}, skipping.")
            sys.stdout.flush()
            return 0

        temp_dir = os.path.join(output_dir, 'temp', cds_id)
        os.makedirs(temp_dir, exist_ok=True)

        # Check which pairs remain to be computed (db cache)
        to_compute = []
        for pair in all_pairs:
            check_key = f"{cds_id}::{pair[0]}::{pair[1]}::{COMPARE_BETWEEN_GROUPS}"
            if not db_has_key(db_conn, check_key):
                to_compute.append(pair)

        if not to_compute:
            print(f"All {len(all_pairs)} pairs for {cds_id} are in DB, skipping codeml.")
            sys.stdout.flush()
            return 0

        print(f"Computing {len(to_compute)} new pairs out of {len(all_pairs)} total for {cds_id}.")
        sys.stdout.flush()

        # Prepare arguments for parallel processing
        pool_args = [
            (pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, None)
            for pair in to_compute
        ]
        num_processes = min(NUM_PARALLEL, len(to_compute))
        print(f"Starting codeml execution on {len(to_compute)} pairs using {num_processes} processes.")
        sys.stdout.flush()

        results_accum = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            print(f"Processing {len(to_compute)} pairs in parallel for {cds_id}.")
            sys.stdout.flush()
            for r in pool.imap_unordered(process_pair, pool_args, chunksize=10):
                if r is not None:
                    results_accum.append(r)
            print(f"Completed processing {len(results_accum)} results for {cds_id}.")
            sys.stdout.flush()

        # Insert all new results into DB
        for r in results_accum:
            seq1, seq2, grp1, grp2, dn, ds, omega, cid = r
            newkey = f"{cid}::{seq1}::{seq2}::{COMPARE_BETWEEN_GROUPS}"
            record_tuple = (seq1, seq2, grp1, grp2, dn, ds, omega, cid)
            db_insert_or_ignore(db_conn, newkey, record_tuple)

        # Now gather final results (entire set) from DB for CSV output
        relevant_keys = []
        for pair in all_pairs:
            final_key = f"{cds_id}::{pair[0]}::{pair[1]}::{COMPARE_BETWEEN_GROUPS}"
            relevant_keys.append(final_key)

        results_for_csv = []
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
                # row is (cache_key, seq1, seq2, group1, group2, dN, dS, omega, cds)
                results_for_csv.append(row[1:])  # skip the cache_key index
            start_index += CHUNK_SIZE

        # Convert to dataframe
        df = pd.DataFrame(
            results_for_csv,
            columns=['Seq1','Seq2','Group1','Group2','dN','dS','omega','CDS']
        )
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        sys.stdout.flush()

        # Build haplotype-level stats
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

        return len(to_compute)

    # --------------------------------------------------------------------------------
    # PROCESS EACH ALLOWED FILE (Unified Parallel Approach)
    # --------------------------------------------------------------------------------

    # This gathers all pairwise tasks from all final_phy_files in one pool,
    # then we write each file's CSV after all computations complete.

    all_pairs_tasks = []
    file_info_list = []

    
    with multiprocessing.Pool(processes=min(NUM_PARALLEL, len(final_phy_files))) as pool:
        results = pool.map(parallel_handle_file, final_phy_files)
    
    for res in results:
        if res is not None:
            info_entry, tasks = res
            if info_entry is not None:
                file_info_list.append(info_entry)
            for t in tasks:
                all_pairs_tasks.append(t)

    # Run all remaining pairwise comparisons in a single Pool
    if all_pairs_tasks:
        print(f"Running a single parallel pool for {len(all_pairs_tasks)} comparisons across all CDS.")
        sys.stdout.flush()
        num_processes = min(NUM_PARALLEL, len(all_pairs_tasks))
        results_done = 0  # Tracks number of processed comparisons for ETA
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Buffer to store results for batch insertion into SQLite
            records_buffer = []
            for result in pool.imap_unordered(process_pair, all_pairs_tasks, chunksize=10):
                if result is not None:
                    seq1, seq2, grp1, grp2, dn, ds, omega, cid = result
                    newkey = f"{cid}::{seq1}::{seq2}::{COMPARE_BETWEEN_GROUPS}"
                    record_tuple = (seq1, seq2, grp1, grp2, dn, ds, omega, cid)
                    records_buffer.append((newkey, record_tuple))
                results_done += 1  # Increment after each processed result
                if results_done % 1000 == 0:  # Only update ETA every few results
                    completed_comparisons = db_count_keys(db_conn)
                    print_eta(completed_comparisons, GLOBAL_COUNTERS['total_comparisons'], ETA_DATA['start_time'], ETA_DATA)
            # Perform batch insertion of all collected results into SQLite
            if records_buffer:
                db_conn.executemany(
                    "INSERT OR IGNORE INTO pairwise_cache (cache_key, seq1, seq2, group1, group2, dN, dS, omega, cds) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(key, rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6], rec[7]) for key, rec in records_buffer]
                )
                db_conn.commit()
    else:
        print("No new comparisons to compute across all CDS.")
        sys.stdout.flush()

    # Build final CSV and haplotype stats for each file
    for info_entry in file_info_list:
        cds_id = info_entry['cds_id']
        output_csv = info_entry['output_csv']
        haplotype_output_csv = info_entry['haplotype_csv']
        all_pairs = info_entry['all_pairs']
        sequences = info_entry['sequences']
        sample_groups = info_entry['sample_groups']

        if os.path.exists(output_csv):
            print(f"Skipping CSV creation, file {output_csv} already present.")
            sys.stdout.flush()
            continue

        # Gather all results for these pairs from DB
        relevant_keys = []
        for pair in all_pairs:
            final_key = f"{cds_id}::{pair[0]}::{pair[1]}::{COMPARE_BETWEEN_GROUPS}"
            relevant_keys.append(final_key)

        results_for_csv = []
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
                results_for_csv.append(row[1:])
            start_index += CHUNK_SIZE

        df = pd.DataFrame(
            results_for_csv,
            columns=['Seq1','Seq2','Group1','Group2','dN','dS','omega','CDS']
        )
        df.to_csv(output_csv, index=False)
        print(f"Results for {cds_id} saved to {output_csv}")
        sys.stdout.flush()

        # Build haplotype-level stats
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
        print(f"Haplotype stats for {cds_id} saved to {haplotype_output_csv}")
        sys.stdout.flush()

    # --------------------------------------------------------------------------------
    # FINAL SUMMARY & CLEANUP
    # --------------------------------------------------------------------------------
    end_time = time.time()

    total_seqs = GLOBAL_COUNTERS['total_seqs']
    invalid_seqs = GLOBAL_COUNTERS['invalid_seqs']
    duplicates = GLOBAL_COUNTERS['duplicates']
    total_cds_processed = GLOBAL_COUNTERS['total_cds']
    total_comps_planned = GLOBAL_COUNTERS['total_comparisons']
    stop_codons = GLOBAL_COUNTERS['stop_codons']

    valid_seqs_final = total_seqs - invalid_seqs
    final_invalid_pct = (invalid_seqs / total_seqs * 100) if total_seqs > 0 else 0

    logging.info("=== END OF RUN SUMMARY ===")
    logging.info(f"Total PHYLIP found: {total_files}")
    logging.info(f"Total seq processed: {total_seqs}")
    logging.info(f"Invalid seq: {invalid_seqs} ({final_invalid_pct:.2f}%)")
    logging.info(f"Sequences with stop codons: {stop_codons}")
    logging.info(f"Duplicates: {duplicates}")
    logging.info(f"Total CDS (final) after gene association: {total_cds_processed}")
    logging.info(f"Planned comparisons: {total_comps_planned}")
    current_db_count = db_count_keys(db_conn)
    logging.info(f"Completed comps: {current_db_count}")

    if ETA_DATA['start_time']:
        run_minutes = (end_time - ETA_DATA['start_time']) / 60
        logging.info(f"Total time: {run_minutes:.2f} min")
    else:
        total_run_min = (end_time - start_time) / 60
        logging.info(f"Total time: {total_run_min:.2f} min")

    save_cache(cache_file, GLOBAL_COUNTERS)

    try:
        with open(VALIDATION_CACHE_FILE, 'wb') as f:
            pickle.dump(VALIDATION_CACHE, f)
        print(f"Validation cache saved with {len(VALIDATION_CACHE)} entries.")
    except Exception as e:
        print(f"Could not save validation cache: {e}")

    # Consolidate all final CSVs if desired
    def consolidate_all_csvs(csv_dir, final_csv='all_pairwise_results.csv'):
        all_dfs = []
        potential_csvs = glob.glob(os.path.join(csv_dir, '*.csv'))
        for cf in potential_csvs:
            # Skip haplotype stats
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

    consolidate_all_csvs(args.output_dir)

    db_conn.close()
    logging.info("dN/dS analysis done.")
    print("dN/dS analysis done.")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
