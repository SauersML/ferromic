#!/usr/bin/env python3
"""
dN/dS Analysis Script using PAML's CODEML

This script calculates pairwise dN/dS values using PAML's CODEML program.
It processes PHYLIP files containing nucleotide sequences, computes pairwise
comparisons (within or between groups), and produces pairwise and haplotype-level
statistics.

Changes from original:
- Removed statistical tests and histogram generation.
- Added Transcript_ID column from a given CDS to Transcript map.
- If Transcript_ID not found, skip that CDS.
- No empty columns are dropped; we ensure all columns are populated or file is skipped.
- Detailed start-of-run summary and final summary.
- Show progress and ETA during the run.
- Duplicate sample names handled by incrementing the third character of the sample name.

Requirements:
- PAML (codeml) installed.
- Python packages: pandas, numpy, psutil, tqdm.

Usage:
    python3 dnds_analysis.py --phy_dir PATH_TO_PHY_FILES --output_dir OUTPUT_DIRECTORY --codeml_path PATH_TO_CODEML
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

COMPARE_BETWEEN_GROUPS = False=  # Set to True to enable between-group comparisons

cds_to_transcript_map = {}  # Fix

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s [%(levelname)s] %(message)s',
   handlers=[
       logging.FileHandler('dnds_analysis.log'),
       logging.StreamHandler(sys.stdout)
   ]
)

# Global counters for summary stats
GLOBAL_INVALID_SEQS = 0
GLOBAL_DUPLICATES = 0
GLOBAL_TOTAL_SEQS = 0
GLOBAL_TOTAL_CDS = 0
GLOBAL_TOTAL_COMPARISONS = 0

def validate_sequence(seq):
    global GLOBAL_INVALID_SEQS
    if len(seq) % 3 != 0:
        logging.warning(f"Sequence length {len(seq)} not divisible by 3. Skipping sequence.")
        GLOBAL_INVALID_SEQS += 1
        return None

    valid_bases = set('ATCGNatcgn-')
    if not set(seq).issubset(valid_bases):
        invalid_chars = set(seq) - valid_bases
        logging.warning(f"Invalid characters {invalid_chars} found in sequence. Skipping sequence.")
        GLOBAL_INVALID_SEQS += 1
        return None

    return seq.upper()

def extract_group_from_sample(sample_name):
    match = re.search(r'_(0|1)$', sample_name)
    if match:
        return int(match.group(1))
    else:
        logging.warning(f"Group suffix not found in sample name: {sample_name}")
        return None

def create_paml_ctl(seqfile, outfile, working_dir):
    ctl_content = f"""
      seqfile = {seqfile}
      treefile = tree.txt
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
    return ctl_path

def run_codeml(ctl_path, working_dir, codeml_path):
    try:
        process = subprocess.Popen(
            [codeml_path],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=300)
        if process.returncode != 0:
            logging.error(f"CODEML failed: {stderr.decode('utf-8')}")
            return False
        return True
    except subprocess.TimeoutExpired:
        process.kill()
        logging.error("CODEML process timed out.")
        return False
    except Exception as e:
        logging.error(f"Error running CODEML: {e}")
        return False

def parse_codeml_output(outfile_dir):
    rst_file = os.path.join(outfile_dir, 'rst')
    if not os.path.exists(rst_file):
        logging.error(f"CODEML output file not found: {rst_file}")
        return None, None, None

    try:
        with open(rst_file, 'r') as f:
            content = f.read()

        pattern = re.compile(
            r"t=\s*[\d\.]+\s+S=\s*([\d\.]+)\s+N=\s*([\d\.]+)\s+"
            r"dN/dS=\s*([\d\.]+)\s+dN=\s*([\d\.]+)\s+dS=\s*([\d\.]+)"
        )
        match = pattern.search(content)
        if match:
            dN = float(match.group(4))
            dS = float(match.group(5))
            omega = float(match.group(3))
            return dN, dS, omega
        else:
            logging.error("Could not parse CODEML output.")
            return None, None, None
    except Exception as e:
        logging.error(f"Error parsing CODEML output: {e}")
        return None, None, None

def get_safe_process_count():
    total_cpus = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    safe_processes = max(1, min(total_cpus // 2, int(mem.available / (2 * 1024**3))))
    return safe_processes

def parse_phy_file(filepath):
    """
    Parse a PHYLIP file to extract sequences.

    Returns:
    (dict, bool): sequences dict and bool indicating if duplicates were found.
    """
    global GLOBAL_DUPLICATES, GLOBAL_TOTAL_SEQS
    sequences = {}
    duplicates_found = False
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return {}, False

    with open(filepath, 'r') as file:
        lines = file.readlines()
        if not lines:
            logging.error(f"PHYLIP file is empty: {filepath}")
            return {}, False

        try:
            num_sequences, seq_length = map(int, lines[0].strip().split())
            sequence_lines = lines[1:]
        except ValueError:
            logging.warning(f"No valid header found in {filepath}. Processing without header.")
            sequence_lines = lines

        for line in sequence_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                sample_name = parts[0]
                sequence = ''.join(parts[1:])
            else:
                sample_name = line[:10].strip()
                sequence = line[10:].replace(" ", "")
            validated_seq = validate_sequence(sequence)
            if validated_seq:
                # Counting total valid sequences
                GLOBAL_TOTAL_SEQS += 1
                if sample_name in sequences:
                    duplicates_found = True
                    # Increment the third character of sample_name based on how many duplicates
                    base_name = sample_name[:2] + sample_name[3:] 
                    dup_count = sum(1 for s in sequences.keys() if s[:2] + s[3:] == base_name)
                    new_name = sample_name[:2] + str(dup_count) + sample_name[3:]
                    logging.info(f"Duplicate sample name found. Renaming {sample_name} to {new_name}")
                    sequences[new_name] = validated_seq
                    GLOBAL_DUPLICATES += 1
                else:
                    sequences[sample_name] = validated_seq
    return sequences, duplicates_found

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        logging.info(f"Cache loaded from {cache_file}")
    else:
        cache = {}
    return cache

def save_cache(cache_file, cache_data):
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    logging.info(f"Cache saved to {cache_file}")

def process_pair(args):
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, cache = args
    seq1_name, seq2_name = pair

    cache_key = (cds_id, seq1_name, seq2_name, COMPARE_BETWEEN_GROUPS)
    if cache_key in cache:
        return cache[cache_key]

    if seq1_name not in sequences or seq2_name not in sequences:
        logging.error(f"Sequences not found for pair: {seq1_name}, {seq2_name}")
        return None

    group1 = sample_groups.get(seq1_name)
    group2 = sample_groups.get(seq2_name)

    if not COMPARE_BETWEEN_GROUPS and group1 != group2:
        return None

    if sequences[seq1_name] == sequences[seq2_name]:
        # Identical sequences
        result = (seq1_name, seq2_name, group1, group2, 0.0, 0.0, -1.0, cds_id)
        cache[cache_key] = result
        return result

    working_dir = os.path.join(temp_dir, f'{seq1_name}_{seq2_name}')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    seqfile = os.path.join(working_dir, 'seqfile.phy')
    with open(seqfile, 'w') as f:
        f.write(f" 2 {len(sequences[seq1_name])}\n")
        f.write(f"{seq1_name} {sequences[seq1_name]}\n")
        f.write(f"{seq2_name} {sequences[seq2_name]}\n")

    treefile = os.path.join(working_dir, 'tree.txt')
    with open(treefile, 'w') as f:
        f.write(f"({seq1_name},{seq2_name});\n")

    ctl_path = create_paml_ctl(seqfile, 'mlc', working_dir)

    success = run_codeml(ctl_path, working_dir, codeml_path)
    if not success:
        result = (seq1_name, seq2_name, group1, group2, np.nan, np.nan, np.nan, cds_id)
        cache[cache_key] = result
        return result

    dn, ds, omega = parse_codeml_output(working_dir)
    if omega is None:
        omega = np.nan

    result = (seq1_name, seq2_name, group1, group2, dn, ds, omega, cds_id)
    cache[cache_key] = result
    return result

def estimate_total_comparisons(phy_dir):
    """
    Estimate total comparisons before running.
    Also count total invalid sequences and duplicates.
    """
    global GLOBAL_TOTAL_CDS, GLOBAL_TOTAL_COMPARISONS
    phy_files = glob.glob(os.path.join(phy_dir, '*.phy'))
    total_comparisons = 0
    for phy_file in phy_files:
        # Parse quickly
        sequences, duplicates = parse_phy_file(phy_file)
        if not sequences:
            # no sequences => no comparisons
            continue

        # Assign groups temporarily
        sample_groups = {}
        all_samples = sequences.keys()
        # If any sample doesn't have a group, just skip computations
        skip_file = False
        for s in all_samples:
            g = extract_group_from_sample(s)
            if g is None:
                skip_file = True
                break
            sample_groups[s] = g
        if skip_file:
            continue

        if COMPARE_BETWEEN_GROUPS:
            # all vs all
            pairs = list(combinations(all_samples, 2))
        else:
            group0_samples = [s for s, g in sample_groups.items() if g == 0]
            group1_samples = [s for s, g in sample_groups.items() if g == 1]
            pairs = list(combinations(group0_samples, 2)) + list(combinations(group1_samples, 2))
        total_comparisons += len(pairs)
        GLOBAL_TOTAL_CDS += 1
    GLOBAL_TOTAL_COMPARISONS = total_comparisons

def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
    parser.add_argument('--codeml_path', type=str, default='codeml', help='Path to codeml executable.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pre-run analysis to print summary before running:
    # First, load the cache
    cache_file = os.path.join(args.output_dir, 'results_cache.pkl')
    cache = load_cache(cache_file)

    # Estimate total comparisons, sequences, invalid, duplicates
    # (Already counted globally by parse_phy_file calls in estimate_total_comparisons)
    estimate_total_comparisons(args.phy_dir)

    phy_files = glob.glob(os.path.join(args.phy_dir, '*.phy'))
    total_files = len(phy_files)
    logging.info("=== START-OF-RUN SUMMARY ===")
    logging.info(f"Found {total_files} PHYLIP files.")
    logging.info(f"Total sequences encountered so far: {GLOBAL_TOTAL_SEQS}")
    logging.info(f"Invalid sequences (not divisible by 3 or invalid chars): {GLOBAL_INVALID_SEQS}")
    logging.info(f"Duplicates encountered: {GLOBAL_DUPLICATES}")
    logging.info(f"Total CDS files that will be processed: {GLOBAL_TOTAL_CDS}")
    logging.info(f"Total expected CODEML comparisons: {GLOBAL_TOTAL_COMPARISONS}")

    cached_results_count = len(cache)
    remaining_runs = GLOBAL_TOTAL_COMPARISONS - cached_results_count
    logging.info(f"Cache has {cached_results_count} results. {remaining_runs} remain to be computed.")

    # Check if Transcript_ID is available for all CDS

    # Start processing
    start_time = time.time()

    work_args = []
    # Filter only those CDS we haven't processed fully yet
    for phy_file in phy_files:
        phy_filename = os.path.basename(phy_file)
        cds_id = phy_filename.replace('.phy', '')

        # Check Transcript_ID
        transcript_id = cds_to_transcript_map.get(cds_id)
        if transcript_id is None:
            logging.warning(f"No Transcript_ID found for {cds_id}. Skipping this file.")
            continue

        mode_suffix = "_all" if COMPARE_BETWEEN_GROUPS else ""
        output_csv = os.path.join(args.output_dir, f'{cds_id}{mode_suffix}.csv')
        haplotype_output_csv = os.path.join(args.output_dir, f'{cds_id}{mode_suffix}_haplotype_stats.csv')

        if os.path.exists(output_csv) and os.path.exists(haplotype_output_csv):
            logging.info(f"Skipping {phy_file} - output files already exist")
            continue
        work_args.append((phy_file, args.output_dir, args.codeml_path, total_files, len(work_args)+1, cache))

    total_new_files = len(work_args)
    completed_comparisons = cached_results_count  # start with cached ones done

    # Function to estimate ETA
    def print_eta(completed, total, start):
        elapsed = time.time() - start
        if elapsed > 0 and completed > 0:
            rate = completed / elapsed  # comparisons per second
            remaining = total - completed
            if rate > 0:
                eta_seconds = remaining / rate
                hours = eta_seconds / 3600
                logging.info(f"Progress: {completed}/{total} comparisons done. ETA: {hours:.2f} hours")
            else:
                logging.info(f"Progress: {completed}/{total} comparisons done. ETA: N/A")
        else:
            logging.info(f"Progress: {completed}/{total} comparisons done. ETA: N/A")

    for idx, args_tuple in enumerate(work_args, 1):
        phy_file = args_tuple[0]
        logging.info(f"Processing file {idx}/{total_new_files}: {phy_file}")
        haplotype_file = process_phy_file(args_tuple)

        # After processing the file, new comparisons might be completed
        # Let's count how many results we have now in cache
        # process_phy_file writes directly to CSV
        # The new comparisons done are the differences in cache size
        old_cache_size = len(cache)
        save_cache(cache_file, cache)
        new_cache_size = len(cache)

        newly_completed = new_cache_size - old_cache_size
        completed_comparisons += newly_completed
        # Print overall progress and ETA
        total_comparisons = GLOBAL_TOTAL_COMPARISONS
        percent = (completed_comparisons / total_comparisons)*100 if total_comparisons > 0 else 0
        logging.info(f"Overall Progress: {completed_comparisons}/{total_comparisons} comparisons ({percent:.2f}%)")
        print_eta(completed_comparisons, total_comparisons, start_time)

    # Final summary
    elapsed_time = time.time() - start_time
    final_percent_invalid = (GLOBAL_INVALID_SEQS / GLOBAL_TOTAL_SEQS * 100) if GLOBAL_TOTAL_SEQS > 0 else 0

    logging.info("=== END-OF-RUN SUMMARY ===")
    logging.info(f"Total PHYLIP files: {total_files}")
    logging.info(f"Total sequences processed: {GLOBAL_TOTAL_SEQS}")
    logging.info(f"Invalid sequences: {GLOBAL_INVALID_SEQS} ({final_percent_invalid:.2f}% of total)")
    logging.info(f"Duplicates encountered and renamed: {GLOBAL_DUPLICATES}")
    logging.info(f"Total CDS processed: {GLOBAL_TOTAL_CDS}")
    logging.info(f"Total comparisons expected: {GLOBAL_TOTAL_COMPARISONS}")
    logging.info(f"Total comparisons completed: {completed_comparisons}")
    logging.info(f"Total time: {elapsed_time/60:.2f} minutes")

    logging.info("dN/dS analysis completed.")

if __name__ == '__main__':
    main()
