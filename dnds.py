#!/usr/bin/env python3
"""
dN/dS Analysis Script using PAML's CODEML

This script calculates pairwise dN/dS values using PAML's CODEML program.


1. Input Files:
   Each input file is a PHYLIP-like file with lines consisting of:
   SAMPLE_NAME_L/R + SEQUENCE (no spaces). For example:
       ABC_XYZ_HG01352_LACGGAGTAC...
   Where each sample name ends with "_0" or "_1" before the sequence.

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
from functools import partial

COMPARE_BETWEEN_GROUPS = False

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

def increment_counter(key, amount=1):
    print(f"Incrementing counter {key} by {amount}")
    sys.stdout.flush()
    GLOBAL_COUNTERS[key] = GLOBAL_COUNTERS.get(key, 0) + amount

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
    sys.stdout.flush()
    if not seq:
        print("Invalid sequence: empty")
        sys.stdout.flush()
        increment_counter('invalid_seqs')
        return None

    seq = seq.upper()
    line_len = len(full_line)
    seq_len = len(seq)

    MAX_CDS_LENGTH = 150000
    if seq_len > MAX_CDS_LENGTH:
        print("Invalid sequence: exceeds max length")
        sys.stdout.flush()
        increment_counter('invalid_seqs')
        return None

    if seq_len > line_len:
        print("Invalid sequence: sequence length greater than line length?")
        sys.stdout.flush()
        increment_counter('invalid_seqs')
        return None

    if seq_len % 3 != 0:
        print("Invalid sequence: length not divisible by 3")
        sys.stdout.flush()
        increment_counter('invalid_seqs')
        return None

    valid_bases = set('ATCGN-')
    invalid_chars = set(seq) - valid_bases
    if invalid_chars:
        print(f"Invalid sequence: contains invalid characters {invalid_chars}")
        sys.stdout.flush()
        increment_counter('invalid_seqs')
        return None

    stop_codons = {'TAA', 'TAG', 'TGA'}
    for i in range(0, seq_len-2, 3):
        codon = seq[i:i+3]
        if codon in stop_codons:
            print(f"Invalid sequence: in-frame stop codon {codon}")
            sys.stdout.flush()
            increment_counter('stop_codons')
            increment_counter('invalid_seqs')
            return None

    print("Sequence validated successfully.")
    sys.stdout.flush()
    return seq


def create_paml_ctl(seqfile, outfile, working_dir):
    print(f"Creating PAML control file in {working_dir}")
    sys.stdout.flush()
    seqfile = os.path.abspath(seqfile)
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
    print(f"PAML control file created at {ctl_path}")
    sys.stdout.flush()
    return ctl_path

def run_codeml(ctl_path, working_dir, codeml_path):
    print(f"Running codeml with control file: {ctl_path}")
    sys.stdout.flush()
    command = [codeml_path, ctl_path]
    try:
        process = subprocess.Popen(
            command,
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=300)
        if process.returncode != 0:
            error_msg = stderr.decode('utf-8')
            logging.error(f"CODEML failed: {error_msg}")
            print(f"CODEML failed with error: {error_msg}")
            sys.stdout.flush()
            return False
        else:
            print("CODEML completed successfully.")
            sys.stdout.flush()
            return True
    except subprocess.TimeoutExpired:
        process.kill()
        logging.error("CODEML timed out.")
        print("CODEML timed out.")
        sys.stdout.flush()
        return False
    except FileNotFoundError as fnf:
        logging.error(f"CODEML not found: {fnf}")
        print(f"CODEML not found: {fnf}")
        sys.stdout.flush()
        return False
    except Exception as e:
        logging.error(f"Error running CODEML: {e}")
        print(f"Error running CODEML: {e}")
        sys.stdout.flush()
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
            content = f.read()
        pattern = re.compile(
            r"t=\s*[\d\.]+\s+S=\s*([\d\.]+)\s+N=\s*([\d\.]+)\s+dN/dS=\s*([\d\.]+)\s+dN=\s*([\d\.]+)\s+dS=\s*([\d\.]+)"
        )
        match = pattern.search(content)
        if match:
            dN = float(match.group(4))
            dS = float(match.group(5))
            omega = float(match.group(3))
            print(f"Parsed dN={dN}, dS={dS}, omega={omega}")
            sys.stdout.flush()
            return dN, dS, omega
        else:
            print("No match for expected pattern in rst file.")
            sys.stdout.flush()
            return None, None, None
    except Exception as e:
        print(f"Error parsing CODEML output: {e}")
        sys.stdout.flush()
        return None, None, None

def get_safe_process_count():
    print("Getting safe process count for multiprocessing...")
    sys.stdout.flush()
    total_cpus = multiprocessing.cpu_count()
    print(f"Detected {total_cpus} CPUs.")
    sys.stdout.flush()
    return total_cpus

def parse_phy_file(filepath):
    print(f"Parsing phy file: {filepath}")
    sys.stdout.flush()
    sequences = {}
    duplicates_found = False
    line_pattern = re.compile(r'^([A-Za-z0-9_]+_[LR])([ATCGNatcgn-]+)$')

    if not os.path.isfile(filepath):
        print("File does not exist.")
        sys.stdout.flush()
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
                # If name parsing fails, skip
                continue
            first = name_parts[0][:3]
            second = name_parts[1][:3]
            hg_part = name_parts[-2]
            group = name_parts[-1]

            hash_val = abs(hash(hg_part)) % 100
            hash_str = f"{hash_val:02d}"
            sample_name = f"{first}{second}{hash_str}_{group}"

            validated_seq = validate_sequence(sequence, filepath, sample_name, line)
            if validated_seq:
                increment_counter('total_seqs')
                if sample_name in sequences:
                    print(f"Duplicate found for {sample_name}, renaming...")
                    sys.stdout.flush()
                    duplicates_found = True
                    base_name = sample_name[:2] + sample_name[3:]
                    dup_count = sum(1 for s in sequences.keys() if s[:2] + s[3:] == base_name)
                    new_name = sample_name[:2] + str(dup_count) + sample_name[3:]
                    sequences[new_name] = validated_seq
                    increment_counter('duplicates')
                else:
                    sequences[sample_name] = validated_seq
    print(f"Finished parsing {filepath}, found {len(sequences)} sequences (duplicates_found={duplicates_found})")
    sys.stdout.flush()
    return sequences, duplicates_found

def load_cache(cache_file):
    print(f"Loading cache from {cache_file}")
    sys.stdout.flush()
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Cache loaded with {len(data)} entries.")
        sys.stdout.flush()
        return data
    else:
        print("No cache file found, starting fresh.")
        sys.stdout.flush()
        return {}

def save_cache(cache_file, cache_data):
    print(f"Saving cache to {cache_file} with {len(cache_data)} entries.")
    sys.stdout.flush()
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("Cache saved.")
    sys.stdout.flush()

def process_pair(args):
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, cache = args
    seq1_name, seq2_name = pair
    cache_key = (cds_id, seq1_name, seq2_name, COMPARE_BETWEEN_GROUPS)
    print(f"Processing pair: {seq1_name}, {seq2_name} for {cds_id}")
    sys.stdout.flush()
    if cache_key in cache:
        print("Pair result found in cache, skipping computation.")
        sys.stdout.flush()
        return cache[cache_key]

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
        print("Sequences are identical, returning omega = -1.")
        sys.stdout.flush()
        result = (seq1_name, seq2_name, group1, group2, 0.0, 0.0, -1.0, cds_id)
        cache[cache_key] = result
        return result

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

    ctl_path = create_paml_ctl(seqfile, 'mlc', working_dir)
    success = run_codeml(ctl_path, working_dir, codeml_path)
    if not success:
        print("Codeml run failed, returning NaN values.")
        sys.stdout.flush()
        result = (seq1_name, seq2_name, group1, group2, np.nan, np.nan, np.nan, cds_id)
        cache[cache_key] = result
        return result

    dn, ds, omega = parse_codeml_output(working_dir)
    if omega is None:
        omega = np.nan
    print(f"Pair processed: dn={dn}, ds={ds}, omega={omega}")
    sys.stdout.flush()
    result = (seq1_name, seq2_name, group1, group2, dn, ds, omega, cds_id)
    cache[cache_key] = result
    return result

def estimate_one_file(phy_file):
    print(f"Estimating comparisons for file {phy_file}")
    sys.stdout.flush()
    sequences, duplicates = parse_phy_file(phy_file)
    if not sequences:
        return (0,0)
    sample_groups = {}
    skip_file = False
    # Force all sequences in this file to use the group number from the filename:
    for s in sequences.keys():
        sample_groups[s] = int(group_num)
    if skip_file:
        return (0,0)

    if COMPARE_BETWEEN_GROUPS:
        pairs = list(combinations(sequences.keys(), 2))
    else:
        g0 = [s for s,g in sample_groups.items() if g==0]
        g1 = [s for s,g in sample_groups.items() if g==1]
        pairs = list(combinations(g0,2)) + list(combinations(g1,2))

    cds_count = 1 if pairs else 0
    pair_count = len(pairs)
    print(f"File {phy_file} estimation: {cds_count} CDS, {pair_count} comparisons")
    sys.stdout.flush()
    return (cds_count, pair_count)

def get_transcript_coordinates(transcript_id):
    print(f"Getting transcript coordinates for {transcript_id}")
    sys.stdout.flush()
    gtf_file = '../hg38.knownGene.gtf'
    min_start = None
    max_end = None
    chrom = None
    try:
        with open(gtf_file, 'r') as f:
            for line in f:
                if line.strip() == '' or line.startswith('#'):
                    continue
                fields = line.split('\t')
                if len(fields) < 9:
                    continue
                feature_type = fields[2]
                if feature_type != 'CDS':
                    continue

                attrs = fields[8].strip()
                tid = None
                for attr in attrs.split(';'):
                    attr = attr.strip()
                    if attr.startswith('transcript_id "'):
                        tid = attr.split('"')[1]
                        break

                if tid == transcript_id:
                    chr_ = fields[0]
                    start = int(fields[3])
                    end = int(fields[4])
                    if chrom is None:
                        chrom = chr_
                    if min_start is None or start < min_start:
                        min_start = start
                    if max_end is None or end > max_end:
                        max_end = end
    except Exception as e:
        print(f"Error reading GTF or processing coordinates: {e}")
        sys.stdout.flush()

    if chrom is not None and min_start is not None and max_end is not None:
        print(f"Coordinates for {transcript_id}: {chrom}, {min_start}, {max_end}")
        sys.stdout.flush()
        return (chrom, min_start, max_end)
    else:
        print(f"No coordinates found for {transcript_id}")
        sys.stdout.flush()
        return (None, None, None)

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
            for j in range(i+1, len(cds_list)):
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

def main():
    print("Starting main process...")
    sys.stdout.flush()
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
    parser.add_argument('--codeml_path', type=str, default='codeml', help='Path to codeml executable.')
    args = parser.parse_args()

    print(f"PHY_DIR: {args.phy_dir}")
    print(f"OUTPUT_DIR: {args.output_dir}")
    print(f"CODEML_PATH: {args.codeml_path}")
    sys.stdout.flush()

    os.makedirs(args.output_dir, exist_ok=True)

    cache_file = os.path.join(args.output_dir, 'results_cache.pkl')
    cache = load_cache(cache_file)

    print("Gathering .phy files...")
    sys.stdout.flush()
    phy_files = glob.glob(os.path.join(args.phy_dir, '*.phy'))
    total_files = len(phy_files)
    print(f"Found {total_files} phy files.")
    sys.stdout.flush()

    filename_pattern = re.compile(r'^group_(\d+)_([A-Za-z0-9\.]+)_chr_([A-Za-z0-9]+)_combined\.phy$')

    cds_meta_all = []
    all_sequences_data = {}

    print("Parsing all phy files for metadata...")
    sys.stdout.flush()
    for phy_file in phy_files:
        basename = os.path.basename(phy_file)
        m = filename_pattern.match(basename)
        if not m:
            continue
        group_num = m.group(1)
        transcript_id = m.group(2)
        chromosome = m.group(3)

        chr_, st, en = get_transcript_coordinates(transcript_id)
        if chr_ is None or st is None or en is None:
            continue

        sequences, duplicates = parse_phy_file(phy_file)
        valid_seq_count = len(sequences)
        if valid_seq_count == 0:
            continue

        cds_id = basename.replace('.phy', '')
        cds_meta_all.append((cds_id, transcript_id, chr_, st, en, sequences))
        all_sequences_data[cds_id] = sequences
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
            _,_,_,_,_,seqs = cds_id_to_data[cid]
            if seqs:
                length = len(next(iter(seqs.values())))
                if length > best_len:
                    best_len = length
                    best_id = cid
        if best_id:
            allowed_cds_ids.add(best_id)

    final_phy_files = [f for f in phy_files if os.path.basename(f).replace('.phy','') in allowed_cds_ids]

    print("Estimating total comparisons...")
    sys.stdout.flush()
    def quick_estimate(phy):
        return estimate_one_file(phy)
    results = [quick_estimate(p) for p in final_phy_files]
    total_cds = sum(r[0] for r in results)
    total_comps = sum(r[1] for r in results)
    GLOBAL_COUNTERS['total_cds'] = total_cds
    GLOBAL_COUNTERS['total_comparisons'] = total_comps

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
    cached_results_count = len(cache)
    remaining = GLOBAL_TOTAL_COMPARISONS - cached_results_count
    logging.info(f"Cache: {cached_results_count} results. {remaining} remain.")

    if GLOBAL_TOTAL_COMPARISONS > 0:
        ETA_DATA['start_time'] = time.time()

    start_time = time.time()
    completed_comparisons = cached_results_count

    def run_cds_file(phy_file, output_dir, codeml_path, cache):
        print(f"Running CDS file: {phy_file}")
        sys.stdout.flush()
        cds_id = os.path.basename(phy_file).replace('.phy','')
        mode_suffix = "_all" if COMPARE_BETWEEN_GROUPS else ""
        output_csv = os.path.join(output_dir, f'{cds_id}{mode_suffix}.csv')
        haplotype_output_csv = os.path.join(output_dir, f'{cds_id}{mode_suffix}_haplotype_stats.csv')

        if os.path.exists(output_csv) and os.path.exists(haplotype_output_csv):
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
            g = extract_group_from_sample(s)
            if g is None:
                print("Failed to extract group from sample, skipping.")
                sys.stdout.flush()
                return
            sample_groups[s] = g

        if COMPARE_BETWEEN_GROUPS:
            all_samples = list(sample_groups.keys())
            pairs = list(combinations(all_samples, 2))
        else:
            g0 = [x for x,y in sample_groups.items() if y==0]
            g1 = [x for x,y in sample_groups.items() if y==1]
            pairs = list(combinations(g0,2)) + list(combinations(g1,2))

        if not pairs:
            print("No pairs to compare, skipping.")
            sys.stdout.flush()
            return

        temp_dir = os.path.join(output_dir, 'temp', cds_id)
        os.makedirs(temp_dir, exist_ok=True)

        pool_args = [(pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, cache) for pair in pairs]
        num_processes = get_safe_process_count()
        print(f"Running codeml on {len(pairs)} pairs with {num_processes} processes.")
        sys.stdout.flush()

        results = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            for r in pool.imap_unordered(process_pair, pool_args, chunksize=10):
                if r is not None:
                    results.append(r)

        df = pd.DataFrame(results, columns=['Seq1','Seq2','Group1','Group2','dN','dS','omega','CDS'])
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        sys.stdout.flush()

        hap_stats = []
        for sample in sequences.keys():
            sample_df = df[(df['Seq1'] == sample) | (df['Seq2'] == sample)]
            omega_vals = sample_df['omega'].dropna()
            omega_vals = omega_vals[~omega_vals.isin([-1,99])]
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

        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary directory cleaned up.")
        sys.stdout.flush()

    print("Processing each allowed CDS file...")
    sys.stdout.flush()
    for idx, phy_file in enumerate(final_phy_files, 1):
        logging.info(f"Processing file {idx}/{len(final_phy_files)}: {phy_file}")
        print(f"Processing file {idx}/{len(final_phy_files)}: {phy_file}")
        sys.stdout.flush()
        old_size = len(cache)
        run_cds_file(phy_file, args.output_dir, args.codeml_path, cache)
        save_cache(cache_file, cache)
        new_size = len(cache)
        newly_done = new_size - old_size
        completed_comparisons += newly_done
        percent = (completed_comparisons / GLOBAL_TOTAL_COMPARISONS * 100) if GLOBAL_TOTAL_COMPARISONS > 0 else 0
        logging.info(f"Overall: {completed_comparisons}/{GLOBAL_TOTAL_COMPARISONS} comps ({percent:.2f}%)")
        print(f"Overall progress: {completed_comparisons}/{GLOBAL_TOTAL_COMPARISONS} comps ({percent:.2f}%)")
        sys.stdout.flush()
        print_eta(completed_comparisons, GLOBAL_TOTAL_COMPARISONS, start_time, ETA_DATA)

    end_time = time.time()
    final_invalid_pct = (GLOBAL_INVALID_SEQS/GLOBAL_TOTAL_SEQS*100) if GLOBAL_TOTAL_SEQS>0 else 0

    logging.info("=== END OF RUN SUMMARY ===")
    logging.info(f"Total PHYLIP: {total_files}")
    logging.info(f"Total seq: {GLOBAL_TOTAL_SEQS}")
    logging.info(f"Invalid seq: {GLOBAL_INVALID_SEQS} ({final_invalid_pct:.2f}%)")
    logging.info(f"Sequences with stop codons: {GLOBAL_COUNTERS['stop_codons']}")
    logging.info(f"Duplicates: {GLOBAL_DUPLICATES}")
    logging.info(f"Total CDS (final): {GLOBAL_TOTAL_CDS}")
    logging.info(f"Expected comps: {GLOBAL_TOTAL_COMPARISONS}")
    logging.info(f"Completed comps: {completed_comparisons}")
    logging.info(f"Total time: {(end_time - start_time)/60:.2f} min")

    logging.info("dN/dS analysis done.")
    print("dN/dS analysis done.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
