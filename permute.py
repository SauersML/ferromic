"""
======================================================================
Permutation-Based dN/dS Analysis
======================================================================

This script implements a transcript-by-transcript label-permutation test
to detect differences in dN/dS (omega) values between two groups
(Group 0 vs. Group 1). It uses median-based statistics to handle
special values (omega = -1 or 99) and outliers. The approach
automatically accounts for sequence-level correlation by preserving
all pairwise distances and simply reassigning group labels at the
sequence level.

Primary Steps:
1. Read CSV data with pairwise comparisons
2. Preprocess, parse group assignments, genomic coords, transcript IDs
3. For each transcript, isolate relevant data
4. Compute:
   - Observed difference in median(0–0) vs median(1–1)
   - Permutation-based p-value from random reassignments
5. Apply multiple testing correction (Benjamini–Hochberg)
6. Print final tables & output CSV
"""

import os
import re
import sys
import json
import pickle
import warnings
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

# For gene annotation lookups
import requests
from urllib.parse import urlencode

# For multiple testing correction
from scipy import stats

# For concurrency
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

warnings.filterwarnings('ignore')

########################################################################
#                         CONFIGURATION SECTION                        #
########################################################################

# CSV file paths
INPUT_CSV_PATH = "all_pairwise_results.csv"  # main pairwise data
INV_INFO_CSV_PATH = "inv_info.csv"           
OUTPUT_DIR = "results"
CACHE_DIR = "cache"

# Minimum # sequences in each group for valid analysis
MIN_SEQUENCES_PER_GROUP = 10

# Whether to exclude special omega values (-1 or 99) from analysis entirely
FILTER_SPECIAL_OMEGA_VALUES = False

# Number of permutations
NUM_PERMUTATIONS = 10000

# Confidence level for significance
FDR_THRESHOLD = 0.05

########################################################################
#                            HELPER FUNCTIONS                          #
########################################################################

def read_and_preprocess_data(file_path):
    """
    Read and preprocess the evolutionary rate data from a CSV file.

    Steps:
    1. Load the CSV into a DataFrame
    2. Deduce group (0, 1, or 2=cross) from 'Group1'/'Group2' columns
    3. Extract genomic coordinates (chrom, start, end) from 'CDS'
    4. Extract transcript ID from 'CDS' (pattern: ENST\d+\.\d+)
    5. Convert 'omega' to numeric
    6. (Optionally) filter out -1, 99 if FILTER_SPECIAL_OMEGA_VALUES
    7. Return the DataFrame

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing pairwise comparisons

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with:
        - Seq1, Seq2, Group1, Group2
        - group (0, 1, or 2 for cross)
        - chrom, start, end
        - transcript_id
        - omega
        - full_cds (original 'CDS' string)
    """
    print(f"{Fore.GREEN}Reading data from {file_path}...{Style.RESET_ALL}")
    df = pd.read_csv(file_path)

    # Keep a copy of the original 'CDS' for reference
    df['full_cds'] = df['CDS']

    # Determine comparison group from Group1/Group2
    df['group'] = np.nan
    df.loc[(df['Group1'] == 0) & (df['Group2'] == 0), 'group'] = 0
    df.loc[(df['Group1'] == 1) & (df['Group2'] == 1), 'group'] = 1
    df.loc[((df['Group1'] == 0) & (df['Group2'] == 1)) |
           ((df['Group1'] == 1) & (df['Group2'] == 0)), 'group'] = 2

    # Regex to extract genomic coordinates
    coord_pattern = r'chr(\w+)_start(\d+)_end(\d+)'
    coords = df['CDS'].str.extract(coord_pattern)
    df['chrom'] = 'chr' + coords[0]
    df['start'] = pd.to_numeric(coords[1], errors='coerce')
    df['end'] = pd.to_numeric(coords[2], errors='coerce')

    # Extract transcript ID
    transcript_pattern = r'(ENST\d+\.\d+)'
    df['transcript_id'] = df['CDS'].str.extract(transcript_pattern)[0]

    # Convert omega to float
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')

    # Count special values
    omega_minus1_count = (df['omega'] == -1).sum()
    omega_99_count = (df['omega'] == 99).sum()
    print(f"{Fore.YELLOW}Rows with omega = -1: {omega_minus1_count}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Rows with omega = 99:  {omega_99_count}{Style.RESET_ALL}")

    # Optionally filter out special values
    if FILTER_SPECIAL_OMEGA_VALUES:
        original_len = len(df)
        df = df[(df['omega'] != -1) & (df['omega'] != 99)]
        filtered_count = original_len - len(df)
        print(f"Filtered out {filtered_count} rows with special omega (-1 or 99).")

    # Drop rows with NaN in 'omega'
    df = df.dropna(subset=['omega'])

    # Some summary info
    print(f"{Fore.CYAN}Total comparisons (rows) after preprocessing: {len(df)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Unique transcripts: {df['transcript_id'].nunique()}{Style.RESET_ALL}")

    return df


def get_gene_info(gene_symbol):
    """
    Retrieve the official gene name from MyGene.info for a given gene symbol.

    Parameters
    ----------
    gene_symbol : str

    Returns
    -------
    str
        Official gene name or "Unknown"
    """
    if not gene_symbol or pd.isna(gene_symbol):
        return "Unknown"
    try:
        url = f"http://mygene.info/v3/query?q=symbol:{gene_symbol}&species=human&fields=name"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                return data['hits'][0].get('name', 'Unknown')
    except Exception as e:
        print(f"{Fore.RED}Error fetching gene info: {e}{Style.RESET_ALL}")
    return "Unknown"


def get_gene_annotation(coordinates):
    """
    Retrieve gene annotation from UCSC for a given genomic coordinate string.

    Format: "chr_X_start_NNN_end_NNN"
    Example: "chr_1_start_100000_end_100500"

    Returns (gene_symbol, gene_name), or (None, None) if no gene found.
    """
    try:
        match = re.search(r'chr_(\w+)_start_(\d+)_end_(\d+)', coordinates)
        if not match:
            return (None, None)
        chrom, start, end = match.groups()
        chrom = 'chr' + chrom
        start, end = int(start), int(end)

        base_url = "https://api.genome.ucsc.edu/getData/track"
        params = {
            'genome': 'hg38',
            'track': 'knownGene',
            'chrom': chrom,
            'start': start,
            'end': end
        }
        url = f"{base_url}?{urlencode(params)}"
        response = requests.get(url, timeout=10)
        if not response.ok:
            return (None, None)

        data = response.json()
        track_data = data.get('knownGene', data)
        if not track_data or isinstance(track_data, str):
            return (None, None)

        overlapping_genes = []
        if isinstance(track_data, list):
            for gene in track_data:
                if not isinstance(gene, dict):
                    continue
                gene_start = gene.get('chromStart', 0)
                gene_end = gene.get('chromEnd', 0)
                if gene_start <= end and gene_end >= start:
                    overlapping_genes.append(gene)

        if not overlapping_genes:
            return (None, None)

        # pick best overlap
        best_gene = max(
            overlapping_genes,
            key=lambda g: max(0, min(g.get('chromEnd', 0), end) - max(g.get('chromStart', 0), start))
        )
        symbol = best_gene.get('geneName', 'Unknown')
        if symbol in ['none', None] or symbol.startswith('ENSG'):
            # try to find a better symbol among other overlapping
            for g in overlapping_genes:
                potential_symbol = g.get('geneName')
                if potential_symbol and potential_symbol != 'none' and not potential_symbol.startswith('ENSG'):
                    symbol = potential_symbol
                    break

        # get the official gene name from MyGene
        name = get_gene_info(symbol)
        return (symbol, name)

    except Exception as e:
        print(f"{Fore.RED}Error in gene annotation: {e}{Style.RESET_ALL}")
        return (None, None)


########################################################################
#                   PERMUTATION-BASED ANALYSIS LOGIC                   #
########################################################################


def compute_observed_statistic(df_transcript, group_assignments):
    """
    Compute the observed difference in median(0--0) vs median(1--1).

    Parameters
    ----------
    df_transcript : pd.DataFrame
        Subset of the main DataFrame for a single transcript
    group_assignments : dict
        Maps each sequence ID -> 0 or 1 (the original labeling).
        (Sequences that appear in cross-group pairs still have a single
         assigned group.)

    Returns
    -------
    (float, list, list)
        T_obs = median(0-0) - median(1-1)
        group0_distances = all 0--0 distances
        group1_distances = all 1--1 distances
    """
    # Build lookup (seqA, seqB) -> omega
    pair_dict = {}
    for row in df_transcript.itertuples():
        pair_dict[(row.Seq1, row.Seq2)] = row.omega
        pair_dict[(row.Seq2, row.Seq1)] = row.omega

    group0_values = []
    group1_values = []

    # Identify all sequences that appear in df_transcript
    all_seqs = pd.unique(df_transcript[['Seq1', 'Seq2']].values.ravel())

    # Gather distances for 0-0 and 1-1
    for i in all_seqs:
        for j in all_seqs:
            if j <= i:
                continue  # avoid double-counting (only i<j)
            g_i = group_assignments[i]
            g_j = group_assignments[j]
            if g_i == 0 and g_j == 0:
                val = pair_dict.get((i, j))
                if val is not None:
                    group0_values.append(val)
            elif g_i == 1 and g_j == 1:
                val = pair_dict.get((i, j))
                if val is not None:
                    group1_values.append(val)

    if len(group0_values) == 0 or len(group1_values) == 0:
        return (np.nan, group0_values, group1_values)

    median_0 = np.median(group0_values)
    median_1 = np.median(group1_values)
    T_obs = median_0 - median_1
    return (T_obs, group0_values, group1_values)


def permutation_test_transcript(df_transcript, group_assignments, n0, n1, B=10000, seed=None):
    """
    Perform a label-permutation test for a single transcript, using median-based
    difference between 0--0 and 1--1.

    Parameters
    ----------
    df_transcript : pd.DataFrame
        Data subset for this transcript
    group_assignments : dict
        Maps each sequence -> original group label (0 or 1)
    n0 : int
        Number of sequences labeled group 0 in the real data (for this transcript)
    n1 : int
        Number of sequences labeled group 1 in the real data (for this transcript)
    B : int
        Number of permutations
    seed : int or None
        Optional random seed

    Returns
    -------
    dict
        {
         'effect_size': float (T_obs),
         'p_value': float,
         'group0_count': len(0-0 pairs),
         'group1_count': len(1-1 pairs),
         'failure_reason': str or None
        }
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_seqs = pd.unique(df_transcript[['Seq1', 'Seq2']].values.ravel())

    # Build pairwise dictionary once
    pair_dict = {}
    for row in df_transcript.itertuples():
        pair_dict[(row.Seq1, row.Seq2)] = row.omega
        pair_dict[(row.Seq2, row.Seq1)] = row.omega

    # ~~~~~~~~~~~~~
    # Observed T
    # ~~~~~~~~~~~~~
    T_obs, group0vals_obs, group1vals_obs = compute_observed_statistic(
        df_transcript, group_assignments
    )
    if np.isnan(T_obs):
        # means no 0-0 or no 1-1
        failure_reason = "No valid 0-0 or 1-1 pairs"
        return {
            'effect_size': np.nan,
            'p_value': np.nan,
            'group0_count': len(group0vals_obs),
            'group1_count': len(group1vals_obs),
            'failure_reason': failure_reason
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Permutation-based p-value
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    T_perm_vals = []
    seq_list = list(all_seqs)

    # We'll want to skip permutations if either group is empty
    # But that should be rare if n0, n1 > 0 and the data is well-populated.
    # Because the real # of sequences is n0 + n1.
    # If n0 or n1 is too small, we already handle that outside.

    # Pre-allocate for speed
    # T_perm_vals = np.zeros(B, dtype=float)

    count_extreme = 0
    total_valid_perm = 0

    for b in range(B):
        random.shuffle(seq_list)  # shuffle in place
        perm_0 = set(seq_list[:n0])
        perm_1 = set(seq_list[n0:])

        # gather distances for perm 0-0 and perm 1-1
        perm_0_vals = []
        perm_1_vals = []

        for i in perm_0:
            for j in perm_0:
                if j <= i:
                    continue
                val = pair_dict.get((i, j))
                if val is not None:
                    perm_0_vals.append(val)

        for i in perm_1:
            for j in perm_1:
                if j <= i:
                    continue
                val = pair_dict.get((i, j))
                if val is not None:
                    perm_1_vals.append(val)

        if len(perm_0_vals) == 0 or len(perm_1_vals) == 0:
            # skip storing T_perm
            continue

        T_b = np.median(perm_0_vals) - np.median(perm_1_vals)
        total_valid_perm += 1
        if abs(T_b) >= abs(T_obs):
            count_extreme += 1

    if total_valid_perm == 0:
        # Means something's off with data coverage
        return {
            'effect_size': T_obs,
            'p_value': np.nan,
            'group0_count': len(group0vals_obs),
            'group1_count': len(group1vals_obs),
            'failure_reason': "No valid permutations (no 0-0 or 1-1 in permutations?)"
        }

    # compute p
    p_val = count_extreme / total_valid_perm

    return {
        'effect_size': T_obs,
        'p_value': p_val,
        'group0_count': len(group0vals_obs),
        'group1_count': len(group1vals_obs),
        'failure_reason': None
    }


def transcript_worker(args):
    """
    Worker function for parallel processing of transcripts.

    Parameters
    ----------
    args : tuple
        (transcript_id, df_transcript, group_assignments, B)

    Returns
    -------
    dict
        {
         'transcript_id': ...,
         'coordinates': str,
         'gene_symbol': str,
         'gene_name': str,
         'n0': int,
         'n1': int,
         'num_comp_group_0': int,
         'num_comp_group_1': int,
         'effect_size': float,
         'p_value': float,
         'corrected_p_value': float or NaN,
         'failure_reason': str or None,
         'matrix_0': optional np.array or None,
         'matrix_1': optional np.array or None,
         'pairwise_comparisons': set of (seq1,seq2)
        }
    """
    transcript_id, df_transcript, group_assignments, B = args

    # Gather all sequences for this transcript
    all_seqs = pd.unique(df_transcript[['Seq1', 'Seq2']].values.ravel())

    # Count how many are group0 vs group1
    seqs_0 = [s for s in all_seqs if group_assignments[s] == 0]
    seqs_1 = [s for s in all_seqs if group_assignments[s] == 1]
    n0, n1 = len(seqs_0), len(seqs_1)

    # If insufficient sequences
    if n0 < MIN_SEQUENCES_PER_GROUP or n1 < MIN_SEQUENCES_PER_GROUP:
        msg = f"Insufficient sequences in group 0 or 1 (n0={n0}, n1={n1})"
        return {
            'transcript_id': transcript_id,
            'coordinates': None,
            'gene_symbol': None,
            'gene_name': None,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': 0,
            'num_comp_group_1': 0,
            'effect_size': np.nan,
            'p_value': np.nan,
            'corrected_p_value': np.nan,
            'failure_reason': msg,
            'matrix_0': None,
            'matrix_1': None,
            'pairwise_comparisons': None
        }

    # Collect unique genomic coordinates in this transcript for a summary
    coords_set = set(f"{row.chrom}:{row.start}-{row.end}"
                     for row in df_transcript.itertuples())
    coords_str = ";".join(sorted(coords_set))

    # Attempt gene annotation from the first row
    first_row = df_transcript.iloc[0]
    coordinates_str = f"chr_{str(first_row.chrom).replace('chr','')}_start_{first_row.start}_end_{first_row.end}"
    gene_symbol, gene_name = get_gene_annotation(coordinates_str)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform the permutation test
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    result_dict = permutation_test_transcript(
        df_transcript=df_transcript,
        group_assignments=group_assignments,
        n0=n0,
        n1=n1,
        B=B
    )
    # build a pairwise dictionary for potential matrix storage
    pairwise_comps = set()
    for row in df_transcript.itertuples():
        pairwise_comps.add((row.Seq1, row.Seq2))

    # Build matrix_0, matrix_1 if you want
    matrix_0, matrix_1 = create_pairwise_matrices(df_transcript, seqs_0, seqs_1)

    # For the final output
    final = {
        'transcript_id': transcript_id,
        'coordinates': coords_str,
        'gene_symbol': gene_symbol,
        'gene_name': gene_name,
        'n0': n0,
        'n1': n1,
        'num_comp_group_0': result_dict['group0_count'],
        'num_comp_group_1': result_dict['group1_count'],
        'effect_size': result_dict['effect_size'],
        'p_value': result_dict['p_value'],
        'corrected_p_value': np.nan, # will fill in after BH
        'failure_reason': result_dict['failure_reason'],
        'matrix_0': matrix_0,
        'matrix_1': matrix_1,
        'pairwise_comparisons': pairwise_comps
    }
    return final


def create_pairwise_matrices(df_transcript, seqs_0, seqs_1):
    """
    Create pairwise matrices for group0 and group1 for potential plotting.

    Returns
    -------
    (matrix_0, matrix_1) or (None, None)
      Where each matrix is a 2D np.array of shape (len(seqs), len(seqs))
      with NaN for missing or diagonal, etc.
    """
    if len(seqs_0) == 0 and len(seqs_1) == 0:
        return (None, None)

    pair_dict = {}
    for row in df_transcript.itertuples():
        pair_dict[(row.Seq1, row.Seq2)] = row.omega
        pair_dict[(row.Seq2, row.Seq1)] = row.omega

    # Build matrix for group 0
    matrix_0 = None
    if len(seqs_0) > 0:
        seqs0_sorted = sorted(seqs_0)
        matrix_0 = np.full((len(seqs0_sorted), len(seqs0_sorted)), np.nan)
        for i in range(len(seqs0_sorted)):
            for j in range(i+1, len(seqs0_sorted)):
                s_i = seqs0_sorted[i]
                s_j = seqs0_sorted[j]
                val = pair_dict.get((s_i, s_j))
                if val is not None:
                    matrix_0[i, j] = val
                    matrix_0[j, i] = val

    # Build matrix for group 1
    matrix_1 = None
    if len(seqs_1) > 0:
        seqs1_sorted = sorted(seqs_1)
        matrix_1 = np.full((len(seqs1_sorted), len(seqs1_sorted)), np.nan)
        for i in range(len(seqs1_sorted)):
            for j in range(i+1, len(seqs1_sorted)):
                s_i = seqs1_sorted[i]
                s_j = seqs1_sorted[j]
                val = pair_dict.get((s_i, s_j))
                if val is not None:
                    matrix_1[i, j] = val
                    matrix_1[j, i] = val

    return (matrix_0, matrix_1)


########################################################################
#                                MAIN                                  #
########################################################################

def main():
    """
    Main execution function.

    Orchestrates:
    1. Data loading
    2. Group assignment build
    3. Parallel transcript analysis
    4. BH correction
    5. Output
    """
    start_time = datetime.now()
    print(f"{Fore.BLUE}=== Permutation-Based Analysis STARTED: {start_time} ==={Style.RESET_ALL}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 1: Read and preprocess
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df = read_and_preprocess_data(INPUT_CSV_PATH)

    # Build a dictionary: sequence -> group label (0 or 1)
    group_assignments = build_group_assignments(df)

    # Group by transcript
    grouped = df.groupby('transcript_id')
    print(f"{Fore.GREEN}Found {len(grouped)} transcripts in the data.{Style.RESET_ALL}")

    # Prepare arguments for parallel execution
    tasks = []
    for transcript_id, df_t in grouped:
        tasks.append((transcript_id, df_t, group_assignments, NUM_PERMUTATIONS))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 2: Parallel transcript analysis
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    results = []
    max_workers = multiprocessing.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(transcript_worker, tasks)
        for r in futures:
            results.append(r)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 3: Multiple testing correction (BH-FDR)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Filter out NaN p-values
    valid_mask = results_df['p_value'].notna() & (results_df['p_value'] > 0)
    valid_results = results_df[valid_mask].copy()
    valid_results = valid_results.sort_values('p_value')

    if len(valid_results) > 0:
        # BH procedure
        m = len(valid_results)
        valid_results['rank'] = np.arange(1, m+1)
        valid_results['bh'] = valid_results['p_value'] * m / valid_results['rank']
        valid_results['bh'] = valid_results['bh'].cummin().clip(upper=1.0)

        # map back
        bh_map = dict(zip(valid_results['transcript_id'], valid_results['bh']))
        results_df['corrected_p_value'] = results_df['transcript_id'].map(bh_map)
    else:
        results_df['corrected_p_value'] = np.nan

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Step 4: Output & Summaries
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Save full results CSV (but remove big data structures)
    drop_cols = ['matrix_0', 'matrix_1', 'pairwise_comparisons']
    output_csv_path = os.path.join(OUTPUT_DIR, "final_results_permutation.csv")
    results_df.drop(columns=drop_cols, errors='ignore').to_csv(output_csv_path, index=False)
    print(f"{Fore.MAGENTA}Saved final results to {output_csv_path}{Style.RESET_ALL}")

    # Summarize significant hits
    sig_mask = results_df['corrected_p_value'] < FDR_THRESHOLD
    num_significant = sig_mask.sum()
    print(f"{Fore.YELLOW}Number of transcripts with FDR < {FDR_THRESHOLD}: {num_significant}{Style.RESET_ALL}")
    if num_significant > 0:
        sig_results = results_df[sig_mask].copy()
        sig_results = sig_results.sort_values('p_value')
        # Save, print
        sig_csv = os.path.join(OUTPUT_DIR, "significant_transcripts.csv")
        sig_results.drop(columns=drop_cols, errors='ignore').to_csv(sig_csv, index=False)
        print(f"{Fore.MAGENTA}Significant transcripts saved to {sig_csv}{Style.RESET_ALL}")

    # Save the big data (matrices, pairwise, etc.) for future plotting
    big_results_dict = {}
    for idx, row in results_df.iterrows():
        coord = row['coordinates']
        if pd.isna(coord):
            coord = f"transcript_{row['transcript_id']}"
        # store a dictionary of relevant data
        big_results_dict[coord] = {
            'matrix_0': row['matrix_0'],
            'matrix_1': row['matrix_1'],
            'pairwise_comparisons': row['pairwise_comparisons'],
            'p_value': row['p_value'],
            'corrected_p_value': row['corrected_p_value'],
            'effect_size': row['effect_size'],
            'gene_symbol': row['gene_symbol'],
            'gene_name': row['gene_name']
        }
    big_results_pkl = os.path.join(CACHE_DIR, "all_cds_results_permutation.pkl")
    with open(big_results_pkl, 'wb') as f:
        pickle.dump(big_results_dict, f)
    print(f"{Fore.MAGENTA}Saved matrix data to {big_results_pkl}{Style.RESET_ALL}")

    # Print a short summary table
    print("\n=== Summary by Transcript ===")
    print(f"{'Transcript':<20} {'n0':>5} {'n1':>5} {'EffectSize':>12} {'p-val':>12} {'FDR':>12} {'Failure':>20}")
    for idx, row in results_df.sort_values('p_value', na_position='last').iterrows():
        tid = str(row['transcript_id'])
        efs = f"{row['effect_size']:.4f}" if not pd.isna(row['effect_size']) else "NA"
        pv  = f"{row['p_value']:.3e}" if not pd.isna(row['p_value']) else "NA"
        cpv = f"{row['corrected_p_value']:.3e}" if not pd.isna(row['corrected_p_value']) else "NA"
        fail = row['failure_reason']
        if pd.isna(fail):
            fail = ""
        print(f"{tid:<20} {row['n0']:>5} {row['n1']:>5} {efs:>12} {pv:>12} {cpv:>12} {fail:>20}")

    end_time = datetime.now()
    print(f"\n{Fore.BLUE}=== Analysis COMPLETED at: {end_time} ==={Style.RESET_ALL}")
    print(f"Total runtime: {end_time - start_time}")


def build_group_assignments(df):
    """
    Build a dict: sequence -> group (0 or 1).

    - Very simple. We have explicit columns 'Group1' / 'Group2' for each row, specifying
    the groups for Seq1, Seq2.

    Returns
    -------
    dict
        sequence_id -> 0 or 1
    """
    # Each sequence has a consistent group across all transcripts.

    seq_group_map = {}
    for row in df.itertuples():
        g1 = row.Group1
        g2 = row.Group2
        s1 = row.Seq1
        s2 = row.Seq2

        if g1 in [0, 1]:
            # store
            prev = seq_group_map.get(s1)
            if prev is None:
                seq_group_map[s1] = g1
            else:
                if prev != g1:
                    # conflict
                    print("!")
                    pass

        if g2 in [0, 1]:
            prev = seq_group_map.get(s2)
            if prev is None:
                seq_group_map[s2] = g2
            else:
                if prev != g2:
                    # conflict
                    print("!")
                    pass

    unassigned = [s for s in df['Seq1'].unique() if s not in seq_group_map] + \
                 [s for s in df['Seq2'].unique() if s not in seq_group_map]
    unassigned = list(set(unassigned))

    for row in df.itertuples():
        if row.group == 2:
            # cross
            # means Seq1 is group row.Group1, Seq2 is group row.Group2
            if row.Group1 in [0,1] and row.Seq1 not in seq_group_map:
                seq_group_map[row.Seq1] = row.Group1
            if row.Group2 in [0,1] and row.Seq2 not in seq_group_map:
                seq_group_map[row.Seq2] = row.Group2

    # any that remain truly unassigned
    for s in unassigned:
        if s not in seq_group_map:
            print(f"Warning: sequence {s} was never assigned. Setting group=0 by default.")

    return seq_group_map


if __name__ == "__main__":
    main()
