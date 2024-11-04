#!/usr/bin/env python3

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
import threading
import time
from collections import defaultdict
from datetime import datetime
from scipy.stats import mannwhitneyu

# Function to parse .phy files
def parse_phy_file(filepath):
    sequences = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        if len(lines) < 1:
            print(f"Error: Empty .phy file {filepath}")
            return sequences
        try:
            num_sequences, seq_length = map(int, lines[0].strip().split())
        except ValueError:
            print(f"Error parsing the header of {filepath}")
            return sequences
        for line in lines[1:]:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                # In case the sequence name and sequence are concatenated
                name = line[:10].strip()
                sequence = line[10:].strip()
            else:
                name = parts[0].strip()
                sequence = parts[1].strip()
            sequences[name] = sequence
    return sequences

# Function to extract group from sample name
def extract_group_from_sample(sample_name):
    # Assuming group is indicated by the last character after '_'
    # e.g., EAS_CHB_NA18534_1
    match = re.search(r'_(\d+)$', sample_name)
    if match:
        return int(match.group(1))
    else:
        return None

# Function to create PAML control file
def create_paml_ctl(seqfile, outfile, working_dir):
    ctl_content = f"""
      seqfile = {seqfile}
      treefile = None
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
    with open(ctl_path, 'w') as ctl_file:
        ctl_file.write(ctl_content)
    return ctl_path

# Function to run PAML codeml
def run_codeml(ctl_path, working_dir, codeml_path):
    try:
        # Run codeml with a timeout
        result = subprocess.run([codeml_path], cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if result.returncode != 0:
            print(f"codeml error in {working_dir}: {result.stderr.decode()}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"codeml timed out in {working_dir}")
        return False
    except Exception as e:
        print(f"Error running codeml in {working_dir}: {e}")
        return False

# Function to parse PAML output
def parse_codeml_output(outfile):
    dN = None
    dS = None
    omega = None
    try:
        with open(outfile, 'r') as f:
            for line in f:
                if "dN =" in line and "dS =" in line and "omega =" in line:
                    parts = line.strip().split()
                    try:
                        dN = float(parts[1])
                        dS = float(parts[4])
                        omega = float(parts[7])
                    except (IndexError, ValueError):
                        pass
                    break
    except FileNotFoundError:
        print(f"Results file {outfile} not found.")
    return dN, dS, omega

# Function to process a single pairwise comparison
def process_pair(args):
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir = args
    seq1_name, seq2_name = pair
    seq1 = sequences[seq1_name]
    seq2 = sequences[seq2_name]

    # Create a temporary directory for this comparison
    working_dir = os.path.join(temp_dir, f'temp_{seq1_name}_{seq2_name}')
    os.makedirs(working_dir, exist_ok=True)

    # Create a temporary .phy file with these two sequences
    temp_phy = os.path.join(working_dir, 'temp.phy')
    with open(temp_phy, 'w') as phy_file:
        phy_file.write(f" 2 {len(seq1)}\n")
        phy_file.write(f"{seq1_name:<10}{seq1}\n")
        phy_file.write(f"{seq2_name:<10}{seq2}\n")

    # Create PAML control file
    ctl_path = create_paml_ctl('temp.phy', 'results.txt', working_dir)

    # Run codeml
    success = run_codeml(ctl_path, working_dir, codeml_path)

    # Parse results
    results_file = os.path.join(working_dir, 'results.txt')
    dN, dS, omega = parse_codeml_output(results_file)

    # Clean up temporary directory
    shutil.rmtree(working_dir)

    if not success:
        return (seq1_name, seq2_name, sample_groups[seq1_name], sample_groups[seq2_name], None, None, None)

    return (seq1_name, seq2_name, sample_groups[seq1_name], sample_groups[seq2_name], dN, dS, omega)

# Function to process a single .phy file
def process_phy_file(args):
    phy_file, output_dir, codeml_path, total_files, file_index = args

    # Extract CDS information from filename
    phy_filename = os.path.basename(phy_file)
    match = re.match(r'group_(\d+)_chr_(.+)_start_(\d+)_end_(\d+)\.phy', phy_filename)
    if match:
        group = int(match.group(1))
        chr_num = match.group(2)
        start = match.group(3)
        end = match.group(4)
        cds_id = f'chr{chr_num}_start{start}_end{end}'
    else:
        cds_id = phy_filename.replace('.phy', '')
        group = None  # Group will be determined from sample names

    # Output CSV file for this CDS
    output_csv = os.path.join(output_dir, f'{cds_id}.csv')
    haplotype_output_csv = os.path.join(output_dir, f'{cds_id}_haplotype_stats.csv')

    if os.path.exists(output_csv) and os.path.exists(haplotype_output_csv):
        print(f"[{file_index}/{total_files}] Skipping {phy_file} as output CSV already exists.")
        return haplotype_output_csv

    print(f"[{file_index}/{total_files}] Processing {phy_file}...")

    # Read sequences
    sequences = parse_phy_file(phy_file)
    if not sequences:
        print(f"No sequences found in {phy_file}. Skipping.")
        return None
    sample_names = list(sequences.keys())

    # Extract groups from sample names
    sample_groups = {}
    for sample in sample_names:
        sample_group = extract_group_from_sample(sample)
        if sample_group is not None:
            sample_groups[sample] = sample_group
        else:
            sample_groups[sample] = group  # Use group from filename if not in sample name

    # Generate all pairwise combinations
    pairs = list(combinations(sample_names, 2))
    total_pairs = len(pairs)

    if total_pairs == 0:
        print(f"No pairs to process in {phy_file}.")
        return None

    # Create a temporary directory for this CDS
    temp_dir = os.path.join(output_dir, f'temp_{cds_id}_{datetime.now().strftime("%Y%m%d%H%M%S%f")}')
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    pool_args = []
    for pair in pairs:
        pool_args.append((pair, sequences, sample_groups, cds_id, codeml_path, temp_dir))

    # Use multiprocessing to process pairs
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(process_pair, pool_args), 1):
            results.append(res)
            if i % max(1, total_pairs // 10) == 0 or i == total_pairs:
                percent = (i / total_pairs) * 100
                print(f"  Processed {i}/{total_pairs} pairs ({percent:.2f}%) in {phy_file}")

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    # Collect results
    df = pd.DataFrame(results, columns=['Seq1', 'Seq2', 'Group1', 'Group2', 'dN', 'dS', 'omega'])
    df['CDS'] = cds_id

    # Save individual CDS results
    df.to_csv(output_csv, index=False)

    # Compute mean and median dN/dS for each haplotype
    haplotype_stats = []
    for sample in sample_names:
        # Get all comparisons involving this sample
        sample_df = df[(df['Seq1'] == sample) | (df['Seq2'] == sample)]
        omega_values = sample_df['omega'].dropna()
        mean_omega = omega_values.mean()
        median_omega = omega_values.median()
        haplotype_stats.append({
            'Haplotype': sample,
            'Group': sample_groups[sample],
            'CDS': cds_id,
            'Mean_dNdS': mean_omega,
            'Median_dNdS': median_omega
        })

    haplotype_df = pd.DataFrame(haplotype_stats)
    haplotype_df.to_csv(haplotype_output_csv, index=False)

    # Print stats for this CDS
    print(f"Statistics for CDS {cds_id}:")
    group0 = haplotype_df[haplotype_df['Group'] == 0]['Mean_dNdS'].dropna()
    group1 = haplotype_df[haplotype_df['Group'] == 1]['Mean_dNdS'].dropna()
    if not group0.empty:
        print(f"  Group 0 - Mean dN/dS: {group0.mean():.4f}, Median: {group0.median():.4f}, SD: {group0.std():.4f}")
    if not group1.empty:
        print(f"  Group 1 - Mean dN/dS: {group1.mean():.4f}, Median: {group1.median():.4f}, SD: {group1.std():.4f}")

    return haplotype_output_csv

# Function to perform statistical tests
def perform_statistical_tests(haplotype_stats_files):
    # Combine all haplotype stats into a single DataFrame
    haplotype_dfs = []
    for f in haplotype_stats_files:
        try:
            df = pd.read_csv(f)
            haplotype_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not haplotype_dfs:
        print("No haplotype statistics files to process.")
        return

    haplotype_df = pd.concat(haplotype_dfs, ignore_index=True)

    # Save final CSV
    haplotype_df.to_csv('final_haplotype_stats.csv', index=False)

    # Compare group 0 and group 1 across all CDS
    group0 = haplotype_df[haplotype_df['Group'] == 0]['Mean_dNdS'].dropna()
    group1 = haplotype_df[haplotype_df['Group'] == 1]['Mean_dNdS'].dropna()

    print("\nOverall Statistical Analysis:")
    if not group0.empty:
        print(f"Group 0 - Mean dN/dS: {group0.mean():.4f}, Median: {group0.median():.4f}, SD: {group0.std():.4f}")
    if not group1.empty:
        print(f"Group 1 - Mean dN/dS: {group1.mean():.4f}, Median: {group1.median():.4f}, SD: {group1.std():.4f}")

    # Perform statistical test (e.g., Mann-Whitney U test)
    if not group0.empty and not group1.empty:
        stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
        print(f"Mann-Whitney U test: Statistic={stat}, p-value={p_value}")

        if p_value < 0.05:
            print("There is a significant difference between Group 0 and Group 1.")
        else:
            print("There is no significant difference between Group 0 and Group 1.")
    else:
        print("Not enough data to perform statistical tests.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
    parser.add_argument('--codeml_path', type=str, default='/home/hsiehph/sauer354/di/paml/bin/codeml', help='Path to the codeml executable.')
    args = parser.parse_args()

    phy_dir = args.phy_dir
    output_dir = args.output_dir
    codeml_path = args.codeml_path

    os.makedirs(output_dir, exist_ok=True)

    # Get list of all .phy files
    phy_files = glob.glob(os.path.join(phy_dir, '*.phy'))

    total_cds = len(phy_files)
    print(f"Total CDS files to process: {total_cds}")

    # Prepare arguments for multiprocessing
    pool_args = []
    for idx, phy_file in enumerate(phy_files, 1):
        pool_args.append((phy_file, output_dir, codeml_path, total_cds, idx))

    # Use multiprocessing to process CDS files
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_processes) as pool:
        haplotype_stats_files = []
        for i, result in enumerate(pool.imap_unordered(process_phy_file, pool_args), 1):
            if result:
                haplotype_stats_files.append(result)
            percent = (i / total_cds) * 100
            print(f"Completed {i}/{total_cds} CDS files ({percent:.2f}%)")

    # Perform statistical tests
    perform_statistical_tests(haplotype_stats_files)

if __name__ == '__main__':
    main()
