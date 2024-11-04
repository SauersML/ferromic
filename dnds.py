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

# Function to parse .phy files
def parse_phy_file(filepath):
    sequences = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        num_sequences, seq_length = map(int, lines[0].strip().split())
        for line in lines[1:]:
            if line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                # In case the sequence name and sequence are concatenated
                name = parts[0][:10].strip()
                sequence = parts[0][10:].strip()
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
      noisy = 9
      verbose = 1
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
def run_codeml(ctl_path, working_dir):
    try:
        # Change directory to the working directory
        subprocess.run(['codeml', ctl_path], cwd=working_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error running codeml: {e}")

# Function to parse PAML output
def parse_codeml_output(outfile):
    dN = None
    dS = None
    omega = None
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
    return dN, dS, omega

# Function to process a single pairwise comparison
def process_pair(args):
    pair, sequences, phy_filename, output_dir = args
    seq1_name, seq2_name = pair
    seq1 = sequences[seq1_name]
    seq2 = sequences[seq2_name]

    # Create a temporary directory for this comparison
    working_dir = os.path.join(output_dir, f'temp_{seq1_name}_{seq2_name}')
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
    run_codeml(ctl_path, working_dir)

    # Parse results
    results_file = os.path.join(working_dir, 'results.txt')
    dN, dS, omega = parse_codeml_output(results_file)

    # Clean up temporary directory
    shutil.rmtree(working_dir)

    return (seq1_name, seq2_name, dN, dS, omega)

# Function to process a single .phy file
def process_phy_file(phy_file, output_dir, progress_queue):
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
    if os.path.exists(output_csv):
        print(f"Skipping {phy_file} as output CSV already exists.")
        progress_queue.put(1)
        return

    print(f"Processing {phy_file}...")

    # Read sequences
    sequences = parse_phy_file(phy_file)
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

    # Prepare arguments for multiprocessing
    pool_args = []
    for pair in pairs:
        pool_args.append((pair, sequences, phy_filename, output_dir))

    # Use multiprocessing to process pairs
    with multiprocessing.Pool() as pool:
        results = pool.map(process_pair, pool_args)

    # Collect results
    df = pd.DataFrame(results, columns=['Seq1', 'Seq2', 'dN', 'dS', 'omega'])
    df['CDS'] = cds_id
    df['Group1'] = df['Seq1'].apply(lambda x: sample_groups[x])
    df['Group2'] = df['Seq2'].apply(lambda x: sample_groups[x])

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
    haplotype_output_csv = os.path.join(output_dir, f'{cds_id}_haplotype_stats.csv')
    haplotype_df.to_csv(haplotype_output_csv, index=False)

    progress_queue.put(1)

# Function to monitor progress
def progress_monitor(total_tasks, progress_queue):
    completed = 0
    while completed < total_tasks:
        progress_queue.get()
        completed += 1
        percent = (completed / total_tasks) * 100
        print(f"Progress: {completed}/{total_tasks} CDS files processed ({percent:.2f}%)")

# Function to perform statistical tests
def perform_statistical_tests(all_haplotype_stats):
    # Combine all haplotype stats into a single DataFrame
    haplotype_df = pd.concat(all_haplotype_stats, ignore_index=True)

    # Save final CSV
    haplotype_df.to_csv('final_haplotype_stats.csv', index=False)

    # Compare group 0 and group 1
    group0 = haplotype_df[haplotype_df['Group'] == 0]['Mean_dNdS'].dropna()
    group1 = haplotype_df[haplotype_df['Group'] == 1]['Mean_dNdS'].dropna()

    print("\nStatistical Analysis:")
    print(f"Group 0 - Mean dN/dS: {group0.mean():.4f}, Median: {group0.median():.4f}, SD: {group0.std():.4f}")
    print(f"Group 1 - Mean dN/dS: {group1.mean():.4f}, Median: {group1.median():.4f}, SD: {group1.std():.4f}")

    # Perform statistical test (e.g., Mann-Whitney U test)
    from scipy.stats import mannwhitneyu

    stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
    print(f"Mann-Whitney U test: Statistic={stat}, p-value={p_value}")

    if p_value < 0.05:
        print("There is a significant difference between Group 0 and Group 1.")
    else:
        print("There is no significant difference between Group 0 and Group 1.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
    args = parser.parse_args()

    phy_dir = args.phy_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Get list of all .phy files
    phy_files = glob.glob(os.path.join(phy_dir, '*.phy'))

    total_cds = len(phy_files)
    progress_queue = multiprocessing.Queue()

    # Start progress monitor thread
    progress_thread = threading.Thread(target=progress_monitor, args=(total_cds, progress_queue))
    progress_thread.start()

    # Process each .phy file
    all_haplotype_stats = []
    processes = []
    for phy_file in phy_files:
        p = multiprocessing.Process(target=process_phy_file, args=(phy_file, output_dir, progress_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    progress_thread.join()

    # Collect all haplotype stats
    haplotype_stats_files = glob.glob(os.path.join(output_dir, '*_haplotype_stats.csv'))
    all_haplotype_stats = [pd.read_csv(f) for f in haplotype_stats_files]

    # Perform statistical tests
    perform_statistical_tests(all_haplotype_stats)

if __name__ == '__main__':
    main()
