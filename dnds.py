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
import hashlib
from scipy.stats import mannwhitneyu, levene
import matplotlib.pyplot as plt
import pickle

# ----------------------------
# Setup Logging
# ----------------------------

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s [%(levelname)s] %(message)s',
   handlers=[
       logging.FileHandler('dnds_analysis.log'),
       logging.StreamHandler(sys.stdout)
   ]
)

# ----------------------------
# Utility Functions
# ----------------------------

def validate_sequence(seq):
    """
    Validate the nucleotide sequence.

    Checks if the sequence is codon-aligned (length divisible by 3)
    and contains only valid nucleotide characters.

    Parameters:
    seq (str): The nucleotide sequence.

    Returns:
    str or None: Returns the validated sequence or None if invalid.
    """
    if len(seq) % 3 != 0:
        logging.warning(f"Sequence length {len(seq)} not divisible by 3. Skipping sequence.")
        return None

    valid_bases = set('ATCGNatcgn-')
    if not set(seq).issubset(valid_bases):
        invalid_chars = set(seq) - valid_bases
        logging.warning(f"Invalid characters {invalid_chars} found in sequence. Skipping sequence.")
        return None

    return seq.upper()

def extract_group_from_sample(sample_name):
    """
    Extract the group number from the sample name.

    Assumes sample names end with '_0' or '_1' indicating the group.

    Parameters:
    sample_name (str): The sample name.

    Returns:
    int or None: Returns the group number or None if not found.
    """
    match = re.search(r'_(0|1)$', sample_name)
    if match:
        return int(match.group(1))
    else:
        logging.warning(f"Group suffix not found in sample name: {sample_name}")
        return None

def create_paml_ctl(seqfile, outfile, working_dir):
    """
    Create the CODEML control file with necessary parameters.

    Parameters:
    seqfile (str): Path to the sequence file.
    outfile (str): Name of the output file.
    working_dir (str): The working directory.

    Returns:
    str: Path to the control file.
    """
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
    """
    Run the CODEML program.

    Parameters:
    ctl_path (str): Path to the control file.
    working_dir (str): The working directory.
    codeml_path (str): Path to the codeml executable.

    Returns:
    bool: True if CODEML ran successfully, False otherwise.
    """
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
    """
    Parse the output from CODEML.

    Parameters:
    outfile_dir (str): The directory containing CODEML output files.

    Returns:
    tuple: (dN, dS, omega)
    """
    rst_file = os.path.join(outfile_dir, 'rst')
    if not os.path.exists(rst_file):
        logging.error(f"CODEML output file not found: {rst_file}")
        return None, None, None

    try:
        with open(rst_file, 'r') as f:
            content = f.read()

        # Pattern to extract dN, dS, and omega
        pattern = re.compile(
            r"t=\s*[\d\.]+\s+S=\s*([\d\.]+)\s+N=\s*([\d\.]+)\s+"
            r"dN/dS=\s*([\d\.]+)\s+dN=\s*([\d\.]+)\s+dS=\s*([\d\.]+)"
        )
        match = pattern.search(content)
        if match:
            S = float(match.group(1))
            N = float(match.group(2))
            omega = float(match.group(3))
            dN = float(match.group(4))
            dS = float(match.group(5))
            return dN, dS, omega
        else:
            logging.error("Could not parse CODEML output.")
            return None, None, None
    except Exception as e:
        logging.error(f"Error parsing CODEML output: {e}")
        return None, None, None

def get_safe_process_count():
    """
    Determine a safe number of processes based on system resources.

    Returns:
    int: Number of processes to use.
    """
    total_cpus = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    safe_processes = max(1, min(total_cpus // 2, int(mem.available / (2 * 1024**3))))
    return safe_processes

def parse_phy_file(filepath):
    """
    Parse a PHYLIP file to extract sequences.

    Parameters:
    filepath (str): Path to the PHYLIP file.

    Returns:
    dict: Dictionary of sequences keyed by sample name.
    """
    sequences = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        if not lines:
            logging.error(f"PHYLIP file is empty: {filepath}")
            return sequences

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
                sequences[sample_name] = validated_seq
    return sequences

# ----------------------------
# Caching Mechanism
# ----------------------------

def load_cache(cache_file):
    """
    Load the cache from a file.

    Parameters:
    cache_file (str): Path to the cache file.

    Returns:
    dict: Cached data.
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        logging.info(f"Cache loaded from {cache_file}")
    else:
        cache = {}
    return cache

def save_cache(cache_file, cache_data):
    """
    Save the cache to a file.

    Parameters:
    cache_file (str): Path to the cache file.
    cache_data (dict): Data to cache.
    """
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    logging.info(f"Cache saved to {cache_file}")

# ----------------------------
# Core Processing Functions
# ----------------------------

def process_pair(args):
    """
    Process a pair of sequences.

    Parameters:
    args (tuple): Contains all arguments needed for processing.

    Returns:
    tuple: Results of the processing.
    """
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, cache = args
    seq1_name, seq2_name = pair

    # Check if result is already cached
    cache_key = (cds_id, seq1_name, seq2_name)
    if cache_key in cache:
        return cache[cache_key]

    if seq1_name not in sequences or seq2_name not in sequences:
        logging.error(f"Sequences not found for pair: {seq1_name}, {seq2_name}")
        return None

    group1 = sample_groups.get(seq1_name)
    group2 = sample_groups.get(seq2_name)

    if group1 != group2:
        logging.error(f"Sequences from different groups: {seq1_name}, {seq2_name}")
        return None

    if sequences[seq1_name] == sequences[seq2_name]:
        # Identical sequences
        result = (seq1_name, seq2_name, group1, group2, 0.0, 0.0, -1.0, cds_id)
        cache[cache_key] = result
        return result

    # Prepare working directory
    working_dir = os.path.join(temp_dir, f'{seq1_name}_{seq2_name}')
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # Create sequence file
    seqfile = os.path.join(working_dir, 'seqfile.phy')
    with open(seqfile, 'w') as f:
        f.write(f" 2 {len(sequences[seq1_name])}\n")
        f.write(f"{seq1_name} {sequences[seq1_name]}\n")
        f.write(f"{seq2_name} {sequences[seq2_name]}\n")

    # Create tree file
    treefile = os.path.join(working_dir, 'tree.txt')
    with open(treefile, 'w') as f:
        f.write(f"({seq1_name},{seq2_name});\n")

    # Create control file
    ctl_path = create_paml_ctl(seqfile, 'mlc', working_dir)

    # Run CODEML
    success = run_codeml(ctl_path, working_dir, codeml_path)
    if not success:
        result = (seq1_name, seq2_name, group1, group2, np.nan, np.nan, np.nan, cds_id)
        cache[cache_key] = result
        return result

    # Parse output
    dn, ds, omega = parse_codeml_output(working_dir)
    if omega is None:
        omega = np.nan

    result = (seq1_name, seq2_name, group1, group2, dn, ds, omega, cds_id)
    cache[cache_key] = result
    return result

def process_phy_file(args):
    """
    Process a PHYLIP file.

    Parameters:
    args (tuple): Contains all arguments needed for processing.

    Returns:
    str: Path to the haplotype statistics CSV file.
    """
    phy_file, output_dir, codeml_path, total_files, file_index, cache = args
    start_time = time.time()

    # Extract CDS ID
    phy_filename = os.path.basename(phy_file)
    cds_id = phy_filename.replace('.phy', '')

    # Output files
    output_csv = os.path.join(output_dir, f'{cds_id}.csv')
    haplotype_output_csv = os.path.join(output_dir, f'{cds_id}_haplotype_stats.csv')

    # Check if results already exist
    if os.path.exists(output_csv) and os.path.exists(haplotype_output_csv):
        logging.info(f"Results already exist for {cds_id}. Skipping.")
        return haplotype_output_csv

    # Parse sequences
    sequences = parse_phy_file(phy_file)
    if not sequences:
        logging.error(f"No valid sequences in {phy_file}. Skipping.")
        return None

    # Assign groups
    sample_groups = {}
    for sample_name in sequences.keys():
        group = extract_group_from_sample(sample_name)
        if group is None:
            logging.error(f"Group not found for sample {sample_name}. Skipping.")
            return None
        sample_groups[sample_name] = group

    # Generate pairs within groups
    group0_samples = [s for s, g in sample_groups.items() if g == 0]
    group1_samples = [s for s, g in sample_groups.items() if g == 1]

    group0_pairs = list(combinations(group0_samples, 2))
    group1_pairs = list(combinations(group1_samples, 2))

    pairs = group0_pairs + group1_pairs

    if not pairs:
        logging.error(f"No pairs to process in {phy_file}. Skipping.")
        return None

    # Create temporary directory
    temp_dir = os.path.join(output_dir, 'temp', cds_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare arguments for multiprocessing
    pool_args = []
    for pair in pairs:
        pool_args.append((pair, sequences, sample_groups, cds_id, codeml_path, temp_dir, cache))

    # Process pairs with multiprocessing
    num_processes = get_safe_process_count()
    manager = multiprocessing.Manager()
    progress_counter = manager.Value('i', 0)
    total_pairs = len(pool_args)
    results = []

    def update_progress(result):
        if result:
            results.append(result)
        with progress_counter.get_lock():
            progress_counter.value += 1
            progress = (progress_counter.value / total_pairs) * 100
            logging.info(f"Progress: {progress_counter.value}/{total_pairs} pairs processed ({progress:.2f}%)")

    with multiprocessing.Pool(processes=num_processes) as pool:
        for _ in pool.imap_unordered(process_pair, pool_args):
            update_progress(_)

    # Save pairwise results
    df = pd.DataFrame(results, columns=['Seq1', 'Seq2', 'Group1', 'Group2', 'dN', 'dS', 'omega', 'CDS'])
    df.to_csv(output_csv, index=False)

    # Calculate haplotype statistics
    haplotype_stats = []
    for sample in sequences.keys():
        sample_df = df[(df['Seq1'] == sample) | (df['Seq2'] == sample)]
        omega_values = sample_df['omega'].dropna()
        # Exclude -1 and 99 before averaging
        omega_values = omega_values[~omega_values.isin([-1, 99])]
        if not omega_values.empty:
            mean_omega = omega_values.mean()
            median_omega = omega_values.median()
        else:
            mean_omega = np.nan
            median_omega = np.nan
        haplotype_stats.append({
            'Haplotype': sample,
            'Group': sample_groups[sample],
            'CDS': cds_id,
            'Mean_omega': mean_omega,
            'Median_omega': median_omega,
            'Num_Comparisons': len(omega_values)
        })

    haplotype_df = pd.DataFrame(haplotype_stats)
    haplotype_df.to_csv(haplotype_output_csv, index=False)

    # Clean up temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)

    end_time = time.time()
    logging.info(f"Processed {phy_file} in {end_time - start_time:.2f} seconds.")

    return haplotype_output_csv

def perform_statistical_tests(haplotype_stats_files, output_dir):
    """
    Perform statistical tests on the combined haplotype statistics.

    Parameters:
    haplotype_stats_files (list): List of haplotype statistics files.
    output_dir (str): Directory to save outputs.

    Returns:
    None
    """
    haplotype_dfs = []
    for f in haplotype_stats_files:
        try:
            df = pd.read_csv(f)
            haplotype_dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read {f}: {e}")

    if not haplotype_dfs:
        logging.warning("No haplotype statistics to analyze.")
        return

    combined_df = pd.concat(haplotype_dfs, ignore_index=True)

    # Prepare datasets
    datasets = {
        "All data": combined_df,
        "Excluding dN/dS = -1": combined_df[combined_df['Mean_omega'] != -1],
        "Excluding dN/dS = 99": combined_df[combined_df['Mean_omega'] != 99],
        "Excluding both -1 and 99": combined_df[(combined_df['Mean_omega'] != -1) & (combined_df['Mean_omega'] != 99)]
    }

    for dataset_name, df in datasets.items():
        logging.info(f"Analyzing dataset: {dataset_name}")
        group0 = df[df['Group'] == 0]['Mean_omega'].dropna()
        group1 = df[df['Group'] == 1]['Mean_omega'].dropna()

        if group0.empty or group1.empty:
            logging.warning(f"Insufficient data for {dataset_name}. Skipping.")
            continue

        # Perform statistical tests
        stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
        logging.info(f"Mann-Whitney U test for {dataset_name}: p-value = {p_value:.4f}")

        # Levene's test for variance
        if dataset_name == "Excluding both -1 and 99":
            stat, p_value = levene(group0, group1)
            logging.info(f"Levene's test for {dataset_name}: p-value = {p_value:.4f}")

        # Generate histograms
        plt.figure()
        plt.hist(group0, bins=20, alpha=0.5, label='Group 0')
        plt.hist(group1, bins=20, alpha=0.5, label='Group 1')
        plt.legend()
        plt.title(f"Histogram of Mean Omega - {dataset_name}")
        plt.xlabel('Mean Omega')
        plt.ylabel('Frequency')
        plt.tight_layout()
        histogram_file = os.path.join(output_dir, f"histogram_{dataset_name.replace(' ', '_')}.png")
        plt.savefig(histogram_file)
        plt.close()
        logging.info(f"Histogram saved: {histogram_file}")

def analyze_cds_per_individual(haplotype_stats_files, output_dir):
    """
    Analyze each CDS individually and perform statistical tests.

    Parameters:
    haplotype_stats_files (list): List of haplotype statistics files.
    output_dir (str): Directory to save outputs.

    Returns:
    None
    """
    cds_results = []
    for f in haplotype_stats_files:
        try:
            df = pd.read_csv(f)
            cds_id = df['CDS'].iloc[0]
            # Exclude -1 and 99
            df = df[(df['Mean_omega'] != -1) & (df['Mean_omega'] != 99)]

            group0 = df[df['Group'] == 0]['Mean_omega'].dropna()
            group1 = df[df['Group'] == 1]['Mean_omega'].dropna()

            if len(group0) >= 3 and len(group1) >= 3:
                # Perform Mann-Whitney U test
                stat, p_value = mannwhitneyu(group0, group1, alternative='two-sided')
                significant = p_value < 0.05
                cds_results.append({
                    'CDS': cds_id,
                    'p_value': p_value,
                    'Significant': significant
                })
        except Exception as e:
            logging.error(f"Error processing {f}: {e}")

    # Calculate proportion of significant CDSs
    if cds_results:
        results_df = pd.DataFrame(cds_results)
        total_cds = len(results_df)
        significant_cds = results_df[results_df['Significant']].shape[0]
        proportion_significant = significant_cds / total_cds

        logging.info(f"Total CDSs analyzed: {total_cds}")
        logging.info(f"Number of significant CDSs: {significant_cds}")
        logging.info(f"Proportion of significant CDSs: {proportion_significant:.2%}")

        # Save results
        results_file = os.path.join(output_dir, 'cds_statistical_results.csv')
        results_df.to_csv(results_file, index=False)
        logging.info(f"CDS statistical results saved to {results_file}")

# ----------------------------
# Main Function
# ----------------------------

def main():
    """
    Main function to orchestrate the dN/dS analysis.
    """
    parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
    parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
    parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
    parser.add_argument('--codeml_path', type=str, default='codeml', help='Path to codeml executable.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load cache
    cache_file = os.path.join(args.output_dir, 'results_cache.pkl')
    cache = load_cache(cache_file)

    # Preliminary analysis
    existing_haplotype_files = glob.glob(os.path.join(args.output_dir, '*_haplotype_stats.csv'))
    if existing_haplotype_files:
        logging.info("Performing preliminary analysis on existing results.")
        perform_statistical_tests(existing_haplotype_files, args.output_dir)
        analyze_cds_per_individual(existing_haplotype_files, args.output_dir)
    else:
        logging.info("No existing results found. Proceeding with full analysis.")

    # Find PHYLIP files
    phy_files = glob.glob(os.path.join(args.phy_dir, '*.phy'))
    total_files = len(phy_files)
    logging.info(f"Found {total_files} PHYLIP files to process.")

    # Process files
    work_args = []
    for idx, phy_file in enumerate(phy_files, 1):
        work_args.append((phy_file, args.output_dir, args.codeml_path, total_files, idx, cache))

    new_haplotype_files = []
    for args_tuple in work_args:
        haplotype_file = process_phy_file(args_tuple)
        if haplotype_file:
            new_haplotype_files.append(haplotype_file)
        # Save cache after each file
        save_cache(cache_file, cache)

    # Final analysis
    all_haplotype_files = glob.glob(os.path.join(args.output_dir, '*_haplotype_stats.csv'))
    if all_haplotype_files:
        logging.info("Performing final analysis on all results.")
        perform_statistical_tests(all_haplotype_files, args.output_dir)
        analyze_cds_per_individual(all_haplotype_files, args.output_dir)
    else:
        logging.warning("No haplotype statistics files found for analysis.")

    logging.info("dN/dS analysis completed.")

if __name__ == '__main__':
    main()
