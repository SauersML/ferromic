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
from scipy.stats import mannwhitneyu, wilcoxon, levene

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
    """Check if the sequence is codon-aligned and contains only valid nucleotides."""
    if len(seq) % 3 != 0:
        logging.warning(f"Skipping sequence of length {len(seq)}: not divisible by 3")
        return None

    valid_bases = set('ATCGNatcgn-')
    invalid_chars = set(seq) - valid_bases
    if invalid_chars:
        logging.warning(f"Skipping sequence with invalid nucleotides: {invalid_chars}")
        return None

    return seq

def generate_checksum(full_name):
    """Create a 3-character checksum from the sample name."""
    hash_object = hashlib.md5(full_name.encode())
    checksum = hash_object.hexdigest()[:3].upper()
    return checksum

def extract_group_from_sample(sample_name):
    """Retrieve the group number from the sample name ending with _0 or _1."""
    sample_name = sample_name.strip()
    match = re.search(r'_(0|1)$', sample_name)
    if match:
        return int(match.group(1))
    logging.warning(f"Group suffix not found in sample name: {sample_name}")
    return None

def create_paml_ctl(seqfile, outfile, working_dir):
    """Generate the CODEML control file with necessary parameters."""
    ctl_content = f"""      seqfile = {seqfile}
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
    """Execute CODEML and handle its output."""
    logging.info(f"Running CODEML in directory: {working_dir}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            [codeml_path],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )

        stdout, stderr = process.communicate(input=b'\n', timeout=3600)
        runtime = time.time() - start_time
        logging.info(f"CODEML completed in {runtime:.2f} seconds")
        
        if process.returncode != 0:
            logging.error(f"CODEML encountered an error: {stderr.decode().strip()}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        logging.error("CODEML execution timed out.")
        process.kill()
        return False
    except Exception as e:
        logging.error(f"Error running CODEML: {e}")
        return False

def parse_codeml_output(outfile_dir):
    """Extract dN, dS, and omega from CODEML output files."""
    logging.info(f"Parsing CODEML output in {outfile_dir}")
    
    results = {
        'dN': None,
        'dS': None,
        'omega': None,
        'N': None,
        'S': None,
        'lnL': None
    }
    
    rst_path = os.path.join(outfile_dir, 'rst')
    if os.path.exists(rst_path):
        logging.info("Parsing RST file")
        try:
            with open(rst_path, 'r') as f:
                content = f.read()
                
            pairwise_match = re.search(
                r'(\d+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)',
                content
            )
            
            if pairwise_match:
                results['N'] = float(pairwise_match.group(3))
                results['S'] = float(pairwise_match.group(4))
                results['dN'] = float(pairwise_match.group(5))
                results['dS'] = float(pairwise_match.group(6))
                results['omega'] = float(pairwise_match.group(7))
                results['lnL'] = float(pairwise_match.group(8))
                
                logging.info("RST file parsed successfully")
                logging.info(f"N sites: {results['N']}")
                logging.info(f"S sites: {results['S']}")
                logging.info(f"dN: {results['dN']}")
                logging.info(f"dS: {results['dS']}")
                logging.info(f"omega: {results['omega']}")
                logging.info(f"lnL: {results['lnL']}")
                
                return (results['dN'], results['dS'], results['omega'])
            logging.warning("Pairwise comparison section not found in RST file")
        
        except Exception as e:
            logging.error(f"Error parsing RST file: {e}")
    
    logging.info("RST file not found or parsing failed. Attempting to parse results.txt")
    results_path = os.path.join(outfile_dir, 'results.txt')
    
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                content = f.read()
                
            ml_match = re.search(
                r't=\s*[\d\.]+\s+S=\s*([\d\.]+)\s+N=\s*([\d\.]+)\s+dN/dS=\s*([\d\.]+)\s+dN\s*=\s*([\d\.]+)\s+dS\s*=\s*([\d\.]+)',
                content
            )
            
            if ml_match:
                results['S'] = float(ml_match.group(1))
                results['N'] = float(ml_match.group(2))
                results['omega'] = float(ml_match.group(3))
                results['dN'] = float(ml_match.group(4))
                results['dS'] = float(ml_match.group(5))
                
                logging.info("results.txt parsed successfully")
                logging.info(f"N sites: {results['N']}")
                logging.info(f"S sites: {results['S']}")
                logging.info(f"dN: {results['dN']}")
                logging.info(f"dS: {results['dS']}")
                logging.info(f"omega: {results['omega']}")
                
                return (results['dN'], results['dS'], results['omega'])
            logging.error("ML output section not found in results.txt")
                
        except Exception as e:
            logging.error(f"Error parsing results.txt: {e}")
    else:
        logging.error("results.txt not found")
    
    logging.info("Attempting to parse 2ML.dN and 2ML.dS files")
    try:
        dN_path = os.path.join(outfile_dir, '2ML.dN')
        dS_path = os.path.join(outfile_dir, '2ML.dS')
        
        with open(dN_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                results['dN'] = float(lines[2].strip().split()[-1])
            else:
                logging.warning("2ML.dN does not have enough lines")
        
        with open(dS_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 3:
                results['dS'] = float(lines[2].strip().split()[-1])
            else:
                logging.warning("2ML.dS does not have enough lines")
        
        if results['dS'] and results['dS'] != 0:
            results['omega'] = results['dN'] / results['dS']
        else:
            results['omega'] = np.nan
        
        logging.info("2ML.dN and 2ML.dS parsed successfully")
        logging.info(f"dN: {results['dN']}")
        logging.info(f"dS: {results['dS']}")
        logging.info(f"omega: {results['omega']}")
        
        return (results['dN'], results['dS'], results['omega'])
        
    except Exception as e:
        logging.error(f"Error parsing 2ML.* files: {e}")
    
    logging.error("Failed to parse CODEML output from all sources")
    return (None, None, None)

def get_safe_process_count():
    """Calculate an appropriate number of parallel processes based on system resources."""
    total_cpus = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    
    safe_cpu_count = max(1, min(total_cpus // 4, 8))
    mem_based_count = max(1, int(mem.available / (4 * 1024 ** 3)))
    
    process_count = min(safe_cpu_count, mem_based_count)
    logging.info(f"System has {total_cpus} CPUs and {mem.available / (1024 ** 3):.1f}GB free RAM")
    logging.info(f"Using {process_count} parallel processes")
    
    return process_count

# ----------------------------
# Core Processing Functions
# ----------------------------

def process_pair(args):
    """Handle the processing of a single pair of sequences."""
    pair, sequences, sample_groups, cds_id, codeml_path, temp_dir = args
    seq1_name, seq2_name = pair
    
    if seq1_name not in sequences or seq2_name not in sequences:
        missing = [name for name in [seq1_name, seq2_name] if name not in sequences]
        logging.error(f"Missing sequences for pair: {missing}")
        return None

    group1 = sample_groups.get(seq1_name)
    group2 = sample_groups.get(seq2_name)
    
    if group1 != group2:
        logging.error(f"Cross-group comparison detected: {seq1_name} (Group {group1}) vs {seq2_name} (Group {group2})")
        return None
        
    if sequences[seq1_name] == sequences[seq2_name]:
        logging.info(f"Identical sequences found: {seq1_name} and {seq2_name} - assigning omega = -1")
        return (
            seq1_name,
            seq2_name,
            group1,
            group1,
            0.0,
            0.0,
            -1.0,
            cds_id
        )
    
    timestamp = int(time.time())
    working_dir = os.path.join(temp_dir, f'temp_{seq1_name}_{seq2_name}_{timestamp}')
    os.makedirs(working_dir, exist_ok=True)

    phy_path = os.path.join(working_dir, 'temp.phy')
    with open(phy_path, 'w') as f:
        seq_len = len(sequences[seq1_name])
        f.write(f"2 {seq_len}\n")
        f.write(f"{seq1_name[:10].ljust(10)}  {sequences[seq1_name]}\n")
        f.write(f"{seq2_name[:10].ljust(10)}  {sequences[seq2_name]}\n")

    ctl_path = create_paml_ctl('temp.phy', 'results.txt', working_dir)
    
    tree_path = os.path.join(working_dir, 'tree.txt')
    with open(tree_path, 'w') as f:
        f.write('(Seq1,Seq2);')

    success = run_codeml(ctl_path, working_dir, codeml_path)

    if success:
        dn, ds, omega = parse_codeml_output(working_dir)
        if omega is None or not isinstance(omega, (int, float)):
            omega = np.nan
        return (
            seq1_name,
            seq2_name, 
            group1,
            group1,
            dn if dn is not None else np.nan,
            ds if ds is not None else np.nan,
            omega,
            cds_id
        )
    return (
        seq1_name,
        seq2_name,
        group1,
        group1,
        np.nan,
        np.nan,
        np.nan,
        cds_id
    )

def process_phy_file(args):
    """Manage the processing of a single PHYLIP file."""
    phy_file, output_dir, codeml_path, total_files, file_index = args

    logging.info(f"Processing file {file_index}/{total_files}: {phy_file}")
    start_time = time.time()

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
        group = None
    logging.info(f"Identified CDS: {cds_id}")

    output_csv = os.path.join(output_dir, f'{cds_id}.csv')
    haplotype_output_csv = os.path.join(output_dir, f'{cds_id}_haplotype_stats.csv')

    if os.path.exists(output_csv) and os.path.exists(haplotype_output_csv):
        logging.info(f"Skipping {phy_file} - output files already exist")
        return haplotype_output_csv

    sequences = parse_phy_file(phy_file)
    if not sequences:
        logging.error(f"No valid sequences found in {phy_file}")
        return None

    sample_names = list(sequences.keys())
    logging.info(f"Found {len(sample_names)} samples")

    sample_groups = {}
    for sample in sample_names:
        sample_group = extract_group_from_sample(sample)
        sample_groups[sample] = sample_group if sample_group is not None else group
    logging.info(f"Assigned {len(sample_groups)} samples to groups")

    group0_names = [name for name, grp in sample_groups.items() if grp == 0]
    group1_names = [name for name, grp in sample_groups.items() if grp == 1]
    
    logging.info(f"Found {len(group0_names)} sequences in group 0")
    logging.info(f"Found {len(group1_names)} sequences in group 1")

    group0_pairs = list(combinations(group0_names, 2))
    group1_pairs = list(combinations(group1_names, 2))
    pairs = group0_pairs + group1_pairs
    
    total_pairs = len(pairs)
    logging.info(f"Generated {len(group0_pairs)} pairs within group 0")
    logging.info(f"Generated {len(group1_pairs)} pairs within group 1")
    logging.info(f"Total pairs to process: {total_pairs}")

    if total_pairs == 0:
        logging.error(f"No valid pairs to process in {phy_file}")
        return None

    timestamp = int(time.time())
    temp_dir = os.path.join(output_dir, f'temp_{cds_id}_{timestamp}')
    os.makedirs(temp_dir, exist_ok=True)

    pool_args = [(pair, sequences, sample_groups, cds_id, codeml_path, temp_dir) for pair in pairs]
    num_processes = get_safe_process_count()
    logging.info(f"Processing {total_pairs} pairs using {num_processes} processes")
    
    results = []
    completed_pairs = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(process_pair, pool_args):
            if result is not None:
                if result[2] == result[3]:
                    results.append(result)
                else:
                    logging.warning(f"Mismatched groups for pair: {result[0]} vs {result[1]}")
            completed_pairs += 1
            progress = (completed_pairs / total_pairs) * 100
            logging.info(f"Progress: {completed_pairs}/{total_pairs} pairs processed ({progress:.2f}%)")

    logging.info(f"All pairs processed for {phy_file}")
    
    try:
        shutil.rmtree(temp_dir)
        logging.info(f"Temporary directory removed: {temp_dir}")
    except Exception as e:
        logging.warning(f"Could not remove temporary directory {temp_dir}: {e}")

    df = pd.DataFrame(results, columns=['Seq1', 'Seq2', 'Group1', 'Group2', 'dN', 'dS', 'omega', 'CDS'])
    
    cross_group = df[df['Group1'] != df['Group2']]
    if not cross_group.empty:
        logging.error("Cross-group comparisons detected in results")
        logging.error(cross_group)
        return None

    df.to_csv(output_csv, index=False)
    logging.info(f"Pairwise results saved to {output_csv}")

    haplotype_stats = []
    for idx, sample in enumerate(sample_names, 1):
        sample_df = df[(df['Seq1'] == sample) | (df['Seq2'] == sample)]
        omega_values = pd.to_numeric(sample_df['omega'], errors='coerce')
        # Filter out special values before averaging
        regular_omegas = omega_values[~omega_values.isin([-1, 99])]

        if not regular_omegas.empty:
            mean_omega = regular_omegas.mean()
            median_omega = regular_omegas.median()
        else:
            mean_omega = np.nan
            median_omega = np.nan
        
        haplotype_stats.append({
            'Haplotype': sample,
            'Group': sample_groups[sample],
            'CDS': cds_id,
            'Mean_dNdS': mean_omega,
            'Median_dNdS': median_omega,
            'Num_Comparisons': len(omega_values)
        })
        
        mean_str = f"{mean_omega:.4f}" if pd.notna(mean_omega) else "N/A"
        median_str = f"{median_omega:.4f}" if pd.notna(median_omega) else "N/A"
        logging.info(f"Sample {idx}/{len(sample_names)}: {sample} (Group {sample_groups[sample]}): mean dN/dS = {mean_str}, median = {median_str}, comparisons = {len(omega_values)}")

    haplotype_df = pd.DataFrame(haplotype_stats)
    haplotype_df.to_csv(haplotype_output_csv, index=False)
    logging.info(f"Haplotype statistics saved to {haplotype_output_csv}")

    group0 = haplotype_df[haplotype_df['Group'] == 0]['Mean_dNdS'].dropna()
    group1 = haplotype_df[haplotype_df['Group'] == 1]['Mean_dNdS'].dropna()

    logging.info(f"Within-Group Statistics for CDS {cds_id}:")
    if not group0.empty:
        logging.info(f"Group 0: n={len(group0)}, Mean={group0.mean():.4f}, Median={group0.median():.4f}, SD={group0.std():.4f}")
    else:
        logging.info("Group 0: No valid Mean_dNdS values.")
    if not group1.empty:
        logging.info(f"Group 1: n={len(group1)}, Mean={group1.mean():.4f}, Median={group1.median():.4f}, SD={group1.std():.4f}")
    else:
        logging.info("Group 1: No valid Mean_dNdS values.")

    total_time = time.time() - start_time
    logging.info(f"Completed processing {phy_file} in {total_time:.1f} seconds")
    
    return haplotype_output_csv

def perform_statistical_tests(haplotype_stats_files, output_dir):
    """Conduct statistical analyses on the aggregated haplotype statistics."""
    haplotype_dfs = []
    total_files = len(haplotype_stats_files)
    logging.info(f"Starting statistical tests on {total_files} haplotype stats files")
    for idx, f in enumerate(haplotype_stats_files, 1):
        try:
            df = pd.read_csv(f)
            haplotype_dfs.append(df)
            logging.info(f"Loaded haplotype stats from {f} ({idx}/{total_files})")
        except Exception as e:
            logging.error(f"Failed to read {f}: {e}")

    if not haplotype_dfs:
        logging.warning("No haplotype statistics files available for analysis.")
        return

    combined_df = pd.concat(haplotype_dfs, ignore_index=True)
    logging.info(f"Combined haplotype data has {len(combined_df)} entries")

    logging.info(f"Number of -1 values: {sum(combined_df['Mean_dNdS'] == -1)}, Number of 99 values: {sum(combined_df['Mean_dNdS'] == 99)}")

    if 'Mean_dNdS' not in combined_df.columns:
        logging.error("Combined DataFrame lacks 'Mean_dNdS' column.")
        return

    logging.info("Mean_dNdS Summary:")
    logging.info(combined_df['Mean_dNdS'].describe())

    if 'Group' not in combined_df.columns:
        logging.error("Combined DataFrame lacks 'Group' column.")
        return

    combined_df['Group'] = pd.to_numeric(combined_df['Group'], errors='coerce')
    unique_groups = combined_df['Group'].dropna().unique()
    logging.info(f"Groups present in data: {unique_groups}")

    if not set(unique_groups).intersection({0,1}):
        logging.error("No valid groups (0 or 1) found in 'Group' column.")
        return

    # Create filtered datasets - FIXED by using boolean indexing properly
    datasets = {
        "All data": combined_df,
        "Excluding dN/dS = -1": combined_df[~(combined_df['Mean_dNdS'] == -1)].copy(),
        "Excluding dN/dS = 99": combined_df[~(combined_df['Mean_dNdS'] == 99)].copy(),
        "Excluding both -1 and 99": combined_df[~(combined_df['Mean_dNdS'].isin([-1, 99]))].copy()
    }

    for dataset_name, dataset_df in datasets.items():
        logging.info(f"Analyzing dataset: {dataset_name}")
        logging.info(f"Entries: {len(dataset_df)}")
        
        stats = {}
        for group in [0, 1]:
            group_data = dataset_df[dataset_df['Group'] == group]['Mean_dNdS'].dropna()
            if not group_data.empty:
                stats[group] = {
                    'n': len(group_data),
                    'mean': group_data.mean(),
                    'median': group_data.median(),
                    'std': group_data.std()
                }
                logging.info(f"Group {group}: n={stats[group]['n']}, Mean={stats[group]['mean']:.4f}, "
                           f"Median={stats[group]['median']:.4f}, SD={stats[group]['std']:.4f}")
            else:
                logging.info(f"Group {group}: No valid Mean_dNdS values")

        if 0 in stats and 1 in stats:
            # Get the data for both groups outside the try block
            group0_data = dataset_df[dataset_df['Group'] == 0]['Mean_dNdS'].dropna()
            group1_data = dataset_df[dataset_df['Group'] == 1]['Mean_dNdS'].dropna()
            
            try:
                # Mann-Whitney U test
                stat, p_value = mannwhitneyu(group0_data, group1_data, alternative='two-sided')
                logging.info(f"Mann-Whitney U test: Statistic={stat}, p-value={p_value:.6f}")
                
                # Effect size calculation
                effect_size = abs(stats[0]['mean'] - stats[1]['mean']) / \
                            np.sqrt((stats[0]['std']**2 + stats[1]['std']**2) / 2)
                logging.info(f"Effect size (Cohen's d): {effect_size:.4f}")
                
                if p_value < 0.05:
                    logging.info("Significant difference between Group 0 and Group 1")
                else:
                    logging.info("No significant difference between Group 0 and Group 1")
                
                # Range information
                min_g0 = group0_data.min()
                max_g0 = group0_data.max()
                min_g1 = group1_data.min()
                max_g1 = group1_data.max()
                logging.info(f"Group 0 range: {min_g0:.4f} to {max_g0:.4f}")
                logging.info(f"Group 1 range: {min_g1:.4f} to {max_g1:.4f}")
                
                # Quartile information
                g0_quartiles = group0_data.quantile([0.25, 0.75])
                g1_quartiles = group1_data.quantile([0.25, 0.75])
                logging.info(f"Group 0 quartiles: Q1={g0_quartiles[0.25]:.4f}, Q3={g0_quartiles[0.75]:.4f}")
                logging.info(f"Group 1 quartiles: Q1={g1_quartiles[0.25]:.4f}, Q3={g1_quartiles[0.75]:.4f}")
                
                # Only do Levene's test for the specific dataset
                if dataset_name == "Excluding both -1 and 99":
                    if len(group0_data) >= 2 and len(group1_data) >= 2:
                        levene_stat, levene_p = levene(group0_data, group1_data)
                        logging.info(f"Levene's test for variance homogeneity: "
                                   f"Statistic={levene_stat:.4f}, p-value={levene_p:.6f}")
                    else:
                        logging.warning("Insufficient data for Levene's test")
                
            except Exception as e:
                logging.error(f"Error during statistical tests: {e}")

    return combined_df


def parse_phy_file(filepath):
    """Extract and validate sequences from a PHYLIP file, keeping sample names unchanged."""
    logging.info(f"Parsing PHYLIP file: {filepath}")
    sequences = {}
    
    with open(filepath, 'r') as file:
        lines = file.readlines()
        if not lines:
            logging.error(f"PHYLIP file is empty: {filepath}")
            return sequences

        try:
            num_sequences, seq_length = map(int, lines[0].strip().split())
            sequence_lines = [line.strip() for line in lines[1:] if line.strip()]
            logging.info(f"File has {num_sequences} sequences each of length {seq_length}")
        except ValueError:
            logging.warning(f"No valid header found in {filepath}. Processing without header.")
            sequence_lines = [line.strip() for line in lines if line.strip()]
    
        for line in sequence_lines:
            match = re.match(r'^(.+?_[01])\s*(.*)$', line)
            if match:
                full_name = match.group(1)
                sequence = match.group(2)
            else:
                parts = line.split()
                if len(parts) >= 2:
                    full_name = parts[0]
                    sequence = ''.join(parts[1:])
                else:
                    full_name = line[:10].strip()
                    sequence = line[10:].replace(" ", "")
            
            validated_seq = validate_sequence(sequence)
            if validated_seq:
                sequences[full_name] = validated_seq
                logging.info(f"Added sequence: {full_name} (Length: {len(validated_seq)})")

    logging.info(f"Total valid sequences parsed: {len(sequences)}")
    return sequences


# ----------------------------
# Statistical Analysis Functions
# ----------------------------

def analyze_cds_dataset(df, cds_id, dataset_name):
   """Analyze a single CDS dataset with proper statistical treatment."""
   results = {
       'cds_id': cds_id,
       'dataset': dataset_name,
       'group0_stats': None,
       'group1_stats': None,
       'comparison_stats': None
   }

   # Analyze each group separately
   for group in [0, 1]:
       group_data = df[df['Group'] == group]['Mean_dNdS'].dropna()
       if len(group_data) >= 3:  # Minimum sample size requirement
           stats = {
               'n': len(group_data),
               'mean': group_data.mean(),
               'median': group_data.median(),
               'std': group_data.std(),
               'q1': group_data.quantile(0.25),
               'q3': group_data.quantile(0.75),
               'min': group_data.min(),
               'max': group_data.max()
           }
           if group == 0:
               results['group0_stats'] = stats
           else:
               results['group1_stats'] = stats

   # Only perform comparison if both groups have sufficient data
   if results['group0_stats'] and results['group1_stats']:
       group0_data = df[df['Group'] == 0]['Mean_dNdS'].dropna()
       group1_data = df[df['Group'] == 1]['Mean_dNdS'].dropna()
       
       try:
           stat, p_value = mannwhitneyu(group0_data, group1_data, alternative='two-sided')
           effect_size = abs(results['group0_stats']['mean'] - results['group1_stats']['mean']) / \
                        np.sqrt((results['group0_stats']['std']**2 + results['group1_stats']['std']**2) / 2)
           
           results['comparison_stats'] = {
               'mw_statistic': stat,
               'p_value': p_value,
               'effect_size': effect_size
           }
       except Exception as e:
           logging.error(f"Statistical comparison failed for CDS {cds_id}: {e}")
   
   return results

def perform_statistical_analysis(haplotype_stats_files, output_dir):
   """Perform statistical analysis properly organized by CDS."""
   # First, organize data by CDS
   cds_data = {}
   for filepath in haplotype_stats_files:
       try:
           df = pd.read_csv(filepath)
           cds_id = df['CDS'].iloc[0]  # Each file should contain data for one CDS
           cds_data[cds_id] = df
       except Exception as e:
           logging.error(f"Error reading {filepath}: {e}")
           continue
   
   logging.info(f"Found {len(cds_data)} unique CDS regions to analyze")
   
   # Analysis results for each CDS
   all_results = []
   
   for cds_id, df in cds_data.items():
       logging.info(f"\nAnalyzing CDS: {cds_id}")
       
       # Create different datasets for exclusion criteria
       datasets = {
           "All data": df,
           "Excluding dN/dS = -1": df[df['Mean_dNdS'] != -1].copy(),
           "Excluding dN/dS = 99": df[df['Mean_dNdS'] != 99].copy(),
           "Excluding both -1 and 99": df[(df['Mean_dNdS'] != -1) & 
                                        (df['Mean_dNdS'] != 99)].copy()
       }
       
       # Analyze each dataset
       for dataset_name, dataset_df in datasets.items():
           results = analyze_cds_dataset(dataset_df, cds_id, dataset_name)
           if results['comparison_stats']:  # Only add if comparison was possible
               all_results.append(results)
               
               # Log detailed results for this CDS and dataset
               logging.info(f"\nResults for {cds_id} - {dataset_name}:")
               if results['group0_stats']:
                   g0 = results['group0_stats']
                   logging.info(f"Group 0: n={g0['n']}, Mean={g0['mean']:.4f}, "
                              f"Median={g0['median']:.4f}, SD={g0['std']:.4f}")
                   logging.info(f"Group 0 range: {g0['min']:.4f} to {g0['max']:.4f}")
                   logging.info(f"Group 0 quartiles: Q1={g0['q1']:.4f}, Q3={g0['q3']:.4f}")
               
               if results['group1_stats']:
                   g1 = results['group1_stats']
                   logging.info(f"Group 1: n={g1['n']}, Mean={g1['mean']:.4f}, "
                              f"Median={g1['median']:.4f}, SD={g1['std']:.4f}")
                   logging.info(f"Group 1 range: {g1['min']:.4f} to {g1['max']:.4f}")
                   logging.info(f"Group 1 quartiles: Q1={g1['q1']:.4f}, Q3={g1['q3']:.4f}")
               
               if results['comparison_stats']:
                   comp = results['comparison_stats']
                   logging.info(f"Mann-Whitney U test: Statistic={comp['mw_statistic']}, "
                              f"p-value={comp['p_value']:.6f}")
                   logging.info(f"Effect size (Cohen's d): {comp['effect_size']:.4f}")
               
               if dataset_name == "Excluding both -1 and 99":
                   stat, p_value = levene(group0_data, group1_data)
                   logging.info(f"Levene's test for variance: Statistic={stat:.4f}, p-value={p_value:.6f}")
   
       
       # Save comprehensive results
       results_df = pd.DataFrame([
           {
               'CDS': r['cds_id'],
               'Dataset': r['dataset'],
               'Group0_n': r['group0_stats']['n'],
               'Group0_mean': r['group0_stats']['mean'],
               'Group0_median': r['group0_stats']['median'],
               'Group0_sd': r['group0_stats']['std'],
               'Group1_n': r['group1_stats']['n'],
               'Group1_mean': r['group1_stats']['mean'],
               'Group1_median': r['group1_stats']['median'],
               'Group1_sd': r['group1_stats']['std'],
               'MWU_statistic': r['comparison_stats']['mw_statistic'],
               'p_value': r['comparison_stats']['p_value'],
               'significant': r['comparison_stats']['p_value'] < 0.05,
               'effect_size': r['comparison_stats']['effect_size']
           }
           for r in all_results
       ])
       
       # Summary statistics
       logging.info(f"\nAnalysis Summary:")
       logging.info(f"Total CDS analyzed: {len(cds_data)}")
       
       return results_df
   return None

def check_existing_results(output_dir):
   """Check existing results and determine if analysis should proceed."""
   comprehensive_results = os.path.join(output_dir, 'comprehensive_cds_analysis.csv')
   
   if os.path.exists(comprehensive_results):
       try:
           results_df = pd.read_csv(comprehensive_results)
           logging.info(f"Found existing comprehensive analysis with {len(results_df)} entries")
           return results_df
       except Exception as e:
           logging.error(f"Error reading existing results: {e}")
   
   return None

# ----------------------------
# Main Function
# ----------------------------

def main():
   """Main function to orchestrate the dN/dS analysis."""
   parser = argparse.ArgumentParser(description="Calculate pairwise dN/dS using PAML.")
   parser.add_argument('--phy_dir', type=str, default='.', help='Directory containing .phy files.')
   parser.add_argument('--output_dir', type=str, default='paml_output', help='Directory to store output files.')
   parser.add_argument('--codeml_path', type=str, default='../../../../paml/bin/codeml', help='Path to codeml executable.')
   args = parser.parse_args()

   os.makedirs(args.output_dir, exist_ok=True)

   logging.info("Starting preliminary analysis of existing results")
   existing_haplotype_files = glob.glob(os.path.join(args.output_dir, '*_haplotype_stats.csv'))
   if existing_haplotype_files:
       logging.info(f"Found {len(existing_haplotype_files)} existing haplotype files")
       perform_statistical_tests(existing_haplotype_files, args.output_dir)
       perform_statistical_analysis(existing_haplotype_files, args.output_dir)
       logging.info("Preliminary analysis completed. Continuing with remaining files")
   else:
       logging.info("No existing results found. Proceeding with full analysis")

   phy_files = glob.glob(os.path.join(args.phy_dir, '*.phy'))
   total_files = len(phy_files)
   logging.info(f"Found {total_files} PHYLIP files to process")

   # Check which files need processing
   files_to_process = []
   for phy_file in phy_files:
       phy_filename = os.path.basename(phy_file)
       match = re.match(r'group_(\d+)_chr_(.+)_start_(\d+)_end_(\d+)\.phy', phy_filename)
       if match:
           cds_id = f'chr{match.group(2)}_start{match.group(3)}_end{match.group(4)}'
       else:
           cds_id = phy_filename.replace('.phy', '')
       output_csv = os.path.join(args.output_dir, f'{cds_id}.csv')
       haplotype_output_csv = os.path.join(args.output_dir, f'{cds_id}_haplotype_stats.csv')
       if not os.path.exists(output_csv) or not os.path.exists(haplotype_output_csv):
           files_to_process.append(phy_file)
       else:
           logging.info(f"Skipping {phy_file} - output files already exist")

   if not files_to_process:
       logging.info("All PHYLIP files have been processed. Analysis complete.")
       return

   # Process new files
   logging.info(f"Processing {len(files_to_process)} new files")
   work_args = []
   for idx, phy_file in enumerate(files_to_process, 1):
       work_args.append((phy_file, args.output_dir, args.codeml_path, total_files, idx))

   new_haplotype_files = []
   total_new_files = len(work_args)
   for idx, args_tuple in enumerate(work_args, 1):
       logging.info(f"Processing file {idx}/{total_new_files}: {args_tuple[0]}")
       result = process_phy_file(args_tuple)
       if result:
           new_haplotype_files.append(result)
       progress = (idx / total_new_files) * 100
       logging.info(f"Overall Progress: {idx}/{total_new_files} files processed ({progress:.2f}%)")

   if new_haplotype_files:
       # Run both analyses on updated full dataset
       all_haplotype_files = glob.glob(os.path.join(args.output_dir, '*_haplotype_stats.csv'))
       perform_statistical_tests(all_haplotype_files, args.output_dir)
       perform_statistical_analysis(all_haplotype_files, args.output_dir)
       logging.info("Final analysis completed")
   else:
       logging.warning("No new haplotype statistics files were generated")

if __name__ == '__main__':
   main()
