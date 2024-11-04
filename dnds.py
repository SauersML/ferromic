import os
import sys
import glob
import subprocess
import multiprocessing
import signal
import psutil
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
import logging

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s [%(levelname)s] %(message)s',
   handlers=[
       logging.FileHandler('dnds_analysis.log'),
       logging.StreamHandler(sys.stdout)
   ]
)

def parse_phy_file(filepath):
   print(f"\n=== Starting to parse file: {filepath} ===")
   sequences = {}
   with open(filepath, 'r') as file:
       lines = file.readlines()
       if len(lines) < 1:
           print(f"ERROR: Empty .phy file {filepath}")
           return sequences
       try:
           num_sequences, seq_length = map(int, lines[0].strip().split())
           print(f"File contains {num_sequences} sequences of length {seq_length}")
       except ValueError:
           print(f"ERROR: Failed parsing header of {filepath}")
           return sequences
       
       for i, line in enumerate(lines[1:], 1):
           if line.strip() == '':
               continue
           try:
               parts = line.strip().split()
               if len(parts) < 2:
                   name = line[:10].strip()
                   sequence = line[10:].strip()
               else:
                   name = parts[0].strip()
                   sequence = parts[1].strip()
               sequences[name] = sequence
               print(f"Parsed sequence {i}: {name} (length: {len(sequence)})")
           except Exception as e:
               print(f"ERROR parsing line {i}: {str(e)}")
   
   print(f"Successfully parsed {len(sequences)} sequences")
   return sequences

def extract_group_from_sample(sample_name):
   match = re.search(r'_(\d+)$', sample_name)
   if match:
       return int(match.group(1))
   else:
       print(f"WARNING: Could not extract group from sample name: {sample_name}")
       return None

def create_paml_ctl(seqfile, outfile, working_dir):
   print(f"Creating PAML control file in {working_dir}")
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
   print(f"PAML control file created: {ctl_path}")
   return ctl_path

def run_codeml(ctl_path, working_dir, codeml_path):
   print(f"\n=== Running CODEML in {working_dir} ===")
   start_time = time.time()
   
   cpu_percent = psutil.cpu_percent()
   mem = psutil.virtual_memory()
   print(f"System status before CODEML: CPU {cpu_percent}%, Memory {mem.percent}%")
   
   try:
       process = subprocess.Popen(
           [codeml_path],
           cwd=working_dir,
           stdout=subprocess.PIPE,
           stderr=subprocess.PIPE,
           preexec_fn=os.setsid
       )

       try:
           stdout, stderr = process.communicate(timeout=60)
           print(f"CODEML runtime: {time.time() - start_time:.2f}s")
           
           if process.returncode != 0:
               print(f"CODEML error: {stderr.decode()}")
               return False
               
           print("CODEML completed successfully")
           return True
           
       except subprocess.TimeoutExpired:
           print("CODEML process timed out - terminating")
           os.killpg(os.getpgid(process.pid), signal.SIGTERM)
           return False
           
   except Exception as e:
       print(f"ERROR running CODEML: {str(e)}")
       return False

def parse_codeml_output(outfile):
   print(f"Parsing CODEML output: {outfile}")
   dN = None
   dS = None 
   omega = None
   try:
       with open(outfile, 'r') as f:
           content = f.read()
           print(f"CODEML output file size: {len(content)} bytes")
           for line in content.split('\n'):
               if "dN =" in line and "dS =" in line and "omega =" in line:
                   parts = line.strip().split()
                   try:
                       dN = float(parts[1])
                       dS = float(parts[4])
                       omega = float(parts[7])
                       print(f"Parsed values: dN={dN}, dS={dS}, omega={omega}")
                   except (IndexError, ValueError) as e:
                       print(f"ERROR parsing CODEML values: {str(e)}")
                   break
   except FileNotFoundError:
       print(f"ERROR: Results file {outfile} not found")
   
   return dN, dS, omega

def get_safe_process_count():
   total_cpus = multiprocessing.cpu_count()
   mem = psutil.virtual_memory()
   
   # Conservative allocation - use at most 25% of CPUs and ensure 4GB per process
   safe_cpu_count = max(1, min(total_cpus // 4, 8))
   mem_based_count = max(1, int(mem.available / (4 * 1024 * 1024 * 1024)))
   
   process_count = min(safe_cpu_count, mem_based_count)
   print(f"\nSystem resources: {total_cpus} CPUs, {mem.available/1024/1024/1024:.1f}GB free RAM")
   print(f"Using {process_count} parallel processes")
   
   return process_count

def process_pair(args):
   pair, sequences, sample_groups, cds_id, codeml_path, temp_dir = args
   seq1_name, seq2_name = pair
   
   print(f"\n=== Processing pair: {seq1_name} vs {seq2_name} ===")
   
   working_dir = os.path.join(temp_dir, f'temp_{seq1_name}_{seq2_name}_{int(time.time())}')
   os.makedirs(working_dir, exist_ok=True)
   print(f"Created working directory: {working_dir}")

   temp_phy = os.path.join(working_dir, 'temp.phy')
   with open(temp_phy, 'w') as phy_file:
       phy_file.write(f" 2 {len(sequences[seq1_name])}\n")
       phy_file.write(f"{seq1_name:<10}{sequences[seq1_name]}\n")
       phy_file.write(f"{seq2_name:<10}{sequences[seq2_name]}\n")
   print(f"Written sequences to: {temp_phy}")

   ctl_path = create_paml_ctl('temp.phy', 'results.txt', working_dir)
   success = run_codeml(ctl_path, working_dir, codeml_path)

   results_file = os.path.join(working_dir, 'results.txt')
   if success:
       dN, dS, omega = parse_codeml_output(results_file)
   else:
       dN, dS, omega = None, None, None

   try:
       shutil.rmtree(working_dir)
       print(f"Cleaned up directory: {working_dir}")
   except Exception as e:
       print(f"WARNING: Failed to clean up {working_dir}: {str(e)}")

   return (seq1_name, seq2_name, sample_groups[seq1_name], sample_groups[seq2_name], dN, dS, omega)

def process_phy_file(args):
   phy_file, output_dir, codeml_path, total_files, file_index = args

   print(f"\n====== Processing file {file_index}/{total_files}: {phy_file} ======")
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
   print(f"Identified CDS: {cds_id}")

   output_csv = os.path.join(output_dir, f'{cds_id}.csv')
   haplotype_output_csv = os.path.join(output_dir, f'{cds_id}_haplotype_stats.csv')

   if os.path.exists(output_csv) and os.path.exists(haplotype_output_csv):
       print(f"Skipping {phy_file} - output files already exist")
       return haplotype_output_csv

   sequences = parse_phy_file(phy_file)
   if not sequences:
       print(f"ERROR: No sequences found in {phy_file}")
       return None
       
   sample_names = list(sequences.keys())
   print(f"Found {len(sample_names)} samples")

   sample_groups = {}
   for sample in sample_names:
       sample_group = extract_group_from_sample(sample)
       sample_groups[sample] = sample_group if sample_group is not None else group
   print(f"Assigned {len(sample_groups)} samples to groups")

   pairs = list(combinations(sample_names, 2))
   total_pairs = len(pairs)
   print(f"Generated {total_pairs} pairwise combinations")

   if total_pairs == 0:
       print(f"ERROR: No pairs to process in {phy_file}")
       return None

   temp_dir = os.path.join(output_dir, f'temp_{cds_id}_{datetime.now().strftime("%Y%m%d%H%M%S%f")}')
   os.makedirs(temp_dir, exist_ok=True)
   print(f"Created temporary directory: {temp_dir}")

   pool_args = []
   for pair in pairs:
       pool_args.append((pair, sequences, sample_groups, cds_id, codeml_path, temp_dir))

   num_processes = get_safe_process_count()
   print(f"Processing {total_pairs} pairs using {num_processes} processes")
   
   with multiprocessing.Pool(processes=num_processes) as pool:
       results = []
       completed = 0
       for result in pool.imap_unordered(process_pair, pool_args):
           results.append(result)
           completed += 1
           if completed % max(1, total_pairs // 20) == 0:
               print(f"Progress: {completed}/{total_pairs} pairs ({(completed/total_pairs)*100:.1f}%)")
               print(f"Current runtime: {time.time() - start_time:.1f}s")

   print(f"All pairs processed for {phy_file}")
   
   try:
       shutil.rmtree(temp_dir)
       print(f"Cleaned up temporary directory: {temp_dir}")
   except Exception as e:
       print(f"WARNING: Failed to clean up {temp_dir}: {str(e)}")

   df = pd.DataFrame(results, columns=['Seq1', 'Seq2', 'Group1', 'Group2', 'dN', 'dS', 'omega'])
   df['CDS'] = cds_id
   df.to_csv(output_csv, index=False)
   print(f"Saved pairwise results to: {output_csv}")

   haplotype_stats = []
   for sample in sample_names:
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
       print(f"Sample {sample}: mean dN/dS = {mean_omega:.4f}, median = {median_omega:.4f}")

   haplotype_df = pd.DataFrame(haplotype_stats)
   haplotype_df.to_csv(haplotype_output_csv, index=False)
   print(f"Saved haplotype statistics to: {haplotype_output_csv}")

   group0 = haplotype_df[haplotype_df['Group'] == 0]['Mean_dNdS'].dropna()
   group1 = haplotype_df[haplotype_df['Group'] == 1]['Mean_dNdS'].dropna()
   
   print(f"\nStatistics for CDS {cds_id}:")
   if not group0.empty:
       print(f"Group 0: n={len(group0)}, Mean={group0.mean():.4f}, Median={group0.median():.4f}, SD={group0.std():.4f}")
   if not group1.empty:
       print(f"Group 1: n={len(group1)}, Mean={group1.mean():.4f}, Median={group1.median():.4f}, SD={group1.std():.4f}")

   total_time = time.time() - start_time
   print(f"Completed processing {phy_file} in {total_time:.1f} seconds")
   
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
