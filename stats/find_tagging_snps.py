#!/usr/bin/env python3
"""
Find Tagging SNPs for Inversions and Generate a TSV Report

This script processes a directory of PHYLIP files (`.phy.gz`) to identify
tagging SNPs for each inversion and compiles the results into a single TSV file.

The script performs the following steps:
1.  Scans the input directory for all `.phy.gz` files.
2.  For each file:
    a. Parses the inversion name, chromosome, and start/end coordinates from the filename.
    b. Reads the gzipped PHYLIP file to get sequences for the 'direct' (group 0) and
       'inverted' (group 1) orientations.
    c. Identifies all variable sites (SNPs) across the alignment.
    d. For each SNP, it calculates:
        - The Pearson correlation (phi coefficient) between allele presence and inversion status.
        - The frequency of the major allele in the 'direct' group.
        - The frequency of the major allele in the 'inverted' group.
        - The absolute difference in allele frequencies between the two groups.
3.  Aggregates the results from all files into a single data frame.
4.  Sorts the results by inversion name and the strength of the correlation.
5.  Writes the final, comprehensive report to the specified output TSV file.
"""

import os
import re
import sys
import gzip
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from glob import glob

def parse_phylip(file_path):
    """
    Parses a PHYLIP file, handling the two-group format.

    Args:
        file_path (str): The path to the PHYLIP file.

    Returns:
        A tuple containing:
        - sequences (np.array): A 2D numpy array of the sequences.
        - n_group0 (int): The number of sequences in the first group.
    """
    sequences = []
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()

    header = lines[0].strip().split()
    n_total, seq_len = int(header[0]), int(header[1])

    # Assumes a two-group format where the first half of sequences
    # belong to one orientation and the second half to the other.
    n_group0 = n_total // 2

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) == 2:
            sequences.append(list(parts[1]))

    return np.array(sequences), n_group0

def find_tagging_snps(phy_dir, output_file):
    """
    Identifies tagging SNPs and writes a comprehensive report to a TSV file.

    Args:
        phy_dir (str): The directory containing the `.phy.gz` files.
        output_file (str): The path to the output TSV file.
    """
    phy_files = glob(os.path.join(phy_dir, '*.phy.gz'))

    if not phy_files:
        print(f"No '.phy.gz' files found in '{phy_dir}'")
        return

    all_results = []
    has_errors = False

    # Regex to extract chr, start, and end from filenames like:
    # 'inversion_group_inversion_11_chr1-12345-67890.phy.gz'
    filename_regex = re.compile(r'(chr[\w]+)-(\d+)-(\d+)')

    for phy_file in phy_files:
        inversion_name = os.path.basename(phy_file).replace('.phy.gz', '')

        match = filename_regex.search(inversion_name)
        if match:
            chromosome, region_start, region_end = match.groups()
        else:
            chromosome, region_start, region_end = 'unknown', '0', '0'

        try:
            sequences, n_group0 = parse_phylip(phy_file)
        except Exception as e:
            print(f"Error parsing {phy_file}: {e}")
            has_errors = True
            continue

        n_total, n_sites = sequences.shape
        n_group1 = n_total - n_group0

        if n_total < 2 or n_sites == 0 or n_group0 == 0 or n_group1 == 0:
            continue

        inversion_status = np.array([0] * n_group0 + [1] * n_group1)

        variable_sites = [i for i in range(n_sites) if len(np.unique(sequences[:, i])) > 1]

        if not variable_sites:
            continue

        for site in variable_sites:
            alleles = sequences[:, site]
            major_allele = np.unique(alleles)[0]
            encoded_alleles = (alleles == major_allele).astype(int)

            # Calculate correlation
            r, _ = stats.pearsonr(inversion_status, encoded_alleles)
            if np.isnan(r):
                continue

            # Calculate allele frequencies
            freq_direct = np.mean(encoded_alleles[:n_group0])
            freq_inverted = np.mean(encoded_alleles[n_group0:])
            freq_diff = abs(freq_direct - freq_inverted)

            all_results.append({
                'inversion_name': inversion_name,
                'chromosome': chromosome,
                'region_start': int(region_start),
                'region_end': int(region_end),
                'snp_site_index': site,
                'snp_position': int(region_start) + site,
                'correlation': r,
                'allele_freq_direct': freq_direct,
                'allele_freq_inverted': freq_inverted,
                'allele_freq_difference': freq_diff,
            })

    if not all_results:
        print("No variable SNPs found across all files.")
    else:
        df = pd.DataFrame(all_results)
        df['abs_correlation'] = df['correlation'].abs()
        df = df.sort_values(['inversion_name', 'abs_correlation'], ascending=[True, False])
        df = df.drop(columns=['abs_correlation'])

        df.to_csv(output_file, sep='\t', index=False, float_format='%.4f')
        print(f"Successfully wrote tagging SNP report to {output_file}")

    if has_errors:
        print("\nErrors occurred during processing. Exiting with non-zero status.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Find tagging SNPs and generate a TSV report.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('phy_dir', help="Directory containing the .phy.gz files.")
    parser.add_argument(
        '--output',
        default='tagging_snps.tsv',
        help="Path to the output TSV file (default: tagging_snps.tsv)."
    )
    args = parser.parse_args()

    find_tagging_snps(args.phy_dir, args.output)

if __name__ == '__main__':
    main()
