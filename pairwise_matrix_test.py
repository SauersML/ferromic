import pandas as pd
import numpy as np
import re
from collections import defaultdict
from tqdm.auto import tqdm
import warnings
import os
import json
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)
from datetime import datetime
from scipy import stats
import requests
from urllib.parse import urlencode
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM

# Suppress warnings
warnings.filterwarnings('ignore')

def read_and_preprocess_data(file_path):
    """Read and preprocess the CSV file."""
    print("Reading data...")
    df = pd.read_csv(file_path)
    
    # Store original CDS as full_cds
    df['full_cds'] = df['CDS']
    
    # Extract transcript ID, chromosome, start, and end positions from CDS
    df['group'] = df['CDS'].apply(lambda x: 1 if x.startswith('group_1') else 0)
    
    # Extract coordinates using regex
    coord_pattern = r'chr_(\w+)_start_(\d+)_end_(\d+)'
    coords = df['CDS'].str.extract(coord_pattern)
    df['chrom'] = 'chr' + coords[0]
    df['start'] = pd.to_numeric(coords[1])
    df['end'] = pd.to_numeric(coords[2])
    
    # Extract transcript ID
    transcript_pattern = r'(ENST\d+\.\d+)'
    df['transcript_id'] = df['CDS'].str.extract(transcript_pattern)[0]
    
    # Convert omega to numeric, coerce non-numeric to NaN
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')
    
    # Filter valid omega values
    df = df[
        (df['omega'] != -1) &
        (df['omega'] != 99)
    ].dropna(subset=['omega'])

    print(f"Total valid comparisons: {len(df)}")
    print(f"Unique coordinates found: {df.groupby(['chrom', 'start', 'end']).ngroups}")
    return df

def get_pairwise_value(seq1, seq2, pairwise_dict):
    """Get omega value for a pair of sequences."""
    key = (seq1, seq2) if (seq1, seq2) in pairwise_dict else (seq2, seq1)
    val = pairwise_dict.get(key)
    return val

def create_matrices(sequences_0, sequences_1, pairwise_dict):
    """Create matrices for two groups based on sequence assignments."""
    n0, n1 = len(sequences_0), len(sequences_1)
    
    if n0 == 0 and n1 == 0:
        return None, None
        
    matrix_0 = np.full((n0, n0), np.nan) if n0 > 0 else None
    matrix_1 = np.full((n1, n1), np.nan) if n1 > 0 else None

    # Fill matrix 0
    if n0 > 0:
        for i in range(n0):
            for j in range(i + 1, n0):
                val = get_pairwise_value(sequences_0[i], sequences_0[j], pairwise_dict)
                if val is not None:
                    matrix_0[i, j] = matrix_0[j, i] = float(val)

    # Fill matrix 1
    if n1 > 0:
        for i in range(n1):
            for j in range(i + 1, n1):
                val = get_pairwise_value(sequences_1[i], sequences_1[j], pairwise_dict)
                if val is not None:
                    matrix_1[i, j] = matrix_1[j, i] = float(val)

    return matrix_0, matrix_1

def get_gene_info(gene_symbol):
    """Get human-readable gene info from MyGene.info using gene symbol."""
    try:
        url = f"http://mygene.info/v3/query?q=symbol:{gene_symbol}&species=human&fields=name"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                return data['hits'][0].get('name', 'Unknown')
    except Exception as e:
        print(f"Error fetching gene info: {str(e)}")
    return 'Unknown'

def get_gene_annotation(coordinates):
    """
    Get gene annotation for a genomic location.
    Returns (gene_symbol, gene_name).
    """
    try:
        # Parse coordinates
        match = re.search(r'chr_(\w+)_start_(\d+)_end_(\d+)', coordinates)
        if not match:
            return None, None
            
        chrom, start, end = match.groups()
        chrom = 'chr' + chrom
        start, end = int(start), int(end)
        
        # Query UCSC API
        base_url = "https://api.genome.ucsc.edu/getData/track"
        params = {'genome': 'hg38', 'track': 'knownGene', 'chrom': chrom, 'start': start, 'end': end}
        
        response = requests.get(f"{base_url}?{urlencode(params)}", timeout=10)
        if not response.ok:
            return None, None
            
        data = response.json()
        
        # Handle different response structures
        track_data = data.get('knownGene', data)
        
        if isinstance(track_data, str) or not track_data:
            return None, None
            
        # Find overlapping genes
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
            return None, None
            
        # Get the best overlapping gene
        best_gene = max(
            overlapping_genes,
            key=lambda gene: max(0, min(gene.get('chromEnd', 0), end) - max(gene.get('chromStart', 0), start))
        )
        
        symbol = best_gene.get('geneName', 'Unknown')
        if symbol in ['none', None] or symbol.startswith('ENSG'):
            for gene in overlapping_genes:
                potential_symbol = gene.get('geneName')
                if potential_symbol and potential_symbol != 'none' and not potential_symbol.startswith('ENSG'):
                    symbol = potential_symbol
                    break
                    
        name = get_gene_info(symbol)
        return symbol, name
        
    except Exception as e:
        print(f"Error in gene annotation: {str(e)}")
        return None, None

def analysis_worker(args):
    """Mixed effects analysis for a single coordinate with crossed random effects."""
    all_sequences, pairwise_dict, sequences_0, sequences_1 = args
    
    n0, n1 = len(sequences_0), len(sequences_1)
    
    # Check if we only have sequences in one group
    if n0 == 0 or n1 == 0:
        group_with_data = '0' if n1 == 0 else '1'
        print(f"All sequences in group {group_with_data}, need both groups for comparison.")
        return {
            'effect_size': np.nan,
            'p_value': np.nan,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': sum(1 for (seq1, seq2) in pairwise_dict.keys() if seq1 in sequences_0 and seq2 in sequences_0),
            'num_comp_group_1': sum(1 for (seq1, seq2) in pairwise_dict.keys() if seq1 in sequences_1 and seq2 in sequences_1),
            'std_err': np.nan,
            'failure_reason': f"All sequences in group {group_with_data}"
        }
    
    # Prepare data for mixed-model analysis
    data = []
    for (seq1, seq2), omega in pairwise_dict.items():
        if seq1 in sequences_0 and seq2 in sequences_0:
            group = 0
        elif seq1 in sequences_1 and seq2 in sequences_1:
            group = 1
        else:
            continue
        data.append({
            'omega_value': omega,
            'group': group,
            'seq1': seq1,
            'seq2': seq2
        })

    df = pd.DataFrame(data)
    
    # Initialize values
    effect_size = np.nan
    p_value = np.nan
    std_err = np.nan
    failure_reason = None

    # Check for valid data
    if df.empty or df['group'].nunique() < 2 or df['omega_value'].nunique() < 2:
        if df.empty:
            failure_reason = "No valid pairwise comparisons found"
        elif df['group'].nunique() < 2:
            failure_reason = "Missing one of the groups in pairwise comparisons"
        elif df['omega_value'].nunique() < 2:
            print("RAW DATA for Not enough omega value variation for statistical analysis:")
            print(df)
            failure_reason = "Not enough omega value variation for statistical analysis"

        
        print(f"WARNING: {failure_reason}")
        
        return {
            'effect_size': effect_size,
            'p_value': p_value,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': (df['group'] == 0).sum() if not df.empty else 0,
            'num_comp_group_1': (df['group'] == 1).sum() if not df.empty else 0,
            'std_err': std_err,
            'failure_reason': failure_reason
        }

    # Categorize sequences for random effects modeling
    df['seq1_code'] = pd.Categorical(df['seq1']).codes
    df['seq2_code'] = pd.Categorical(df['seq2']).codes

    try:
        # Set up mixed model with sequence random effects
        df['groups'] = 1
        vc = {
            'seq1': '0 + C(seq1_code)',
            'seq2': '0 + C(seq2_code)'
        }
        
        # Fit mixed model
        model = MixedLM.from_formula(
            'omega_value ~ group',
            groups='groups',
            vc_formula=vc,
            re_formula='0',
            data=df
        )
        result = model.fit(reml=False)
        
        # Extract results
        effect_size = result.fe_params['group']
        p_value = result.pvalues['group']
        std_err = result.bse['group']
        
    except Exception as e:
        failure_reason = f"Statistical model error: {str(e)[:100]}..."
        print(f"Model fitting failed with error: {str(e)}")

    return {
        'effect_size': effect_size,
        'p_value': p_value,
        'n0': n0,
        'n1': n1,
        'std_err': std_err,
        'num_comp_group_0': (df['group'] == 0).sum(),
        'num_comp_group_1': (df['group'] == 1).sum(),
        'failure_reason': failure_reason
    }


def analyze_transcript(args):
    """Analyze a specific transcript."""
    df_transcript, transcript_id = args

    print(f"\nAnalyzing transcript: {transcript_id}")

    # Group sequences by their group (0 or 1)
    group_0_df = df_transcript[df_transcript['group'] == 0]
    group_1_df = df_transcript[df_transcript['group'] == 1]

    # Get unique sequences for each group
    sequences_0 = pd.concat([group_0_df['Seq1'], group_0_df['Seq2']]).unique()
    sequences_1 = pd.concat([group_1_df['Seq1'], group_1_df['Seq2']]).unique()

    # Create pairwise dictionary
    pairwise_dict = {}
    for _, row in df_transcript.iterrows():
        pairwise_dict[(row['Seq1'], row['Seq2'])] = row['omega']

    # All sequences for analysis
    all_sequences = (
        np.concatenate([sequences_0, sequences_1])
        if len(sequences_0) > 0 and len(sequences_1) > 0
        else (sequences_0 if len(sequences_0) > 0 else sequences_1)
    )

    # Collect coordinate references (for display only)
    unique_coords = set(
        f"{r['chrom']}:{r['start']}-{r['end']}" for _, r in df_transcript.iterrows()
    )
    coords_str = ";".join(sorted(unique_coords))

    # Create matrices for visualization or further analysis if needed
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)

    # For annotation, pick the first row
    first_row = df_transcript.iloc[0]
    coordinates_str = f"chr_{str(first_row['chrom']).replace('chr', '')}_start_{first_row['start']}_end_{first_row['end']}"
    gene_symbol, gene_name = get_gene_annotation(coordinates_str)
    # Perform statistical analysis
    analysis_result = analysis_worker((all_sequences, pairwise_dict, sequences_0, sequences_1))

    # Combine results
    result = {
        'transcript_id': transcript_id,
        'coordinates': coords_str,
        'gene_symbol': gene_symbol,
        'gene_name': gene_name,
        'n0': len(sequences_0),
        'n1': len(sequences_1),
        'num_comp_group_0': analysis_result['num_comp_group_0'],
        'num_comp_group_1': analysis_result['num_comp_group_1'],
        'effect_size': analysis_result['effect_size'],
        'p_value': analysis_result['p_value'],
        'std_err': analysis_result['std_err'],
        'failure_reason': analysis_result['failure_reason']
    }

    return result

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")

    # Read and preprocess data
    df = read_and_preprocess_data('all_pairwise_results.csv')
    
    # Group by transcript
    transcript_groups = df.groupby('transcript_id')
    print(f"\nFound {len(transcript_groups)} unique transcripts")
    
    # Prepare arguments for parallel processing
    transcript_args = [(transcript_group, transcript_id) for transcript_id, transcript_group in transcript_groups]
    
    # Process each transcript in parallel
    results = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for result in tqdm(executor.map(analyze_transcript, transcript_args), 
                          total=len(transcript_args), 
                          desc="Processing transcripts"):
            results.append(result)

    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    valid_results = results_df[results_df['p_value'].notna() & (results_df['p_value'] > 0)]
    num_valid_tests = len(valid_results)
    
    if num_valid_tests > 0:
        results_df['bonferroni_p_value'] = (results_df['p_value'] * num_valid_tests).clip(upper=1.0)
    else:
        results_df['bonferroni_p_value'] = results_df['p_value']
    
    # Add -log10(p) for easier interpretation
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/final_results.csv', index=False)
    
    # Print summary table
    print("\n=== Group Assignment Summary by Transcript ===")
    print(f"{'Transcript/Coordinates':<50} {'Group 0':<10} {'Group 1':<10} {'Total':<10} {'P-value':<15} {'Effect Size':<15} {'Gene'}")
    print("-" * 120)
    
    # Calculate totals
    total_group_0 = results_df['n0'].sum()
    total_group_1 = results_df['n1'].sum()
    
    # Sort by p-value for display
    sorted_results = results_df.sort_values('p_value')
    
    for _, row in sorted_results.iterrows():
        transcript_str = str(row['transcript_id']) if 'transcript_id' in row and pd.notna(row['transcript_id']) else ""
        coords_str = str(row['coordinates']) if 'coordinates' in row and pd.notna(row['coordinates']) else ""
        summary_label = f"{transcript_str} / {coords_str}".strip(" /")
    
        group_0_count = row['n0']
        group_1_count = row['n1']
        total = group_0_count + group_1_count
        
        p_value = f"{row['p_value']:.6e}" if not pd.isna(row['p_value']) else "N/A"
        effect_size = f"{row['effect_size']:.4f}" if not pd.isna(row['effect_size']) else "N/A"
        
        gene_info = f"{row['gene_symbol']}" if 'gene_symbol' in row and pd.notna(row['gene_symbol']) else ""
    
        print(f"{summary_label:<50} {group_0_count:<10} {group_1_count:<10} {total:<10} {p_value:<15} {effect_size:<15} {gene_info}")
    
    print("-" * 120)
    print(f"{'TOTAL':<50} {total_group_0:<10} {total_group_1:<10} {total_group_0 + total_group_1:<10}")
    
    # Print Bonferroni results
    significant_count = (results_df['bonferroni_p_value'] < 0.05).sum()
    print(f"\nSignificant results after Bonferroni correction (p < 0.05): {significant_count}")
    
    # Print significant results
    if significant_count > 0:
        print("\nSignificant results after Bonferroni correction:")
        print(f"{'Transcript/Coordinates':<50} {'P-value':<15} {'Corrected P':<15} {'Effect Size':<15} {'Gene'}")
        print("-" * 120)
        
        sig_results = results_df[results_df['bonferroni_p_value'] < 0.05].sort_values('p_value')
        
        for _, row in sig_results.iterrows():
            transcript_str = str(row['transcript_id']) if 'transcript_id' in row and pd.notna(row['transcript_id']) else ""
            coords_str = str(row['coordinates']) if 'coordinates' in row and pd.notna(row['coordinates']) else ""
            label = f"{transcript_str} / {coords_str}".strip(" /")
            p_value = f"{row['p_value']:.6e}" if not pd.isna(row['p_value']) else "N/A"
            corrected_p = f"{row['bonferroni_p_value']:.6e}" if not pd.isna(row['bonferroni_p_value']) else "N/A"
            effect_size = f"{row['effect_size']:.4f}" if not pd.isna(row['effect_size']) else "N/A"
            
            gene_info = ""
            if 'gene_symbol' in row and pd.notna(row['gene_symbol']) and 'gene_name' in row and pd.notna(row['gene_name']):
                gene_info = f"{row['gene_symbol']}: {row['gene_name']}"
            gene_info = gene_info[:40]
            
            print(f"{label:<50} {p_value:<15} {corrected_p:<15} {effect_size:<15} {gene_info}")
    
    # Print summary of failure reasons
    failure_counts = results_df['failure_reason'].value_counts()
    if not failure_counts.empty:
        print("\n=== Analysis Failure Summary ===")
        for reason, count in failure_counts.items():
            if pd.notna(reason):
                print(f"- {reason}: {count} coordinates")
    
    print(f"\nAnalysis completed at {datetime.now()}")
    print(f"Total runtime: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
