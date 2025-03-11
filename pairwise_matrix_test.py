import pandas as pd
import numpy as np
import re
from collections import defaultdict
from tqdm.auto import tqdm
import warnings
import os
import json
import pickle
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

warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION PARAMETERS
# =====================================================================

# Minimum number of sequences required in each group for valid analysis
MIN_SEQUENCES_PER_GROUP = 10

# Flag to determine whether to filter out special omega values (-1 and 99)
FILTER_SPECIAL_OMEGA_VALUES = False

# Flag to determine whether to calculate omega manually from dN/dS
CALCULATE_OMEGA_MANUALLY = False

def read_and_preprocess_data(file_path):
    """
    Read and preprocess the evolutionary rate data from a CSV file.

    This function performs several key preprocessing steps:
    1. Reads the CSV containing pairwise sequence comparisons
    2. Extracts group assignments (0 or 1) from CDS identifiers 
    3. Parses genomic coordinates (chromosome, start, end)
    4. Extracts transcript identifiers
    5. Validates omega values and handles special cases (-1, 99)

    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing raw pairwise comparison data

    Returns:
    --------
    DataFrame
        Processed DataFrame with additional columns for analysis:
        - group: Binary indicator (0 or 1) of sequence group assignment
        - chrom: Chromosome identifier (e.g., 'chr1')
        - start: Start coordinate of the genomic region
        - end: End coordinate of the genomic region
        - transcript_id: Ensembl transcript identifier

    Note:
    -----
    The function retains all omega values, including special cases like -1 and 99,
    which often indicate calculation limitations. Only NaN values are dropped.
    """
    print("Reading data...")
    df = pd.read_csv(file_path)

    # Store original CDS as full_cds for reference and troubleshooting
    df['full_cds'] = df['CDS']

    # Determine comparison group based on Group1 and Group2 columns
    df['group'] = None
    
    # Within-group comparisons
    df.loc[(df['Group1'] == 0) & (df['Group2'] == 0), 'group'] = 0  # Group 0 vs Group 0
    df.loc[(df['Group1'] == 1) & (df['Group2'] == 1), 'group'] = 1  # Group 1 vs Group 1
    
    # Cross-group comparisons - used for better random effect estimation
    df.loc[((df['Group1'] == 0) & (df['Group2'] == 1)) | 
           ((df['Group1'] == 1) & (df['Group2'] == 0)), 'group'] = 2  # Cross-group

    # Extract genomic coordinates using regex pattern
    # Format expected: chrX_startNNN_endNNN where X is chromosome and NNN are positions
    coord_pattern = r'chr(\w+)_start(\d+)_end(\d+)'
    coords = df['CDS'].str.extract(coord_pattern)
    df['chrom'] = 'chr' + coords[0]
    df['start'] = pd.to_numeric(coords[1])
    df['end'] = pd.to_numeric(coords[2])

    # Extract transcript ID using Ensembl format pattern (ENSTXXXXX.X)
    transcript_pattern = r'(ENST\d+\.\d+)'
    df['transcript_id'] = df['CDS'].str.extract(transcript_pattern)[0]

    # Convert omega to numeric values, coercing non-numeric entries to NaN
    # Omega is the ratio of non-synonymous to synonymous substitution rates
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')
    df['dN'] = pd.to_numeric(df['dN'], errors='coerce')
    df['dS'] = pd.to_numeric(df['dS'], errors='coerce')
    
    # Calculate omega manually if flag is set
    if CALCULATE_OMEGA_MANUALLY:
        print("Calculating omega manually from dN and dS values...")
        # Store original omega for reference
        df['original_omega'] = df['omega']
        
        # Calculate omega as dN/dS with special case handling
        # Case 1: dN = 0, dS = 0 -> omega = -1 (identical sequences)
        # Case 2: dN = any value, dS = 0 -> omega = 99 (infinite/undefined)
        # Case 3: Normal case -> omega = dN/dS
        df['omega'] = df.apply(
            lambda row: -1.0 if row['dN'] == 0 and row['dS'] == 0 else
                       (99.0 if row['dS'] == 0 else row['dN'] / row['dS']),
            axis=1
        )
    
        # Report differences between original and calculated omega
        different_count = (df['original_omega'] != df['omega']).sum()
        print(f"Manual calculation resulted in {different_count} different omega values")
    
    # Report special omega values
    # -1 means identical sequences
    # 99 means inf (or very high) omega
    omega_minus1_count = len(df[df['omega'] == -1])
    omega_99_count = len(df[df['omega'] == 99])
    print(f"Rows with omega = -1: {omega_minus1_count}")
    print(f"Rows with omega = 99: {omega_99_count}")
    
    # Filter out special omega values if flag is set
    if FILTER_SPECIAL_OMEGA_VALUES:
        original_len = len(df)
        df = df[(df['omega'] != -1) & (df['omega'] != 99)]
        filtered_count = original_len - len(df)
        print(f"Filtered out {filtered_count} rows with special omega values (-1 or 99)")
    
    # Keep all valid omega values, only dropping NaN entries
    # This preserves special codes while ensuring numeric analysis
    df = df.dropna(subset=['omega'])

    # Report dataset dimensions after preprocessing
    print(f"Total comparisons (including all omega values): {len(df)}")
    print(f"Unique coordinates found: {df.groupby(['chrom', 'start', 'end']).ngroups}")

    # Load the inversion info CSV
    inv_info_df = pd.read_csv('inv_info.csv')


    # Summarize sequence counts by group assignment
    # This is important to verify sufficient sample sizes for statistical analysis
    group0_seqs = set(pd.concat([df[df['group'] == 0]['Seq1'], df[df['group'] == 0]['Seq2']]).unique())
    group1_seqs = set(pd.concat([df[df['group'] == 1]['Seq1'], df[df['group'] == 1]['Seq2']]).unique())
    print(f"Sequences in group 0: {len(group0_seqs)}")
    print(f"Sequences in group 1: {len(group1_seqs)}")

    return df


def get_pairwise_value(seq1, seq2, pairwise_dict):
    """
    Retrieve the omega value for a specific pair of sequences from the pairwise dictionary.
    
    This function handles the bidirectional nature of sequence comparisons by checking
    both possible orderings of the sequence pair.
    
    Parameters:
    -----------
    seq1, seq2 : str
        Identifiers for the two sequences being compared
    pairwise_dict : dict
        Dictionary with sequence pairs as keys and omega values as values
        
    Returns:
    --------
    float or None
        The omega value for the sequence pair, or None if not found
        
    -----
    Since sequence comparisons can be stored with sequences in either order,
    this function checks both (seq1, seq2) and (seq2, seq1) as potential keys.
    """
    key = (seq1, seq2) if (seq1, seq2) in pairwise_dict else (seq2, seq1)
    val = pairwise_dict.get(key)
    return val

def create_matrices(sequences_0, sequences_1, pairwise_dict):
    """
    Create pairwise omega value matrices for the two sequence groups.
    
    This function generates square matrices for each group, where each cell [i,j]
    contains the omega value between sequences i and j in that group. These matrices
    are symmetric along the diagonal.
    
    Parameters:
    -----------
    sequences_0 : list
        List of sequence identifiers in group 0
    sequences_1 : list
        List of sequence identifiers in group 1
    pairwise_dict : dict
        Dictionary mapping sequence pairs to their omega values
        
    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray) or (None, None)
        Two matrices containing pairwise omega values:
        - matrix_0: Square matrix for group 0 sequences
        - matrix_1: Square matrix for group 1 sequences
        Returns (None, None) if both sequence lists are empty
        
    Note:
    -----
    - Matrix cells are initialized with NaN and only valid comparisons are filled
    - The matrices are symmetric (matrix[i,j] = matrix[j,i])
    - Diagonal elements (self-comparisons) remain as NaN
    """
    n0, n1 = len(sequences_0), len(sequences_1)
    
    # Return None for both matrices if there are no sequences
    if n0 == 0 and n1 == 0:
        return None, None
        
    # Initialize matrices with NaN values
    # Only create matrices for groups that have sequences
    matrix_0 = np.full((n0, n0), np.nan) if n0 > 0 else None
    matrix_1 = np.full((n1, n1), np.nan) if n1 > 0 else None

    # Fill matrix for group 0 sequences with pairwise omega values
    if n0 > 0:
        for i in range(n0):
            for j in range(i + 1, n0):  # Only process upper triangle
                val = get_pairwise_value(sequences_0[i], sequences_0[j], pairwise_dict)
                if val is not None:
                    # Fill both positions to make a symmetric matrix
                    matrix_0[i, j] = matrix_0[j, i] = float(val)

    # Fill matrix for group 1 sequences with pairwise omega values
    if n1 > 0:
        for i in range(n1):
            for j in range(i + 1, n1):  # Only process upper triangle
                val = get_pairwise_value(sequences_1[i], sequences_1[j], pairwise_dict)
                if val is not None:
                    # Fill both positions to make a symmetric matrix
                    matrix_1[i, j] = matrix_1[j, i] = float(val)

    return matrix_0, matrix_1

def get_gene_info(gene_symbol):
    """
    Retrieve human-readable gene information from MyGene.info API using gene symbol.
    
    Parameters:
    -----------
    gene_symbol : str
        The gene symbol (e.g., "TP53") to look up
        
    Returns:
    --------
    str
        Official gene name if found, or "Unknown" if not found or on error
        
    Note:
    -----
    - Uses MyGene.info REST API with 10-second timeout
    - Returns "Unknown" on any exception (network error, parsing error, etc.)
    - Filters specifically for human genes
    """
    try:
        # Query the MyGene.info API with the gene symbol, species constraint, and field selection
        url = f"http://mygene.info/v3/query?q=symbol:{gene_symbol}&species=human&fields=name"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                return data['hits'][0].get('name', 'Unknown')
    except Exception as e:
        print(f"Error fetching gene info: {str(e)}")
    return 'Unknown'  # Default return value for any error case

def get_gene_annotation(coordinates):
    """
    Retrieve gene annotation information for a genomic location.
    
    This function parses genomic coordinates and queries the UCSC Genome Browser API
    to identify genes overlapping with the specified region. It selects the best
    overlapping gene based on the extent of overlap.
    
    Parameters:
    -----------
    coordinates : str
        String representation of genomic coordinates in format:
        "chr_X_start_NNNNN_end_NNNNN"
        
    Returns:
    --------
    tuple (str, str)
        A tuple containing (gene_symbol, gene_name), or (None, None) if no gene found
        
    Note:
    -----
    - Uses UCSC API to query the knownGene track on hg38 genome build
    - Selects the gene with maximum overlap with the target region
    - Handles Ensembl gene IDs by preferring standard gene symbols when available
    - Error handling returns (None, None) for any exception
    """
    try:
        # Parse coordinates from the input string
        match = re.search(r'chr_(\w+)_start_(\d+)_end_(\d+)', coordinates)
        if not match:
            return None, None
            
        chrom, start, end = match.groups()
        chrom = 'chr' + chrom
        start, end = int(start), int(end)
        
        # Query the UCSC Genome Browser API for gene annotations
        base_url = "https://api.genome.ucsc.edu/getData/track"
        params = {'genome': 'hg38', 'track': 'knownGene', 'chrom': chrom, 'start': start, 'end': end}
        
        response = requests.get(f"{base_url}?{urlencode(params)}", timeout=10)
        if not response.ok:
            return None, None
            
        data = response.json()
        
        # Handle different response structures from UCSC API
        track_data = data.get('knownGene', data)
        
        if isinstance(track_data, str) or not track_data:
            return None, None
            
        # Find genes that overlap with our target region
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
            
        # Select gene with maximum overlap with our target region
        best_gene = max(
            overlapping_genes,
            key=lambda gene: max(0, min(gene.get('chromEnd', 0), end) - max(gene.get('chromStart', 0), start))
        )
        
        # Get gene symbol, avoiding Ensembl IDs when possible
        symbol = best_gene.get('geneName', 'Unknown')
        if symbol in ['none', None] or symbol.startswith('ENSG'):
            # Look for better gene names in other overlapping genes
            for gene in overlapping_genes:
                potential_symbol = gene.get('geneName')
                if potential_symbol and potential_symbol != 'none' and not potential_symbol.startswith('ENSG'):
                    symbol = potential_symbol
                    break
                    
        # Get full gene name from MyGene.info for readability
        name = get_gene_info(symbol)
        return symbol, name
        
    except Exception as e:
        print(f"Error in gene annotation: {str(e)}")
        return None, None

def analysis_worker(args):
    """
    Perform mixed-effects statistical analysis for a group of sequences.
    
    This worker function implements a crossed random effects model to compare
    omega values between two groups while accounting for sequence-specific effects.
    It's designed to be used with parallel processing.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing:
        - all_sequences: Combined list of all sequence IDs
        - pairwise_dict: Dictionary of pairwise omega values
        - sequences_0: List of sequence IDs in group 0
        - sequences_1: List of sequence IDs in group 1
        
    Returns:
    --------
    dict
        Dictionary with analysis results including:
        - effect_size: Estimated difference in omega between groups
        - p_value: Statistical significance of the effect
        - n0, n1: Number of sequences in each group
        - num_comp_group_0, num_comp_group_1: Number of comparisons in each group
        - std_err: Standard error of the effect size estimate
        - failure_reason: Description of analysis failure (if any)
        
    Note:
    -----
    - Uses mixed linear model with crossed random effects for sequence identities
    - Returns NaN for effect_size and p_value if analysis cannot be completed
    - Analysis may fail due to insufficient sequences, lack of variation, or model errors
    """
    all_sequences, pairwise_dict, sequences_0, sequences_1 = args
    
    n0, n1 = len(sequences_0), len(sequences_1)
    
    # Validate minimum sequence requirements for statistical power
    if n0 < MIN_SEQUENCES_PER_GROUP or n1 < MIN_SEQUENCES_PER_GROUP:
        insufficient_groups = []
        if n0 < MIN_SEQUENCES_PER_GROUP:
            insufficient_groups.append('0')
        if n1 < MIN_SEQUENCES_PER_GROUP:
            insufficient_groups.append('1')
            
        groups_str = " and ".join(insufficient_groups)
        print(f"Insufficient sequences in group(s) {groups_str}, need at least {MIN_SEQUENCES_PER_GROUP} sequences per group.")

        return {
            'effect_size': np.nan,
            'p_value': np.nan,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': sum(1 for (seq1, seq2) in pairwise_dict.keys() if seq1 in sequences_0 and seq2 in sequences_0),
            'num_comp_group_1': sum(1 for (seq1, seq2) in pairwise_dict.keys() if seq1 in sequences_1 and seq2 in sequences_1),
            'std_err': np.nan,
            'failure_reason': f"Insufficient sequences in group(s) {groups_str} (minimum {MIN_SEQUENCES_PER_GROUP} required)"
        }
    
    # Prepare data for mixed-model analysis by collecting all pairwise comparisons, including cross-group
    data = []
    for (seq1, seq2), omega in pairwise_dict.items():
        pair_group = None
        if seq1 in sequences_0 and seq2 in sequences_0:
            pair_group = 0
        elif seq1 in sequences_1 and seq2 in sequences_1:
            pair_group = 1
        else:
            pair_group = 2
        data.append({
            'omega_value': omega,
            'group': pair_group,
            'seq1': seq1,
            'seq2': seq2
        })

    df = pd.DataFrame(data)
    
    # Apply different transformations based on our filtering flag
    if FILTER_SPECIAL_OMEGA_VALUES:
        # Use raw omega values directly when special values are filtered out
        df['analysis_var'] = df['omega_value']
    else:
        # Use rank transform when keeping special values
        df['ranked_omega'] = df['omega_value'].rank(method='average')
        # Analysis variable will be the rank
        df['analysis_var'] = df['ranked_omega']

    # Initialize results variables
    effect_size = np.nan
    p_value = np.nan
    std_err = np.nan
    failure_reason = None

    # Validate data requirements for statistical analysis
    if df.empty or df['group'].nunique() < 2 or df['analysis_var'].nunique() < 2:
        if df.empty:
            failure_reason = "No valid pairwise comparisons found"
        elif df['group'].nunique() < 2:
            failure_reason = "Missing one of the groups in pairwise comparisons"
        elif df['analysis_var'].nunique() < 2:
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

    # Prepare categorical codes for sequence identifiers to use in random effects
    all_unique_seqs = pd.unique(pd.concat([df['seq1'], df['seq2']]))
    seq_to_code = {seq: i for i, seq in enumerate(all_unique_seqs)}

    # Apply the same coding to both columns
    df['seq1_code'] = df['seq1'].map(seq_to_code)
    df['seq2_code'] = df['seq2'].map(seq_to_code)

    try:
        # Set up mixed model with crossed random effects for sequence identities
        # This controls for sequence-specific effects that might confound group differences
        df['groups'] = 1  # Dummy grouping variable for statsmodels implementation
        vc = {
            'seq1': '0 + C(seq1_code)',  # Random effect for first sequence
            'seq2': '0 + C(seq2_code)'   # Random effect for second sequence
        }
        
        # Fit mixed linear model with group as fixed effect and sequence IDs as random effect
        # Create dummy variables for each group explicitly
        df['is_group1'] = (df['group'] == 1).astype(int)
        df['is_group2'] = (df['group'] == 2).astype(int)  # Cross-group comparison
        
        # Use these dummy variables in the model
        model = MixedLM.from_formula(
            'analysis_var ~ is_group1 + is_group2',  # Group 0 is reference
            groups='groups',
            vc_formula=vc,
            re_formula='0',
            data=df
        )
        result = model.fit(reml=False)
        
        # Extract ONLY the group 0 vs group 1 effect
        effect_size = result.params.get('is_group1', np.nan)
        p_value = result.pvalues.get('is_group1', np.nan)
        std_err = result.bse.get('is_group1', np.nan)
        
    except Exception as e:
        # Handle statistical model failures
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
    """
    Analyze evolutionary rates for a specific transcript.
    
    This function processes all pairwise comparisons for a single transcript,
    organizing sequences by group, performing statistical analysis, retrieving
    gene annotations
    
    Parameters:
    -----------
    args : tuple
        Tuple containing:
        - df_transcript: DataFrame subset for this transcript
        - transcript_id: Identifier of the transcript being analyzed
        
    Returns:
    --------
    dict
        Dictionary with analysis results including:
        - transcript_id: Ensembl transcript identifier
        - coordinates: Genomic coordinates of the transcript
        - gene_symbol: Gene symbol if annotation available
        - gene_name: Full gene name if annotation available
        - n0, n1: Number of sequences in each group
        - num_comp_group_0, num_comp_group_1: Number of comparisons in each group
        - effect_size: Estimated difference in omega between groups
        - p_value: Statistical significance of the effect
        - std_err: Standard error of the effect size estimate
        - failure_reason: Description of analysis failure (if any)
        
    Note:
    -----
    - Creates pairwise dictionary of omega values for statistical analysis
    - Identifies gene annotations using UCSC and MyGene.info APIs
    - Delegates statistical analysis to analysis_worker function
    """
    df_transcript, transcript_id = args

    print(f"\nAnalyzing transcript: {transcript_id}")

    # Separate sequences by their group assignment
    group_0_df = df_transcript[df_transcript['group'] == 0]
    group_1_df = df_transcript[df_transcript['group'] == 1]

    # Get unique sequence identifiers for each group
    sequences_0 = pd.concat([group_0_df['Seq1'], group_0_df['Seq2']]).unique()
    sequences_1 = pd.concat([group_1_df['Seq1'], group_1_df['Seq2']]).unique()

    # Create pairwise dictionary mapping sequence pairs to their omega values
    pairwise_dict = {}
    for _, row in df_transcript.iterrows():
        pairwise_dict[(row['Seq1'], row['Seq2'])] = row['omega']

    # Combine all sequences for comprehensive analysis
    all_sequences = (
        np.concatenate([sequences_0, sequences_1])
        if len(sequences_0) > 0 and len(sequences_1) > 0
        else (sequences_0 if len(sequences_0) > 0 else sequences_1)
    )

    # Collect unique genomic coordinates for reporting
    unique_coords = set(
        f"{r['chrom']}:{r['start']}-{r['end']}" for _, r in df_transcript.iterrows()
    )
    coords_str = ";".join(sorted(unique_coords))

    # Create matrices for possible visualization or additional analysis
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)

    # Get gene annotation information from the first entry's coordinates
    first_row = df_transcript.iloc[0]
    coordinates_str = f"chr_{str(first_row['chrom']).replace('chr', '')}_start_{first_row['start']}_end_{first_row['end']}"
    gene_symbol, gene_name = get_gene_annotation(coordinates_str)
    
    # Perform statistical analysis on the transcript data
    analysis_result = analysis_worker((all_sequences, pairwise_dict, sequences_0, sequences_1))

    # Compute normal-only median and mean for each group (excluding -1 and 99)
    group_0_normal = group_0_df[(group_0_df['omega'] != -1) & (group_0_df['omega'] != 99)]
    median_0_normal = group_0_normal['omega'].median()
    mean_0_normal = group_0_normal['omega'].mean()
    group_1_normal = group_1_df[(group_1_df['omega'] != -1) & (group_1_df['omega'] != 99)]
    median_1_normal = group_1_normal['omega'].median()
    mean_1_normal = group_1_normal['omega'].mean()

    # Compute percentage of identical (-1) and no synonymous variation (99) in each group
    pct_identical_0 = 100.0 * (group_0_df['omega'] == -1).mean()
    pct_nosyn_0 = 100.0 * (group_0_df['omega'] == 99).mean()
    pct_identical_1 = 100.0 * (group_1_df['omega'] == -1).mean()
    pct_nosyn_1 = 100.0 * (group_1_df['omega'] == 99).mean()
    
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
        'failure_reason': analysis_result['failure_reason'],
        'matrix_0': matrix_0,
        'matrix_1': matrix_1,
        'pairwise_comparisons': set(pairwise_dict.keys()),
        'median_0_normal': median_0_normal,
        'mean_0_normal': mean_0_normal,
        'median_1_normal': median_1_normal,
        'mean_1_normal': mean_1_normal,
        'pct_identical_0': pct_identical_0,
        'pct_nosyn_0': pct_nosyn_0,
        'pct_identical_1': pct_identical_1,
        'pct_nosyn_1': pct_nosyn_1
    }


    return result


def main():
    """
    Main execution function for evolutionary rate analysis.
    
    This function orchestrates the entire analysis workflow:
    1. Reading and preprocessing input data
    2. Organizing analysis by transcript
    3. Performing parallel analysis across transcripts
    4. Applying multiple testing correction
    5. Generating summary statistics and reports
    6. Saving results to CSV
    
    Parameters:
    -----------
    None
    
    Returns:
    --------
    None
        Results are saved to CSV and printed to console
        
    Note:
    -----
    - Uses parallel processing via ProcessPoolExecutor
    - Applies correction for multiple hypothesis testing
    - Outputs both comprehensive and significant result summaries
    - Tracks and reports analysis runtime
    """
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")

    # Read and preprocess the input dataset
    df = read_and_preprocess_data('all_pairwise_results.csv')
    
    # Group the data by transcript for independent analysis
    transcript_groups = df.groupby('transcript_id')
    print(f"\nFound {len(transcript_groups)} unique transcripts")
    
    # Prepare arguments for parallel processing of transcripts
    transcript_args = [(transcript_group, transcript_id) for transcript_id, transcript_group in transcript_groups]
    
    # Process each transcript in parallel using all available CPU cores
    results = []
    cds_results = {}
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for result in tqdm(executor.map(analyze_transcript, transcript_args), 
                          total=len(transcript_args), 
                          desc="Processing transcripts"):
            results.append(result)
            
            # Extract CDS from coordinates and store in CDS-based dictionary
            coords_split = result['coordinates'].split(';')
            for coord in coords_split:
                if coord:
                    # Use CDS as key for matrix visualization compatibility
                    cds_results[coord] = {
                        'matrix_0': result['matrix_0'],
                        'matrix_1': result['matrix_1'],
                        'pairwise_comparisons': result['pairwise_comparisons'],
                        'p_value': result['p_value'],
                        'observed_effect_size': result['effect_size'],
                        'corrected_p_value': result['p_value'] * len(transcript_args) if not pd.isna(result['p_value']) else np.nan,
                        'gene_symbol': result['gene_symbol'],
                        'gene_name': result['gene_name']
                    }

    # Create results dataframe for further analysis and reporting
    results_df = pd.DataFrame(results)
    
    # Apply Benjamini-Hochberg procedure for FDR control
    valid_results = results_df[results_df['p_value'].notna() & (results_df['p_value'] > 0)]
    num_valid_tests = len(valid_results)
    
    if num_valid_tests > 0:
        # Sort p-values
        valid_results = valid_results.sort_values('p_value').copy()

        # Calculate ranks
        valid_results['rank'] = np.arange(1, len(valid_results) + 1)
        
        # Calculate BH adjusted p-values
        valid_results['corrected_p_value'] = valid_results['p_value'] * num_valid_tests / valid_results['rank']

        # Monotonicity of p-values (step-up procedure)
        valid_results['corrected_p_value'] = valid_results['corrected_p_value'].iloc[::-1].cummin().iloc[::-1]

        # Create a mapping from transcript_id to adjusted p-value
        corrected_p_value_map = dict(zip(valid_results['transcript_id'], valid_results['corrected_p_value']))
    
        # Map adjusted p-values back to the original dataframe
        results_df['corrected_p_value'] = results_df['transcript_id'].map(corrected_p_value_map)

        # Cap adjusted p-values at 1.0
        results_df['corrected_p_value'] = results_df['corrected_p_value'].clip(upper=1.0)
    else:
        results_df['corrected_p_value'] = results_df['p_value']
    
    # Calculate -log10(p) for visualization and interpretation
    # Larger values indicate more significant results
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    
    # Save results to CSV file
    os.makedirs('results', exist_ok=True)
    # Create a copy without data structures for CSV export
    csv_results_df = results_df.drop(['matrix_0', 'matrix_1', 'pairwise_comparisons'], axis=1)
    csv_results_df.to_csv('results/final_results.csv', index=False)
    
    # Create and save filtered results sorted by absolute effect size
    valid_p_results = results_df[results_df['p_value'].notna()]
    sorted_by_effect = valid_p_results.copy()
    sorted_by_effect['abs_effect_size'] = sorted_by_effect['effect_size'].abs()
    sorted_by_effect = sorted_by_effect.sort_values('abs_effect_size', ascending=False)
    sorted_by_effect = sorted_by_effect.drop('abs_effect_size', axis=1)  # Remove the temporary column
    # Remove data structures before saving to CSV
    csv_sorted_by_effect = sorted_by_effect.drop(['matrix_0', 'matrix_1', 'pairwise_comparisons'], axis=1)
    csv_sorted_by_effect.to_csv('results/significant_by_effect.csv', index=False)

    
    # Calculate total sequence counts by group
    total_group_0 = results_df['n0'].sum()
    total_group_1 = results_df['n1'].sum()
    
    # Sort results by p-value for more intuitive display
    sorted_results = results_df.sort_values('p_value')
    
    # Print detailed header for main results table
    print("\n=== Group Assignment Summary by Transcript ===")
    print(f"{'Transcript/Coordinates':<50} {'Group 0':<10} {'Group 1':<10} {'Total':<10} {'P-value/Status':<40} {'Effect Size':<15}")
    print("-" * 160)
    
    # Print detailed information for each transcript
    for _, row in sorted_results.iterrows():
        transcript_str = str(row['transcript_id']) if 'transcript_id' in row and pd.notna(row['transcript_id']) else ""
        coords_str = str(row['coordinates']) if 'coordinates' in row and pd.notna(row['coordinates']) else ""
        summary_label = f"{transcript_str} / {coords_str}".strip(" /")
    
        group_0_count = row['n0']
        group_1_count = row['n1']
        total = group_0_count + group_1_count
        
        # Format p-value display, showing failure reason if analysis failed
        if pd.isna(row['p_value']) and pd.notna(row['failure_reason']):
            p_value = row['failure_reason']
        else:
            p_value = f"{row['p_value']:.6e}" if not pd.isna(row['p_value']) else "N/A"
            
        # Format effect size display
        if pd.isna(row['effect_size']) and pd.notna(row['failure_reason']):
            effect_size = "N/A"
        else:
            effect_size = f"{row['effect_size']:.4f}" if not pd.isna(row['effect_size']) else "N/A"
        
        # Format gene information display
        gene_info = f"{row['gene_symbol']}" if 'gene_symbol' in row and pd.notna(row['gene_symbol']) else ""
        
        # Print row of results table
        print(f"{summary_label:<50} {group_0_count:<10} {group_1_count:<10} {total:<10} {p_value:<40} {effect_size:<15} {gene_info:<15}")
    
    # Print table footer with totals
    print("-" * 160)
    print(f"{'TOTAL':<50} {total_group_0:<10} {total_group_1:<10} {total_group_0 + total_group_1:<10}")
    
    # Summarize significant results after multiple testing correction
    significant_count = (results_df['corrected_p_value'] < 0.05).sum()
    print(f"\nSignificant results after correction (p < 0.05): {significant_count}")
    
    # Print detailed information for significant results
    if significant_count > 0:
        print("\nSignificant results after correction:")
        print(
            f"{'Transcript/Coordinates':<50} "
            f"{'P-value':<15} "
            f"{'Corrected P':<15} "
            f"{'Effect Size':<15} "
            f"{'Median_0':<10} "
            f"{'Mean_0':<10} "
            f"{'Median_1':<10} "
            f"{'Mean_1':<10} "
            f"{'Gene':<15} "
        )
        print("-" * 160)

        # Select and sort significant results
        sig_results = results_df[results_df['corrected_p_value'] < 0.05].sort_values('p_value')
        
        # Print each significant result with detailed information
        for _, row in sig_results.iterrows():
            transcript_str = str(row['transcript_id']) if 'transcript_id' in row and pd.notna(row['transcript_id']) else ""
            coords_str = str(row['coordinates']) if 'coordinates' in row and pd.notna(row['coordinates']) else ""
            label = f"{transcript_str} / {coords_str}".strip(" /")
            p_value = f"{row['p_value']:.6e}" if not pd.isna(row['p_value']) else "N/A"
            corrected_p = f"{row['corrected_p_value']:.6e}" if not pd.isna(row['corrected_p_value']) else "N/A"
            effect_size = f"{row['effect_size']:.4f}" if not pd.isna(row['effect_size']) else "N/A"
            
            median_0 = row['median_0_normal']
            mean_0 = row['mean_0_normal']
            median_1 = row['median_1_normal']
            mean_1 = row['mean_1_normal']

            median_0_str = f"{median_0:.3f}" if not pd.isna(median_0) else "N/A"
            mean_0_str = f"{mean_0:.3f}" if not pd.isna(mean_0) else "N/A"
            median_1_str = f"{median_1:.3f}" if not pd.isna(median_1) else "N/A"
            mean_1_str = f"{mean_1:.3f}" if not pd.isna(mean_1) else "N/A"

            # Format gene information with name if available
            gene_info = ""
            if 'gene_symbol' in row and pd.notna(row['gene_symbol']) and 'gene_name' in row and pd.notna(row['gene_name']):
                gene_info = f"{row['gene_symbol']}: {row['gene_name']}"
            gene_info = gene_info[:40]
            
            print(f"{label:<50} {p_value:<15} {corrected_p:<15} {effect_size:<15} {median_0_str:<10} {mean_0_str:<10} {median_1_str:<10} {mean_1_str:<10} {gene_info:<15}")
                
    # Summarize analysis failures by reason
    failure_counts = results_df['failure_reason'].value_counts()
    if not failure_counts.empty:
        print("\n=== Analysis Failure Summary ===")
        for reason, count in failure_counts.items():
            if pd.notna(reason):
                print(f"- {reason}: {count} coordinates")

    # Save the CDS results to pickle file for matrix visualization later
    os.makedirs('cache', exist_ok=True)
    cds_results_file = 'cache/all_cds_results.pkl'
    print(f"\nSaving CDS results to {cds_results_file}...")
    with open(cds_results_file, 'wb') as f:
        pickle.dump(cds_results, f)
    print(f"Saved results for {len(cds_results)} CDSs")
    
    # Print completion information and runtime
    print(f"\nAnalysis completed at {datetime.now()}")
    print(f"Total runtime: {datetime.now() - start_time}")
if __name__ == "__main__":
    main()
