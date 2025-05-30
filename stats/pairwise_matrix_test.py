import pandas as pd
import numpy as np
import re
from collections import defaultdict
from tqdm.auto import tqdm
import warnings
import os
import json
import pickle
import glob
import hashlib
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

# Path to folder containing PCA files
PCA_FOLDER = "pca"

# Number of PCs to use as covariates
NUM_PCS_TO_USE = 5

# Flag to enable PC correction for population structure
ENABLE_PC_CORRECTION = True

# Minimum number of sequences required in each group for valid analysis
MIN_SEQUENCES_PER_GROUP = 10

# Flag to determine whether to filter out omega = 99 values
FILTER_OMEGA_99 = True  # <-- True to filter 99

# Flag to determine whether to use ranked omega for the main statistical analysis
USE_RANKED_OMEGA_ANALYSIS = True # <-- True to use ranks

# Flag to determine whether to calculate omega manually from dN/dS
CALCULATE_OMEGA_MANUALLY = False # This should do nothing

# Flag to enable Low-Middle-High omega categorization analysis
PERFORM_OMEGA_CATEGORY_ANALYSIS = True

FILTER_ON_CROSS_GROUP_OMEGA = False

def read_and_preprocess_data(file_path):
    """
    Read and preprocess the evolutionary rate data from a CSV file.

    This function performs several key preprocessing steps:
    1. Reads the CSV containing pairwise sequence comparisons
    2. Extracts group assignments (0 or 1) from CDS identifiers 
    3. Parses genomic coordinates (chromosome, start, end)
    4. Extracts transcript identifiers
    5. Validates omega values and handles special cases (-1, 99)
    6. Extracts chromosome identifiers for PC matching

    
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
        - chromosome: Plain chromosome number for matching with PCA data

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
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce').replace(-1, -1.0)
    df['dN'] = pd.to_numeric(df['dN'], errors='coerce')
    df['dS'] = pd.to_numeric(df['dS'], errors='coerce')
    
    # Calculate omega manually if flag is set
    if CALCULATE_OMEGA_MANUALLY:
        print("Calculating omega manually from dN and dS values...")
        # Store original omega for reference
        df['original_omega'] = df['omega']
        
        # Calculate omega as dN/dS with special case handling
        # Case 1: dN = 0, dS = 0 -> omega = -1 (identical sequences)
        # Identical sequences can be indicative of strong purifying selection
        # Case 2: dN = any value, dS = 0 -> omega = 99 (infinite/undefined)
        # Case 3: Normal case -> omega = dN/dS
        df['omega'] = df.apply(
            lambda row: 0.0 if row['dN'] == 0 and row['dS'] == 0 else
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
    
    # Filter out omega=99 values if flag is set
    if FILTER_OMEGA_99:
        original_len = len(df)
        df = df[df['omega'] != 99]
        filtered_count = original_len - len(df)
        print(f"Filtered out {filtered_count} rows with omega=99")

    # Keep all valid omega values, only dropping NaN entries
    # This preserves special codes while ensuring numeric analysis
    df = df.dropna(subset=['omega'])

    # Report dataset dimensions after preprocessing
    print("Extracting chromosome identifiers for PC matching...")
    df['chromosome'] = df['CDS'].apply(extract_chromosome)
    
    # Report dataset dimensions after preprocessing
    print(f"Total comparisons (including all omega values): {len(df)}")
    print(f"Unique coordinates found: {df.groupby(['chrom', 'start', 'end']).ngroups}")
    print(f"Unique chromosomes found: {df['chromosome'].nunique()}")

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

def convert_full_name_to_short(full_name):
    """
    Convert PCA sample names to the format used in pairwise results.
    
    Parameters:
    -----------
    full_name : str
        Sample name in PCA format (e.g., 'EUR_GBR_HG00096_L')
        
    Returns:
    --------
    str
        Sample name in shortened format (e.g., 'EURGB93_L')
    """
    parts = full_name.split('_')
    if len(parts) < 4:
        return None
        
    # Extract the population and subpopulation from the beginning
    first = parts[0][:3] if len(parts) > 0 else "UNK"
    second = parts[1][:3] if len(parts) > 1 else "UNK"
    
    # Extract the sample ID (which is typically the second-to-last part)
    # and the haplotype indicator (L/R) which is the last part
    hg_part = parts[-2] if len(parts) > 1 else "UNKWN"
    group = parts[-1] if parts[-1] in ['L', 'R'] else "U"  # L or R for left/right haplotype
    
    # Generate hash like in original code
    md5_val = hashlib.md5(hg_part.encode('utf-8')).hexdigest()
    hash_str = md5_val[:2]
    
    short_name = f"{first}{second}{hash_str}_{group}"
    
    # Print sample name
    if hash(full_name) % 100 == 0:  # Only print ~1% of conversions to avoid overwhelming output
        print(f"Example sample name conversion: {full_name} -> {short_name}")
    
    return short_name

def load_pca_data(pca_folder, n_pcs=3):
    """
    Load PCA data for all chromosomes with name conversion.
    
    Parameters:
    -----------
    pca_folder : str
        Path to folder containing PCA files
    n_pcs : int
        Number of principal components to use
        
    Returns:
    --------
    dict
        Nested dictionary: {chr: {sample_name: [PC1, PC2, ...]}}
    """
    print(f"Loading PCA data from {pca_folder}...")
    pca_data = {}  # Structure: {chr: {sample_name: [PC1, PC2, ...]}}
    
    # Find and process all PCA files
    pca_files = glob.glob(os.path.join(pca_folder, "pca_chr_*.tsv"))
    if not pca_files:
        print(f"ERROR: No PCA files found in {pca_folder}. Check folder path and file naming.")
        print(f"Current working directory: {os.getcwd()}")
        try:
            if os.path.exists(pca_folder):
                for f in os.listdir(pca_folder):
                    print(f"  {f}")
            else:
                print(f"  Directory {pca_folder} does not exist!")
        except Exception as e:
            print(f"  Error listing directory: {e}")
        return pca_data
        
    for pca_file in pca_files:
        chr_name = os.path.basename(pca_file).replace("pca_chr_", "").replace(".tsv", "")
        
        try:
            df = pd.read_csv(pca_file, sep='\t')
            
            # Create chromosome entry
            pca_data[chr_name] = {}
            
            # Create PC column names
            pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
            
            if not all(pc in df.columns for pc in pc_cols):
                print(f"WARNING: Not all required PCs ({pc_cols}) found in {pca_file}")
                pc_cols = [col for col in pc_cols if col in df.columns]
                
            # Process each sample
            conversion_count = 0
            for _, row in df.iterrows():
                full_name = row['Haplotype']
                short_name = convert_full_name_to_short(full_name)
                if short_name:
                    # Store available principal components
                    pca_data[chr_name][short_name] = row[pc_cols].values.tolist()
                    conversion_count += 1
                    
            print(f"  Loaded {conversion_count} samples from chromosome {chr_name} with {len(pc_cols)} PCs")
                
        except Exception as e:
            print(f"ERROR: Failed to process {pca_file}: {e}")
    
    # Count samples with PCA data
    sample_count = sum(len(samples) for samples in pca_data.values())
    chr_count = len(pca_data)
    print(f"Successfully loaded PCA data for {chr_count} chromosomes and {sample_count} samples")
    
    return pca_data

def extract_chromosome(cds_field):
    """
    Extract chromosome number from CDS field in pairwise results.
    
    Parameters:
    -----------
    cds_field : str
        CDS field from pairwise results CSV
        
    Returns:
    --------
    str
        Chromosome identifier (e.g., '1', 'X', etc.)
    """
    if pd.isna(cds_field):
        return None
    match = re.search(r'chr(\w+)_start', cds_field)
    if match:
        return match.group(1)
    return None

def categorize_omega(omega_value):
    """
    Categorize omega values into Low, Middle, and High categories.
    
    Parameters:
    -----------
    omega_value : float
        The omega (dN/dS) value to categorize
        
    Returns:
    --------
    str
        Category label: 'Low', 'Middle', or 'High'
    """
    # Identical sequences (-1) are categorized as Low (strong conservation)
    if omega_value == -1:
        return 'Low'
    # Infinite omega values (99) are categorized as High (potential positive selection)
    elif omega_value == 99:
        return 'High'
    # Regular omega value categories
    elif omega_value < LOW_OMEGA_THRESHOLD:
        return 'Low'
    elif omega_value > HIGH_OMEGA_THRESHOLD:
        return 'High'
    else:
        return 'Middle'

def analyze_omega_categories(group_0_df, group_1_df):
    """
    Analyze the conservation differences between two groups.
    Uses sequence-level median omega values directly for a more powerful
    continuous value analysis rather than categorical binning.
    
    Parameters:
    -----------
    group_0_df : DataFrame
        DataFrame containing group 0 pairwise comparisons
    group_1_df : DataFrame
        DataFrame containing group 1 pairwise comparisons
        
    Returns:
    --------
    dict
        Dictionary with sequence-level conservation analysis results:
        - median_values: Median omega values for each group
        - p_value: Statistical test p-value
        - test_used: Name of statistical test used
        - dominant_difference: Simple conservation comparison
    """
    # Skip analysis if either group is empty
    if group_0_df.empty or group_1_df.empty:
        return {
            'median_values': None,
            'p_value': np.nan,
            'test_used': None,
            'dominant_difference': None
        }
    
    # Extract all unique sequences in each group
    g0_seqs = set(pd.concat([group_0_df['Seq1'], group_0_df['Seq2']]))
    g1_seqs = set(pd.concat([group_1_df['Seq1'], group_1_df['Seq2']]))
    
    # Calculate median omega for each sequence in Group 0
    g0_seq_medians = []
    for seq in g0_seqs:
        # Find all pairwise comparisons involving this sequence
        seq_comparisons = group_0_df[(group_0_df['Seq1'] == seq) | (group_0_df['Seq2'] == seq)]
        if not seq_comparisons.empty:
            g0_seq_medians.append(seq_comparisons['omega'].median())
    
    # Calculate median omega for each sequence in Group 1
    g1_seq_medians = []
    for seq in g1_seqs:
        # Find all pairwise comparisons involving this sequence
        seq_comparisons = group_1_df[(group_1_df['Seq1'] == seq) | (group_1_df['Seq2'] == seq)]
        if not seq_comparisons.empty:
            g1_seq_medians.append(seq_comparisons['omega'].median())
    
    # Check if we have enough sequences for analysis
    g0_total = len(g0_seq_medians)
    g1_total = len(g1_seq_medians)
    
    if g0_total < MIN_SEQUENCES_PER_GROUP or g1_total < MIN_SEQUENCES_PER_GROUP:
        return {
            'median_values': {
                'group_0': np.nan,
                'group_1': np.nan
            },
            'p_value': np.nan,
            'test_used': "Insufficient sequences",
            'dominant_difference': None,
            'sequences_per_group': {
                'group_0': g0_total,
                'group_1': g1_total
            }
        }
    
    # Calculate group-level median values
    median_0 = np.median(g0_seq_medians) if g0_seq_medians else np.nan
    median_1 = np.median(g1_seq_medians) if g1_seq_medians else np.nan
    
    # Determine conservation difference (lower omega = more conservation)
    if not np.isnan(median_0) and not np.isnan(median_1):
        if median_1 < median_0:
            dominant_difference = "Group 1 more conserved than Group 0"
        else:
            dominant_difference = "Group 1 less conserved than Group 0"
    else:
        dominant_difference = None
    
    # Perform Mann-Whitney U test (more robust for non-normal distributions)
    test_used = "Mann-Whitney U"
    p_value = np.nan
    
    try:
        # Only perform test if we have enough sequences
        if g0_total >= MIN_SEQUENCES_PER_GROUP and g1_total >= MIN_SEQUENCES_PER_GROUP:
            # Handle special omega values (-1 and 99) by converting to ranks internally
            g0_array = np.array(g0_seq_medians)
            g1_array = np.array(g1_seq_medians)
            
            # Mann-Whitney U test (non-parametric, robust to non-normal distributions)
            _, p_value = stats.mannwhitneyu(g0_array, g1_array, alternative='two-sided')
        else:
            test_used = "Insufficient sequences"
    except Exception as e:
        test_used = f"Failed: {str(e)[:50]}"
    
    # Return results
    return {
        'median_values': {
            'group_0': median_0,
            'group_1': median_1
        },
        'p_value': p_value,
        'test_used': test_used,
        'dominant_difference': dominant_difference,
        'sequences_per_group': {
            'group_0': g0_total,
            'group_1': g1_total
        }
    }

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

# Persistent cache to store gene information between runs
os.makedirs('gene_cache', exist_ok=True)
GENE_CACHE_FILE = 'gene_cache/gene_info_cache.pkl'

# Load existing gene cache from file if it exists
GENE_INFO_CACHE = {}
if os.path.exists(GENE_CACHE_FILE):
    try:
        with open(GENE_CACHE_FILE, 'rb') as f:
            GENE_INFO_CACHE = pickle.load(f)
        print(f"Loaded {len(GENE_INFO_CACHE)} cached gene annotations")
    except Exception as e:
        print(f"Error loading gene cache: {e}")
        GENE_INFO_CACHE = {}

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
    - Caches results persistently between runs
    """
    # Check cache first
    if gene_symbol in GENE_INFO_CACHE:
        return GENE_INFO_CACHE[gene_symbol]
        
    try:
        # Query the MyGene.info API with the gene symbol, species constraint, and field selection
        url = f"http://mygene.info/v3/query?q=symbol:{gene_symbol}&species=human&fields=name"
        response = requests.get(url, timeout=10)
        if response.ok:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                name = data['hits'][0].get('name', 'Unknown')
                # Cache the result
                GENE_INFO_CACHE[gene_symbol] = name
                # Save updated cache to disk
                with open(GENE_CACHE_FILE, 'wb') as f:
                    pickle.dump(GENE_INFO_CACHE, f)
                return name
    except Exception as e:
        print(f"Error fetching gene info: {str(e)}")
    
    # Cache the negative result too
    GENE_INFO_CACHE[gene_symbol] = 'Unknown'
    # Save updated cache to disk, including negative results
    with open(GENE_CACHE_FILE, 'wb') as f:
        pickle.dump(GENE_INFO_CACHE, f)
    return 'Unknown'  # Default return value for any error case

def get_gene_info_from_transcript(transcript_id):
    """
    Retrieve human-readable gene information from MyGene.info API using transcript ID.
    
    Parameters:
    -----------
    transcript_id : str
        Ensembl transcript identifier (e.g., "ENST00000519106.2")
        
    Returns:
    --------
    tuple (str, str)
        A tuple containing (gene_symbol, gene_name), or ("Unknown", "Unknown") if not found
        
    Note:
    -----
    - Uses MyGene.info REST API with 10-second timeout
    - Removes version number from transcript ID (e.g., ENST00000519106.2 -> ENST00000519106)
    - Returns ("Unknown", "Unknown") on any exception (network error, parsing error, etc.)
    - Filters specifically for human genes
    - Caches results persistently between runs
    """
    # Check cache first
    if transcript_id in GENE_INFO_CACHE:
        return GENE_INFO_CACHE[transcript_id]
    
    try:
        # Remove version number if present
        base_id = transcript_id.split('.')[0]
        
        # Query the MyGene.info API with the transcript ID, species constraint, and field selection
        url = f"http://mygene.info/v3/query?q=ensembl.transcript:{base_id}&species=human&fields=_id,name,symbol,summary"
        response = requests.get(url, timeout=10)
        
        if response.ok:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                hit = data['hits'][0]
                result = (hit.get('symbol', 'Unknown'), hit.get('name', 'Unknown'))
                # Cache the result
                GENE_INFO_CACHE[transcript_id] = result
                # Save updated cache to disk
                with open(GENE_CACHE_FILE, 'wb') as f:
                    pickle.dump(GENE_INFO_CACHE, f)
                return result
    except Exception as e:
        print(f"Error fetching gene info for transcript {transcript_id}: {str(e)}")
    
    # Cache the negative result too
    result = ('Unknown', 'Unknown')
    GENE_INFO_CACHE[transcript_id] = result
    # Save updated cache to disk, including negative results
    with open(GENE_CACHE_FILE, 'wb') as f:
        pickle.dump(GENE_INFO_CACHE, f)
    return result

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
    omega values between two groups while accounting for sequence-specific effects
    and controlling for population structure using principal components.
    It's designed to be used with parallel processing.

    Parameters:
    -----------
    args : tuple
        Tuple containing:
        - all_sequences: Combined list of all sequence IDs (Not directly used in this version, but part of signature)
        - pairwise_dict: Dictionary of pairwise omega values
        - sequences_0: List of sequence IDs in group 0
        - sequences_1: List of sequence IDs in group 1
        - chromosome: Chromosome identifier for PCA matching
        - pc_data: Dictionary of PC values by chromosome and sample {chr: {sample: [PC1,..]}}
        - enable_pc_correction: Flag to enable PC-based correction

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
        - pc_corrected: Boolean indicating if PC correction was applied

    Note:
    -----
    - Uses mixed linear model with crossed random effects for sequence identities
    - Includes PCs as covariates when PC data is available (using optimized merge)
    - Returns NaN for effect_size and p_value if analysis cannot be completed
    - Analysis may fail due to insufficient sequences, lack of variation, or model errors
    """
    # Unpack arguments
    all_sequences, pairwise_dict, sequences_0, sequences_1, chromosome, pc_data, enable_pc_correction = args
    func_start_time = datetime.now()
    print(f"[{func_start_time}] Worker started for chromosome {chromosome}.")

    n0, n1 = len(sequences_0), len(sequences_1)

    # --- 1. Validate minimum sequence requirements ---
    if n0 < MIN_SEQUENCES_PER_GROUP or n1 < MIN_SEQUENCES_PER_GROUP:
        insufficient_groups = [g for g, count in [('0', n0), ('1', n1)] if count < MIN_SEQUENCES_PER_GROUP]
        groups_str = " and ".join(insufficient_groups)
        reason = f"Insufficient sequences in group(s) {groups_str} ({n0=}, {n1=}, minimum {MIN_SEQUENCES_PER_GROUP} required)"
        print(f"[{datetime.now()}] Worker exiting early: {reason}")
        num_comp_0 = sum(1 for (s1, s2) in pairwise_dict.keys() if s1 in sequences_0 and s2 in sequences_0)
        num_comp_1 = sum(1 for (s1, s2) in pairwise_dict.keys() if s1 in sequences_1 and s2 in sequences_1)
        return {
            'effect_size': np.nan, 'p_value': np.nan, 'n0': n0, 'n1': n1,
            'num_comp_group_0': num_comp_0, 'num_comp_group_1': num_comp_1,
            'std_err': np.nan, 'failure_reason': reason, 'pc_corrected': False
        }

    # --- 2. Prepare DataFrame from pairwise data ---
    print(f"[{datetime.now()}] Preparing DataFrame for {len(pairwise_dict)} comparisons...")
    # Using list comprehension for potentially slightly better performance than append loop
    data = [
        {'omega_value': omega,
         'group': 0 if seq1 in sequences_0 and seq2 in sequences_0 else (1 if seq1 in sequences_1 and seq2 in sequences_1 else 2),
         'seq1': seq1,
         'seq2': seq2}
        for (seq1, seq2), omega in pairwise_dict.items()
    ]
    df = pd.DataFrame(data)
    print(f"[{datetime.now()}] DataFrame prepared with {len(df)} rows.")

    # --- 3. Determine Analysis Variable (Ranked or Raw) ---
    if USE_RANKED_OMEGA_ANALYSIS:
        print(f"[{datetime.now()}] Using RANKED omega values for analysis.")
        df['analysis_var'] = df['omega_value'].rank(method='average')
    else:
        print(f"[{datetime.now()}] Using RAW omega values for analysis.")
        df['analysis_var'] = df['omega_value']

    # --- 4. Initialize Results & Validate Data for Model ---
    effect_size, p_value, std_err = np.nan, np.nan, np.nan
    failure_reason = None
    pc_corrected = False # Initialize here

    # Check for sufficient data variance and groups *before* complex steps
    if df.empty or df['group'].nunique() < 2 or df['analysis_var'].nunique() < 2:
        if df.empty: failure_reason = "No valid pairwise comparisons found after initial prep"
        elif df['group'].nunique() < 2: failure_reason = "Missing one of the comparison groups (0 or 1)"
        elif df['analysis_var'].nunique() < 2:
             print("DEBUG: Data with low variation:")
             print(df[['omega_value', 'group', 'analysis_var']].head())
             failure_reason = f"Not enough variation in '{'ranked_omega' if USE_RANKED_OMEGA_ANALYSIS else 'omega_value'}' for statistical analysis (nunique={df['analysis_var'].nunique()})"
        print(f"[{datetime.now()}] WARNING: {failure_reason}")
        return {
            'effect_size': np.nan, 'p_value': np.nan, 'n0': n0, 'n1': n1,
            'num_comp_group_0': (df['group'] == 0).sum() if not df.empty else 0,
            'num_comp_group_1': (df['group'] == 1).sum() if not df.empty else 0,
            'std_err': np.nan, 'failure_reason': failure_reason, 'pc_corrected': False
        }

    # --- 5. Prepare Sequence Codes for Random Effects ---
    print(f"[{datetime.now()}] Preparing sequence codes for random effects...")
    all_unique_seqs = pd.unique(pd.concat([df['seq1'], df['seq2']]))
    seq_to_code = {seq: i for i, seq in enumerate(all_unique_seqs)}
    df['seq1_code'] = df['seq1'].map(seq_to_code).astype('category') # Using category might help statsmodels slightly
    df['seq2_code'] = df['seq2'].map(seq_to_code).astype('category')
    print(f"[{datetime.now()}] Codes prepared for {len(all_unique_seqs)} unique sequences.")

    # --- 6. Prepare Model Formula (Fixed Effects + Optional PCs) ---
    print(f"[{datetime.now()}] Preparing model formula...")
    # Base fixed effects: intercept is implicit, test group 1 vs group 0 (ref), include group 2
    df['is_group1'] = (df['group'] == 1).astype(int)
    df['is_group2'] = (df['group'] == 2).astype(int) # Cross-group comparisons
    fixed_effects = ['is_group1', 'is_group2']
    all_pc_cols = []

    # --- 6a. PC Addition ---
    should_apply_pc = enable_pc_correction and pc_data is not None and chromosome in pc_data and pc_data[chromosome]

    if should_apply_pc:
        print(f"[{datetime.now()}] Applying PC correction for chromosome {chromosome} using optimized merge...")
        try:
            # Convert PC data for this chromosome into a DataFrame
            pc_dict_chrom = pc_data[chromosome]
            pc_df = pd.DataFrame.from_dict(pc_dict_chrom, orient='index')
            # Rename columns to PC1, PC2, ...
            pc_df.columns = [f"PC{i+1}" for i in range(len(pc_df.columns))]
            # Select only the required number of PCs
            pc_cols_base = [f"PC{i+1}" for i in range(NUM_PCS_TO_USE)]
            if not all(pc in pc_df.columns for pc in pc_cols_base):
                 print(f"Warning: Requested {NUM_PCS_TO_USE} PCs, but only {len(pc_df.columns)} available in data for chr {chromosome}.")
                 pc_cols_base = [pc for pc in pc_cols_base if pc in pc_df.columns] # Use available PCs
            pc_df = pc_df[pc_cols_base]

            # Define final PC column names
            pc_cols_s1 = [f"{pc}_s1" for pc in pc_cols_base]
            pc_cols_s2 = [f"{pc}_s2" for pc in pc_cols_base]
            all_pc_cols = pc_cols_s1 + pc_cols_s2

            # Merge for seq1
            df = pd.merge(df, pc_df, left_on='seq1', right_index=True, how='left', suffixes=('', '_s1_temp'))
            rename_dict_s1 = {old: new for old, new in zip(pc_cols_base, pc_cols_s1)}
            df.rename(columns=rename_dict_s1, inplace=True)

            # Merge for seq2
            df = pd.merge(df, pc_df, left_on='seq2', right_index=True, how='left', suffixes=('', '_s2_temp'))
            rename_dict_s2 = {old: new for old, new in zip(pc_cols_base, pc_cols_s2)}
            df.rename(columns=rename_dict_s2, inplace=True)

            # Fill NaNs introduced by merge (sequences missing PC data) with 0.0
            df[all_pc_cols] = df[all_pc_cols].fillna(0.0)

            fixed_effects.extend(all_pc_cols)
            pc_corrected = True
            print(f"[{datetime.now()}] Added {len(all_pc_cols)} PC covariates via merge.")

        except Exception as e:
            print(f"[{datetime.now()}] ERROR during PC data merge for chr {chromosome}: {e}. Proceeding without PC correction.")
            pc_corrected = False
            # Ensure fixed_effects only contains base group indicators if PC merge failed
            fixed_effects = ['is_group1', 'is_group2']
    else:
        # Log why PC wasn't applied
        if not enable_pc_correction: reason = "PC correction disabled globally"
        elif pc_data is None: reason = "No PC data loaded"
        elif chromosome not in pc_data: reason = f"No PC data available for chromosome {chromosome}"
        elif not pc_data[chromosome]: reason = f"PC data for chromosome {chromosome} is empty"
        else: reason = "PC check logic error (unexpected)"
        print(f"[{datetime.now()}] Skipping PC correction: {reason}.")

    # Construct the final formula string
    formula = f"analysis_var ~ {' + '.join(fixed_effects)}"

    # --- 7. Define and Fit the Mixed Model ---
    # Random effects structure (crossed effects for seq1 and seq2)
    vc_formula = {'seq1': '0 + C(seq1_code)', 'seq2': '0 + C(seq2_code)'}

    print(f"[{datetime.now()}] Preparing to fit MixedLM model.")
    print(f"  Data dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Unique sequences (random effects levels): {len(all_unique_seqs)}")
    print(f"  Formula: {formula}")
    print(f"  VC Formula: {vc_formula}")

    fit_start_time = datetime.now()
    print(f"[{fit_start_time}] >>> Starting model.fit()...")
    try:
        model = MixedLM.from_formula(
            formula=formula,
            groups=np.ones(len(df)), # Dummy group for crossed effects in statsmodels
            vc_formula=vc_formula,
            re_formula='0', # Indicates only variance components, no separate random intercepts
            data=df
        )
        # Consider adding method='cg' or others if default ('lbfgs') is slow/fails
        # result = model.fit(reml=False, method='lbfgs', maxiter=100) # Example with options
        result = model.fit(reml=False) # Use default first

        fit_end_time = datetime.now()
        print(f"[{fit_end_time}] <<< Finished model.fit() (Duration: {fit_end_time - fit_start_time}).")

        # --- 8. Extract Results ---
        print(f"[{datetime.now()}] Extracting results from fitted model.")
        # Use .get() with default np.nan for robustness if a term wasn't estimated
        effect_size = result.params.get('is_group1', np.nan)
        p_value = result.pvalues.get('is_group1', np.nan)
        std_err = result.bse.get('is_group1', np.nan)

        # Sanity check for NaN results which might indicate fit issues
        if pd.isna(p_value) and pd.isna(effect_size):
             failure_reason = "Model fit succeeded but key results (p_value, effect_size) are NaN."
             print(f"[{datetime.now()}] WARNING: {failure_reason}")
             # Optionally print result.summary() here for debugging
             # print(result.summary())

    except Exception as e:
        fit_fail_time = datetime.now()
        print(f"[{fit_fail_time}] !!! FAILED model.fit() (Duration before fail: {fit_fail_time - fit_start_time}).")
        failure_reason = f"StatsModels MixedLM Error: {str(e)[:150]}..." # Limit error length
        print(f"  Error details: {e}")
        # Reset results to NaN if fit failed
        effect_size, p_value, std_err = np.nan, np.nan, np.nan

    # --- 9. Return Final Dictionary ---
    final_result = {
        'effect_size': effect_size,
        'p_value': p_value,
        'n0': n0,
        'n1': n1,
        'std_err': std_err,
        'num_comp_group_0': (df['group'] == 0).sum(),
        'num_comp_group_1': (df['group'] == 1).sum(),
        'failure_reason': failure_reason,
        'pc_corrected': pc_corrected # Use the flag set during PC processing
    }
    func_end_time = datetime.now()
    print(f"[{func_end_time}] Worker finished for chromosome {chromosome} (Total time: {func_end_time - func_start_time}).")
    return final_result

def analyze_transcript(args):
    """
    Analyze evolutionary rates for a specific transcript.
    
    This function processes all pairwise comparisons for a single transcript,
    organizing sequences by group, performing statistical analysis, retrieving
    gene annotations, and controlling for population structure via PCs.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing:
        - df_transcript: DataFrame subset for this transcript
        - transcript_id: Identifier of the transcript being analyzed
        - pc_data: Dictionary of PC values by chromosome and sample
        
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
        - pc_corrected: Boolean indicating if PC correction was applied
        
    Note:
    -----
    - Creates pairwise dictionary of omega values for statistical analysis
    - Identifies gene annotations using UCSC and MyGene.info APIs
    - Delegates statistical analysis to analysis_worker function
    - Controls for population structure using PCA data when available
    """
    df_transcript, transcript_id, pc_data = args

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

    # Get gene information directly from transcript ID
    gene_symbol, gene_name = get_gene_info_from_transcript(transcript_id)
    
    # Get chromosome for this transcript to match with PCA data
    chromosome = df_transcript['chromosome'].iloc[0] if not df_transcript.empty else None

    # --- Apply Filter Logic (if global flag is True) ---
    perform_main_analysis = True # Default: perform analysis
    skip_reason = None
    # Calculate cross-group median omega
    cross_group_df = df_transcript[df_transcript['group'] == 2]
    num_comp_cross_group = len(cross_group_df)
    median_cross_group_omega = np.nan
    if not cross_group_df.empty:
        median_cross_group_omega = cross_group_df['omega'].median() # Includes -1, 99

    if FILTER_ON_CROSS_GROUP_OMEGA:
        if num_comp_cross_group == 0:
            perform_main_analysis = False
            skip_reason = 'Skipped (Filter Active): No cross-group comparisons'
        elif pd.isna(median_cross_group_omega): # Handles edge case if all cross-group omegas were NaN initially
             perform_main_analysis = False
             skip_reason = 'Skipped (Filter Active): Median cross-group omega is NaN'
        elif median_cross_group_omega <= 0.0:
            perform_main_analysis = False
            skip_reason = f'Skipped (Filter Active): Median cross-group omega ({median_cross_group_omega:.4f}) <= 0.0'

        if not perform_main_analysis:
             print(f"  {skip_reason} for {transcript_id}.")
        else:
             # Only print proceed message if filter was active and passed
             print(f"  Proceeding with main analysis for {transcript_id}: Median cross-group omega ({median_cross_group_omega:.4f}) > 0.0.")

    # --- Perform Main Statistical Analysis (if not skipped) ---
    if perform_main_analysis:
        # Call the worker function for the main statistical test
        analysis_result = analysis_worker((all_sequences, pairwise_dict, sequences_0, sequences_1,
                                          chromosome, pc_data, ENABLE_PC_CORRECTION))
    else:
        # If skipped, create a placeholder analysis_result dictionary with NaNs and the skip reason
        analysis_result = {
            'effect_size': np.nan, 'p_value': np.nan, 'std_err': np.nan,
            'failure_reason': skip_reason, # Use the reason determined by the filter
            'pc_corrected': False,
            # Ensure keys expected downstream exist, even if NaN
             'num_comp_group_0': len(group_0_df), # Use counts calculated earlier
             'num_comp_group_1': len(group_1_df)
        }


    # Compute normal-only median and mean for each group (excluding -1 and 99) - for reporting
    group_0_normal = group_0_df[(group_0_df['omega'] > -1) & (group_0_df['omega'] < 99)]
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
            'chromosome': chromosome,
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
            'pc_corrected': analysis_result.get('pc_corrected', False),
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
            'pct_nosyn_1': pct_nosyn_1,
            'num_comp_cross_group': num_comp_cross_group,
            'median_cross_group_omega': median_cross_group_omega
        }

    # Perform omega category analysis if enabled
    if PERFORM_OMEGA_CATEGORY_ANALYSIS:
        category_result = analyze_omega_categories(group_0_df, group_1_df)
        
        # Add category analysis results to the output
        result.update({
            'category_p_value': category_result['p_value'],
            'corrected_category_p_value': np.nan,  # Will be filled in later during multiple testing correction
            'category_test': category_result['test_used'],
            'median_values': category_result['median_values'],
            'category_difference': category_result['dominant_difference']
        })
    return result


def main():
    """
    Main execution function for evolutionary rate analysis.
    
    This function orchestrates the entire analysis workflow:
    1. Reading and preprocessing input data
    2. Loading PCA data for population structure control
    3. Organizing analysis by transcript
    4. Performing parallel analysis across transcripts with PC correction
    5. Applying multiple testing correction
    6. Generating summary statistics and reports
    7. Saving results to CSV
    
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
    - Controls for population structure using PCA data
    - Applies correction for multiple hypothesis testing
    - Outputs both comprehensive and significant result summaries
    - Tracks and reports analysis runtime
    """
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")

    # Load PCA data if enabled
    pc_data = None
    if ENABLE_PC_CORRECTION:
        pc_data = load_pca_data(PCA_FOLDER, NUM_PCS_TO_USE)

    # Read and preprocess the input dataset
    df = read_and_preprocess_data('all_pairwise_results.csv')
    
    # Group the data by transcript for independent analysis
    transcript_groups = df.groupby('transcript_id')
    print(f"\nFound {len(transcript_groups)} unique transcripts")

    # Prepare arguments for parallel processing of transcripts
    transcript_args = [(transcript_group, transcript_id, pc_data) for transcript_id, transcript_group in transcript_groups]
    
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
    # For main analysis p-values
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
        # If no valid tests were performed (num_valid_tests is 0), set corrected p-value to NaN for all
        results_df['corrected_p_value'] = np.nan
    
    # Apply the same procedure to category p-values if category analysis was performed
    if PERFORM_OMEGA_CATEGORY_ANALYSIS:
        valid_cat_results = results_df[results_df['category_p_value'].notna() & (results_df['category_p_value'] > 0)]
        num_valid_cat_tests = len(valid_cat_results)
        
        if num_valid_cat_tests > 0:
            # Sort category p-values
            valid_cat_results = valid_cat_results.sort_values('category_p_value').copy()
            
            # Calculate ranks for category p-values
            valid_cat_results['cat_rank'] = np.arange(1, len(valid_cat_results) + 1)
            
            # Calculate BH adjusted category p-values
            valid_cat_results['corrected_category_p_value'] = valid_cat_results['category_p_value'] * num_valid_cat_tests / valid_cat_results['cat_rank']
            
            # Monotonicity of category p-values (step-up procedure)
            valid_cat_results['corrected_category_p_value'] = valid_cat_results['corrected_category_p_value'].iloc[::-1].cummin().iloc[::-1]
            
            # Create a mapping from transcript_id to adjusted category p-value
            corrected_cat_p_value_map = dict(zip(valid_cat_results['transcript_id'], valid_cat_results['corrected_category_p_value']))
            
            # Map adjusted category p-values back to the original dataframe
            results_df['corrected_category_p_value'] = results_df['transcript_id'].map(corrected_cat_p_value_map)
            
            # Cap adjusted category p-values at 1.0
            results_df['corrected_category_p_value'] = results_df['corrected_category_p_value'].clip(upper=1.0)
        else:
            results_df['corrected_category_p_value'] = results_df['category_p_value']
    
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
    
    # Summarize significant results after multiple testing correction for the main analysis
    significant_count = (results_df['corrected_p_value'] < 0.05).sum()
    # Use num_valid_tests calculated during BH correction
    print(f"\nSignificant results (main analysis) after correction (corrected p < 0.05): {significant_count} out of {num_valid_tests} tested transcripts")
    
    # Summarize filtering impact based on the global flag
    print(f"Filter Status (FILTER_ON_CROSS_GROUP_OMEGA): {'ENABLED' if FILTER_ON_CROSS_GROUP_OMEGA else 'DISABLED'}")
    if FILTER_ON_CROSS_GROUP_OMEGA:
        # Count skips specifically due to the filter logic
        skipped_filter_count = (results_df['failure_reason'].str.startswith('Skipped (Filter Active)', na=False)).sum()
        total_transcripts_initial = len(results_df) # Total transcripts initially processed
        print(f"  Transcripts skipped by cross-group omega filter: {skipped_filter_count} out of {total_transcripts_initial}")
    
    # Summarize significant results from category analysis
    if PERFORM_OMEGA_CATEGORY_ANALYSIS:
        print("\n\n=== Conservation Analysis Results ===")
        print("Comparing evolutionary conservation between groups (Low vs High omega ratio)")
        
        cat_significant_count = (results_df['category_p_value'] < 0.05).sum()
        cat_significant_corrected_count = (results_df['corrected_category_p_value'] < 0.05).sum()
        print(f"Significant conservation differences (raw p < 0.05): {cat_significant_count}")
        print(f"Significant conservation differences after correction (corrected p < 0.05): {cat_significant_corrected_count}")
        
        # Show breakdown of conservation patterns after correction
        if cat_significant_corrected_count > 0:
            cat_results = results_df[results_df['corrected_category_p_value'] < 0.05]
            pattern_counts = cat_results['category_difference'].value_counts()
            
            print("\nConservation patterns (after multiple testing correction):")
            for pattern, count in pattern_counts.items():
                print(f"  {pattern}: {count} genes")
            
            # Print simple list of significant hits
            print("\nSignificant conservation differences (after correction):")
            print(f"{'Gene':<15} {'Raw P':<15} {'Corrected P':<15} {'Median Grp0':<12} {'Median Grp1':<12} {'Pattern':<30}")
            print("-" * 95)
            for _, row in cat_results.sort_values('corrected_category_p_value').iterrows():
                gene_name = row['gene_symbol'] if pd.notna(row['gene_symbol']) else "Unknown"
                cat_p_val = f"{row['category_p_value']:.6e}" if pd.notna(row['category_p_value']) else "N/A"
                cat_corr_p_val = f"{row['corrected_category_p_value']:.6e}" if pd.notna(row['corrected_category_p_value']) else "N/A"
                cat_diff = row.get('category_difference', 'N/A')
                
                # Get median values if available
                median_0 = median_1 = "N/A"
                if (row.get('median_values') is not None and 
                    'group_0' in row['median_values'] and 
                    'group_1' in row['median_values']):
                    median_0 = f"{row['median_values']['group_0']:.4f}" if pd.notna(row['median_values']['group_0']) else "N/A"
                    median_1 = f"{row['median_values']['group_1']:.4f}" if pd.notna(row['median_values']['group_1']) else "N/A"
                
                print(f"{gene_name:<15} {cat_p_val:<15} {cat_corr_p_val:<15} {median_0:<12} {median_1:<12} {cat_diff:<30}")
    
    # Print detailed information for significant results
    if significant_count > 0:
        print("\nSignificant results after correction:")
        print(
            f"{'Chrom':<10} "
            f"{'P-value':<15} "
            f"{'Corrected P':<15} "
            f"{'Effect Size':<15} "
            f"{'PC Corrected':<12} "
            f"{'Median_0':<10} "
            f"{'Mean_0':<10} "
            f"{'Median_1':<10} "
            f"{'Mean_1':<10} "
            f"{'Pct_id_0':<10} "
            f"{'Pct_noSyn_0':<10} "
            f"{'Pct_id_1':<10} "
            f"{'Pct_noSyn_1':<10} "
            f"{'Gene':<15} "
        )
        print("-" * 160)

        # Select and sort significant results
        sig_results = results_df[results_df['corrected_p_value'] < 0.05].sort_values('p_value')
        
        # Print each significant result with detailed information
        for _, row in sig_results.iterrows():
            coords_str = str(row['coordinates']) if 'coordinates' in row and pd.notna(row['coordinates']) else ""
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

            pct_id_0_val = row['pct_identical_0']
            pct_noSyn_0_val = row['pct_nosyn_0']
            pct_id_1_val = row['pct_identical_1']
            pct_noSyn_1_val = row['pct_nosyn_1']
            pct_id_0_str = f"{pct_id_0_val:.2f}%" if not pd.isna(pct_id_0_val) else "N/A"
            pct_noSyn_0_str = f"{pct_noSyn_0_val:.2f}%" if not pd.isna(pct_noSyn_0_val) else "N/A"
            pct_id_1_str = f"{pct_id_1_val:.2f}%" if not pd.isna(pct_id_1_val) else "N/A"
            pct_noSyn_1_str = f"{pct_noSyn_1_val:.2f}%" if not pd.isna(pct_noSyn_1_val) else "N/A"

            chrom_str = ""
            if pd.notna(coords_str):
                chrom_str = coords_str.split(":")[0]

            gene_info = ""
            if 'gene_symbol' in row and pd.notna(row['gene_symbol']) and 'gene_name' in row and pd.notna(row['gene_name']):
                gene_info = f"{row['gene_symbol']}: {row['gene_name']}"
            gene_info = gene_info[:40]
            
            # Add PC correction indicator
            pc_corrected = row.get('pc_corrected', False)
            pc_str = "Yes" if pc_corrected else "No"
            
            print(
                f"{chrom_str:<10} "
                f"{p_value:<15} "
                f"{corrected_p:<15} "
                f"{effect_size:<15} "
                f"{pc_str:<12} "
                f"{median_0_str:<10} "
                f"{mean_0_str:<10} "
                f"{median_1_str:<10} "
                f"{mean_1_str:<10} "
                f"{pct_id_0_str:<10} "
                f"{pct_noSyn_0_str:<10} "
                f"{pct_id_1_str:<10} "
                f"{pct_noSyn_1_str:<10} "
                f"{gene_info:<15}"
            )
                 
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
    
    # Print summary of PC correction usage
    if ENABLE_PC_CORRECTION:
        pc_corrected_count = results_df['pc_corrected'].sum() if 'pc_corrected' in results_df.columns else 0
        total_transcripts = len(results_df)
        
        # Count transcripts with sufficient data for analysis
        transcripts_with_data = results_df[results_df['failure_reason'].isna()].shape[0]
        transcripts_with_insufficient_data = results_df[results_df['failure_reason'].str.contains('Insufficient sequences', na=False)].shape[0] if 'failure_reason' in results_df.columns else 0
        
        print(f"\nPopulation structure correction summary:")
        print(f"  - PC correction enabled: {ENABLE_PC_CORRECTION}")
        print(f"  - Transcripts with sufficient data for analysis: {transcripts_with_data}/{total_transcripts} ({transcripts_with_data/total_transcripts*100:.1f}%)")
        print(f"  - Transcripts with insufficient sequence data: {transcripts_with_insufficient_data}/{total_transcripts} ({transcripts_with_insufficient_data/total_transcripts*100:.1f}%)")
        print(f"  - Transcripts with PC correction applied: {pc_corrected_count}/{transcripts_with_data} ({pc_corrected_count/transcripts_with_data*100:.1f}% of valid transcripts)")
        print(f"  - Number of PCs used per sequence: {NUM_PCS_TO_USE} (total of {NUM_PCS_TO_USE*2} PC covariates in the model)")
        
        # Calculate how PC correction affected significance
        if pc_corrected_count > 0:
            pc_corrected_df = results_df[results_df['pc_corrected'] == True]
            pc_sig_count = (pc_corrected_df['corrected_p_value'] < 0.05).sum()
            pc_sig_pct = pc_sig_count / len(pc_corrected_df) * 100 if len(pc_corrected_df) > 0 else 0
            
            non_pc_df = results_df[results_df['pc_corrected'] == False]
            non_pc_sig_count = (non_pc_df['corrected_p_value'] < 0.05).sum()
            non_pc_sig_pct = non_pc_sig_count / len(non_pc_df) * 100 if len(non_pc_df) > 0 else 0
            
            print(f"  - Significant results with PC correction: {pc_sig_count}/{len(pc_corrected_df)} ({pc_sig_pct:.1f}%)")

    # Print completion information and runtime
    print(f"\nAnalysis completed at {datetime.now()}")
    print(f"Total runtime: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
