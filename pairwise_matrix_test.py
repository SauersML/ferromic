import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
import warnings
import os
import json
from datetime import datetime
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import combine_pvalues
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as mpatches
import requests
from urllib.parse import urlencode
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Create necessary directories
for directory in [CACHE_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(exist_ok=True)

def get_cache_path(cds):
    """Generate cache file path for a CDS."""
    return CACHE_DIR / f"{cds.replace('/', '_')}.pkl"

def load_cached_result(cds):
    """Load cached result for a CDS if it exists."""
    cache_path = get_cache_path(cds)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_result = pickle.load(f)
            return cached_result
        except:
            return None
    return None

def save_cached_result(cds, result):
    """Save result for a CDS to cache."""
    cache_path = get_cache_path(cds)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

def read_and_preprocess_data(file_path):
    """Read and preprocess the CSV file."""
    print("Reading data...")
    df = pd.read_csv(file_path)

    # Convert CDS name from chrX_startY_endZ to chrX:Y-Z
    df['CDS'] = df['CDS'].str.replace('_start', ':', regex=False)
    df['CDS'] = df['CDS'].str.replace('_end', '-', regex=False)

    # Convert omega to numeric, coerce non-numeric to NaN
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')
    
    # Now filter valid omega values
    df = df[
        (df['omega'] != -1) &
        (df['omega'] != 99)
    ].dropna(subset=['omega'])


    print(f"Total valid comparisons: {len(df):,}")
    print(f"Unique CDSs found: {df['CDS'].nunique():,}")
    return df

def get_pairwise_value(seq1, seq2, pairwise_dict):
    """Get omega value for a pair of sequences."""
    key = (seq1, seq2) if (seq1, seq2) in pairwise_dict else (seq2, seq1)
    val = pairwise_dict.get(key)
    #if val is None:
        #print(f"\n=== DEBUG: Failed pairwise lookup ===")
        #print(f"Tried keys: {(seq1, seq2)}, {(seq2, seq1)}")
        #print(f"Key type attempted: {type((seq1, seq2))}")
        #print(f"Sample dict key type: {type(list(pairwise_dict.keys())[0])}")
    return val

def create_matrices(sequences_0, sequences_1, pairwise_dict):
    """Create matrices for two groups based on sequence assignments."""
    print("\n=== DEBUG: create_matrices ===")
    print(f"Number of sequences: Group 0={len(sequences_0)}, Group 1={len(sequences_1)}")
    print("Sample sequences_0:", sequences_0[:3])
    print("Sample sequences_1:", sequences_1[:3])
    print("Sample pairwise_dict keys:", list(pairwise_dict.keys())[:3])
    print("Sample pairwise_dict values:", list(pairwise_dict.values())[:3])

    if len(sequences_0) == 0 or len(sequences_1) == 0:
        return None, None

    n0, n1 = len(sequences_0), len(sequences_1)
    matrix_0 = np.full((n0, n0), np.nan)
    matrix_1 = np.full((n1, n1), np.nan)

    # Fill matrix 0
    for i in range(n0):
        for j in range(i + 1, n0):
            val = get_pairwise_value(sequences_0[i], sequences_0[j], pairwise_dict)
            if val is not None:
                matrix_0[i, j] = matrix_0[j, i] = float(val)

    # Fill matrix 1
    for i in range(n1):
        for j in range(i + 1, n1):
            val = get_pairwise_value(sequences_1[i], sequences_1[j], pairwise_dict)
            if val is not None:
                matrix_1[i, j] = matrix_1[j, i] = float(val)

    return matrix_0, matrix_1

def analysis_worker(args):
    """Mixed effects analysis for a single CDS with crossed random effects."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    import numpy as np

    all_sequences, n0, pairwise_dict, sequences_0, sequences_1 = args

    # Prepare data
    data = []
    for (seq1, seq2), omega in pairwise_dict.items():
        # Determine group of the pair
        if seq1 in sequences_0 and seq2 in sequences_0:
            group = 0
        elif seq1 in sequences_1 and seq2 in sequences_1:
            group = 1
        else:
            continue  # Skip pairs not within the same group

        data.append({
            'omega_value': omega,
            'group': group,
            'seq1': seq1,
            'seq2': seq2
        })

    df = pd.DataFrame(data)

    # Initialize variables
    effect_size = np.nan
    p_value = np.nan
    std_err = np.nan

    # Check if DataFrame has sufficient data
    if df.empty or df['group'].nunique() < 2 or df['omega_value'].nunique() < 2:
        return {
            'observed_effect_size': effect_size,
            'p_value': p_value,
            'n0': n0,
            'n1': len(all_sequences) - n0,
            'num_comp_group_0': (df['group'] == 0).sum() if not df.empty else 0,
            'num_comp_group_1': (df['group'] == 1).sum() if not df.empty else 0,
            'std_err': std_err
        }

    # Convert sequences to categorical codes
    df['seq1_code'] = pd.Categorical(df['seq1']).codes
    df['seq2_code'] = pd.Categorical(df['seq2']).codes

    # Print diagnostic info
    print(f"\nAnalyzing CDS with:")
    print(f"Data shape: {df.shape}")
    print(f"Unique seq1_codes: {len(df['seq1_code'].unique())}")
    print(f"Unique seq2_codes: {len(df['seq2_code'].unique())}")
    print(f"Group counts:\n{df['group'].value_counts()}")

    try:
        # Prepare data for MixedLM
        # Introduce a dummy 'groups' variable since we'll specify random effects via 'vc_formula'
        df['groups'] = 1  # All data belongs to the same group for variance components

        # Define variance components for crossed random effects
        vc = {
            'seq1': '0 + C(seq1_code)',
            'seq2': '0 + C(seq2_code)'
        }

        # Fit the mixed model using MixedLM.from_formula
        model = MixedLM.from_formula(
            'omega_value ~ group',
            groups='groups',
            vc_formula=vc,
            re_formula='0',  # No random intercept for 'groups' since it's a dummy
            data=df
        )
        result = model.fit(reml=False)  # Use ML estimation

        # Extract results
        effect_size = result.fe_params['group']
        p_value = result.pvalues['group']
        std_err = result.bse['group']

        print(f"Successfully fit model:")
        print(f"Effect size: {effect_size:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"Std error: {std_err:.4f}")

    except Exception as e:
        print(f"Model fitting failed with error: {str(e)}")
        # Variables remain as initialized (np.nan)

    return {
        'observed_effect_size': effect_size,
        'p_value': p_value,
        'n0': n0,
        'n1': len(all_sequences) - n0,
        'std_err': std_err,
        'num_comp_group_0': (df['group'] == 0).sum(),
        'num_comp_group_1': (df['group'] == 1).sum(),
    }

def compute_cliffs_delta(x, y):
    """Compute Cliff's Delta effect size."""
    n_x = len(x)
    n_y = len(y)
    n_pairs = n_x * n_y

    # Efficient computation using broadcasting
    x = np.array(x)
    y = np.array(y)

    difference_matrix = x[:, np.newaxis] - y
    num_greater = np.sum(difference_matrix > 0)
    num_less = np.sum(difference_matrix < 0)

    cliffs_delta = (num_greater - num_less) / n_pairs

    return cliffs_delta




def get_gene_info(gene_symbol):
    """Get human-readable gene info from MyGene.info using gene symbol"""
    try:
        # Query by symbol to get gene info
        url = f"http://mygene.info/v3/query?q=symbol:{gene_symbol}&species=human&fields=name"
        print(f"\nQuerying: {url}")  # Debug print
        response = requests.get(url, timeout=10)
        print(f"Response status: {response.status_code}")  # Debug print
        if response.ok:
            data = response.json()
            print(f"Raw response: {response.text}")  # Debug print
            if data.get('hits') and len(data['hits']) > 0:
                return data['hits'][0].get('name', 'Unknown')
    except Exception as e:
        print(f"Error fetching gene info: {str(e)}")
    return 'Unknown'




def get_gene_annotation(cds, cache_file='gene_name_cache.json'):
    """
    Get gene annotation for a CDS with caching and detailed error reporting
    Returns (gene_symbol, gene_name, error_log) where error_log contains any warnings/errors
    Uses UCSC API to look up genes that overlap with the given coordinates
    """
    error_log = []
    
    # Load cache if it exists
    cache = {}
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
    except json.JSONDecodeError as e:
        error_log.append(f"WARNING: Cache file corrupt or invalid JSON: {str(e)}")
    except Exception as e:
        error_log.append(f"WARNING: Failed to load cache file: {str(e)}")

    # Check cache first
    if cds in cache:
        error_log.append(f"INFO: Found entry in cache for {cds}")
        return cache[cds]['symbol'], cache[cds]['name'], error_log

    def parse_coords(coord_str):
        """Parse coordinate string into chr, start, end"""
        if not coord_str:
            return None, "ERROR: Empty coordinate string provided"
            
        try:
            parts = coord_str.split(':')
            if len(parts) != 2:
                return None, "ERROR: Invalid coordinate format - missing ':'"
                
            chr = parts[0]
            start_end = parts[1].split('-')
            if len(start_end) != 2:
                return None, "ERROR: Invalid coordinate format - missing '-'"
                
            start = int(start_end[0])
            end = int(start_end[1])
            
            if start > end:
                return None, f"ERROR: Invalid coordinates - start ({start}) greater than end ({end})"
                
            return {'chr': chr, 'start': start, 'end': end}, None
            
        except ValueError as e:
            return None, f"ERROR: Failed to parse coordinates - invalid numbers: {str(e)}"
        except Exception as e:
            return None, f"ERROR: Failed to parse coordinates: {str(e)}"

    def query_ucsc(chr, start, end):
        """Query UCSC API for genes at location"""
        base_url = "https://api.genome.ucsc.edu/getData/track"
        params = {
            'genome': 'hg38',
            'track': 'knownGene',
            'chrom': chr,
            'start': start,
            'end': end
        }
        
        try:
            url = f"{base_url}?{urlencode(params)}"
            print(f"\nQuerying UCSC API URL: {url}")  # Debug print
            response = requests.get(url, timeout=10)
            
            print(f"\nResponse status: {response.status_code}")  # Debug print
            #print(f"\nRaw API Response:\n{response.text}")  # Debug print
            
            if not response.ok:
                return None, f"ERROR: API request failed with status {response.status_code}: {response.text}"
                
            data = response.json()
            
            # Print the structure of the response
            print("\nResponse keys:", data.keys())
            
            if not data:
                return None, "ERROR: Empty response from API"
                
            # Look for the track data - handle both possible response structures
            track_data = None
            if 'knownGene' in data:
                track_data = data['knownGene']
            elif isinstance(data, list):
                track_data = data
                
            if not track_data:
                return None, "No gene data found in response"
            
            # Filter for genes that overlap our region
            overlapping_genes = []
            for gene in track_data:
                gene_start = gene.get('chromStart', 0)
                gene_end = gene.get('chromEnd', 0)
                
                # Check if our region falls within the gene's coordinates
                if gene_start <= end and gene_end >= start:
                    overlapping_genes.append(gene)
                    #print(f"\nFound overlapping gene: {gene}")  # Debug print
            
            if not overlapping_genes:
                return None, "No overlapping genes found"
                
            return overlapping_genes, None
            
        except requests.Timeout:
            return None, "ERROR: API request timed out"
        except requests.RequestException as e:
            return None, f"ERROR: API request failed: {str(e)}"
        except json.JSONDecodeError as e:
            return None, f"ERROR: Failed to parse API response: {str(e)}"
        except Exception as e:
            return None, f"ERROR: Unexpected error during API query: {str(e)}"

    # Parse coordinates
    loc, parse_error = parse_coords(cds)
    if parse_error:
        error_log.append(parse_error)
        return None, None, error_log
    
    # Query API
    genes, query_error = query_ucsc(loc['chr'], loc['start'], loc['end'])
    if query_error:
        error_log.append(query_error)
        return None, None, error_log

    if not genes:
        error_log.append(f"WARNING: No genes found for coordinates {loc['chr']}:{loc['start']}-{loc['end']}")
        return None, None, error_log

    # Get the most relevant gene (the one that best contains our region)
    best_gene = None
    best_overlap = 0
    region_length = loc['end'] - loc['start']
    
    for gene in genes:
        gene_start = gene.get('chromStart', 0)
        gene_end = gene.get('chromEnd', 0)
        
        # Calculate how much of our region is contained within this gene
        overlap_start = max(gene_start, loc['start'])
        overlap_end = min(gene_end, loc['end'])
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_gene = gene

    if not best_gene:
        error_log.append("WARNING: Could not determine best matching gene")
        return None, None, error_log

    # Extract gene information 
    symbol = best_gene.get('geneName')
    if symbol == 'none' or symbol.startswith('ENSG'):
        # Try to find another gene with a proper symbol
        for gene in genes:
            potential_symbol = gene.get('geneName')
            if potential_symbol != 'none' and not potential_symbol.startswith('ENSG'):
                symbol = potential_symbol
                break

    name = get_gene_info(symbol) # Returns full human readable name

    if symbol == 'Unknown':
        error_log.append("WARNING: No symbol found in gene data")
    if name == 'Unknown':
        error_log.append("WARNING: No name found in gene data")

    # Update cache
    try:
        cache[cds] = {'symbol': symbol, 'name': name}
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        error_log.append(f"WARNING: Failed to update cache file: {str(e)}")

    return symbol, name, error_log




def create_visualization(matrix_0, matrix_1, cds, result):
    # Retrieve gene annotation
    gene_symbol, gene_name, error_log = get_gene_annotation(cds)
    if error_log:
        print(f"\nWarnings/Errors for CDS {cds}:")
        for msg in error_log:
            if msg.startswith("ERROR"):
                print(f"🚫 {msg}")
            elif msg.startswith("WARNING"):
                print(f"⚠️ {msg}")
            else:
                print(f"ℹ️ {msg}")
    if gene_symbol in [None, 'Unknown'] or gene_name in [None, 'Unknown']:
        print(f"❌ No annotation found for CDS: {cds}")
        gene_symbol = None
        gene_name = None
    else:
        print(f"✅ Found annotation:")
        print(f"   Symbol: {gene_symbol}")
        print(f"   Name: {gene_name}")

    # Re-read all data for this CDS to retrieve special values too
    df_all = pd.read_csv('all_pairwise_results.csv')
    df_all['CDS'] = df_all['CDS'].str.replace('_start', ':', regex=False)
    df_all['CDS'] = df_all['CDS'].str.replace('_end', '-', regex=False)
    df_cds_all = df_all[df_all['CDS'] == cds]

    # Create pairwise dict including all omega values (including special values)
    pairwise_dict_all = {(row['Seq1'], row['Seq2']): row['omega']
                         for _, row in df_cds_all.iterrows()}

    # Determine sequences for each group
    sequences = pd.concat([df_cds_all['Seq1'], df_cds_all['Seq2']]).dropna().unique()
    sequences = [str(seq) for seq in sequences]
    sequences_direct = np.array([seq for seq in sequences if not seq.endswith('1')])
    sequences_inverted = np.array([seq for seq in sequences if seq.endswith('1')])

    # Recreate the full matrices (including special values)
    matrix_0_full, matrix_1_full = create_matrices(sequences_direct, sequences_inverted, pairwise_dict_all)
    if matrix_0_full is None or matrix_1_full is None:
        print(f"No data available for CDS: {cds}")
        return

    # Define colors for special values
    color_minus_one = (242/255, 235/255, 250/255)  # lavender for identical sequences (-1)
    color_ninety_nine = (1, 192/255, 192/255)      # light red for no non-syn variation (99)
    special_patches = [
        mpatches.Patch(color=color_minus_one, label='Identical sequences'),
        mpatches.Patch(color=color_ninety_nine, label='No non-synonymous variation')
    ]

    from matplotlib.colors import LogNorm
    cmap_normal = sns.color_palette("viridis", as_cmap=True)

    # Create a large figure so that text appears relatively smaller.
    # We'll use a 2-row layout: top row for matrices, bottom row for the histogram.
    plt.rcParams.update(plt.rcParamsDefault)
    fig = plt.figure(figsize=(20, 12))

    # GridSpec: 
    # Top row: 3 columns (matrix_0, matrix_1, colorbar)
    # Bottom row: 1 column spanning all three top columns for the histogram
    gs = plt.GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.05],
        height_ratios=[1, 0.4],
        hspace=0.3, wspace=0.3
    )

    # Main Title
    if gene_symbol and gene_name:
        title_str = f'{gene_symbol}: {gene_name}\n{cds}'
    else:
        title_str = f'Pairwise Comparison Analysis: {cds}'
    fig.suptitle(title_str, fontsize=24, fontweight='bold', y=0.98)

    def plot_matrices(ax, matrix, title_str):
        """
        Plot the given matrix:
        - Upper triangle: normal omega values (log scale colormap)
        - Lower triangle: special values (-1 and 99) with distinct colors
        """
        n = matrix.shape[0]
        upper_triangle = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        lower_triangle = np.tril(np.ones_like(matrix, dtype=bool), k=-1)

        special_minus_one = (matrix == -1)
        special_ninety_nine = (matrix == 99)
        normal_mask = (~np.isnan(matrix)) & (~special_minus_one) & (~special_ninety_nine)

        # NORMAL VALUES (upper triangle)
        normal_data = matrix.copy()
        normal_data[~(normal_mask & upper_triangle)] = np.nan
        normal_data = np.where(normal_data < 0.01, 0.01, normal_data)
        normal_data = np.where(normal_data > 50, 50, normal_data)
        normal_data_inv = normal_data[::-1, :]

        sns.heatmap(
            normal_data_inv, cmap=cmap_normal, norm=LogNorm(vmin=0.01, vmax=50),
            ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False,
            mask=np.isnan(normal_data_inv)
        )

        # SPECIAL VALUES (lower triangle)
        from matplotlib.colors import ListedColormap
        special_cmap = ListedColormap([color_minus_one, color_ninety_nine])
        special_data = np.full_like(matrix, np.nan, dtype=float)
        special_data[special_minus_one] = 0
        special_data[special_ninety_nine] = 1
        special_data[~lower_triangle] = np.nan
        special_data_inv = special_data[::-1, :]

        sns.heatmap(
            special_data_inv, cmap=special_cmap, ax=ax, cbar=False, square=True,
            xticklabels=False, yticklabels=False, mask=np.isnan(special_data_inv)
        )

        ax.set_title(title_str, fontsize=18, pad=15)
        ax.tick_params(axis='both', which='both', length=0)

    # Plot Direct matrix (top row, first column)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_matrices(ax1, matrix_0_full, f'Direct Sequences (n={len(sequences_direct)})')

    # Plot Inverted matrix (top row, second column)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_matrices(ax2, matrix_1_full, f'Inverted Sequences (n={len(sequences_inverted)})')

    # Add the colorbar in the top row, third column
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Create a ScalarMappable with a logarithmic scale color normalization
    sm = ScalarMappable(norm=LogNorm(vmin=0.01, vmax=50), cmap=cmap_normal)
    sm.set_array([])
    
    # Generate the colorbar without predefined ticks and labels
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    # Specify the exact ticks and corresponding labels
    desired_ticks = [0.01, 1, 3, 10, 50]
    desired_labels = ['0', '1', '3', '10', '50']

    # Apply the fixed ticks and labels to the colorbar
    cbar.ax.yaxis.set_major_locator(FixedLocator(desired_ticks))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(desired_labels))
    
    # Set the colorbar label and tick label size
    cbar.set_label('Omega Value', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    legend = ax1.legend(
        handles=special_patches,
        title='Special Values',
        loc='upper left',
        bbox_to_anchor=(0.0, -0.05),
        ncol=1, frameon=True, fontsize=10
    )
    legend.get_title().set_fontsize(10)


    # Extract normal omega values for distribution plot
    values_direct = matrix_0_full[np.tril_indices_from(matrix_0_full, k=-1)]
    values_inverted = matrix_1_full[np.tril_indices_from(matrix_1_full, k=-1)]
    values_direct = values_direct[~np.isnan(values_direct)]
    values_direct = values_direct[(values_direct != -1) & (values_direct != 99)]
    values_inverted = values_inverted[~np.isnan(values_inverted)]
    values_inverted = values_inverted[(values_inverted != -1) & (values_inverted != 99)]

    # Histogram (actually KDE plot) on the second row, spanning all columns
    ax3 = fig.add_subplot(gs[1, :])
    sns.kdeplot(values_direct, ax=ax3, label='Direct', fill=True, common_norm=False,
                color='#1f77b4', alpha=0.6)
    sns.kdeplot(values_inverted, ax=ax3, label='Inverted', fill=True, common_norm=False,
                color='#ff7f0e', alpha=0.6)
    ax3.set_title('Distribution of Omega Values', fontsize=20, pad=15)
    ax3.set_xlabel('Omega Value', fontsize=16)
    ax3.set_ylabel('Density', fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.legend(title='Groups', title_fontsize=16, fontsize=14)

    # Add P-value text inside the histogram subplot (top-right corner)
    p_value_val = result.get('p_value', np.nan)
    bonf_p_value_val = result.get('bonferroni_p_value', np.nan)
    bonf_p_value_str = f"Corrected p-value: {bonf_p_value_val:.4e}" if not np.isnan(bonf_p_value_val) else "P-value: N/A"
    ax3.text(
        0.97, 0.97, f"{bonf_p_value_str}",
        transform=ax3.transAxes,
        ha='right', va='top', fontsize=14,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Adjust layout to reduce clutter
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(PLOTS_DIR / f'analysis_{cds.replace("/", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)





def analyze_cds_parallel(args):
    """Analyze a single CDS"""
    df_cds, cds = args

    # Check cache first
    cached_result = load_cached_result(cds)
    if cached_result is not None:
        return cds, cached_result

    # Create pairwise dictionary
    pairwise_dict = {(row['Seq1'], row['Seq2']): row['omega']
                     for _, row in df_cds.iterrows()}

    # Collect all unique sequences from both 'Seq1' and 'Seq2' columns
    all_seqs = pd.concat([df_cds['Seq1'], df_cds['Seq2']]).unique()
    sequences_0 = np.array([seq for seq in all_seqs if not seq.endswith('1')])
    sequences_1 = np.array([seq for seq in all_seqs if seq.endswith('1')])
    all_sequences = np.concatenate([sequences_0, sequences_1])

    n0 = len(sequences_0)
    n1 = len(sequences_1)

    # Generate matrices for visualization (do this first)
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)

    # Initialize base result dictionary with matrices
    result = {
        'matrix_0': matrix_0,
        'matrix_1': matrix_1,
        'pairwise_comparisons': set(pairwise_dict.keys())
    }


    # If either matrix is None, it means one group had no sequences.
    if matrix_0 is None or matrix_1 is None:
        # Not enough data to proceed
        result.update({
            'observed_effect_size': np.nan,
            'p_value': np.nan,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': 0,
            'num_comp_group_1': 0,
            'std_err': np.nan
        })
        # Return early
        save_cached_result(cds, result)
        return cds, result
    
    valid_per_seq_group_0 = np.sum(~np.isnan(matrix_0), axis=1)
    valid_per_seq_group_1 = np.sum(~np.isnan(matrix_1), axis=1)

    # Set minimum required sequences per group
    min_sequences_per_group = 3

    if (n0 < min_sequences_per_group or 
        n1 < min_sequences_per_group or
        np.nansum(~np.isnan(matrix_0)) < 3 or 
        np.nansum(~np.isnan(matrix_1)) < 3 or
        not all(valid_per_seq_group_0 >= 5) or
        not all(valid_per_seq_group_1 >= 5)):
    
        # Not enough sequences/valid comparisons
        result.update({
            'observed_effect_size': np.nan,
            'p_value': np.nan,
            'n0': n0,
            'n1': n1,
            'num_comp_group_0': 0,
            'num_comp_group_1': 0,
            'std_err': np.nan
        })
    else:
        # Proceed with analysis
        worker_result = analysis_worker((
            all_sequences, n0, pairwise_dict,
            sequences_0, sequences_1
        ))
        result.update(worker_result)
    
    save_cached_result(cds, result)
    return cds, result


def parse_cds_coordinates(cds_name):
    """Extract chromosome and coordinates from CDS name."""
    try:
        # Try different possible formats
        if '/' in cds_name:  # If it's a path, take the last part
            cds_name = cds_name.split('/')[-1]

        if '_chr_' in cds_name:
            left, right = cds_name.split('_chr_')
            right = right.replace('_combined', '')
            chrom = 'chr' + right
            start = 0
            end = 0
            return chrom, start, end
        elif ':' in cds_name:
            chrom, coords = cds_name.split(':')
            start, end = map(int, coords.replace('-', '..').split('..'))
            return chrom, start, end

        # If parsing fails
        print(f"Failed to parse: {cds_name}")
        return None, None, None
    except Exception as e:
        print(f"Error parsing {cds_name}: {str(e)}")
        return None, None, None


CLUSTERS_CACHE_PATH = CACHE_DIR / "clusters.pkl"

def load_clusters_cache():
    if CLUSTERS_CACHE_PATH.exists():
        try:
            with open(CLUSTERS_CACHE_PATH, 'rb') as f:
                clusters = pickle.load(f)
            return clusters
        except Exception as e:
            print(f"WARNING: Failed to load clusters cache: {e}")
    return None

def save_clusters_cache(clusters):
    with open(CLUSTERS_CACHE_PATH, 'wb') as f:
        pickle.dump(clusters, f)

def build_overlap_clusters(results_df):
    """Build clusters of overlapping CDS regions."""
    cds_coords = []
    for cds in results_df['CDS']:
        chrom, start, end = parse_cds_coordinates(cds)
        if (chrom is not None) and (start is not None) and (end is not None):
            cds_coords.append((chrom, start, end, cds))

    cds_coords.sort(key=lambda x: (x[0], x[1]))

    clusters = {}
    cluster_id = 0
    active = []

    for chrom, start, end, cds in cds_coords:
        active = [(c, e, cid) for c, e, cid in active if not (c == chrom and e < start)]
        overlapping = {cid for c, e, cid in active if c == chrom and e >= start}
        if not overlapping:
            clusters[cluster_id] = {cds}
            active.append((chrom, end, cluster_id))
            cluster_id += 1
        else:
            target = min(overlapping)
            clusters[target].add(cds)
            merged = []
            for c, e, cid in active:
                if cid in overlapping and cid != target:
                    clusters[target].update(clusters[cid])
                    del clusters[cid]
                if cid in overlapping:
                    merged.append((c, e, target))
                else:
                    merged.append((c, e, cid))
            merged.append((chrom, end, target))
            active = merged

    # After merging, clusters dict contains the final clusters
    return clusters





def combine_cluster_evidence(cluster_cdss, results_df, results):
    """Select the single largest CDS by amino acid length from the cluster and use its statistics."""
    # Helper function to load GTF and compute amino acid lengths
    if not hasattr(combine_cluster_evidence, "_aa_length_map"):
        combine_cluster_evidence._aa_length_map = {}
        combine_cluster_evidence._interval_map = defaultdict(list)
        combine_cluster_evidence._chrom_strand = {}
        with open("hg38.knownGene.gtf", "r") as f:
            cds_lengths = defaultdict(int)
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 9:
                    continue
                feature = fields[2]
                if feature != "CDS":
                    continue
                chrom = fields[0]
                start = int(fields[3])
                end = int(fields[4])
                attrs = fields[8].strip().split(';')
                attr_dict = {}
                for a in attrs:
                    a = a.strip()
                    if a:
                        k,v = a.split(' ',1)
                        v = v.strip('"')
                        attr_dict[k] = v
                tid = attr_dict.get("transcript_id", None)
                if tid is None:
                    continue
                combine_cluster_evidence._interval_map[tid].append((start, end))
                combine_cluster_evidence._chrom_strand[tid] = (chrom, fields[6])
                cds_lengths[tid] += (end - start + 1)
            for t, length in cds_lengths.items():
                aa = length / 3.0
                combine_cluster_evidence._aa_length_map[t] = aa
    # Function to find transcript_id for a given CDS coordinate if possible.
    # We'll consider a transcript matches if its intervals cover this CDS fully.
    # This is WRONG. We must fix this later by adding transcript ID or similar to the results.
    def find_transcript_id_for_cds(cds_name):
        chrom, start, end = parse_cds_coordinates(cds_name)
        if chrom is None:
            return None
        best_tid = None
        best_len = -1
        for tid, intervals in combine_cluster_evidence._interval_map.items():
            t_chrom, _ = combine_cluster_evidence._chrom_strand[tid]
            if t_chrom != chrom:
                continue
            merged = []
            for s,e in sorted(intervals, key=lambda x:x[0]):
                if not merged or s > merged[-1][1]+1:
                    merged.append([s,e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            covered = False
            needed = start
            for ms,me in merged:
                if ms <= needed <= me:
                    if me >= end:
                        covered = True
                        break
                    else:
                        needed = me+1
                elif ms > needed:
                    break
            if covered:
                aa_len = combine_cluster_evidence._aa_length_map.get(tid, -1)
                if aa_len > best_len:
                    best_len = aa_len
                    best_tid = tid
        return best_tid
    # Compute the amino acid length for each CDS in the cluster
    best_cds = None
    best_len = -1
    for c in cluster_cdss:
        tid = find_transcript_id_for_cds(c)
        if tid is not None:
            aa_len = combine_cluster_evidence._aa_length_map.get(tid, -1)
            if aa_len > best_len:
                best_len = aa_len
                best_cds = c
    if best_cds is None or best_len <= 0:
        return {
            'combined_pvalue': np.nan,
            'weighted_effect_size': np.nan,
            'n_comparisons': 0,
            'n_valid_cds': 0,
            'cluster_pairs': set()
        }
    chosen = results_df[results_df['CDS'] == best_cds].dropna(subset=['observed_effect_size','p_value'])
    if len(chosen) == 0:
        return {
            'combined_pvalue': np.nan,
            'weighted_effect_size': np.nan,
            'n_comparisons': 0,
            'n_valid_cds': 0,
            'cluster_pairs': set()
        }
    row = chosen.iloc[0]
    pairs = results[best_cds]['pairwise_comparisons'] if best_cds in results else set()
    return {
        'combined_pvalue': row['p_value'],
        'weighted_effect_size': row['observed_effect_size'],
        'n_comparisons': len(pairs),
        'n_valid_cds': 1,
        'cluster_pairs': pairs
    }
    
def compute_overall_significance(cluster_results):
    """Compute overall significance from independent clusters using scipy's combine_pvalues."""
    import numpy as np
    from scipy import stats
    # Initialize default return values
    overall_pvalue_combined = np.nan
    overall_effect = np.nan
    n_valid_clusters = 0
    total_comparisons = 0
    # Filter out clusters with valid combined_pvalue and weighted_effect_size
    valid_clusters = [
        c for c in cluster_results.values()
        if not np.isnan(c['combined_pvalue']) and not np.isnan(c['weighted_effect_size'])
    ]
    if valid_clusters:
        cluster_pvals = np.array([c['combined_pvalue'] for c in valid_clusters])
        # Use scipy.stats.combine_pvalues to combine p-values
        # Choose method: 'fisher', 'stouffer', or others as appropriate etc.
        statistic, overall_pvalue_combined = stats.combine_pvalues(cluster_pvals, method='fisher')
        print(f"\nCombined p-value using Fisher's method: {overall_pvalue_combined:.4e}")
        print(f"Fisher's statistic: {statistic:.4f}")
        # Stouffer's method
        weights = np.array([c['n_comparisons'] for c in valid_clusters], dtype=float)
        # Check for zero weights
        if np.all(weights == 0) or np.isnan(weights).any():
            weights = None
            print("Note: Weights not used in Stouffer's method due to zero or NaN values.")
        # Use Stouffer's method with weights
        statistic_stouffer, pvalue_stouffer = stats.combine_pvalues(cluster_pvals, method='stouffer', weights=weights)
        print(f"Combined p-value using Stouffer's method: {pvalue_stouffer:.4e}")
        print(f"Stouffer's Z-score statistic: {statistic_stouffer:.4f}")
        # Compute weighted effect size
        effect_sizes = np.array([c['weighted_effect_size'] for c in valid_clusters])
        if weights is not None:
            normalized_weights = weights / np.sum(weights)
        else:
            normalized_weights = np.ones_like(effect_sizes) / len(effect_sizes)
        overall_effect = np.average(effect_sizes, weights=normalized_weights)
        # Count comparisons
        all_unique_pairs = set()
        for c in valid_clusters:
            all_unique_pairs.update(c['cluster_pairs'])
        total_comparisons = len(all_unique_pairs)
        n_valid_clusters = len(valid_clusters)
        overall_pvalue = overall_pvalue_combined
    else:
        print("No valid clusters available for significance computation.")
        overall_pvalue = np.nan
        overall_pvalue_combined = np.nan
        overall_effect = np.nan
        pvalue_stouffer = np.nan
    return {
        'overall_pvalue': overall_pvalue,
        'overall_pvalue_fisher': overall_pvalue_combined,
        'overall_pvalue_stouffer': pvalue_stouffer,
        'overall_effect': overall_effect,
        'n_valid_clusters': n_valid_clusters,
        'total_comparisons': total_comparisons
    }



def create_manhattan_plot(results_df, inv_file='inv_info.csv', top_hits_to_annotate=10):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from adjustText import adjust_text
    import os

    # Filter to valid p-values
    valid_mask = results_df['p_value'].notnull() & (results_df['p_value'] > 0)
    if valid_mask.sum() == 0:
        print("No valid p-values found. Cannot create plot.")
        return

    valid_pvals = results_df.loc[valid_mask, 'p_value']
    m = len(valid_pvals)

    # Bonferroni correction
    results_df['bonferroni_p_value'] = np.nan
    results_df.loc[valid_mask, 'bonferroni_p_value'] = results_df.loc[valid_mask, 'p_value'] * m
    results_df['bonferroni_p_value'] = results_df['bonferroni_p_value'].clip(upper=1.0)

    # Compute -log10(p)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))

    # Read inversion file
    inv_df = pd.read_csv(inv_file).dropna(subset=['chr', 'region_start', 'region_end'])

    def overlaps(inv_row, cds_df):
        c = inv_row['chr']
        s = inv_row['region_start']
        e = inv_row['region_end']
        subset = cds_df[(cds_df['chrom'] == c) & 
                        (cds_df['start'] <= e) & 
                        (cds_df['end'] >= s) & 
                        (cds_df['p_value'].notnull()) & (cds_df['p_value'] > 0)]
        return len(subset) > 0

    inv_df = inv_df[inv_df.apply(lambda row: overlaps(row, results_df), axis=1)]

    def chr_sort_key(ch):
        base = ch.replace('chr', '')
        try:
            return (0, int(base))
        except:
            mapping = {'X': 23, 'Y': 24, 'M': 25}
            return (1, mapping.get(base, 99))

    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)

    chrom_to_index = {c: i for i, c in enumerate(unique_chroms)}
    chrom_ranges = {}
    for c in unique_chroms:
        chr_df = results_df[results_df['chrom'] == c]
        c_min = chr_df['start'].min()
        c_max = chr_df['end'].max()
        chrom_ranges[c] = (c_min, c_max)

    xs = []
    for _, row in results_df.iterrows():
        c = row['chrom']
        if c not in chrom_ranges or pd.isnull(c):
            xs.append(np.nan)
            continue
        c_min, c_max = chrom_ranges[c]
        if c_max > c_min:
            rel_pos = (row['start'] - c_min) / (c_max - c_min)
        else:
            rel_pos = 0.5
        xs.append(chrom_to_index[c] + rel_pos)
    results_df['plot_x'] = xs

    eff = results_df['observed_effect_size']
    eff_mean = eff.mean()
    eff_std = eff.std()
    if eff_std == 0 or np.isnan(eff_std):
        eff_std = 1.0
    eff_z = (eff - eff_mean) / eff_std
    eff_z = np.clip(eff_z, -1, 1)

    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue', 'gray', 'red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0.0, vmax=1)

    plt.figure(figsize=(20, 10))
    sns.set_style("ticks")
    ax = plt.gca()

    recurrent_color = 'purple'
    single_color = 'green'
    
    for _, inv in inv_df.iterrows():
        inv_chr = inv['chr']
        if inv_chr not in chrom_to_index:
            continue
        
        c_idx = chrom_to_index[inv_chr]
        c_min, c_max = chrom_ranges[inv_chr]
    
        inv_size = inv['region_end'] - inv['region_start']
        chrom_size = c_max - c_min if c_max > c_min else 1
    
        if c_max > c_min:
            rel_start = (inv['region_start'] - c_min) / (c_max - c_min)
            rel_end = (inv['region_end'] - c_min) / (c_max - c_min)
        else:
            # Fallback positions if parsing fails
            rel_start = 0.4
            rel_end = 0.6
    
        inv_x_start = c_idx + max(0, min(rel_start, 1))
        inv_x_end = c_idx + min(1, max(rel_end, 0))
    
        # Determine inversion type (0 or 1)
        inversion_type = inv.get('0_single_1_recur', 0)
        if inversion_type == 1:
            inv_color = recurrent_color
        else:
            inv_color = single_color
    
        # If inversion >50% of chromosome length, apply a pattern and lower alpha
        if inv_size > 0.5 * chrom_size:
            # Large inversion: use a hatch pattern and a different alpha
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.1, zorder=0, hatch='//')
        else:
            # Normal sized inversion: no hatch
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.2, zorder=0)
    
    # Create legend entries:
    # Normal-sized recurrent and single
    recurrent_patch = mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion')
    single_patch = mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion')
    
    # Large recurrent and single (use the same colors but with hatch and different alpha for legend)
    recurrent_large_patch = mpatches.Patch(facecolor=recurrent_color, hatch='//', edgecolor='black', alpha=0.1, label='Large recurrent inversion')
    single_large_patch = mpatches.Patch(facecolor=single_color, hatch='//', edgecolor='black', alpha=0.1, label='Large single-event inversion')
    
    ax.legend(
        handles=[recurrent_patch, single_patch, recurrent_large_patch, single_large_patch],
        loc='upper left', fontsize=14, frameon=True
    )

    scatter = ax.scatter(
        results_df['plot_x'],
        results_df['neg_log_p'],
        c=eff_z,
        cmap=cmap,
        norm=norm,
        s=50,
        alpha=0.7,
        linewidth=0,
        zorder=2
    )

    cb = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)
    cb.ax.tick_params(labelsize=14)

    sig_threshold = -np.log10(0.05)
    ax.axhline(y=sig_threshold, color='red', linestyle='--', linewidth=2, zorder=3, label='p=0.05')
    if m > 0:
        bonf_threshold = -np.log10(0.05 / m)
        if np.isfinite(bonf_threshold) and bonf_threshold > 0:
            ax.axhline(y=bonf_threshold, color='darkred', linestyle='--', linewidth=2, zorder=3,
                       label='p=0.05 (Bonferroni)')

    current_ylim = ax.get_ylim()
    new_ylim = max(current_ylim[1], sig_threshold+1)
    if m > 0 and np.isfinite(bonf_threshold):
        new_ylim = max(new_ylim, bonf_threshold+1)
    ax.set_ylim(0, new_ylim)

    ax.set_xticks(range(len(unique_chroms)+1), minor=False)
    ax.set_xticks([i+0.5 for i in range(len(unique_chroms))], minor=True)
    ax.set_xticklabels(['']*(len(unique_chroms)+1), minor=False)
    ax.set_xticklabels(unique_chroms, rotation=45, ha='right', fontsize=14, fontweight='bold', minor=True)

    ax.set_xlabel('Chromosome', fontsize=18)
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of CDS Significance', fontsize=24, fontweight='bold', pad=20)

    handles_auto, labels_auto = ax.get_legend_handles_labels()
    all_handles = [recurrent_patch, single_patch, recurrent_large_patch, single_large_patch] + handles_auto
    ax.legend(handles=all_handles, fontsize=14, frameon=True, loc='upper right')

    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', lw=0.5)
    ax.xaxis.grid(False)

    significant_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)

    # Use the existing get_gene_annotation function
    text_objects = []
    label_points_x = []
    label_points_y = []
    for _, hit_row in significant_hits.iterrows():
        cds = hit_row['CDS']
        gene_symbol, gene_name, err_log = get_gene_annotation(cds)
        if gene_symbol and gene_symbol not in [None, 'Unknown']:
            label_txt = f"{gene_symbol}"
        else:
            label_txt = None

        if label_txt:
            txt = ax.text(
                hit_row['plot_x'],
                hit_row['neg_log_p'] + 1.0,
                label_txt,
                fontsize=12,
                ha='center',
                va='bottom',
                color='black',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
            text_objects.append(txt)
            label_points_x.append(hit_row['plot_x'])
            label_points_y.append(hit_row['neg_log_p'])

    adjust_text(
        text_objects,
        x=label_points_x,
        y=label_points_y,
        ax=ax,
        force_text=2.0,
        force_points=2.0,
        expand_points=(2,2),
        expand_text=(2,2),
        lim=200
    )
    
    for txt, (x, y) in zip(text_objects, zip(label_points_x, label_points_y)):
        x_text, y_text = txt.get_position()
        ax.plot([x_text, x], [y_text, y], color='black', lw=0.5, zorder=3, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join("plots", 'manhattan_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")

    # Read data
    df = read_and_preprocess_data('all_pairwise_results.csv')

    # Prepare arguments for parallel processing
    cds_list = df['CDS'].unique()
    cds_args = [(df[df['CDS'] == cds], cds) for cds in cds_list]

    # Process CDSs in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for cds, result in tqdm(
            executor.map(analyze_cds_parallel, cds_args),
            total=len(cds_args),
            desc="Processing CDSs"
        ):
            results[cds] = result

    # Convert results to DataFrame
    results_df = pd.DataFrame([
        {
            'CDS': cds,
            **{k: v for k, v in result.items()
               if k not in ['matrix_0', 'matrix_1', 'pairwise_comparisons']}
        }
        for cds, result in results.items()
    ])

    # Parse coordinates from CDS and add to results_df
    coords = results_df['CDS'].apply(parse_cds_coordinates)
    results_df[['chrom', 'start', 'end']] = pd.DataFrame(coords.tolist(), index=results_df.index)
    # Drop rows where parsing failed
    results_df.dropna(subset=['chrom','start','end'], inplace=True)

    # Save final results
    results_df.to_csv(RESULTS_DIR / 'final_results.csv', index=False)

    # Compute Bonferroni-corrected p-values
    valid_df = results_df[results_df['p_value'].notnull() & (results_df['p_value'] > 0)]
    total_valid_comparisons = len(valid_df)
    results_df['bonferroni_p_value'] = results_df['p_value'] * total_valid_comparisons
    results_df['bonferroni_p_value'] = results_df['bonferroni_p_value'].clip(upper=1.0)

    # Overall analysis
    print("\nBuilding clusters...")
    clusters = load_clusters_cache()
    if clusters is None:
        clusters = build_overlap_clusters(results_df)
        save_clusters_cache(clusters)
    else:
        print("Loaded clusters from cache.")

    cluster_stats = {}
    for cluster_id, cluster_cdss in clusters.items():
        cluster_stats[cluster_id] = combine_cluster_evidence(cluster_cdss, results_df, results)

    print("\nComputing overall significance...")
    overall_results = compute_overall_significance(cluster_stats)

    json_safe_results = {
        'overall_pvalue': float(overall_results['overall_pvalue']) if not np.isnan(overall_results['overall_pvalue']) else None,
        'overall_pvalue_fisher': float(overall_results['overall_pvalue_fisher']) if not np.isnan(overall_results['overall_pvalue_fisher']) else None,
        'overall_effect': float(overall_results['overall_effect']) if not np.isnan(overall_results['overall_effect']) else None,
        'n_valid_clusters': int(overall_results['n_valid_clusters']) if not np.isnan(overall_results['n_valid_clusters']) else None,
        'total_comparisons': int(overall_results['total_comparisons']) if not np.isnan(overall_results['total_comparisons']) else None
    }

    with open(RESULTS_DIR / 'overall_results.json', 'w') as f:
        json.dump(json_safe_results, f, indent=2)

    print("\nOverall Analysis Results:")
    print(f"Number of independent clusters: {overall_results['n_valid_clusters']}")
    print(f"Total unique CDS pairs: {overall_results['total_comparisons']:,}")
    print(f"Overall p-value: {overall_results['overall_pvalue']:.4e}" if overall_results['overall_pvalue'] is not np.nan else "Overall p-value: NaN")
    print(f"Overall effect size: {overall_results['overall_effect']:.4f}" if overall_results['overall_effect'] is not np.nan else "Overall effect size: NaN")

    # Create a Manhattan plot for all CDSs (now that we have chrom, start, end)
    create_manhattan_plot(results_df, inv_file='inv_info.csv')

    # Identify significant results
    significant_results = results_df[
        (results_df['bonferroni_p_value'].notnull()) & (results_df['bonferroni_p_value'] < 0.05)
    ].sort_values('p_value')

    # Create visualizations for top significant results
    for _, row in significant_results.head(30).iterrows():
        cds = row['CDS']
        viz_result = results[cds]  # Get full result with matrices
        create_visualization(
            viz_result['matrix_0'],
            viz_result['matrix_1'],
            cds,
            row
        )

    valid_results = results_df[~results_df['p_value'].isna()]
    print("\nPer-CDS Analysis Summary:")
    print(f"Total CDSs analyzed: {len(results_df):,}")
    print(f"Valid analyses: {len(valid_results):,}")
    print(f"Significant CDSs (p < 0.05): {(valid_results['p_value'] < 0.05).sum():,}")

    end_time = datetime.now()
    print(f"\nAnalysis completed at {end_time}")
    print(f"Total runtime: {end_time - start_time}")

if __name__ == "__main__":
    main()
