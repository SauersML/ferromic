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
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
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

def get_cache_path(transcript):
    """Generate cache file path for a transcript."""
    return CACHE_DIR / f"{transcript}.pkl"

def load_cached_result(transcript):
    """Load cached result for a transcript if it exists."""
    cache_path = get_cache_path(transcript)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_result = pickle.load(f)
            return cached_result
        except:
            return None
    return None

def save_cached_result(transcript, result):
    """Save result for a transcript to cache."""
    cache_path = get_cache_path(transcript)
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

def read_and_preprocess_data(file_path):
    """Read and preprocess the CSV file."""
    print("Reading data...")
    df = pd.read_csv(file_path)

    # Store original CDS as full_cds
    df['full_cds'] = df['CDS']
    # Extract transcript ID into CDS
    df['CDS'] = df['CDS'].str.extract(r'(ENST\d+\.\d+)')[0]
    # Convert omega to numeric, coerce non-numeric to NaN
    df['omega'] = pd.to_numeric(df['omega'], errors='coerce')
    
    # Filter valid omega values
    df = df[
        (df['omega'] != -1) &
        (df['omega'] != 99)
    ].dropna(subset=['omega'])

    print(f"Total valid comparisons: {len(df):,}")
    print(f"Unique transcripts found: {df['CDS'].nunique():,}")
    return df

def get_pairwise_value(seq1, seq2, pairwise_dict):
    """Get omega value for a pair of sequences."""
    key = (seq1, seq2) if (seq1, seq2) in pairwise_dict else (seq2, seq1)
    val = pairwise_dict.get(key)
    return val

def create_matrices(sequences_0, sequences_1, pairwise_dict):
    """Create matrices for two groups based on sequence assignments."""
    print(f"Number of sequences: Group 0={len(sequences_0)}, Group 1={len(sequences_1)}")
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
    """Mixed effects analysis for a single transcript with crossed random effects."""
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    import numpy as np

    all_sequences, n0, pairwise_dict, sequences_0, sequences_1 = args

    # Prepare data
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
    effect_size = np.nan
    p_value = np.nan
    std_err = np.nan

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

    df['seq1_code'] = pd.Categorical(df['seq1']).codes
    df['seq2_code'] = pd.Categorical(df['seq2']).codes

    try:
        df['groups'] = 1
        vc = {
            'seq1': '0 + C(seq1_code)',
            'seq2': '0 + C(seq2_code)'
        }
        model = MixedLM.from_formula(
            'omega_value ~ group',
            groups='groups',
            vc_formula=vc,
            re_formula='0',
            data=df
        )
        result = model.fit(reml=False)
        effect_size = result.fe_params['group']
        p_value = result.pvalues['group']
        std_err = result.bse['group']
    except Exception as e:
        print(f"Model fitting failed with error: {str(e)}")

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
    n_x, n_y = len(x), len(y)
    n_pairs = n_x * n_y
    x, y = np.array(x), np.array(y)
    difference_matrix = x[:, np.newaxis] - y
    num_greater = np.sum(difference_matrix > 0)
    num_less = np.sum(difference_matrix < 0)
    return (num_greater - num_less) / n_pairs

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

def get_gene_annotation(full_cds, cache_file='gene_name_cache.json'):
    """
    Get gene annotation for a transcript with caching and detailed error reporting.
    Returns (gene_symbol, gene_name, error_log).
    """
    error_log = []
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except Exception as e:
            error_log.append(f"WARNING: Failed to load cache file: {str(e)}")

    if full_cds in cache:
        error_log.append(f"INFO: Found entry in cache for {full_cds}")
        return cache[full_cds]['symbol'], cache[full_cds]['name'], error_log

    def parse_coords(coord_str):
        """Parse coordinate string into chr, start, end."""
        if not coord_str:
            return None, "ERROR: Empty coordinate string provided"
        try:
            if ':' in coord_str:
                parts = coord_str.split(':')
                chr = parts[0]
                start_end = parts[1].split('-')
                start = int(start_end[0])
                end = int(start_end[1])
            else:
                parts = coord_str.split('_')
                chr_index = parts.index('chr')
                start_index = parts.index('start')
                end_index = parts.index('end')
                chr = 'chr' + parts[chr_index + 1]
                start = int(parts[start_index + 1])
                end = int(parts[end_index + 1])
            if start > end:
                return None, f"ERROR: Invalid coordinates - start ({start}) > end ({end})"
            return {'chr': chr, 'start': start, 'end': end}, None
        except Exception as e:
            return None, f"ERROR: Failed to parse coordinates: {str(e)}"

    def query_ucsc(chr, start, end):
        """Query UCSC API for genes at location."""
        base_url = "https://api.genome.ucsc.edu/getData/track"
        params = {'genome': 'hg38', 'track': 'knownGene', 'chrom': chr, 'start': start, 'end': end}
        try:
            response = requests.get(f"{base_url}?{urlencode(params)}", timeout=10)
            if not response.ok:
                return None, f"ERROR: API request failed with status {response.status_code}"
            data = response.json()
            track_data = data.get('knownGene', data) if isinstance(data, list) else data
            if not track_data:
                return None, "No gene data found in response"
            overlapping_genes = [
                gene for gene in track_data
                if gene.get('chromStart', 0) <= end and gene.get('chromEnd', 0) >= start
            ]
            return overlapping_genes if overlapping_genes else None, None
        except Exception as e:
            return None, f"ERROR: API query failed: {str(e)}"

    loc, parse_error = parse_coords(full_cds)
    if parse_error:
        error_log.append(parse_error)
        return None, None, error_log

    genes, query_error = query_ucsc(loc['chr'], loc['start'], loc['end'])
    if query_error:
        error_log.append(query_error)
        return None, None, error_log

    if not genes:
        error_log.append(f"WARNING: No genes found for {loc['chr']}:{loc['start']}-{loc['end']}")
        return None, None, error_log

    best_gene = max(
        genes,
        key=lambda gene: max(0, min(gene.get('chromEnd', 0), loc['end']) - max(gene.get('chromStart', 0), loc['start'])),
        default=None
    )
    if not best_gene:
        error_log.append("WARNING: Could not determine best matching gene")
        return None, None, error_log

    symbol = best_gene.get('geneName', 'Unknown')
    if symbol in ['none', None] or symbol.startswith('ENSG'):
        for gene in genes:
            potential_symbol = gene.get('geneName')
            if potential_symbol and potential_symbol != 'none' and not potential_symbol.startswith('ENSG'):
                symbol = potential_symbol
                break
    name = get_gene_info(symbol)

    try:
        cache[full_cds] = {'symbol': symbol, 'name': name}
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        error_log.append(f"WARNING: Failed to update cache file: {str(e)}")

    return symbol, name, error_log

def create_visualization(matrix_0, matrix_1, transcript, result, full_cds):
    """Create visualization for a transcript."""
    gene_symbol, gene_name, error_log = get_gene_annotation(full_cds)
    if error_log:
        print(f"\nWarnings/Errors for transcript {transcript}:")
        for msg in error_log:
            print(f"{'🚫' if 'ERROR' in msg else '⚠️'} {msg}")

    df_all = pd.read_csv('all_pairwise_results.csv')
    df_all['full_cds'] = df_all['CDS']
    df_all['CDS'] = df_all['CDS'].str.extract(r'(ENST\d+\.\d+)')[0]
    df_cds_all = df_all[df_all['CDS'] == transcript]
    pairwise_dict_all = {(row['Seq1'], row['Seq2']): row['omega'] for _, row in df_cds_all.iterrows()}
    sequences = pd.concat([df_cds_all['Seq1'], df_cds_all['Seq2']]).dropna().unique()
    sequences_direct = [seq for seq in sequences if not seq.endswith('1')]
    sequences_inverted = [seq for seq in sequences if seq.endswith('1')]
    matrix_0_full, matrix_1_full = create_matrices(sequences_direct, sequences_inverted, pairwise_dict_all)

    if matrix_0_full is None or matrix_1_full is None:
        print(f"No data available for transcript: {transcript}")
        return

    color_minus_one = (242/255, 235/255, 250/255)
    color_ninety_nine = (1, 192/255, 192/255)
    special_patches = [
        mpatches.Patch(color=color_minus_one, label='Identical sequences'),
        mpatches.Patch(color=color_ninety_nine, label='No non-synonymous variation')
    ]
    cmap_normal = sns.color_palette("viridis", as_cmap=True)

    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 0.4], hspace=0.3, wspace=0.3)
    title_str = f'{gene_symbol}: {gene_name}\n{transcript}' if gene_symbol and gene_name else f'Pairwise Comparison Analysis: {transcript}'
    fig.suptitle(title_str, fontsize=24, fontweight='bold', y=0.98)

    def plot_matrices(ax, matrix, title_str):
        n = matrix.shape[0]
        upper_triangle = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        lower_triangle = np.tril(np.ones_like(matrix, dtype=bool), k=-1)
        special_minus_one = (matrix == -1)
        special_ninety_nine = (matrix == 99)
        normal_mask = (~np.isnan(matrix)) & (~special_minus_one) & (~special_ninety_nine)
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

    ax1 = fig.add_subplot(gs[0, 0])
    plot_matrices(ax1, matrix_0_full, f'Direct Sequences (n={len(sequences_direct)})')
    ax2 = fig.add_subplot(gs[0, 1])
    plot_matrices(ax2, matrix_1_full, f'Inverted Sequences (n={len(sequences_inverted)})')

    cbar_ax = fig.add_subplot(gs[0, 2])
    sm = ScalarMappable(norm=LogNorm(vmin=0.01, vmax=50), cmap=cmap_normal)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_major_locator(FixedLocator([0.01, 1, 3, 10, 50]))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(['0', '1', '3', '10', '50']))
    cbar.set_label('Omega Value', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    ax1.legend(handles=special_patches, title='Special Values', loc='upper left', bbox_to_anchor=(0.0, -0.05), ncol=1, fontsize=10)

    values_direct = matrix_0_full[np.tril_indices_from(matrix_0_full, k=-1)]
    values_inverted = matrix_1_full[np.tril_indices_from(matrix_1_full, k=-1)]
    values_direct = values_direct[~np.isnan(values_direct) & (values_direct != -1) & (values_direct != 99)]
    values_inverted = values_inverted[~np.isnan(values_inverted) & (values_inverted != -1) & (values_inverted != 99)]

    ax3 = fig.add_subplot(gs[1, :])
    sns.kdeplot(values_direct, ax=ax3, label='Direct', fill=True, color='#1f77b4', alpha=0.6)
    sns.kdeplot(values_inverted, ax=ax3, label='Inverted', fill=True, color='#ff7f0e', alpha=0.6)
    ax3.set_title('Distribution of Omega Values', fontsize=20, pad=15)
    ax3.set_xlabel('Omega Value', fontsize=16)
    ax3.set_ylabel('Density', fontsize=16)
    ax3.legend(title='Groups', title_fontsize=16, fontsize=14)

    p_value = result.get('bonferroni_p_value', result.get('p_value', np.nan))
    p_str = f"Corrected p-value: {p_value:.4e}" if not np.isnan(p_value) else "P-value: N/A"
    ax3.text(0.97, 0.97, p_str, transform=ax3.transAxes, ha='right', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOTS_DIR / f'analysis_{transcript}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def analyze_cds_parallel(args):
    """Analyze a single transcript using explicit group assignments."""
    df_cds, transcript = args
    print(f"\nAnalyzing {transcript}")
    cached_result = load_cached_result(transcript)
    if cached_result is not None:
        return transcript, cached_result

    pairwise_dict = {(row['Seq1'], row['Seq2']): row['omega'] for _, row in df_cds.iterrows()}
    all_seqs = pd.concat([df_cds['Seq1'], df_cds['Seq2']]).unique()
    seq_to_group = {}
    for _, row in df_cds.iterrows():
        seq_to_group[row['Seq1']] = row['Group1']
        seq_to_group[row['Seq2']] = row['Group2']
    sequences_0 = np.array([seq for seq in all_seqs if seq_to_group.get(seq, 0) == 0])
    sequences_1 = np.array([seq for seq in all_seqs if seq_to_group.get(seq, 0) == 1])
    all_sequences = np.concatenate([sequences_0, sequences_1])
    n0, n1 = len(sequences_0), len(sequences_1)
    matrix_0, matrix_1 = create_matrices(sequences_0, sequences_1, pairwise_dict)

    result = {'matrix_0': matrix_0, 'matrix_1': matrix_1, 'pairwise_comparisons': set(pairwise_dict.keys())}
    if n0 == 0 or n1 == 0 or matrix_0 is None or matrix_1 is None:
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
        min_seq = 3
        valid_0 = np.nansum(~np.isnan(matrix_0)) >= 3 and all(np.sum(~np.isnan(matrix_0), axis=1) >= 5)
        valid_1 = np.nansum(~np.isnan(matrix_1)) >= 3 and all(np.sum(~np.isnan(matrix_1), axis=1) >= 5)
        if n0 < min_seq or n1 < min_seq or not valid_0 or not valid_1:
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
            result.update(analysis_worker((all_sequences, n0, pairwise_dict, sequences_0, sequences_1)))

    save_cached_result(transcript, result)
    return transcript, result

def parse_cds_coordinates(cds_name):
    """Extract chromosome and coordinates from CDS name."""
    try:
        if ':' in cds_name:
            chrom, coords = cds_name.split(':')
            start, end = map(int, coords.split('-'))
        else:
            parts = cds_name.split('_')
            chr_index = parts.index('chr')
            start_index = parts.index('start')
            end_index = parts.index('end')
            chrom = 'chr' + parts[chr_index + 1]
            start = int(parts[start_index + 1])
            end = int(parts[end_index + 1])
        return chrom, start, end
    except Exception as e:
        print(f"Error parsing {cds_name}: {str(e)}")
        return None, None, None

CLUSTERS_CACHE_PATH = CACHE_DIR / "clusters.pkl"

def load_clusters_cache():
    if CLUSTERS_CACHE_PATH.exists():
        try:
            with open(CLUSTERS_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load clusters cache: {e}")
    return None

def save_clusters_cache(clusters):
    with open(CLUSTERS_CACHE_PATH, 'wb') as f:
        pickle.dump(clusters, f)

def build_overlap_clusters(results_df):
    """Build clusters of overlapping CDS regions."""
    cds_coords = [
        (chrom, start, end, cds)
        for chrom, start, end, cds in results_df[['chrom', 'start', 'end', 'full_cds']].values
        if chrom is not None
    ]
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
                merged.append((c, e, target if cid in overlapping else cid))
            merged.append((chrom, end, target))
            active = merged

    return clusters

def combine_cluster_evidence(cluster_cdss, results_df, results):
    """Select the largest CDS by amino acid length from the cluster."""
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
                if len(fields) < 9 or fields[2] != "CDS":
                    continue
                chrom, start, end = fields[0], int(fields[3]), int(fields[4])
                attrs = {k.strip(): v.strip('"') for k, v in (a.split(' ', 1) for a in fields[8].split(';') if a.strip())}
                tid = attrs.get("transcript_id")
                if tid:
                    combine_cluster_evidence._interval_map[tid].append((start, end))
                    combine_cluster_evidence._chrom_strand[tid] = (chrom, fields[6])
                    cds_lengths[tid] += (end - start + 1)
            for t, length in cds_lengths.items():
                combine_cluster_evidence._aa_length_map[t] = length / 3.0

    def find_transcript_id_for_cds(cds_name):
        chrom, start, end = parse_cds_coordinates(cds_name)
        if chrom is None:
            return None
        best_tid, best_len = None, -1
        for tid, intervals in combine_cluster_evidence._interval_map.items():
            t_chrom, _ = combine_cluster_evidence._chrom_strand[tid]
            if t_chrom != chrom:
                continue
            merged = []
            for s, e in sorted(intervals):
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            covered, needed = False, start
            for ms, me in merged:
                if ms <= needed <= me:
                    if me >= end:
                        covered = True
                        break
                    needed = me + 1
                elif ms > needed:
                    break
            if covered and (aa_len := combine_cluster_evidence._aa_length_map.get(tid, -1)) > best_len:
                best_len, best_tid = aa_len, tid
        return best_tid

    best_cds, best_len = None, -1
    for c in cluster_cdss:
        tid = find_transcript_id_for_cds(c)
        if tid and (aa_len := combine_cluster_evidence._aa_length_map.get(tid, -1)) > best_len:
            best_len, best_cds = aa_len, c

    if not best_cds:
        return {'combined_pvalue': np.nan, 'weighted_effect_size': np.nan, 'n_comparisons': 0, 'n_valid_cds': 0, 'cluster_pairs': set()}

    best_transcript = results_df[results_df['full_cds'] == best_cds]['CDS'].iloc[0]
    chosen = results_df[results_df['CDS'] == best_transcript].dropna(subset=['observed_effect_size', 'p_value'])
    if chosen.empty:
        return {'combined_pvalue': np.nan, 'weighted_effect_size': np.nan, 'n_comparisons': 0, 'n_valid_cds': 0, 'cluster_pairs': set()}
    
    row = chosen.iloc[0]
    pairs = results.get(best_transcript, {}).get('pairwise_comparisons', set())
    return {
        'combined_pvalue': row['p_value'],
        'weighted_effect_size': row['observed_effect_size'],
        'n_comparisons': len(pairs),
        'n_valid_cds': 1,
        'cluster_pairs': pairs
    }

def compute_overall_significance(cluster_results):
    """Compute overall significance from independent clusters."""
    valid_clusters = [
        c for c in cluster_results.values()
        if not np.isnan(c['combined_pvalue']) and not np.isnan(c['weighted_effect_size'])
    ]
    if not valid_clusters:
        return {
            'overall_pvalue': np.nan,
            'overall_pvalue_fisher': np.nan,
            'overall_pvalue_stouffer': np.nan,
            'overall_effect': np.nan,
            'n_valid_clusters': 0,
            'total_comparisons': 0
        }

    cluster_pvals = np.array([c['combined_pvalue'] for c in valid_clusters])
    statistic, overall_pvalue_fisher = stats.combine_pvalues(cluster_pvals, method='fisher')
    weights = np.array([c['n_comparisons'] for c in valid_clusters], dtype=float)
    weights = None if np.all(weights == 0) or np.isnan(weights).any() else weights
    statistic_stouffer, overall_pvalue_stouffer = stats.combine_pvalues(cluster_pvals, method='stouffer', weights=weights)
    effect_sizes = np.array([c['weighted_effect_size'] for c in valid_clusters])
    normalized_weights = weights / np.sum(weights) if weights is not None else np.ones_like(effect_sizes) / len(effect_sizes)
    overall_effect = np.average(effect_sizes, weights=normalized_weights)
    all_unique_pairs = set().union(*(c['cluster_pairs'] for c in valid_clusters))

    return {
        'overall_pvalue': overall_pvalue_fisher,
        'overall_pvalue_fisher': overall_pvalue_fisher,
        'overall_pvalue_stouffer': overall_pvalue_stouffer,
        'overall_effect': overall_effect,
        'n_valid_clusters': len(valid_clusters),
        'total_comparisons': len(all_unique_pairs)
    }

def create_manhattan_plot(results_df, inv_file='inv_info.csv', top_hits_to_annotate=10):
    """Create a Manhattan plot for all transcripts."""
    valid_mask = results_df['p_value'].notnull() & (results_df['p_value'] > 0)
    if not valid_mask.sum():
        print("No valid p-values found. Cannot create plot.")
        return

    m = len(results_df[valid_mask])
    results_df['bonferroni_p_value'] = (results_df['p_value'] * m).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))

    inv_df = pd.read_csv(inv_file).dropna(subset=['chr', 'region_start', 'region_end'])
    inv_df = inv_df[inv_df.apply(lambda row: len(results_df[
        (results_df['chrom'] == row['chr']) &
        (results_df['start'] <= row['region_end']) &
        (results_df['end'] >= row['region_start']) &
        valid_mask
    ]) > 0, axis=1)]

    chr_sort_key = lambda ch: (0, int(ch.replace('chr', ''))) if ch.replace('chr', '').isdigit() else (1, {'X': 23, 'Y': 24, 'M': 25}.get(ch.replace('chr', ''), 99))
    unique_chroms = sorted(results_df['chrom'].dropna().unique(), key=chr_sort_key)

    chrom_to_index = {c: i for i, c in enumerate(unique_chroms)}
    chrom_ranges = {c: (results_df[results_df['chrom'] == c]['start'].min(), results_df[results_df['chrom'] == c]['end'].max()) for c in unique_chroms}
    results_df['plot_x'] = [
        chrom_to_index[row['chrom']] + (row['start'] - chrom_ranges[row['chrom']][0]) / (chrom_ranges[row['chrom']][1] - chrom_ranges[row['chrom']][0] or 1)
        if row['chrom'] in chrom_ranges and not pd.isnull(row['chrom']) else np.nan
        for _, row in results_df.iterrows()
    ]

    eff_z = np.clip((results_df['observed_effect_size'] - results_df['observed_effect_size'].mean()) / (results_df['observed_effect_size'].std() or 1), -1, 1)
    cmap = LinearSegmentedColormap.from_list('custom_diverging', ['blue', 'gray', 'red'])
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    recurrent_color, single_color = 'purple', 'green'

    for _, inv in inv_df.iterrows():
        if inv['chr'] not in chrom_to_index:
            continue
        c_idx = chrom_to_index[inv['chr']]
        c_min, c_max = chrom_ranges[inv['chr']]
        rel_start = (inv['region_start'] - c_min) / (c_max - c_min or 1)
        rel_end = (inv['region_end'] - c_min) / (c_max - c_min or 1)
        inv_x_start, inv_x_end = c_idx + max(0, min(rel_start, 1)), c_idx + min(1, max(rel_end, 0))
        inv_color = recurrent_color if inv.get('0_single_1_recur', 0) == 1 else single_color
        if (inv['region_end'] - inv['region_start']) > 0.5 * (c_max - c_min or 1):
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.1, hatch='//', zorder=0)
        else:
            ax.axvspan(inv_x_start, inv_x_end, color=inv_color, alpha=0.2, zorder=0)

    scatter = ax.scatter(results_df['plot_x'], results_df['neg_log_p'], c=eff_z, cmap=cmap, norm=norm, s=50, alpha=0.7, zorder=2)
    cb = plt.colorbar(scatter, ax=ax, fraction=0.02, pad=0.02)
    cb.set_label('Z-scored Effect Size', fontsize=16)

    sig_threshold = -np.log10(0.05)
    bonf_threshold = -np.log10(0.05 / m) if m > 0 else sig_threshold
    ax.axhline(y=sig_threshold, color='red', linestyle='--', linewidth=2, label='p=0.05')
    if m > 0 and np.isfinite(bonf_threshold):
        ax.axhline(y=bonf_threshold, color='darkred', linestyle='--', linewidth=2, label='p=0.05 (Bonferroni)')
    ax.set_ylim(0, max(ax.get_ylim()[1], bonf_threshold + 1 if np.isfinite(bonf_threshold) else sig_threshold + 1))

    ax.set_xticks([i + 0.5 for i in range(len(unique_chroms))], minor=True)
    ax.set_xticklabels(unique_chroms, rotation=45, ha='right', fontsize=14, minor=True)
    ax.set_xticks(range(len(unique_chroms) + 1), minor=False)
    ax.set_xticklabels([''] * (len(unique_chroms) + 1), minor=False)
    ax.set_xlabel('Chromosome', fontsize=18)
    ax.set_ylabel('-log10(p-value)', fontsize=18)
    ax.set_title('Manhattan Plot of Transcript Significance', fontsize=24, fontweight='bold', pad=20)

    ax.legend(handles=[
        mpatches.Patch(color=recurrent_color, alpha=0.2, label='Recurrent inversion'),
        mpatches.Patch(color=single_color, alpha=0.2, label='Single-event inversion'),
        mpatches.Patch(facecolor=recurrent_color, hatch='//', alpha=0.1, label='Large recurrent inversion'),
        mpatches.Patch(facecolor=single_color, hatch='//', alpha=0.1, label='Large single-event inversion')
    ] + ax.get_legend_handles_labels()[0], fontsize=14, loc='upper right')

    from adjustText import adjust_text
    significant_hits = results_df[valid_mask].sort_values('p_value').head(top_hits_to_annotate)
    texts = []
    for _, row in significant_hits.iterrows():
        symbol, _, _ = get_gene_annotation(row['full_cds'])
        if symbol and symbol != 'Unknown':
            texts.append(ax.text(row['plot_x'], row['neg_log_p'] + 1, symbol, fontsize=12, ha='center', va='bottom'))
    adjust_text(texts, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join("plots", 'manhattan_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main execution function."""
    start_time = datetime.now()
    print(f"Analysis started at {start_time}")

    df = read_and_preprocess_data('all_pairwise_results.csv')
    transcript_args = [(df[df['CDS'] == t], t) for t in df['CDS'].unique()]
    results = {}
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for transcript, result in tqdm(executor.map(analyze_cds_parallel, transcript_args), total=len(transcript_args), desc="Processing transcripts"):
            results[transcript] = result

    results_df = pd.DataFrame([
        {'CDS': t, 'full_cds': df[df['CDS'] == t]['full_cds'].iloc[0], **{k: v for k, v in r.items() if k not in ['matrix_0', 'matrix_1', 'pairwise_comparisons']}}
        for t, r in results.items()
    ])
    coords = results_df['full_cds'].apply(parse_cds_coordinates)
    results_df[['chrom', 'start', 'end']] = pd.DataFrame(coords.tolist(), index=results_df.index)
    results_df.dropna(subset=['chrom', 'start', 'end'], inplace=True)

    valid_df = results_df[results_df['p_value'].notnull() & (results_df['p_value'] > 0)]
    total_valid = len(valid_df)
    results_df['bonferroni_p_value'] = (results_df['p_value'] * total_valid).clip(upper=1.0)
    results_df['neg_log_p'] = -np.log10(results_df['p_value'].replace(0, np.nan))
    results_df.to_csv(RESULTS_DIR / 'final_results.csv', index=False)
    results_df[['CDS', 'p_value', 'observed_effect_size', 'chrom', 'start', 'end', 'bonferroni_p_value', 'neg_log_p']].to_csv(RESULTS_DIR / 'manhattan_plot_data.csv', index=False)

    clusters = load_clusters_cache() or build_overlap_clusters(results_df)
    save_clusters_cache(clusters)
    cluster_stats = {cid: combine_cluster_evidence(cdss, results_df, results) for cid, cdss in clusters.items()}
    overall_results = compute_overall_significance(cluster_stats)

    with open(RESULTS_DIR / 'overall_results.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else None for k, v in overall_results.items()}, f, indent=2)

    create_manhattan_plot(results_df)
    for _, row in results_df[results_df['bonferroni_p_value'] < 0.05].sort_values('p_value').head(30).iterrows():
        create_visualization(results[row['CDS']]['matrix_0'], results[row['CDS']]['matrix_1'], row['CDS'], row, row['full_cds'])

    print(f"\nAnalysis completed at {datetime.now()}")
    print(f"Total runtime: {datetime.now() - start_time}")

if __name__ == "__main__":
    main()
