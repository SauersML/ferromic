import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import sys
import os
import time
from scipy.stats import shapiro

def paired_permutation_test(x, y, num_permutations=10000):
    differences = np.array(x) - np.array(y)
    observed_mean = np.mean(differences)
    count = 0
    for _ in range(num_permutations):
        signs = np.random.choice([1, -1], size=len(differences))
        permuted_mean = np.mean(differences * signs)
        if abs(permuted_mean) >= abs(observed_mean):
            count += 1
    p_value = count / num_permutations
    return p_value

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('fst_flanking_analysis')

# Constants
MIN_LENGTH = 300_000
FLANK_SIZE = 100_000

# File paths
FST_DATA_FILE = 'per_site_fst_output.falsta'
INVERSION_FILE = 'inv_info.tsv'
OUTPUT_PLOT = 'fst_flanking_regions_bar_plot.png'

# Adjusted category mapping to only include relevant categories
cat_mapping = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}

def normalize_chromosome(chrom):
    chrom = chrom.strip()
    if chrom.startswith('chr_'):
        chrom = chrom[4:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    return f"chr{chrom}"

def extract_coordinates_from_header(header):
    parts = header.strip().split('_')
    try:
        chrom = parts[2]
        start = int(parts[4])
        end = int(parts[6])
        return {'chrom': normalize_chromosome(chrom), 'start': start, 'end': end}
    except Exception as e:
        logger.warning(f"Failed parsing header: {header} - {e}")
        return None

def map_regions_to_inversions(inversion_df):
    recurrent_regions = {}
    single_event_regions = {}
    for _, row in inversion_df.iterrows():
        chrom = normalize_chromosome(str(row['chr']))
        start, end = int(row['region_start']), int(row['region_end'])
        if row['0_single_1_recur'] == 1:
            recurrent_regions.setdefault(chrom, []).append((start, end))
        else:
            single_event_regions.setdefault(chrom, []).append((start, end))
    return recurrent_regions, single_event_regions

def is_overlapping(region1, region2):
    return region1[0] <= region2[1] and region1[1] >= region2[0]

def determine_inversion_type(coords, recurrent_regions, single_event_regions):
    chrom, start, end = coords['chrom'], coords['start'], coords['end']
    rec = recurrent_regions.get(chrom, [])
    sing = single_event_regions.get(chrom, [])
    rec_overlap = any(is_overlapping((start, end), r) for r in rec)
    sing_overlap = any(is_overlapping((start, end), s) for s in sing)
    if rec_overlap and not sing_overlap:
        return 'recurrent'
    elif sing_overlap and not rec_overlap:
        return 'single_event'
    elif rec_overlap and sing_overlap:
        return 'ambiguous'
    return 'unknown'

def load_fst_data(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        header = None
        for line in f:
            line = line.strip()
            if 'population_pairwise' in line:
                header = None
                continue
            if line.startswith('>'):
                header = line
            elif header:
                data = np.array([float(x) if x != 'NA' else np.nan for x in line.split(',')])
                # Skip sequences with only NaNs or insufficient length
                if len(data) >= MIN_LENGTH and not np.all(np.isnan(data)):
                    coords = extract_coordinates_from_header(header)
                    if coords:
                        sequences.append({'coords': coords, 'data': data})
                header = None
    logger.info(f"Loaded {len(sequences)} fst sequences")
    return sequences

def calculate_flanking_means(sequences):
    results = []
    for seq in sequences:
        data = seq['data']
        if len(data) < 2 * FLANK_SIZE:
            continue
        begin = data[:FLANK_SIZE]
        end = data[-FLANK_SIZE:]
        middle = data[FLANK_SIZE:-FLANK_SIZE]
        results.append({
            'coords': seq['coords'],
            'beginning_mean': np.nanmean(begin),
            'ending_mean': np.nanmean(end),
            'middle_mean': np.nanmean(middle)
        })
    return results

def categorize_sequences(means, recurrent_regions, single_event_regions):
    categories = {v: [] for v in cat_mapping.values()}
    for seq in means:
        inv_type = determine_inversion_type(seq['coords'], recurrent_regions, single_event_regions)
        if inv_type in ['recurrent', 'single_event']:
            key = f"{inv_type}"
            categories[key].append(seq)
    return categories

def create_bar_plot(categories):
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = list(cat_mapping.keys()) + ['Overall']
    flanking_means, flanking_se, middle_means, middle_se = [], [], [], []

    # Process each category
    for i, label in enumerate(labels[:-1]):
        cat = cat_mapping[label]
        seqs = categories[cat]
        f_means = [np.nanmean([s['beginning_mean'], s['ending_mean']]) for s in seqs]
        m_means = [s['middle_mean'] for s in seqs]
        valid_f_means = [f for f in f_means if not np.isnan(f)]
        valid_m_means = [m for m in m_means if not np.isnan(m)]

        # Calculate means and standard errors
        mean_f = np.mean(valid_f_means) if valid_f_means else np.nan
        std_f = np.std(valid_f_means, ddof=1) if len(valid_f_means) > 1 else 0
        n_f = len(valid_f_means)
        se_f = std_f / np.sqrt(n_f) if n_f > 0 else 0
        mean_m = np.mean(valid_m_means) if valid_m_means else np.nan
        std_m = np.std(valid_m_means, ddof=1) if len(valid_m_means) > 1 else 0
        n_m = len(valid_m_means)
        se_m = std_m / np.sqrt(n_m) if n_m > 0 else 0

        flanking_means.append(mean_f)
        flanking_se.append(se_f)
        middle_means.append(mean_m)
        middle_se.append(se_m)

        # Paired permutation test
        paired_flanking, paired_middle = [], []
        for s in seqs:
            f_mean = np.nanmean([s['beginning_mean'], s['ending_mean']])
            m_mean = s['middle_mean']
            if not np.isnan(f_mean) and not np.isnan(m_mean):
                paired_flanking.append(f_mean)
                paired_middle.append(m_mean)

        max_y = np.nanmax([m + e for m, e in zip(flanking_means, flanking_se)] + 
                         [m + e for m, e in zip(middle_means, middle_se)]) * 1.1 if flanking_means else 1.0
        if len(paired_flanking) >= 2:
            differences = np.array(paired_flanking) - np.array(paired_middle)
            norm_stat, norm_p = shapiro(differences)
            perm_p_value = paired_permutation_test(paired_middle, paired_flanking, num_permutations=20000)
            text = f"Permutation p={perm_p_value:.4g}\nNormality p={norm_p:.4g}"
        else:
            text = "Insufficient data"
        ax.text(i, max_y, text, ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Overall category
    all_seqs = sum(categories.values(), [])
    f_means_all = [np.nanmean([s['beginning_mean'], s['ending_mean']]) for s in all_seqs]
    m_means_all = [s['middle_mean'] for s in all_seqs]
    valid_f_means_all = [f for f in f_means_all if not np.isnan(f)]
    valid_m_means_all = [m for m in m_means_all if not np.isnan(m)]

    mean_f_all = np.mean(valid_f_means_all) if valid_f_means_all else np.nan
    std_f_all = np.std(valid_f_means_all, ddof=1) if len(valid_f_means_all) > 1 else 0
    n_f_all = len(valid_f_means_all)
    se_f_all = std_f_all / np.sqrt(n_f_all) if n_f_all > 0 else 0
    mean_m_all = np.mean(valid_m_means_all) if valid_m_means_all else np.nan
    std_m_all = np.std(valid_m_means_all, ddof=1) if len(valid_m_means_all) > 1 else 0
    n_m_all = len(valid_m_means_all)
    se_m_all = std_m_all / np.sqrt(n_m_all) if n_m_all > 0 else 0

    flanking_means.append(mean_f_all)
    flanking_se.append(se_f_all)
    middle_means.append(mean_m_all)
    middle_se.append(se_m_all)

    # Overall permutation test
    paired_flanking_overall, paired_middle_overall = [], []
    for s in all_seqs:
        f_mean = np.nanmean([s['beginning_mean'], s['ending_mean']])
        m_mean = s['middle_mean']
        if not np.isnan(f_mean) and not np.isnan(m_mean):
            paired_flanking_overall.append(f_mean)
            paired_middle_overall.append(m_mean)

    max_y = np.nanmax([m + e for m, e in zip(flanking_means, flanking_se)] + 
                     [m + e for m, e in zip(middle_means, middle_se)]) * 1.1 if flanking_means else 1.0
    if len(paired_flanking_overall) >= 2:
        overall_diffs = np.array(paired_flanking_overall) - np.array(paired_middle_overall)
        norm_stat_overall, norm_p_overall = shapiro(overall_diffs)
        overall_perm_p = paired_permutation_test(paired_middle_overall, paired_flanking_overall, num_permutations=20000)
        text = f"Permutation p={overall_perm_p:.3g}\nNormality p={norm_p_overall:.3g}"
    else:
        text = "Insufficient data"
    ax.text(len(labels) - 1, max_y, text, ha='center', va='bottom', fontsize=15, fontweight='bold')

    # Plot bars with error bars
    x = np.arange(len(labels))
    ax.bar(x - 0.2, flanking_means, 0.4, yerr=flanking_se, capsize=5, label='Flanking Regions (100K each end)', color='skyblue')
    ax.bar(x + 0.2, middle_means, 0.4, yerr=middle_se, capsize=5, label='Middle Region', color='salmon')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Mean Fst', fontsize=16)
    ax.set_title('Mean Fst: Flanking vs Middle Regions', fontsize=18)
    ax.tick_params(axis='y', labelsize=13)
    ax.legend(title="Regions", fontsize=13, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logger.info(f"Saved plot to {OUTPUT_PLOT}")

def main():
    start_time = time.time()
    logger.info("Starting fst flanking regions analysis...")
    inversion_df = pd.read_csv(INVERSION_FILE, sep='\t')
    recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
    fst_sequences = load_fst_data(FST_DATA_FILE)
    flanking_means = calculate_flanking_means(fst_sequences)
    categories = categorize_sequences(flanking_means, recurrent_regions, single_event_regions)
    create_bar_plot(categories)
    elapsed = time.time() - start_time
    logger.info(f"Analysis completed successfully in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
