import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import sys
import time
from scipy.stats import mannwhitneyu

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('fst_analysis')

# Constants
MIN_LENGTH = 1

# File paths
FST_DATA_FILE = 'per_site_fst_output.falsta'
INVERSION_FILE = 'inv_info.csv'
OUTPUT_PLOT = 'fst_recurrent_vs_single_event.png'

# Category mapping
cat_mapping = {
    'Recurrent': 'recurrent',
    'Single-event': 'single_event'
}

def normalize_chromosome(chrom):
    """Normalize chromosome names to a consistent format."""
    chrom = chrom.strip()
    if chrom.startswith('chr_'):
        chrom = chrom[4:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    return f"chr{chrom}"

def extract_coordinates_from_header(header):
    """Extract chromosome, start, and end coordinates from a header string."""
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
    """Map genomic regions to recurrent or single-event inversions."""
    recurrent_regions = {}
    single_event_regions = {}
    for _, row in inversion_df.iterrows():
        chrom = normalize_chromosome(str(row['Chromosome']))
        start, end = int(row['Start']), int(row['End'])
        if row['0_single_1_recur'] == 1:
            recurrent_regions.setdefault(chrom, []).append((start, end))
        else:
            single_event_regions.setdefault(chrom, []).append((start, end))
    return recurrent_regions, single_event_regions

def is_overlapping(region1, region2):
    """Check if two genomic regions overlap."""
    return region1[0] <= region2[1] and region1[1] >= region2[0]

def determine_inversion_type(coords, recurrent_regions, single_event_regions):
    """Determine if a region is recurrent, single-event, ambiguous, or unknown."""
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
    """Load Fst data from file, filtering sequences by minimum length."""
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
                if len(data) >= MIN_LENGTH and not np.all(np.isnan(data)):
                    coords = extract_coordinates_from_header(header)
                    if coords:
                        sequences.append({'coords': coords, 'data': data})
                header = None
    logger.info(f"Loaded {len(sequences)} fst sequences")
    return sequences

def calculate_sequence_means(sequences):
    """Calculate mean Fst for each sequence, excluding NaN values."""
    results = []
    for seq in sequences:
        data = seq['data']
        valid_fst = data[~np.isnan(data)]  # Exclude NaN values
        if len(valid_fst) > 0:
            mean_fst = np.mean(valid_fst)
            results.append({
                'coords': seq['coords'],
                'mean_fst': mean_fst
            })
    return results

def categorize_sequences(means, recurrent_regions, single_event_regions):
    """Categorize sequences into recurrent or single-event based on inversion type."""
    categories = {v: [] for v in cat_mapping.values()}
    for seq in means:
        inv_type = determine_inversion_type(seq['coords'], recurrent_regions, single_event_regions)
        if inv_type in ['recurrent', 'single_event']:
            categories[inv_type].append(seq['mean_fst'])
    return categories

def create_combined_plot(categories):
    """Create a bar and jitter plot comparing Fst between categories."""
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(cat_mapping.keys())
    means, ses, individual_points = [], [], []

    # Calculate statistics for each category
    for label in labels:
        cat = cat_mapping[label]
        fst_values = categories[cat]
        if fst_values:
            mean_fst = np.mean(fst_values)
            median_fst = np.median(fst_values)
            std_fst = np.std(fst_values, ddof=1) if len(fst_values) > 1 else 0
            se_fst = std_fst / np.sqrt(len(fst_values))
            means.append(mean_fst)
            ses.append(se_fst)
            individual_points.append(fst_values)
            logger.info(f"{label}: Mean Fst = {mean_fst:.4f}, Median Fst = {median_fst:.4f}, N = {len(fst_values)}")
        else:
            means.append(np.nan)
            ses.append(0)
            individual_points.append([])
            logger.info(f"{label}: No valid sequences")

    # Bar plot with error bars
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=ses, capsize=5, color=['#1f77b4', '#ff7f0e'], alpha=0.7, label='Mean Fst')

    # Jitter plot for individual points
    for i, points in enumerate(individual_points):
        if points:
            x_jitter = np.random.normal(i, 0.1, size=len(points))  # Add jitter
            ax.scatter(x_jitter, points, color='black', alpha=0.5, s=20, label='Individual Sequences' if i == 0 else None)

    # Perform Mann-Whitney U test
    recurrent_fst = categories[cat_mapping['Recurrent']]
    single_event_fst = categories[cat_mapping['Single-event']]
    if recurrent_fst and single_event_fst and len(recurrent_fst) > 0 and len(single_event_fst) > 0:
        stat, p_value = mannwhitneyu(recurrent_fst, single_event_fst, alternative='two-sided')
        text = f"Mann-Whitney p={p_value:.4g}"
        logger.info(f"Mann-Whitney U test: statistic={stat}, p-value={p_value:.4g}")
    else:
        text = "Insufficient data for test"
        logger.info("Insufficient data for Mann-Whitney U test")

    # Add p-value text to plot
    max_y = max([m + se for m, se in zip(means, ses)] + [max(p) for p in individual_points if p]) * 1.1
    ax.text(0.5, max_y, text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize plot aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Mean Fst', fontsize=12)
    ax.set_title('Mean Fst in Recurrent vs Single-event Regions', fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logger.info(f"Saved plot to {OUTPUT_PLOT}")

def main():
    """Main function to run the Fst analysis."""
    start_time = time.time()
    logger.info("Starting fst analysis for recurrent vs single-event regions...")
    
    inversion_df = pd.read_csv(INVERSION_FILE)
    recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
    fst_sequences = load_fst_data(FST_DATA_FILE)
    sequence_means = calculate_sequence_means(fst_sequences)
    
    # Calculate overall mean and median FST
    all_fst_values = [seq['mean_fst'] for seq in sequence_means]
    if all_fst_values:
        overall_mean_fst = np.mean(all_fst_values)
        overall_median_fst = np.median(all_fst_values)
        logger.info(f"Overall: Mean Fst = {overall_mean_fst:.4f}, Median Fst = {overall_median_fst:.4f}, N = {len(all_fst_values)}")
    else:
        logger.info("No valid sequences found")
    
    categories = categorize_sequences(sequence_means, recurrent_regions, single_event_regions)
    create_combined_plot(categories)
    
    elapsed = time.time() - start_time
    logger.info(f"Analysis completed successfully in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
