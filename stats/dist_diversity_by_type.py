import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
import logging
import sys
import time
from scipy.stats import shapiro

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('pi_flanking_analysis')

# Constants
MIN_LENGTH = 300_000  # Minimum sequence length (300K bp)
FLANK_SIZE = 100_000  # Size of flanking regions (100K bp)

# File paths
PI_DATA_FILE = 'per_site_output.falsta'
INVERSION_FILE = 'inv_info.csv'
OUTPUT_PLOT = 'pi_flanking_regions_bar_plot.png'

cat_mapping = {
    'Recurrent Inverted': 'recurrent_inverted',
    'Recurrent Direct': 'recurrent_direct',
    'Single-event Inverted': 'single_event_inverted',
    'Single-event Direct': 'single_event_direct'
}

def normalize_chromosome(chrom):
    """
    Normalize chromosome name to consistent format.
    Ensures format is 'chrN' without duplicate prefixes.
    """
    # Strip any leading/trailing whitespace
    chrom = chrom.strip()
    
    # Remove any existing 'chr' or 'chr_' prefix
    if chrom.startswith('chr_'):
        chrom = chrom[4:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    
    # Add 'chr' prefix back
    return f"chr{chrom}"

def extract_coordinates_from_header(header):
    """Extract genomic coordinates from pi sequence header."""
    # Example: ">filtered_pi_chr_1_start_13104251_end_13122521_group_1"
    try:
        parts = header.strip().split('_')
        
        # Extract chromosome - should get "chrN" format
        chr_idx = -1
        chr_num = ""
        for i, part in enumerate(parts):
            if part == 'chr':
                chr_idx = i
                # Get the chromosome number/letter (should be the part after 'chr')
                if i+1 < len(parts):
                    chr_num = parts[i+1]
                break
                
        if chr_idx == -1:
            logger.warning(f"No 'chr' found in header: {header[:50]}...")
            return None
            
        if not chr_num:
            logger.warning(f"Could not extract chromosome number from header: {header[:50]}...")
            return None
            
        # Normalize chromosome name
        chrom = normalize_chromosome(f"chr{chr_num}")
        
        # Find indices for key parts
        start_idx = -1
        end_idx = -1
        group_idx = -1
        
        for i, part in enumerate(parts):
            if part == 'start':
                start_idx = i
            elif part == 'end':
                end_idx = i
            elif part == 'group':
                group_idx = i
                
        if start_idx == -1 or start_idx + 1 >= len(parts):
            logger.warning(f"No valid 'start' found in header: {header[:50]}...")
            return None
            
        if end_idx == -1 or end_idx + 1 >= len(parts):
            logger.warning(f"No valid 'end' found in header: {header[:50]}...")
            return None
            
        start = int(parts[start_idx+1])
        end = int(parts[end_idx+1])
        
        # Extract group information if available
        group = None
        if group_idx != -1 and group_idx + 1 < len(parts):
            group = int(parts[group_idx+1])
            
        result = {
            'chrom': chrom,
            'start': start,
            'end': end
        }
        
        if group is not None:
            result['group'] = group
            
        return result
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to extract coordinates from header: {header[:50]}... - {e}")
        
    return None

def map_regions_to_inversions(inversion_df):
    """
    Create mapping dictionaries for recurrent and single-event inversion regions.
    Returns two dictionaries mapping chromosome to a list of (start, end) tuples.
    """
    logger.info("Creating inversion region mappings...")
    
    # Ensure numeric columns are properly typed
    inversion_df['region_start'] = pd.to_numeric(inversion_df['region_start'], errors='coerce')
    inversion_df['region_end'] = pd.to_numeric(inversion_df['region_end'], errors='coerce')
    inversion_df['0_single_1_recur'] = pd.to_numeric(inversion_df['0_single_1_recur'], errors='coerce')
    
    # Split inversion df by type
    recurrent_inv = inversion_df[inversion_df['0_single_1_recur'] == 1].copy()
    single_event_inv = inversion_df[inversion_df['0_single_1_recur'] == 0].copy()
    
    logger.info(f"Found {len(recurrent_inv)} recurrent and {len(single_event_inv)} single-event inversion regions")
    
    # Create mappings
    recurrent_regions = {}
    single_event_regions = {}
    
    # Process recurrent inversions
    for _, row in recurrent_inv.iterrows():
        if pd.isna(row['region_start']) or pd.isna(row['region_end']):
            continue
            
        # Normalize chromosome name
        chrom = normalize_chromosome(str(row['chr']))
        start = int(row['region_start'])
        end = int(row['region_end'])
        
        if chrom not in recurrent_regions:
            recurrent_regions[chrom] = []
            
        recurrent_regions[chrom].append((start, end))
        
    # Process single-event inversions
    for _, row in single_event_inv.iterrows():
        if pd.isna(row['region_start']) or pd.isna(row['region_end']):
            continue
            
        # Normalize chromosome name
        chrom = normalize_chromosome(str(row['chr']))
        start = int(row['region_start'])
        end = int(row['region_end'])
        
        if chrom not in single_event_regions:
            single_event_regions[chrom] = []
            
        single_event_regions[chrom].append((start, end))
    
    # Count total regions
    recurrent_count = sum(len(regions) for regions in recurrent_regions.values())
    single_event_count = sum(len(regions) for regions in single_event_regions.values())
    
    # Display chromosome names for verification
    if recurrent_regions:
        logger.info(f"Recurrent regions chromosomes: {sorted(list(recurrent_regions.keys()))}")
    if single_event_regions:
        logger.info(f"Single-event regions chromosomes: {sorted(list(single_event_regions.keys()))}")
    
    logger.info(f"Created mappings: {recurrent_count} recurrent regions across {len(recurrent_regions)} chromosomes")
    logger.info(f"{single_event_count} single-event regions across {len(single_event_regions)} chromosomes")
    
    return recurrent_regions, single_event_regions

def is_overlapping(region1, region2):
    """Check if two regions overlap."""
    start1, end1 = region1
    start2, end2 = region2
    
    return start1 <= end2 and end1 >= start2

def determine_inversion_type(coords, recurrent_regions, single_event_regions):
    """
    Determine inversion type (recurrent, single_event, ambiguous, or unknown)
    based on genomic coordinates.
    """
    chrom = coords['chrom']
    start = coords['start']
    end = coords['end']
    
    is_recurrent = False
    is_single_event = False
    
    # Check recurrent regions
    if chrom in recurrent_regions:
        for region_start, region_end in recurrent_regions[chrom]:
            if is_overlapping((start, end), (region_start, region_end)):
                is_recurrent = True
                break
    
    # Check single-event regions
    if chrom in single_event_regions:
        for region_start, region_end in single_event_regions[chrom]:
            if is_overlapping((start, end), (region_start, region_end)):
                is_single_event = True
                break
    
    # Determine type
    if is_recurrent and not is_single_event:
        return 'recurrent'
    elif is_single_event and not is_recurrent:
        return 'single_event'
    elif is_recurrent and is_single_event:
        return 'ambiguous'
    else:
        return 'unknown'


def paired_permutation_test(x, y, num_permutations=10000):
    """
    Perform a paired permutation test.
    
    Parameters:
    x : array-like, sample values for group 1.
    y : array-like, sample values for group 2.
    num_permutations : int, number of permutations to perform.
    
    Returns:
    p_value : float, the p-value of the test.
    """
    differences = np.array(x) - np.array(y)
    observed_mean = np.mean(differences)
    count = 0
    for _ in range(num_permutations):
        signs = np.random.choice([1, -1], size=len(differences))
        permuted = differences * signs
        permuted_mean = np.mean(permuted)
        if abs(permuted_mean) >= abs(observed_mean):
            count += 1
    p_value = count / num_permutations
    return p_value


def load_pi_data(file_path):
    """Load filtered pi data from file, counting ACTUAL data length."""
    logger.info(f"Loading pi data from {file_path}")
    start_time = time.time()
    
    pi_sequences = []
    skipped_short = 0
    skipped_parse_error = 0
    skipped_not_filtered_pi = 0
    
    with open(file_path, 'r') as f:
        line_count = 0
        header_line = None
        
        for line in f:
            line_count += 1
            line = line.strip()
            
            if line.startswith('>'):
                # Only process headers that specifically have filtered_pi
                if 'filtered_pi' in line:
                    header_line = line
                else:
                    skipped_not_filtered_pi += 1
                    header_line = None
            elif header_line:  # This means we have a filtered_pi header
                # This is a data line following a filtered_pi header
                data_values = line.split(',')
                actual_length = len(data_values)  # Count actual length, not from coordinates
                
                if actual_length >= MIN_LENGTH:
                    # Extract coordinates from header
                    coords = extract_coordinates_from_header(header_line)
                    
                    if coords:
                        # Convert data to numeric values
                        try:
                            data = np.array([float(x) if x.upper() != 'NA' and x.strip() != '' else np.nan for x in data_values],
                                         dtype=np.float32)
                            
                            # Determine the group (inverted or direct) based on the header
                            # Group 1 = inverted haplotype, Group 0 = direct haplotype
                            is_inverted = False
                            if 'group' in coords:
                                is_inverted = (coords['group'] == 1)
                            
                            pi_sequences.append({
                                'header': header_line,
                                'coords': coords,
                                'data': data,
                                'length': actual_length,
                                'is_inverted': is_inverted
                            })
                            
                            if len(pi_sequences) % 100 == 0:
                                logger.info(f"Loaded {len(pi_sequences)} pi sequences so far...")
                                
                        except ValueError as e:
                            logger.warning(f"Value error parsing data: {e}")
                            skipped_parse_error += 1
                    else:
                        logger.warning(f"Failed to extract coordinates from header: {header_line}")
                        skipped_parse_error += 1
                else:
                    skipped_short += 1
                
                # Reset header line
                header_line = None
    
    elapsed_time = time.time() - start_time
    logger.info(f"Loaded {len(pi_sequences)} filtered pi sequences with length >= {MIN_LENGTH} in {elapsed_time:.2f} seconds")
    logger.info(f"Skipped {skipped_short} sequences that were too short")
    logger.info(f"Skipped {skipped_parse_error} sequences due to parsing errors")
    logger.info(f"Skipped {skipped_not_filtered_pi} sequences that were not filtered_pi")
    
    # Separate inverted and direct sequences
    inverted = [seq for seq in pi_sequences if seq['is_inverted']]
    direct = [seq for seq in pi_sequences if not seq['is_inverted']]
    
    logger.info(f"Found {len(inverted)} inverted sequences and {len(direct)} direct sequences")
    
    # Log chromosome distribution of sequences
    chrom_counts = {}
    for seq in pi_sequences:
        if 'coords' in seq and 'chrom' in seq['coords']:
            chrom = seq['coords']['chrom']
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
    
    logger.info(f"Chromosome distribution of pi sequences: {chrom_counts}")
    
    # Sample the first few sequences
    if pi_sequences:
        logger.info("Sample pi sequences:")
        for i, seq in enumerate(pi_sequences[:3]):
            logger.info(f"  Sequence {i+1}: {seq['coords']['chrom']} {seq['coords']['start']}-{seq['coords']['end']}, is_inverted={seq['is_inverted']}")
    
    return pi_sequences

def calculate_flanking_means(pi_sequences):
    """Calculate mean pi for beginning, middle, and ending regions."""
    logger.info(f"Calculating flanking and middle means for {len(pi_sequences)} pi sequences")
    start_time = time.time()
    
    results = []
    
    for i, seq in enumerate(pi_sequences):
        data = seq['data']
        
        if len(data) < 2 * FLANK_SIZE + 1:  # Need at least 2*FLANK_SIZE+1 points
            logger.warning(f"Sequence too short for analysis: {seq['header'][:50]}..., length={len(data)}")
            continue
            
        # Extract beginning, middle and ending regions
        beginning_flank = data[:FLANK_SIZE]
        ending_flank = data[-FLANK_SIZE:]
        middle_region = data[FLANK_SIZE:-FLANK_SIZE]
        
        # Calculate means, excluding NaN
        beginning_mean = np.nanmean(beginning_flank)
        ending_mean = np.nanmean(ending_flank)
        middle_mean = np.nanmean(middle_region)
        
        # Add to results with coordinates and inverted status
        results.append({
            'header': seq['header'],
            'coords': seq['coords'],
            'beginning_mean': beginning_mean,
            'middle_mean': middle_mean,
            'ending_mean': ending_mean,
            'is_inverted': seq['is_inverted']
        })
        
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i+1}/{len(pi_sequences)} sequences...")
    
    # Count inverted and direct sequences
    inverted_count = sum(1 for res in results if res['is_inverted'])
    direct_count = len(results) - inverted_count
    logger.info(f"Flanking means calculated: {inverted_count} inverted sequences, {direct_count} direct sequences")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Calculated means for {len(results)} sequences in {elapsed_time:.2f} seconds")
    return results

def categorize_sequences(flanking_means, recurrent_regions, single_event_regions):
    """
    Categorize sequences by inversion type and haplotype status.
    
    Categories:
    1. Recurrent inversion region, inverted haplotype
    2. Recurrent inversion region, direct haplotype
    3. Single-event inversion region, inverted haplotype
    4. Single-event inversion region, direct haplotype
    """
    logger.info("Categorizing sequences by inversion type and haplotype status...")
    start_time = time.time()
    
    # Create dictionary to hold categories
    categories = {
        'recurrent_inverted': [],
        'recurrent_direct': [],
        'single_event_inverted': [],
        'single_event_direct': []
    }
    
    # Keep track of statistics
    total_processed = 0
    no_inversion_type = 0
    no_valid_coords = 0
    
    # Process each sequence
    for seq in flanking_means:
        total_processed += 1
        coords = seq['coords']
        if not coords:
            no_valid_coords += 1
            continue
        
        # Determine if sequence is inverted (group=1) or direct (group=0)
        is_inverted = seq['is_inverted']
        
        # Determine inversion type
        inversion_type = determine_inversion_type(coords, recurrent_regions, single_event_regions)
        
        if inversion_type not in ['recurrent', 'single_event']:
            no_inversion_type += 1
            continue
        
        # Categorize based on inversion type and haplotype
        if inversion_type == 'recurrent':
            if is_inverted:
                categories['recurrent_inverted'].append(seq)
            else:
                categories['recurrent_direct'].append(seq)
        elif inversion_type == 'single_event':
            if is_inverted:
                categories['single_event_inverted'].append(seq)
            else:
                categories['single_event_direct'].append(seq)
    
    # Report category sizes and statistics
    logger.info(f"Total sequences processed: {total_processed}")
    logger.info(f"Sequences without inversion type: {no_inversion_type}")
    logger.info(f"Sequences without valid coordinates: {no_valid_coords}")
    logger.info(f"Categorized sequences:")
    logger.info(f"  Recurrent inverted: {len(categories['recurrent_inverted'])}")
    logger.info(f"  Recurrent direct: {len(categories['recurrent_direct'])}")
    logger.info(f"  Single-event inverted: {len(categories['single_event_inverted'])}")
    logger.info(f"  Single-event direct: {len(categories['single_event_direct'])}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Categorized sequences in {elapsed_time:.2f} seconds")
    
    return categories

def create_bar_plot(categories):
    """Create bar plot comparing mean pi values by category."""
    logger.info("Creating bar plot...")
    start_time = time.time()
    
    # Calculate means for each category and region
    # For flanking regions, combine beginning and ending values
    category_means = {
        'Recurrent Inverted': {
            'Flanking': np.nanmean([np.nanmean([seq['beginning_mean'], seq['ending_mean']]) for seq in categories['recurrent_inverted']]) if categories['recurrent_inverted'] else np.nan,
            'Middle': np.nanmean([seq['middle_mean'] for seq in categories['recurrent_inverted']]) if categories['recurrent_inverted'] else np.nan,
            'Count': len(categories['recurrent_inverted'])
        },
        'Recurrent Direct': {
            'Flanking': np.nanmean([np.nanmean([seq['beginning_mean'], seq['ending_mean']]) for seq in categories['recurrent_direct']]) if categories['recurrent_direct'] else np.nan,
            'Middle': np.nanmean([seq['middle_mean'] for seq in categories['recurrent_direct']]) if categories['recurrent_direct'] else np.nan,
            'Count': len(categories['recurrent_direct'])
        },
        'Single-event Inverted': {
            'Flanking': np.nanmean([np.nanmean([seq['beginning_mean'], seq['ending_mean']]) for seq in categories['single_event_inverted']]) if categories['single_event_inverted'] else np.nan,
            'Middle': np.nanmean([seq['middle_mean'] for seq in categories['single_event_inverted']]) if categories['single_event_inverted'] else np.nan,
            'Count': len(categories['single_event_inverted'])
        },
        'Single-event Direct': {
            'Flanking': np.nanmean([np.nanmean([seq['beginning_mean'], seq['ending_mean']]) for seq in categories['single_event_direct']]) if categories['single_event_direct'] else np.nan,
            'Middle': np.nanmean([seq['middle_mean'] for seq in categories['single_event_direct']]) if categories['single_event_direct'] else np.nan,
            'Count': len(categories['single_event_direct'])
        }
    }

    # Overall category by combining all sequences
    all_sequences = (categories['recurrent_inverted'] + categories['recurrent_direct'] +
                     categories['single_event_inverted'] + categories['single_event_direct'])
    overall_flanking = np.nanmean([np.nanmean([seq['beginning_mean'], seq['ending_mean']]) for seq in all_sequences]) if all_sequences else np.nan
    overall_middle = np.nanmean([seq['middle_mean'] for seq in all_sequences]) if all_sequences else np.nan
    category_means['Overall'] = {
            'Flanking': overall_flanking,
            'Middle': overall_middle,
            'Count': len(all_sequences)
    }
    
    # Print the calculated means
    for category, data in category_means.items():
        logger.info(f"{category}: Flanking={data['Flanking']:.6f}, Middle={data['Middle']:.6f}, Count={data['Count']}")
    
    # Set up figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define category order and labels
    category_order = ['Recurrent Inverted', 'Recurrent Direct', 'Single-event Inverted', 'Single-event Direct', 'Overall']
    x = np.arange(len(category_order))
    width = 0.35
    
    # Extract means for plotting
    flanking_means = [category_means[cat]['Flanking'] for cat in category_order]
    middle_means = [category_means[cat]['Middle'] for cat in category_order]
    counts = [category_means[cat]['Count'] for cat in category_order]
    
    # Create bars, as before
    rects1 = ax.bar(
        x - width/2,
        flanking_means,
        width,
        label='Flanking Regions (100K each end)',
        color='#66c2a5'
    )
    rects2 = ax.bar(
        x + width/2,
        middle_means,
        width,
        label='Middle Region',
        color='#fc8d62'
    )

    for i, cat in enumerate(category_order):
        if cat == "Overall":
            seq_list = categories["recurrent_inverted"] + categories["recurrent_direct"] + categories["single_event_inverted"] + categories["single_event_direct"]
        else:
            key = cat_mapping[cat]
            seq_list = categories[key]


        flanking_vals = [
            np.nanmean([seq['beginning_mean'], seq['ending_mean']])
            for seq in seq_list
            if not np.isnan(seq['beginning_mean']) and not np.isnan(seq['ending_mean'])
        ]
        middle_vals = [
            seq['middle_mean']
            for seq in seq_list
            if not np.isnan(seq['middle_mean'])
        ]

        paired_flanking, paired_middle = [], []
        for seq in seq_list:
            f_val = np.nanmean([seq['beginning_mean'], seq['ending_mean']])
            m_val = seq['middle_mean']
            if not np.isnan(f_val) and not np.isnan(m_val):
                paired_flanking.append(f_val)
                paired_middle.append(m_val)

        jitter_flanking_x = np.random.normal(0, 0.03, size=len(paired_flanking)) + (i - width/2)
        ax.scatter(
            jitter_flanking_x,
            paired_flanking,
            color='#66c2a5',
            edgecolors='black',
            alpha=0.7,
            s=30
        )

        jitter_middle_x = np.random.normal(0, 0.03, size=len(paired_middle)) + (i + width/2)
        ax.scatter(
            jitter_middle_x,
            paired_middle,
            color='#fc8d62',
            edgecolors='black',
            alpha=0.7,
            s=30
        )

        if len(paired_flanking) >= 2:
            differences = np.array(paired_flanking) - np.array(paired_middle)
            norm_stat, norm_p = shapiro(differences)
            logger.info(f"Normality test: stat = {norm_stat:.3g}, p = {norm_p:.3g}")
            perm_p_value = paired_permutation_test(
                np.array(paired_middle),
                np.array(paired_flanking),
                num_permutations=20000
            )
            logger.info(f"Category '{cat}': permutation p-value = {perm_p_value:.4g}")

        # Global maximum value from all bars
        global_max = max(flanking_means + middle_means)
        y_pos = global_max * 1.1  # Position annotations at 110% of the global maximum
        if not np.isnan(perm_p_value):
            ax.text(
                i,
                y_pos,
                f"Permutation p={perm_p_value:.3g}\nNormality p={norm_p:.3g}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logger.info(f"Saved plot to {OUTPUT_PLOT}")
    
    # Add a dashed line for better comparison
    if any(~np.isnan(flanking_means)) or any(~np.isnan(middle_means)):
        all_values = [v for v in flanking_means + middle_means if not np.isnan(v)]
        if all_values:
            max_y = max(all_values)
            ax.axhline(y=max_y/2, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title="Regions")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(OUTPUT_PLOT, dpi=300)
    logger.info(f"Saved plot to {OUTPUT_PLOT}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Created bar plot in {elapsed_time:.2f} seconds")
    
    return fig

def main():
    """Main function to run the analysis."""
    total_start_time = time.time()
    logger.info("Starting pi flanking regions analysis...")
    
    try:
        # Load inversion info
        logger.info(f"Loading inversion info from {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE)
        logger.info(f"Inversion info: {inversion_df.shape[0]} rows, {inversion_df.shape[1]} columns")
        
        # Display first few rows of inversion_df for debugging
        logger.info("First few rows of inversion_df:")
        for i, row in inversion_df.head(3).iterrows():
            logger.info(f"  Row {i}: chr={row['chr']}, region_start={row['region_start']}, region_end={row['region_end']}, type={row['0_single_1_recur']}")
        
        # Map regions to inversions
        recurrent_regions, single_event_regions = map_regions_to_inversions(inversion_df)
        
        # Load pi data (filtering for length > 300K)
        pi_sequences = load_pi_data(PI_DATA_FILE)
        
        if not pi_sequences:
            logger.error("No pi sequences loaded. Exiting.")
            return
        
        # Calculate flanking means (100K from each edge)
        flanking_means = calculate_flanking_means(pi_sequences)
        
        # Categorize sequences into the four required groups
        categories = categorize_sequences(flanking_means, recurrent_regions, single_event_regions)

        # Combine overall paired data from all sequences (ignoring categories)
        overall_flanking = []
        overall_middle = []
        for seq in flanking_means:
            # Compute the overall flanking mean (average of beginning and ending)
            f_val = np.nanmean([seq['beginning_mean'], seq['ending_mean']])
            m_val = seq['middle_mean']
            if not np.isnan(f_val) and not np.isnan(m_val):
                 overall_flanking.append(f_val)
                 overall_middle.append(m_val)
        
        if len(overall_flanking) >= 2:
            overall_perm_p = paired_permutation_test(np.array(overall_middle), np.array(overall_flanking), num_permutations=20000)
            logger.info(f"OVERALL test results: Permutation p={overall_perm_p:.3g}")
        
        # Create bar plot
        fig = create_bar_plot(categories)
        
        # Display plot if running in interactive environment
        if os.name == 'nt':  # Windows
            os.startfile(OUTPUT_PLOT)
        elif os.name == 'posix':  # MacOS/Linux
            cmd = 'open' if 'darwin' in os.sys.platform else 'xdg-open'
            os.system(f'{cmd} "{OUTPUT_PLOT}"')
        
        total_elapsed_time = time.time() - total_start_time
        logger.info(f"Analysis completed successfully in {total_elapsed_time:.2f} seconds")

        logger.info("\n==== SUMMARY OF RESULTS ====")
        logger.info("Category\tDirection of Effect\tP-value")
        
        for cat in ['Recurrent Inverted', 'Recurrent Direct', 'Single-event Inverted', 'Single-event Direct']:
            cat_key = cat_mapping[cat]
            seq_list = categories[cat_key]
            
            paired_flanking, paired_middle = [], []
            for seq in seq_list:
                f_val = np.nanmean([seq['beginning_mean'], seq['ending_mean']])
                m_val = seq['middle_mean']
                if not np.isnan(f_val) and not np.isnan(m_val):
                    paired_flanking.append(f_val)
                    paired_middle.append(m_val)
            
            if len(paired_flanking) >= 2:
                flanking_mean = np.mean(paired_flanking)
                middle_mean = np.mean(paired_middle)
                direction = "Flanking > Middle" if flanking_mean > middle_mean else "Middle > Flanking"
                
                perm_p_value = paired_permutation_test(
                    np.array(paired_middle),
                    np.array(paired_flanking),
                    num_permutations=20000
                )
                
                logger.info(f"{cat}\t{direction}\t{perm_p_value:.4g}")
            else:
                logger.info(f"{cat}\tNot enough data\tN/A")
        
        # Add overall results
        if len(overall_flanking) >= 2:
            overall_flanking_mean = np.mean(overall_flanking)
            overall_middle_mean = np.mean(overall_middle)
            overall_direction = "Flanking > Middle" if overall_flanking_mean > overall_middle_mean else "Middle > Flanking"
            
            overall_perm_p = paired_permutation_test(
                np.array(overall_middle),
                np.array(overall_flanking),
                num_permutations=20000
            )
            
            logger.info(f"Overall\t{overall_direction}\t{overall_perm_p:.4g}")
        else:
            logger.info(f"Overall\tNot enough data\tN/A")
        
        logger.info("===========================")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
