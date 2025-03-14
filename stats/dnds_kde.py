import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import re
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('median_omega_analysis')

# File paths
PAIRWISE_FILE = 'all_pairwise_results.csv'
INVERSION_FILE = 'inv_info.csv'
OUTPUT_PLOT = 'median_omega_bar_plot.png'

def extract_coordinates_from_cds(cds_string):
    """Extract genomic coordinates from CDS string."""
    pattern = r'(chr\w+)_start(\d+)_end(\d+)'
    match = re.search(pattern, cds_string)
    if match:
        return {
            'chrom': match.group(1),
            'start': int(match.group(2)),
            'end': int(match.group(3))
        }
    logger.warning(f"Failed to extract coordinates from CDS: {cds_string}")
    return None

def load_input_files():
    """Load input files and perform basic validation."""
    try:
        logger.info(f"Loading pairwise results from {PAIRWISE_FILE}")
        pairwise_df = pd.read_csv(PAIRWISE_FILE)

        logger.info(f"Loading inversion info from {INVERSION_FILE}")
        inversion_df = pd.read_csv(INVERSION_FILE)

        logger.info(f"Pairwise results: {pairwise_df.shape[0]} rows, {pairwise_df.shape[1]} columns")
        logger.info(f"Inversion info: {inversion_df.shape[0]} rows, {inversion_df.shape[1]} columns")

        # Ensure numeric columns are properly typed
        inversion_df['region_start'] = pd.to_numeric(inversion_df['region_start'], errors='coerce')
        inversion_df['region_end'] = pd.to_numeric(inversion_df['region_end'], errors='coerce')
        inversion_df['0_single_1_recur'] = pd.to_numeric(inversion_df['0_single_1_recur'], errors='coerce')

        return pairwise_df, inversion_df

    except Exception as e:
        logger.error(f"Error loading input files: {e}")
        return None, None

def map_cds_to_inversions_excluding_ambiguous(pairwise_df, inversion_df):
    """
    Map CDS strings to inversion types, excluding any CDS that map to both types.
    """
    logger.info("Mapping CDS regions to inversion types...")

    # Extract unique CDS strings and their coordinates
    unique_cds = pairwise_df['CDS'].unique()
    logger.info(f"Found {len(unique_cds)} unique CDS strings")
                                                                                                                                                                                 
    # Create dictionary to store CDS coordinates                                                                                                                                 
    cds_coords = {}                                                                                                                                                              
    for cds in unique_cds:
        coords = extract_coordinates_from_cds(cds)                                                                                                                                   
        if coords:                                                                                                                                                                       
            cds_coords[cds] = coords                                                                                                                                                                                                                                                                                                                          
    logger.info(f"Successfully extracted coordinates for {len(cds_coords)} CDS strings")
                                                                                                                                                                                 
    # Split inversion df by type
    recurrent_inv = inversion_df[inversion_df['0_single_1_recur'] == 1].copy()
    single_event_inv = inversion_df[inversion_df['0_single_1_recur'] == 0].copy()

    logger.info(f"Inversion regions: {len(recurrent_inv)} recurrent, {len(single_event_inv)} single-event")

    # Map CDS to inversion types, excluding ambiguous cases
    recurrent_cds = set()
    single_event_cds = set()
    ambiguous_cds = set()

    for cds, coords in cds_coords.items():
        chrom = coords['chrom']
        start = coords['start']
        end = coords['end']                                                                                                                                                                                                                                                                                                                                       
        # Find matching inversion regions using overlap criteria                                                                                                                     
        rec_matches = recurrent_inv[                                                                                                                                                     
            (recurrent_inv['chr'] == chrom) &
            (                                                                                                                                                                                
                # CDS overlaps or is contained within inversion                                                                                                                              
                ((start <= recurrent_inv['region_end']) & (end >= recurrent_inv['region_start']))                                                                                        
            )                                                                                                                                                                        
        ]
                                                                                                                                                                                     
        single_matches = single_event_inv[                                                                                                                                               
            (single_event_inv['chr'] == chrom) &                                                                                                                                         
            (                                                                                                                                                                                
                # CDS overlaps or is contained within inversion
                ((start <= single_event_inv['region_end']) & (end >= single_event_inv['region_start']))                                                                                  
            )                                                                                                                                                                        
        ]                                                                                                                                                                                                                                                                                                                                                         
        # Determine inversion type
        if len(rec_matches) > 0 and len(single_matches) == 0:
            recurrent_cds.add(cds)
        elif len(single_matches) > 0 and len(rec_matches) == 0:
            single_event_cds.add(cds)
        elif len(rec_matches) > 0 and len(single_matches) > 0:
            # Ambiguous case - exclude from both sets
            ambiguous_cds.add(cds)

    # Report mapping results
    logger.info(f"CDS mapping results: {len(recurrent_cds)} recurrent, {len(single_event_cds)} single-event")
    logger.info(f"Ambiguous CDS (excluded): {len(ambiguous_cds)}")

    # Create mapping dictionary
    cds_to_type = {}
    for cds in recurrent_cds:
        cds_to_type[cds] = 'recurrent'
    for cds in single_event_cds:
        cds_to_type[cds] = 'single_event'

    return cds_to_type

def calculate_sequence_median_omega(pairwise_df):
    """
    Calculate median omega value for each unique sequence in each CDS.
    Each sequence (appearing in either Seq1 or Seq2) contributes to the median.
    """
    logger.info("Calculating median omega values for each sequence in each CDS...")

    # Apply GROUP 1 filtering
    group1_df = pairwise_df[(pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1)]
    logger.info(f"After GROUP 1 filtering: {group1_df.shape[0]} rows")

    # Find all rows where each sequence appears (in either Seq1 or Seq2)
    sequence_omega_values = {}  # {(cds, sequence): [omega values]}

    # Process each row in the GROUP 1 dataframe
    for _, row in group1_df.iterrows():
        cds = row['CDS']
        seq1 = row['Seq1']
        seq2 = row['Seq2']
        omega = row['omega']

        # Skip omega = 99 values (likely error codes)
        if omega == 99:
            continue

        # Add omega value for each sequence in this pair
        if (cds, seq1) not in sequence_omega_values:
            sequence_omega_values[(cds, seq1)] = []
        sequence_omega_values[(cds, seq1)].append(omega)

        if (cds, seq2) not in sequence_omega_values:
            sequence_omega_values[(cds, seq2)] = []
        sequence_omega_values[(cds, seq2)].append(omega)

    # Calculate median for each sequence
    sequence_median_omega = {}
    for (cds, seq), omega_values in sequence_omega_values.items():
        sequence_median_omega[(cds, seq)] = np.median(omega_values)

    logger.info(f"Calculated median omega for {len(sequence_median_omega)} CDS-sequence pairs")

    # Create a dataframe for analysis
    median_data = []
    for (cds, seq), median_omega in sequence_median_omega.items():
        median_data.append({
            'CDS': cds,
            'Sequence': seq,
            'median_omega': median_omega
        })

    median_df = pd.DataFrame(median_data)
    logger.info(f"Created median omega dataframe with {len(median_df)} rows")

    return median_df

def categorize_median_omega_values(median_df, cds_to_type):
    """
    Categorize median omega values into the three requested categories by inversion type.
    Only includes GROUP 1 data.
    """
    logger.info("Categorizing median omega values...")

    # Add inversion type column
    median_df['inversion_type'] = median_df['CDS'].map(cds_to_type)

    # Get rows for each inversion type
    recurrent_df = median_df[median_df['inversion_type'] == 'recurrent']
    single_event_df = median_df[median_df['inversion_type'] == 'single_event']

    logger.info(f"Median omega rows by type: {len(recurrent_df)} recurrent, {len(single_event_df)} single-event")

    # Define function to count categories
    def count_omega_categories(df):
        """Count median omega values in each category."""
        minus_one = (df['median_omega'] == -1).sum()
        zero_to_one = ((df['median_omega'] >= 0) & (df['median_omega'] <= 1)).sum()
        above_one = (df['median_omega'] > 1).sum()

        # Check for omega = 99 values
        omega_99 = (df['median_omega'] == 99).sum()
        if omega_99 > 0:
            logger.warning(f"Found {omega_99} rows with median_omega = 99 (not included in any category)")

        # Verify all rows are accounted for
        total = minus_one + zero_to_one + above_one + omega_99
        if total != len(df):
            logger.warning(f"Row count mismatch: {total} categorized vs {len(df)} total")

        return {
            "Exactly -1": minus_one,
            "0 to 1": zero_to_one,
            "Above 1": above_one
        }

    # Count categories for each type
    recurrent_counts = count_omega_categories(recurrent_df)
    single_event_counts = count_omega_categories(single_event_df)

    # Report counts
    logger.info("Recurrent median omega category counts:")
    for category, count in recurrent_counts.items():
        logger.info(f"  {category}: {count}")

    logger.info("Single-event median omega category counts:")
    for category, count in single_event_counts.items():
        logger.info(f"  {category}: {count}")

    return {
        'recurrent': recurrent_counts,
        'single_event': single_event_counts
    }

def plot_median_omega_categories_with_kde(count_data, pairwise_df, cds_to_type):
    """
    Create a two-panel figure:
    1. Top: Bar plot comparing median omega value categories between recurrent and single-event inversions
    2. Bottom: KDE plot comparing recurrent Group 1 inversions vs everything in Group 0
       (filtering out omega = -1 and omega = 99 values)
    """
    import matplotlib.patheffects as path_effects
    
    logger.info("Creating bar plot with KDE subplot...")
    
    # Define display categories (rename "-1" to "Identical Sequences")
    display_categories = ["Identical Sequences", "0 to 1", "Above 1"]
    internal_categories = ["Exactly -1", "0 to 1", "Above 1"]
    
    # Calculate totals for bar plot
    recurrent_total = sum(count_data['recurrent'].values())
    single_event_total = sum(count_data['single_event'].values())
    
    # Calculate percentages
    def calc_percentages(counts, total, categories):
        if total > 0:
            return [counts[cat] / total * 100 for cat in categories]
        else:
            return [0 for _ in categories]
    
    recurrent_pct = calc_percentages(count_data['recurrent'], recurrent_total, internal_categories)
    single_event_pct = calc_percentages(count_data['single_event'], single_event_total, internal_categories)
    
    # Log percentages
    logger.info("Recurrent percentages:")
    for i, cat in enumerate(internal_categories):
        if recurrent_total > 0:
            logger.info(f"  {cat}: {recurrent_pct[i]:.1f}% ({count_data['recurrent'][cat]}/{recurrent_total})")
    
    logger.info("Single-event percentages:")
    for i, cat in enumerate(internal_categories):
        if single_event_total > 0:
            logger.info(f"  {cat}: {single_event_pct[i]:.1f}% ({count_data['single_event'][cat]}/{single_event_total})")
    
    # Prepare KDE data - GROUP 1 RECURRENT vs ALL GROUP 0
    group1_df = pairwise_df[(pairwise_df['Group1'] == 1) & (pairwise_df['Group2'] == 1)].copy()
    group0_df = pairwise_df[(pairwise_df['Group1'] == 0) & (pairwise_df['Group2'] == 0)].copy()
    
    # Map CDS to inversion type for GROUP 1
    group1_df['inversion_type'] = group1_df['CDS'].map(cds_to_type)
    
    # Filter by inversion type and remove omega = -1 and omega = 99 values for KDE
    group1_recurrent_omega = group1_df[
        (group1_df['inversion_type'] == 'recurrent') & 
        (group1_df['omega'] != -1) & 
        (group1_df['omega'] != 99)
    ]['omega']
    
    group0_all_omega = group0_df[
        (group0_df['omega'] != -1) & 
        (group0_df['omega'] != 99)
    ]['omega']
    
    # Calculate means for KDE vertical lines
    group1_recurrent_mean = group1_recurrent_omega.mean() if len(group1_recurrent_omega) > 0 else 0
    group0_all_mean = group0_all_omega.mean() if len(group0_all_omega) > 0 else 0
    
    # Log KDE data
    logger.info(f"KDE data: {len(group1_recurrent_omega)} Group 1 recurrent values, {len(group0_all_omega)} Group 0 values")
    logger.info(f"Mean omega values: Group 1 recurrent = {group1_recurrent_mean:.3f}, Group 0 all = {group0_all_mean:.3f}")
    
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1]})
    
    # Define colors and patterns
    recurrent_color = 'salmon'
    single_event_color = 'skyblue'
    group0_color = 'purple'
    recurrent_pattern = '//'  # stripes
    single_event_pattern = '...'  # dots
    
    # PLOT 1: Bar plot of median omega categories
    x = np.arange(len(display_categories))
    width = 0.35  # width of bars
    
    # Plot bars
    # Recurrent
    rects1 = ax1.bar(x - width/2, recurrent_pct, width, 
                   label=f'Inverted Haplotypes Recurrent ({recurrent_total} sequences)', 
                   color=recurrent_color, hatch=recurrent_pattern, alpha=0.8)
    
    # Single-event
    rects2 = ax1.bar(x + width/2, single_event_pct, width, 
                   label=f'Inverted Haplotypes Single-event ({single_event_total} sequences)', 
                   color=single_event_color, hatch=single_event_pattern, alpha=0.8)
    
    # Add labels and title to bar plot
    ax1.set_ylabel('Percentage (%)', fontsize=14)
    ax1.set_title('Median Omega Value Distribution by Inversion Type (Inverted Haplotypes)', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_categories, fontsize=12)
    ax1.legend(fontsize=12)
    
    # Add value labels on bars
    def autolabel(rects, values, raw_counts):
        """Add percentage and count labels on bars."""
        for i, (rect, value) in enumerate(zip(rects, values)):
            height = rect.get_height()
            if height > 0:  # Only add label if bar has height
                # Add percentage on top of bar
                ax1.annotate(f'{value:.1f}%',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10)
                
                # Add count near base of bar with black text + white outline
                ax1.annotate(f'{raw_counts[i]} sequences',
                            xy=(rect.get_x() + rect.get_width()/2, height/2),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center', va='center',
                            fontsize=9, color='black', fontweight='bold',
                            path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    
    # Add value labels with counts
    autolabel(rects1, recurrent_pct, [count_data['recurrent'][cat] for cat in internal_categories])
    autolabel(rects2, single_event_pct, [count_data['single_event'][cat] for cat in internal_categories])
    
    # Add grid lines to bar plot
    ax1.grid(axis='y', alpha=0.3)
    
    # PLOT 2: KDE plot comparing GROUP 1 recurrent vs GROUP 0 all
    # Filter for visualization: limit to values between 0 and 10 for better visualization
    # but still calculate stats on all values
    group1_recurrent_viz = group1_recurrent_omega[(group1_recurrent_omega >= 0) & (group1_recurrent_omega <= 10)]
    group0_all_viz = group0_all_omega[(group0_all_omega >= 0) & (group0_all_omega <= 10)]
    
    # Plot KDEs with high bandwidth for smoothness
    if len(group1_recurrent_viz) > 1:
        sns.kdeplot(group1_recurrent_viz, ax=ax2, color=recurrent_color, 
                   label=f'Inverted Haplotypes Recurrent ({len(group1_recurrent_omega)} pairwise values, mean={group1_recurrent_mean:.3f})',
                   bw_adjust=1.5, fill=True, alpha=0.3)
    
    if len(group0_all_viz) > 1:
        sns.kdeplot(group0_all_viz, ax=ax2, color=group0_color, 
                   label=f'All Standard Orientation ({len(group0_all_omega)} pairwise values, mean={group0_all_mean:.3f})',
                   bw_adjust=1.5, fill=True, alpha=0.3)
    
    # Add vertical lines at means
    if len(group1_recurrent_omega) > 0:
        ax2.axvline(group1_recurrent_mean, color=recurrent_color, linestyle='--', linewidth=2,
                   path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])
    
    if len(group0_all_omega) > 0:
        ax2.axvline(group0_all_mean, color=group0_color, linestyle='--', linewidth=2,
                   path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])
    
    # Add labels and title to KDE plot
    ax2.set_xlabel('Omega Value', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title('Distribution of Pairwise Omega Values (Excluding -1 and 99)', fontsize=16)
    ax2.legend(fontsize=12)
    
    # Let the x range be determined naturally from the data
    # (No manual setting of x-axis limits)
    
    # Add grid lines to KDE plot
    ax2.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and display plot
    try:
        plt.savefig(OUTPUT_PLOT, dpi=300)
        logger.info(f"Saved plot to {OUTPUT_PLOT}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    
    plt.show()
    logger.info("Plot displayed")

def main():
    """Main execution function."""
    logger.info("Starting median omega bar plot analysis...")

    # Load input files
    pairwise_df, inversion_df = load_input_files()
    if pairwise_df is None or inversion_df is None:
        logger.error("Failed to load input files")
        return

    # Map CDS to inversion types, excluding ambiguous cases
    cds_to_type = map_cds_to_inversions_excluding_ambiguous(pairwise_df, inversion_df)
    if not cds_to_type:
        logger.error("Failed to map CDS to inversion types")
        return

    # Calculate median omega value for each sequence in each CDS
    median_df = calculate_sequence_median_omega(pairwise_df)

    # Categorize median omega values
    count_data = categorize_median_omega_values(median_df, cds_to_type)

    # Create the plots (bar plot with KDE subplot)
    plot_median_omega_categories_with_kde(count_data, pairwise_df, cds_to_type)

    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main()
