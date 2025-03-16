import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import time
from pathlib import Path

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('pi_flanking_analysis')

# Constants
PI_DATA_FILE = 'per_site_output.falsta'
OUTPUT_PLOT = 'difference_vs_flank_size.png'

### Functions

#### Load Data
def load_pi_data(file_path):
    """Load filtered pi data, ensuring sequences meet minimum length."""
    logger.info(f"Loading pi data from {file_path}")
    start_time = time.time()
    
    pi_sequences = []
    with open(file_path, 'r') as f:
        header_line = None
        for line in f:
            line = line.strip()
            if line.startswith('>') and 'filtered_pi' in line:
                header_line = line
            elif header_line:
                data_values = line.split(',')
                actual_length = len(data_values)
                try:
                    data = np.array(
                        [float(x) if x.upper() != 'NA' and x.strip() != '' else np.nan for x in data_values],
                        dtype=np.float32
                    )
                    pi_sequences.append({
                        'header': header_line,
                        'data': data,
                        'length': actual_length
                    })
                except ValueError:
                    logger.warning(f"Parse error in line following {header_line}")
                header_line = None
    
    elapsed_time = time.time() - start_time
    logger.info(f"Loaded {len(pi_sequences)} sequences in {elapsed_time:.2f}s")
    return pi_sequences

#### Bootstrap Confidence Intervals
def bootstrap_ci(data, num_iterations=1000, ci=95):
    """Calculate 95% confidence interval for the mean using bootstrapping."""
    bootstrap_means = []
    for _ in range(num_iterations):
        resample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(resample))
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper

#### Main Analysis and Plotting
def main():
    """Run analysis and generate enhanced plot."""
    logger.info("Starting analysis...")
    total_start = time.time()
    
    # Load data
    pi_sequences = load_pi_data(PI_DATA_FILE)
    if not pi_sequences:
        logger.error("No sequences loaded. Exiting.")
        return
    
    # Generate 20 flanking sizes
    F_values = np.round(np.geomspace(200, 500000, num=20)).astype(int)
    logger.info(f"Analyzing flanking sizes: {F_values}")
    
    # Store results
    F_list = []
    avg_diff_list = []
    ci_lower_list = []
    ci_upper_list = []
    n_list = []
    
    # Process each flanking size
    for F in F_values:
        min_length = 3 * F
        filtered_sequences = [seq for seq in pi_sequences if seq['length'] >= min_length]
        
        if not filtered_sequences:
            logger.warning(f"No sequences for F={F}")
            continue
        
        differences = []
        for seq in filtered_sequences:
            data = seq['data']
            beginning_flank = data[:F]
            ending_flank = data[-F:]
            middle_region = data[F:-F]
            
            beginning_mean = np.nanmean(beginning_flank)
            ending_mean = np.nanmean(ending_flank)
            flanking_mean = np.nanmean([beginning_mean, ending_mean])
            middle_mean = np.nanmean(middle_region)
            
            if not np.isnan(flanking_mean) and not np.isnan(middle_mean):
                difference = middle_mean - flanking_mean
                differences.append(difference)
        
        if differences:
            avg_diff = np.mean(differences)
            lower, upper = bootstrap_ci(differences)
            n = len(differences)
            F_list.append(F)
            avg_diff_list.append(avg_diff)
            ci_lower_list.append(lower)
            ci_upper_list.append(upper)
            n_list.append(n)
            logger.info(f"F={F}, n={n}, avg_diff={avg_diff:.6f}, CI=[{lower:.6f}, {upper:.6f}]")
    
    # Create enhanced plot
    if F_list:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot with error bars
        ax.errorbar(
            F_list, avg_diff_list,
            yerr=[np.array(avg_diff_list) - ci_lower_list, ci_upper_list - np.array(avg_diff_list)],
            fmt='o-', 
            color='darkblue', 
            ecolor='skyblue', 
            capsize=5, 
            linewidth=2, 
            markersize=8, 
            label='Middle - Flanking'
        )
        
        # Add n= labels
        for F, avg_diff, n in zip(F_list, avg_diff_list, n_list):
            ax.text(
                F, avg_diff + 0.00015, f'n={n}', 
                ha='center', va='bottom', fontsize=9, color='black', 
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        # Customize plot
        ax.set_xscale('log')
        ax.set_xlabel('Flanking Region Size (bp)', fontsize=14, labelpad=10)
        ax.set_ylabel('Average Difference (Middle - Flanking)', fontsize=14, labelpad=10)
        ax.set_title(
            'Difference Between Middle and Flanking Regions\nAcross Flanking Sizes',
            fontsize=16, pad=15
        )
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=12, frameon=True, edgecolor='black')
        
        # Enhance ticks and grid
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {OUTPUT_PLOT}")
    
    total_time = time.time() - total_start
    logger.info(f"Completed in {total_time:.2f}s")

if __name__ == "__main__":
    main()
