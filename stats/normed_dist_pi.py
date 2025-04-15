import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import multiprocessing # For parallel processing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, sem # Spearman correlation, Standard Error of Mean
from statsmodels.nonparametric.smoothers_lowess import lowess # LOWESS smoother

# --- Configuration ---

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Create a specific logger for this analysis
logger = logging.getLogger('pi_overall_distance_edge_analysis_fast_lenfilter')

# Constants
MIN_LENGTH = 150_000 # <<<--- ADDED BACK Minimum sequence length filter
NUM_BINS = 100       # Number of bins for normalized distance (High resolution for smoothing)
LOWESS_FRAC = 0.3    # Fraction of data used for LOWESS smoothing (Increased for more smoothness)
CONF_INTERVAL_ALPHA = 0.15 # Transparency for the SEM confidence band
SCATTER_ALPHA = 0.15       # Transparency for underlying bin points (make fainter)
SCATTER_SIZE = 8          # Size for underlying bin points

# Performance Setting
# Use N-1 cores for parallel processing to leave one for system tasks
N_CORES = max(1, multiprocessing.cpu_count() - 1)

# File paths (Update these paths if necessary)
PI_DATA_FILE = 'per_site_output.falsta'
# No inversion file needed for overall analysis
OUTPUT_DIR = Path('pi_analysis_results_overall_distance_edge_fast_lenfilter') # Unique output directory

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
PLOT_COLOR = plt.cm.viridis(0.6) # Use a distinct color for the overall trend

# --- Helper Functions (Optimized & Minimal) ---

def extract_coordinates_from_header(header: str) -> dict | None:
    """Extracts coordinates, requires 'filtered_pi' (reused, minimal)."""
    match = re.search(r'>.*?filtered_pi.*?_chr_?([\w\.\-]+)_start_(\d+)_end_(\d+)(?:_group_[01])?', header, re.IGNORECASE)
    if match:
        chrom_part = match.group(1)
        start_str = match.group(2)
        end_str = match.group(3)
        chrom = normalize_chromosome(chrom_part)
        if start_str.isdigit() and end_str.isdigit():
             start = int(start_str)
             end = int(end_str)
             if chrom is not None and start < end:
                  return {'chrom': chrom, 'start': start, 'end': end}
    return None

def normalize_chromosome(chrom: str) -> str | None:
    """Normalizes chromosome names (reused, minimal)."""
    if not isinstance(chrom, str): chrom = str(chrom)
    chrom_lower = chrom.strip().lower()
    if chrom_lower.startswith('chr_'):
        chrom_part = chrom_lower[4:]
    elif chrom_lower.startswith('chr'):
        chrom_part = chrom_lower[3:]
    else:
        chrom_part = chrom_lower
    if re.fullmatch(r'[a-z0-9_.-]+', chrom_part):
        return f"chr{chrom_part}"
    return None


def parse_pi_data_line(line: str) -> np.ndarray | None:
    """
    Parses comma-separated pi values into a numpy array. Optimized slightly.
    Handles 'NA' strings efficiently. Returns None if no valid numbers found.
    """
    try:
        values = line.split(',')
        float_values = []
        for x in values:
            val_str = x.strip()
            if val_str and val_str.upper() != 'NA':
                try:
                    float_values.append(float(val_str))
                except ValueError:
                    float_values.append(np.nan)
            else:
                float_values.append(np.nan)
        data = np.array(float_values, dtype=np.float32)
        if np.all(np.isnan(data)): return None
        return data
    except Exception as e:
        logger.error(f"Error parsing data line segment: {line[:50]}... Error: {e}", exc_info=False)
        return None

def load_pi_data(file_path: str | Path) -> list[dict]:
    """
    Loads 'filtered_pi' sequences. Applies MIN_LENGTH filter.
    Optimized for speed.
    """
    logger.info(f"Loading pi data from {file_path}")
    # <<<--- UPDATED LOG MESSAGE
    logger.info(f"Applying filters: Header must contain 'filtered_pi', Sequence length >= {MIN_LENGTH}")
    start_time = time.time()
    pi_sequences = []
    # <<<--- ADDED skipped_short counter
    skipped_not_filtered_pi, skipped_coord_error, skipped_data_error, skipped_short = 0, 0, 0, 0
    headers_read = 0
    buffer = {'header': None, 'parts': []}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                if line.startswith('>'):
                    headers_read += 1
                    # --- Process previous sequence stored in buffer ---
                    if buffer['header']:
                        full_sequence_line = "".join(buffer['parts'])
                        pi_data = parse_pi_data_line(full_sequence_line)
                        if pi_data is not None:
                            actual_length = len(pi_data) # Get length from parsed data
                            # <<<--- ADDED length check
                            if actual_length >= MIN_LENGTH:
                                coords = extract_coordinates_from_header(buffer['header'])
                                pi_sequences.append({
                                    'header': buffer['header'],
                                    'coords': coords,
                                    'data': pi_data,
                                    'length': actual_length, # Store actual length
                                })
                            else:
                                skipped_short += 1 # Increment if too short
                        else:
                            skipped_data_error += 1

                    # --- Reset buffer and check new header ---
                    buffer['header'] = None
                    buffer['parts'] = []
                    if 'filtered_pi' in line.lower():
                        buffer['header'] = line
                    else:
                        skipped_not_filtered_pi += 1

                # --- Append data line if current header is considered valid ---
                elif buffer['header']:
                    buffer['parts'].append(line)

            # --- Process the very last sequence left in the buffer after EOF ---
            if buffer['header']:
                full_sequence_line = "".join(buffer['parts'])
                pi_data = parse_pi_data_line(full_sequence_line)
                if pi_data is not None:
                    actual_length = len(pi_data)
                    # <<<--- ADDED length check for last sequence
                    if actual_length >= MIN_LENGTH:
                        coords = extract_coordinates_from_header(buffer['header'])
                        pi_sequences.append({
                            'header': buffer['header'],
                            'coords': coords,
                            'data': pi_data,
                            'length': actual_length,
                        })
                    else:
                        skipped_short += 1
                else:
                    skipped_data_error += 1

    except FileNotFoundError:
        logger.error(f"Fatal Error: Pi data file not found at {file_path}"); return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}", exc_info=True); return []

    elapsed_time = time.time() - start_time
    logger.info(f"Read {headers_read} headers in {elapsed_time:.2f} seconds.")
    # <<<--- UPDATED LOG MESSAGE
    logger.info(f"Loaded {len(pi_sequences)} sequences ('filtered_pi' header, length >= {MIN_LENGTH}).")
    # <<<--- UPDATED LOG MESSAGE
    logger.info(f"Skipped: {skipped_not_filtered_pi} (not 'filtered_pi'), {skipped_short} (too short), "
                f"{skipped_coord_error} (coord parse error - if check enabled), {skipped_data_error} (data parse error).")
    return pi_sequences

# --- Core Logic Functions (Optimized) ---

def calculate_norm_dist_from_center(index: int, total_length: int) -> float:
    """Calculates normalized distance from center (0=center, 1=edge)."""
    if total_length <= 1: return 0.0
    center_index = (total_length - 1) / 2.0
    distance_from_center = abs(index - center_index)
    half_length = total_length / 2.0
    if half_length <= 0: return 0.0
    normalized_distance = distance_from_center / half_length
    return min(normalized_distance, 1.0) # Clamp at 1.0

def _calculate_binned_pi_worker(args: Tuple[np.ndarray, int, int]) -> Optional[np.ndarray]:
    """
    Internal worker function for parallel processing.
    Calculates binned pi vs distance from center for a single sequence.
    Optimized with vectorized distance calculation.
    """
    sequence_data, sequence_length, num_bins = args
    if sequence_data is None or sequence_length < 1 or sequence_length != len(sequence_data):
        return None

    original_indices = np.arange(sequence_length, dtype=np.float64)
    center_index = (float(sequence_length) - 1.0) / 2.0
    half_length = float(sequence_length) / 2.0
    if half_length <= 0: return None

    norm_distances_from_center = np.minimum(1.0, np.abs(original_indices - center_index) / half_length)
    valid_indices_pi = ~np.isnan(sequence_data)
    valid_pi = sequence_data[valid_indices_pi].astype(np.float64)
    valid_distances = norm_distances_from_center[valid_indices_pi]
    if len(valid_pi) == 0: return None

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_edges[-1] += 1e-9
    bin_indices = np.digitize(valid_distances, bin_edges[1:], right=False)
    binned_pi_means = np.full(num_bins, np.nan, dtype=np.float64)
    bin_indices_int = bin_indices.astype(int)
    if np.any(bin_indices_int < 0) or np.any(bin_indices_int >= num_bins):
         # logger.warning("Worker issue: Bin indices out of bounds. Skipping sequence.") # Can be noisy
         return None
    try:
        sum_per_bin = np.bincount(bin_indices_int, weights=valid_pi, minlength=num_bins)
        count_per_bin = np.bincount(bin_indices_int, minlength=num_bins)
    except Exception as e:
        # logger.warning(f"Worker issue: Error during bincount: {e}. Skipping sequence.") # Can be noisy
        return None
    valid_counts_mask = count_per_bin > 0
    if np.any(valid_counts_mask):
        binned_pi_means[valid_counts_mask] = sum_per_bin[valid_counts_mask] / count_per_bin[valid_counts_mask]

    return binned_pi_means

def calculate_all_binned_pi_parallel(pi_sequences: List[Dict], num_bins: int) -> List[Optional[np.ndarray]]:
    """
    Calculates binned pi vs distance from center for all sequences in parallel.
    """
    if not pi_sequences: return []
    logger.info(f"Calculating binned Pi vs. distance from center for {len(pi_sequences)} sequences ({num_bins} bins) using {N_CORES} cores...")
    start_time = time.time()
    tasks = [(seq.get('data'), seq.get('length', 0), num_bins) for seq in pi_sequences]
    results = []
    try:
        chunk_size = max(1, len(tasks) // (N_CORES * 4))
        with multiprocessing.Pool(processes=N_CORES) as pool:
            results = pool.map(_calculate_binned_pi_worker, tasks, chunksize=chunk_size)
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel binning calculation finished in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}", exc_info=True)
        return []
    return results

def aggregate_binned_pi(all_binned_results: List[Optional[np.ndarray]], num_bins: int) -> pd.DataFrame:
    """
    Aggregates the per-sequence binned pi results (list of arrays/Nones).
    Optimized for performance assuming many sequences.
    """
    logger.info("Aggregating results across sequences...")
    start_time = time.time()
    valid_binned_data = [res for res in all_binned_results if isinstance(res, np.ndarray) and res.shape == (num_bins,)]
    if not valid_binned_data:
        logger.warning("No valid binned results found for aggregation.")
        return pd.DataFrame(columns=['bin_index', 'bin_center', 'mean_pi', 'stderr_pi', 'n_sequences'])
    try:
        stacked_binned_data = np.array([arr.astype(np.float64) for arr in valid_binned_data])
        logger.info(f"Stacked binned data shape: {stacked_binned_data.shape}")
    except ValueError as e:
         logger.error(f"Error stacking binned data, likely inconsistent shapes or types: {e}")
         raise

    n_sequences_per_bin = np.sum(~np.isnan(stacked_binned_data), axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_pi_per_bin = np.nanmean(stacked_binned_data, axis=0)
        stderr_pi_per_bin = np.full(num_bins, np.nan, dtype=np.float64)
        valid_sem_indices = n_sequences_per_bin > 1
        if np.any(valid_sem_indices):
             stderr_pi_per_bin[valid_sem_indices] = sem(stacked_binned_data[:, valid_sem_indices], axis=0, nan_policy='omit')

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    results_df = pd.DataFrame({
        'bin_index': np.arange(num_bins),
        'bin_center': bin_centers,
        'mean_pi': mean_pi_per_bin,
        'stderr_pi': stderr_pi_per_bin,
        'n_sequences': n_sequences_per_bin.astype(int)
    })
    results_df.loc[results_df['n_sequences'] <= 1, ['stderr_pi']] = np.nan
    results_df.loc[results_df['n_sequences'] == 0, ['mean_pi']] = np.nan

    elapsed_time = time.time() - start_time
    logger.info(f"Aggregation finished in {elapsed_time:.2f} seconds.")
    return results_df

def test_pi_decrease_from_center(binned_results_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Tests if pi significantly decreases with distance from center (Spearman)."""
    if not isinstance(binned_results_df, pd.DataFrame):
        logger.error("Invalid input to test_pi_decrease_from_center: Expected DataFrame.")
        return None, None
    valid_bins = binned_results_df.dropna(subset=['bin_center', 'mean_pi'])
    if len(valid_bins) < 5:
        logger.warning(f"Correlation test skipped: only {len(valid_bins)} valid bins with data.")
        return None, None
    try:
        rho, p_value = spearmanr(valid_bins['bin_center'], valid_bins['mean_pi'])
        return (rho, p_value) if pd.notna(rho) and pd.notna(p_value) else (None, None)
    except Exception as e:
        logger.error(f"Spearman correlation failed: {e}", exc_info=False)
        return None, None

# --- Plotting Function (Adjusted Labels and Legend) ---

def plot_overall_pi_trend_smoothed(
    aggregated_df: pd.DataFrame,
    correlation_result: Tuple[Optional[float], Optional[float]],
    num_bins: int
    ) -> Optional[plt.Figure]:
    """Plots the overall smoothed pi trend vs. distance from center."""
    logger.info("Generating smoothed plot (Overall Pi vs. Distance from Center)...")
    start_time = time.time()

    if aggregated_df.empty: logger.warning("Aggregated data empty, skipping plot."); return None
    plot_data = aggregated_df.dropna(subset=['bin_center', 'mean_pi']).copy()
    if len(plot_data) < 5: logger.warning(f"Insufficient data ({len(plot_data)} points) for LOWESS, skipping plot."); return None

    x_dist = plot_data['bin_center'].values
    y_pi = plot_data['mean_pi'].values
    y_err = aggregated_df.loc[plot_data.index, 'stderr_pi'].values
    # y_err NaNs align with y_pi NaNs for interpolation later
    y_err = np.where(np.isnan(y_pi), np.nan, y_err)

    try:
        smoothed = lowess(y_pi, x_dist, frac=LOWESS_FRAC, it=1, return_sorted=True)
        x_smooth = smoothed[:, 0]
        y_smooth = smoothed[:, 1]
    except Exception as e:
        logger.error(f"LOWESS smoothing failed: {e}. Skipping plot.", exc_info=False)
        return None

    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Underlying binned data points (faint scatter)
    ax.scatter(x_dist, y_pi, color=PLOT_COLOR, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, label=f'Binned Mean π ({num_bins} Bins)', zorder=1, marker='.')

    # 2. LOWESS smoothed line
    ax.plot(x_smooth, y_smooth, color=PLOT_COLOR, linewidth=2.5, label=f'Smoothed Trend (LOWESS, f={LOWESS_FRAC})', zorder=10)

    # 3. Confidence Interval (SEM of Binned Means around Smoothed Line)
    try:
        sort_idx = np.argsort(x_dist)
        valid_err_indices_orig = ~np.isnan(y_err[sort_idx])
        if np.any(valid_err_indices_orig):
            # Interpolate SEM onto the smoothed x-coordinates using only valid SEM points
            sem_interp = np.interp(x_smooth, x_dist[sort_idx][valid_err_indices_orig], y_err[sort_idx][valid_err_indices_orig], left=np.nan, right=np.nan)
            # interpolated SEM is only used where y_smooth is also valid
            valid_ci_indices = ~np.isnan(sem_interp) & ~np.isnan(y_smooth)
            if np.any(valid_ci_indices):
                ax.fill_between(x_smooth[valid_ci_indices],
                                (y_smooth - sem_interp)[valid_ci_indices],
                                (y_smooth + sem_interp)[valid_ci_indices],
                                color=PLOT_COLOR, alpha=CONF_INTERVAL_ALPHA, label='±1 SEM (of Binned Means)', zorder=5, edgecolor='none')
            else: logger.warning("Could not generate confidence interval band (interpolation/NaN issue).")
        else: logger.warning("No valid SEM values found for interpolation.")

    except Exception as e:
        logger.warning(f"Error generating SEM band: {e}", exc_info=False)

    # --- Customize Plot (Readable Labels, Legend Bottom Left) ---
    ax.set_xlabel("Normalized Distance from Segment Center (0 = Center, 1 = Edge)", fontsize=12)
    ax.set_ylabel("Mean Nucleotide Diversity (π per site)", fontsize=12)
    ax.set_title("Overall Trend: Nucleotide Diversity vs. Distance from Segment Center", fontsize=14, pad=15)

    # Add correlation info text box in upper left
    rho, p_val = correlation_result
    if rho is not None and p_val is not None:
        p_val_str = f"{p_val:.2g}" if p_val >= 0.001 else "< 0.001"
        corr_text = f"Trend Test (π vs Dist. from Center):\nSpearman ρ = {rho:.3f}, p = {p_val_str}"
        ax.text(0.03, 0.97, corr_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='grey'))

    # Place legend in bottom left
    ax.legend(loc='lower left', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
    ax.set_xlim(-0.05, 1.05)

    # Adjust y-limits based on smoothed data + error band
    # Use valid interpolated SEM values for limit calculation
    min_y_limit = np.nanmin(y_smooth[valid_ci_indices] - sem_interp[valid_ci_indices]) if np.any(valid_ci_indices) else np.nanmin(y_smooth)
    max_y_limit = np.nanmax(y_smooth[valid_ci_indices] + sem_interp[valid_ci_indices]) if np.any(valid_ci_indices) else np.nanmax(y_smooth)

    if pd.notna(min_y_limit) and pd.notna(max_y_limit):
        y_range = max_y_limit - min_y_limit
        if y_range > 0:
            y_buffer = y_range * 0.1
            y_min = max(0, min_y_limit - y_buffer)
            y_max = max_y_limit + y_buffer
            ax.set_ylim(y_min, y_max)
        else: ax.set_ylim(bottom=0)
    else: ax.set_ylim(bottom=0)

    sns.despine(ax=ax)
    try: fig.tight_layout()
    except Exception as e: logger.warning(f"Tight layout failed: {e}")

    # Save plot
    output_filename = OUTPUT_DIR / f"overall_pi_vs_dist_center_smoothed_{num_bins}bins_fast.png"
    try:
        plt.savefig(output_filename, dpi=300)
        logger.info(f"Saved overall smoothed pi trend plot to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save overall smoothed pi trend plot: {e}")

    elapsed_time = time.time() - start_time
    logger.info(f"Created and saved overall plot in {elapsed_time:.2f} seconds.")
    return fig


# --- Main Execution ---

def main():
    total_start_time = time.time()
    logger.info("--- Starting Overall Pi vs. Distance from Center Analysis (Fast, WITH Length Filter) ---") # Updated Title

    # --- Load Pi Data (Faster, WITH Length Filter) ---
    pi_file_path = Path(PI_DATA_FILE)
    if not pi_file_path.is_file(): logger.error(f"Pi data file not found: {pi_file_path}."); return
    pi_sequences = load_pi_data(pi_file_path)
    if not pi_sequences: logger.error("No valid sequences loaded (check filters). Exiting."); return # Updated message

    # --- Calculate Binned Pi per Sequence (Parallel) ---
    all_binned_results = calculate_all_binned_pi_parallel(pi_sequences, NUM_BINS)
    num_processed_sequences = sum(1 for r in all_binned_results if r is not None)
    if num_processed_sequences == 0:
        logger.error("Binned pi calculation failed for all loaded sequences. Cannot proceed.")
        return
    logger.info(f"Successfully calculated binned data for {num_processed_sequences} out of {len(pi_sequences)} loaded sequences.")

    # --- Aggregate Binned Results for Overall ---
    overall_agg_df = aggregate_binned_pi(all_binned_results, NUM_BINS)
    if overall_agg_df.empty:
        logger.error("Overall aggregation resulted in empty DataFrame. Cannot proceed.")
        return

    # --- Test Overall Trend ---
    logger.info("Testing overall trend (Spearman correlation)...")
    rho, p_val = test_pi_decrease_from_center(overall_agg_df)

    # --- Generate Plot ---
    fig = plot_overall_pi_trend_smoothed(overall_agg_df, (rho, p_val), NUM_BINS)

    # --- Final Summary ---
    logger.info("\n--- Overall Distance Analysis Summary ---")
    logger.info(f"Input Pi File: {PI_DATA_FILE}")
    # <<<--- UPDATED LOG MESSAGE
    logger.info(f"Filters Applied: Header contains 'filtered_pi', Min Length >= {MIN_LENGTH}")
    logger.info(f"Number of Sequences Processed for Binning: {num_processed_sequences}")
    logger.info(f"Number of Bins for Distance (0=Center, 1=Edge): {NUM_BINS}")
    logger.info(f"LOWESS Smoothing Fraction: {LOWESS_FRAC}")
    logger.info(f"Parallel Cores Used: {N_CORES}")

    logger.info("\nSpearman Correlation (Mean π vs. Normalized Distance from Center):")
    logger.info("-" * 60)
    if rho is not None and p_val is not None:
        p_val_str = f"{p_val:.3g}" if p_val >= 0.001 else "< 0.001"
        logger.info(f"  Spearman Rho (ρ): {rho:.4f}")
        logger.info(f"  P-value (2-sided): {p_val_str}")
        if rho < -0.1:
            logger.info("  Interpretation: Negative correlation suggests pi diversity tends")
            logger.info("                to decrease further away from the segment center.")
        elif rho > 0.1:
            logger.info("  Interpretation: Positive correlation suggests pi diversity tends")
            logger.info("                to *increase* further away from the segment center.")
        else:
             logger.info("  Interpretation: Correlation close to zero or non-significant.")
    else:
        logger.info("  Correlation test could not be performed (e.g., insufficient data).")
    logger.info("-" * 60)

    # Save aggregated data
    agg_csv_path = OUTPUT_DIR / "overall_pi_vs_dist_center_aggregated_data_fast_lenfilter.csv" # Updated filename
    try:
        overall_agg_df[['bin_index', 'bin_center', 'mean_pi', 'stderr_pi', 'n_sequences']].to_csv(
            agg_csv_path, index=False, float_format='%.6g'
        )
        logger.info(f"Aggregated binned data saved to {agg_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save aggregated data CSV: {e}")

    total_elapsed_time = time.time() - total_start_time
    logger.info(f"--- Analysis finished in {total_elapsed_time:.2f} seconds ---")

    if fig: plt.close(fig)

# multiprocessing works correctly when run as script
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
