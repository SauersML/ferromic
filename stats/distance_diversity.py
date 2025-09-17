import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from numba import njit
from tqdm import tqdm
import time

# JIT-compiled function for linear distances
@njit
def compute_distances(positions, sequence_length):
    """Calculate linear distance from nearest edge."""
    dists = np.empty(len(positions), dtype=np.float32)
    for i in range(len(positions)):
        dists[i] = min(positions[i], sequence_length - 1 - positions[i])
    return dists

# Load data efficiently, limited to some sequences
def load_data(file_path, max_sequences=5000):
    """Load up to some theta and pi data sequences from file."""
    print(f"INFO: Loading data from {file_path} (max {max_sequences} sequences)")
    start_time = time.time()
    theta_labels, theta_data = [], []
    pi_labels, pi_data = [], []
    total_sequences = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        print(f"INFO: File read: {len(lines)} lines in {time.time() - start_time:.2f}s")
        for i in tqdm(range(len(lines) - 1), desc="Parsing lines", unit="line"):
            if total_sequences >= max_sequences:
                break
            if 'filtered_theta' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                theta_labels.append(lines[i][1:].strip())
                theta_data.append(np.fromstring(values, sep=',', dtype=np.float32))
                total_sequences += 1
                print(f"DEBUG: Theta line {len(theta_labels)} loaded: {theta_labels[-1]}, {len(theta_data[-1])} values")
            elif 'filtered_pi' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                pi_labels.append(lines[i][1:].strip())
                pi_data.append(np.fromstring(values, sep=',', dtype=np.float32))
                total_sequences += 1
                print(f"DEBUG: Pi line {len(pi_labels)} loaded: {pi_labels[-1]}, {len(pi_data[-1])} values")
    
    print(f"INFO: Loaded {len(theta_labels)} theta and {len(pi_labels)} pi lines (total {total_sequences}) in {time.time() - start_time:.2f}s")
    return (np.array(theta_labels, dtype=object), np.array(theta_data, dtype=object)), \
           (np.array(pi_labels, dtype=object), np.array(pi_data, dtype=object))

# Process data, keeping only points within some number from edge and filtering sequences by mean
def process_data(data_values, max_dist=5000):
    """Process sequences, keeping only points within some number from edge."""
    num_sd = 4  # Define number of standard deviations for filtering
    print(f"INFO: Processing {len(data_values)} sequences, max distance = {max_dist}")
    start_time = time.time()
    
    # Compute mean of each sequence and filter based on overall mean and SD
    sequence_means = [np.nanmean(values) for values in data_values]
    overall_mean = np.nanmean(sequence_means)
    overall_std = np.nanstd(sequence_means)
    keep_mask = [abs(mean - overall_mean) <= num_sd * overall_std for mean in sequence_means]
    filtered_data_values = [data_values[i] for i in range(len(data_values)) if keep_mask[i]]
    print(f"INFO: Filtered to {len(filtered_data_values)} sequences (removed {len(data_values) - len(filtered_data_values)} outliers > {num_sd} SD)")
    
    line_nz_data, line_zero_data = [], []
    all_nz_dists, all_nz_vals = [], []
    all_closest, all_furthest = [], []
    max_seq_len = 0
    
    for idx, values in enumerate(tqdm(filtered_data_values, desc="Processing sequences", unit="seq")):
        print(f"DEBUG: Sequence {idx + 1}: Length = {len(values)}")
        seq_len = len(values)
        max_seq_len = max(max_seq_len, seq_len)
        positions = np.arange(seq_len, dtype=np.int32)
        dists = compute_distances(positions, seq_len)
        
        # Filter for distances <= some number
        mask = dists <= max_dist
        dists = dists[mask]
        values = values[mask]
        print(f"DEBUG: Sequence {idx + 1}: {len(dists)} points within {max_dist} from edge")
        
        valid = ~np.isnan(values)
        nz = valid & (values != 0)
        zeros = valid & (values == 0)
        print(f"DEBUG: Sequence {idx + 1}: {np.sum(valid)} valid, {np.sum(nz)} non-zero, {np.sum(zeros)} zeros")
        
        if np.any(nz):
            nz_dists = dists[nz]
            nz_vals = values[nz]
            sort_idx = np.argsort(nz_dists)
            line_nz_data.append((nz_dists[sort_idx], nz_vals[sort_idx]))
            all_nz_dists.append(nz_dists)
            all_nz_vals.append(nz_vals)
            all_closest.append(nz_dists == np.min(nz_dists))
            all_furthest.append(nz_dists == np.max(nz_dists))
            print(f"DEBUG: Sequence {idx + 1}: Added {len(nz_dists)} non-zero points")
        else:
            line_nz_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            print(f"DEBUG: Sequence {idx + 1}: No non-zero data")
        
        if np.any(valid):
            valid_dists = dists[valid]
            valid_vals = values[valid]
            zero_density = (valid_vals == 0).astype(np.float32)
            sort_idx = np.argsort(valid_dists)
            line_zero_data.append((valid_dists[sort_idx], zero_density[sort_idx]))
            print(f"DEBUG: Sequence {idx + 1}: Added {len(valid_dists)} zero-density points")
        else:
            line_zero_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            print(f"DEBUG: Sequence {idx + 1}: No valid data for zero-density")
    
    all_nz_dists = np.concatenate(all_nz_dists) if all_nz_dists else np.array([], dtype=np.float32)
    all_nz_vals = np.concatenate(all_nz_vals) if all_nz_vals else np.array([], dtype=np.float32)
    all_closest = np.concatenate(all_closest) if all_closest else np.array([], dtype=bool)
    all_furthest = np.concatenate(all_furthest) if all_furthest else np.array([], dtype=bool)
    print(f"INFO: Processed {len(line_nz_data)} lines, {len(all_nz_dists)} non-zero points within {max_dist}, max_seq_len={max_seq_len} in {time.time() - start_time:.2f}s")
    
    return line_nz_data, line_zero_data, all_nz_dists, all_nz_vals, all_closest, all_furthest, max_seq_len

# Compute overall line, limited to some number
def compute_overall_line(line_data, sigma=50):
    """Compute overall line for distances up to some number."""
    print(f"INFO: Computing overall line for {len(line_data)} lines")
    start_time = time.time()
    if not line_data or all(len(dists) == 0 for dists, _ in line_data):
        print("WARNING: No valid data found")
        return np.array([]), np.array([])
    
    common_x = np.linspace(0, 5000, 500)  # Fixed range 0 to some number
    smoothed_vals = np.full((len(line_data), 500), np.nan, dtype=np.float32)
    
    for i, (dists, vals) in enumerate(tqdm(line_data, desc="Smoothing lines", unit="line")):
        if len(dists) > 0:
            smoothed = gaussian_filter1d(vals, sigma=sigma, mode='nearest')
            interpolated_smoothed = np.interp(common_x, dists, smoothed, left=np.nan, right=np.nan)
            smoothed_vals[i] = interpolated_smoothed
    
    overall_line = np.nanmean(smoothed_vals, axis=0)
    print(f"INFO: Overall line computed in {time.time() - start_time:.2f}s")
    return common_x, overall_line

# Generate plot, limited to some number
def create_plot(line_nz_data, line_zero_data, all_nz_dists, all_nz_vals, closest, furthest, metric, suffix, sigma=50):
    """Create plot for points within some number from edge."""
    print(f"\n=== Creating {metric} Plot (0-some number) ===")
    start_time = time.time()
    plt.style.use('seaborn-v0_8-white')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    ax1.tick_params(axis='both', labelsize=14)
    
    if len(all_nz_dists) == 0:
        print("WARNING: No valid data to plot")
        plt.close(fig)
        return None
    
    max_points = 5000
    if metric == 'Theta':
        for i, (dists, density) in enumerate(tqdm(line_zero_data, desc="Plotting Theta lines", unit="line")):
            if len(dists) > 0:
                if len(density) > max_points:
                    idx = np.linspace(0, len(density) - 1, max_points, dtype=int)
                    dists, density = dists[idx], density[idx]
                smoothed = gaussian_filter1d(density, sigma=sigma, mode='nearest')
                ax2.plot(dists, smoothed, color='red', ls='--', lw=0.5, alpha=0.8, label='Zero-Density' if i == 0 else None)
        
        common_x, overall_line = compute_overall_line(line_zero_data, sigma=sigma)
        if len(common_x) > 0:
            valid_idx = ~np.isnan(overall_line)
            if np.any(valid_idx):
                ax2.plot(common_x[valid_idx], overall_line[valid_idx], color='black', lw=2, alpha=0.8, label='Overall Zero-Density')
        ax2.legend(loc='upper right')
    else:  # Pi
        z_scores = np.clip((all_nz_vals - np.nanmean(all_nz_vals)) / np.nanstd(all_nz_vals), -5, 5)
        colors = plt.cm.coolwarm(plt.Normalize(-5, 5)(z_scores))
        ax1.scatter(all_nz_dists[closest], all_nz_vals[closest], c='black', s=15, alpha=0.7, edgecolors='none')
        ax1.scatter(all_nz_dists[furthest & ~closest], all_nz_vals[furthest & ~closest], c=colors[furthest & ~closest], 
                    s=15, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax1.scatter(all_nz_dists[~closest & ~furthest], all_nz_vals[~closest & ~furthest], c=colors[~closest & ~furthest], 
                    s=15, alpha=0.7, edgecolors='none')
        
        for i, (dists, vals) in enumerate(tqdm(line_nz_data, desc="Plotting Pi lines", unit="line")):
            if len(dists) > 0:
                if len(vals) > max_points:
                    idx = np.linspace(0, len(vals) - 1, max_points, dtype=int)
                    dists, vals = dists[idx], vals[idx]
                smoothed = gaussian_filter1d(vals, sigma=sigma, mode='nearest')
                ax1.plot(dists, smoothed, color='black', lw=0.5, alpha=1.0, label='Non-Zero' if i == 0 else None)
        
        common_x, overall_line = compute_overall_line(line_nz_data, sigma=sigma)
        if len(common_x) > 0:
            valid_idx = ~np.isnan(overall_line)
            if np.any(valid_idx):
                ax1.plot(common_x[valid_idx], overall_line[valid_idx], color='black', lw=2, alpha=0.8, label='Overall Non-Zero')
        ax1.legend(loc='upper left')
    
    ax1.set_title(f'{metric} vs. Distance from Edge (0-some number)', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Distance from Nearest Edge', fontsize=16)
    ax1.set_ylabel(f'{metric} Value', fontsize=16)
    ax2.set_ylabel('Zero-Density (Proportion)', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red', labelsize=13)
    ax1.set_xlim(0, 5000)  # Restrict x-axis to 0-some number
    
    plot_path = Path.home() / f'distance_plot_{suffix}_some number.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"INFO: {metric} plot completed in {time.time() - start_time:.2f}s")
    return plot_path

# Main function
def main():
    print("=== Starting Execution ===")
    start_time = time.time()
    file_path = 'per_site_diversity_output.falsta'
    (theta_labels, theta_data), (pi_labels, pi_data) = load_data(file_path)
    
    if not theta_data.size and not pi_data.size:
        print("WARNING: No data to process. Exiting.")
        return
    
    theta_nz, theta_zero, theta_dists, theta_vals, theta_close, theta_far, theta_max_len = process_data(theta_data)
    pi_nz, pi_zero, pi_dists, pi_vals, pi_close, pi_far, pi_max_len = process_data(pi_data)
    
    theta_plot = create_plot(theta_nz, theta_zero, theta_dists, theta_vals, theta_close, theta_far, 'Theta', 'theta')
    pi_plot = create_plot(pi_nz, pi_zero, pi_dists, pi_vals, pi_close, pi_far, 'Pi', 'pi')
    
    for plot in [theta_plot, pi_plot]:
        if plot:
            if os.name == 'nt':
                os.startfile(plot)
            elif os.name == 'posix':
                os.system(f'open "{plot}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot}"')
    print(f"INFO: Total execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
