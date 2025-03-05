import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from numba import njit

# JIT-compiled function for log distances
@njit
def compute_log_distances(positions, sequence_length):
    """Calculate log10 distance from nearest edge for each position."""
    log_dists = np.empty(len(positions), dtype=np.float32)
    for i in range(len(positions)):
        log_dists[i] = min(positions[i], sequence_length - 1 - positions[i])
    return np.log10(log_dists + 1)  # +1 to avoid log(0)

# Load data efficiently
def load_data(file_path):
    """Load filtered theta and pi data from file."""
    print("Loading data...")
    theta_labels, theta_data = [], []
    pi_labels, pi_data = [], []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        for i in range(len(lines) - 1):  # Avoid empty last line
            if 'filtered_theta' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                theta_labels.append(lines[i][1:].strip())
                theta_data.append(np.fromstring(values, sep=',', dtype=np.float32))
            elif 'filtered_pi' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                pi_labels.append(lines[i][1:].strip())
                pi_data.append(np.fromstring(values, sep=',', dtype=np.float32))
    
    print(f"Loaded {len(theta_labels)} theta and {len(pi_labels)} pi data lines")
    return (np.array(theta_labels, dtype=object), np.array(theta_data, dtype=object)), \
           (np.array(pi_labels, dtype=object), np.array(pi_data, dtype=object))

# Process data efficiently
def process_data(data_values):
    """Process sequences for non-zero values and zero-density."""
    line_nz_data = []  # Non-zero data per line
    line_zero_data = []  # Zero-density data per line
    all_nz_logs = []  # All non-zero log distances
    all_nz_vals = []  # All non-zero values
    all_closest = []  # Flags for closest to edge
    all_furthest = []  # Flags for furthest from edge
    
    for values in data_values:
        seq_len = len(values)
        positions = np.arange(seq_len, dtype=np.int32)
        log_dists = compute_log_distances(positions, seq_len)
        
        # Masks
        valid = ~np.isnan(values)  # Valid (non-NaN) positions
        nz = valid & (values != 0)  # Non-zero and valid
        zeros = valid & (values == 0)  # Zero and valid
        
        # Non-zero data
        if np.any(nz):
            nz_logs = log_dists[nz]
            nz_vals = values[nz]
            sort_idx = np.argsort(nz_logs)
            line_nz_data.append((nz_logs[sort_idx], nz_vals[sort_idx]))
            all_nz_logs.append(nz_logs)
            all_nz_vals.append(nz_vals)
            all_closest.append(nz_logs == np.min(nz_logs))
            all_furthest.append(nz_logs == np.max(nz_logs))
        else:
            line_nz_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
        
        # Zero-density data (percentage of zeros among valid positions)
        if np.any(valid):
            valid_logs = log_dists[valid]
            valid_vals = values[valid]
            zero_density = (valid_vals == 0).astype(np.float32) # zero_density = (valid_vals == 0).astype(np.float32) * (100.0 / np.sum(valid))
            sort_idx = np.argsort(valid_logs)
            line_zero_data.append((valid_logs[sort_idx], zero_density[sort_idx]))
        else:
            line_zero_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
    
    # Concatenate all non-zero data
    all_nz_logs = np.concatenate(all_nz_logs) if all_nz_logs else np.array([], dtype=np.float32)
    all_nz_vals = np.concatenate(all_nz_vals) if all_nz_vals else np.array([], dtype=np.float32)
    all_closest = np.concatenate(all_closest) if all_closest else np.array([], dtype=bool)
    all_furthest = np.concatenate(all_furthest) if all_furthest else np.array([], dtype=bool)
    
    return line_nz_data, line_zero_data, all_nz_logs, all_nz_vals, all_closest, all_furthest

# Generate plot efficiently
def create_plot(line_nz_data, line_zero_data, all_nz_logs, all_nz_vals, closest, furthest, metric, suffix, sigma=200, max_lines=100):
    """Create scatter plot with per-line smoothed signals and zero-density."""
    print(f"Creating {metric} plot with per-line smoothed signal and zero-density...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    if len(all_nz_logs) > 0:
        # Z-scores for coloring
        z_scores = np.clip((all_nz_vals - np.nanmean(all_nz_vals)) / np.nanstd(all_nz_vals), -5, 5)
        colors = plt.cm.coolwarm(plt.Normalize(-5, 5)(z_scores))
        
        # Scatter plot with batching
        ax1.scatter(all_nz_logs[closest], all_nz_vals[closest], c='black', s=15, alpha=0.7, edgecolors='none')
        ax1.scatter(all_nz_logs[furthest & ~closest], all_nz_vals[furthest & ~closest], c=colors[furthest & ~closest], 
                    s=15, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax1.scatter(all_nz_logs[~closest & ~furthest], all_nz_vals[~closest & ~furthest], c=colors[~closest & ~furthest], 
                    s=15, alpha=0.7, edgecolors='none')
        
        # Smoothed non-zero lines (limited to max_lines)
        for i, (logs, vals) in enumerate(line_nz_data[:max_lines]):
            if len(logs) > 0:
                smoothed = gaussian_filter1d(vals, sigma=sigma, mode='nearest')
                ax1.plot(logs, smoothed, color='black', lw=0.5, alpha=1.0)
        
        # Smoothed zero-density lines (limited to max_lines)
        for i, (logs, density) in enumerate(line_zero_data[:max_lines]):
            if len(logs) > 0:
                smoothed = gaussian_filter1d(density, sigma=sigma, mode='nearest')
                ax2.plot(logs, smoothed, color='red', ls='--', lw=0.5, alpha=0.8)
    else:
        print(f"No valid data for {metric} plot")
        plt.close(fig)
        return None
    
    # Customize plot
    ax1.set_title(f'{metric} vs. Log Distance from Edge', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Log10(Distance from Nearest Edge + 1)', fontsize=14)
    ax1.set_ylabel(f'{metric} Value', fontsize=14)
    ax2.set_ylabel('Smoothed % Zeros', fontsize=14, color='red')
    ax1.grid(True, ls='--', alpha=0.4)
    ax2.tick_params(axis='y', colors='red')
    
    # Save plot
    plot_path = Path.home() / f'distance_plot_{suffix}.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"{metric} plot saved to: {plot_path}")
    return plot_path

# Main function
def main():
    file_path = 'per_site_output.falsta'
    (theta_labels, theta_data), (pi_labels, pi_data) = load_data(file_path)
    
    if not theta_data.size and not pi_data.size:
        print("No data to process. Exiting.")
        return
    
    # Process data
    theta_nz, theta_zero, theta_logs, theta_vals, theta_close, theta_far = process_data(theta_data)
    pi_nz, pi_zero, pi_logs, pi_vals, pi_close, pi_far = process_data(pi_data)
    
    # Generate plots
    theta_plot = create_plot(theta_nz, theta_zero, theta_logs, theta_vals, theta_close, theta_far, 'Theta', 'theta')
    pi_plot = create_plot(pi_nz, pi_zero, pi_logs, pi_vals, pi_close, pi_far, 'Pi', 'pi')
    
    # Open plots
    for plot in [theta_plot, pi_plot]:
        if plot:
            if os.name == 'nt':
                os.startfile(plot)
            elif os.name == 'posix':
                os.system(f'open "{plot}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot}"')

if __name__ == "__main__":
    main()
