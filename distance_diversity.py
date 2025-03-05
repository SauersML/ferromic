import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from numba import njit
from tqdm import tqdm

# JIT-compiled function for log distances
@njit
def compute_log_distances(positions, sequence_length):
    """Calculate log10 distance from nearest edge for each position."""
    print(f"Computing log distances for sequence length {sequence_length}...")
    log_dists = np.empty(len(positions), dtype=np.float32)
    for i in range(len(positions)):
        log_dists[i] = min(positions[i], sequence_length - 1 - positions[i])
    return np.log10(log_dists + 1)  # +1 to avoid log(0)

# Load data efficiently
def load_data(file_path):
    """Load filtered theta and pi data from file."""
    print(f"Starting to load data from {file_path}...")
    theta_labels, theta_data = [], []
    pi_labels, pi_data = [], []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        print(f"Read {len(lines)} lines from file.")
        for i in tqdm(range(len(lines) - 1), desc="Parsing data lines"):
            if 'filtered_theta' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                theta_labels.append(lines[i][1:].strip())
                theta_data.append(np.fromstring(values, sep=',', dtype=np.float32))
                print(f"Loaded theta line {len(theta_labels)}: {theta_labels[-1]}")
            elif 'filtered_pi' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                pi_labels.append(lines[i][1:].strip())
                pi_data.append(np.fromstring(values, sep=',', dtype=np.float32))
                print(f"Loaded pi line {len(pi_labels)}: {pi_labels[-1]}")
    
    print(f"Completed loading: {len(theta_labels)} theta and {len(pi_labels)} pi data lines")
    return (np.array(theta_labels, dtype=object), np.array(theta_data, dtype=object)), \
           (np.array(pi_labels, dtype=object), np.array(pi_data, dtype=object))

# Process data efficiently
def process_data(data_values):
    """Process sequences for non-zero values and zero-density."""
    print(f"Processing {len(data_values)} data sequences...")
    line_nz_data = []
    line_zero_data = []
    all_nz_logs = []
    all_nz_vals = []
    all_closest = []
    all_furthest = []
    
    for idx, values in enumerate(tqdm(data_values, desc="Processing sequences")):
        print(f"Sequence {idx + 1}: Length = {len(values)}")
        seq_len = len(values)
        positions = np.arange(seq_len, dtype=np.int32)
        log_dists = compute_log_distances(positions, seq_len)
        
        valid = ~np.isnan(values)
        nz = valid & (values != 0)
        zeros = valid & (values == 0)
        print(f"Sequence {idx + 1}: {np.sum(valid)} valid, {np.sum(nz)} non-zero, {np.sum(zeros)} zeros")
        
        if np.any(nz):
            nz_logs = log_dists[nz]
            nz_vals = values[nz]
            sort_idx = np.argsort(nz_logs)
            line_nz_data.append((nz_logs[sort_idx], nz_vals[sort_idx]))
            all_nz_logs.append(nz_logs)
            all_nz_vals.append(nz_vals)
            all_closest.append(nz_logs == np.min(nz_logs))
            all_furthest.append(nz_logs == np.max(nz_logs))
            print(f"Sequence {idx + 1}: Added {len(nz_logs)} non-zero points")
        else:
            line_nz_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            print(f"Sequence {idx + 1}: No non-zero data")
        
        if np.any(valid):
            valid_logs = log_dists[valid]
            valid_vals = values[valid]
            zero_density = (valid_vals == 0).astype(np.float32)
            sort_idx = np.argsort(valid_logs)
            line_zero_data.append((valid_logs[sort_idx], zero_density[sort_idx]))
            print(f"Sequence {idx + 1}: Added {len(valid_logs)} zero-density points")
        else:
            line_zero_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            print(f"Sequence {idx + 1}: No valid data for zero-density")
    
    print("Concatenating non-zero data...")
    all_nz_logs = np.concatenate(all_nz_logs) if all_nz_logs else np.array([], dtype=np.float32)
    all_nz_vals = np.concatenate(all_nz_vals) if all_nz_vals else np.array([], dtype=np.float32)
    all_closest = np.concatenate(all_closest) if all_closest else np.array([], dtype=bool)
    all_furthest = np.concatenate(all_furthest) if all_furthest else np.array([], dtype=bool)
    print(f"Processed: {len(all_nz_logs)} total non-zero points")
    
    return line_nz_data, line_zero_data, all_nz_logs, all_nz_vals, all_closest, all_furthest

# Optimized function to compute overall smoothed line
def compute_overall_smoothed_line(line_data, sigma):
    """Compute an overall smoothed line by interpolating raw data, averaging, then smoothing once."""
    print(f"Computing overall smoothed line for {len(line_data)} lines...")
    if not line_data or all(len(logs) == 0 for logs, _ in line_data):
        print("No valid line data to process.")
        return np.array([]), np.array([])
    
    print("Defining common x-axis...")
    all_logs = np.concatenate([logs for logs, _ in line_data if len(logs) > 0])
    common_x = np.linspace(min(all_logs), max(all_logs), 500)
    print(f"Common x-axis: {len(common_x)} points, range [{min(all_logs):.2f}, {max(all_logs):.2f}]")
    
    interpolated_vals = []
    for i, (logs, vals) in enumerate(tqdm(line_data, desc="Interpolating lines")):
        if len(logs) > 0:
            print(f"Line {i + 1}: Interpolating {len(vals)} raw values to {len(common_x)} points...")
            interp_vals = np.interp(common_x, logs, vals, left=vals[0], right=vals[-1])
            interpolated_vals.append(interp_vals)
            print(f"Line {i + 1}: Interpolated to {len(interp_vals)} points")
    
    print("Averaging interpolated values...")
    overall_vals = np.mean(interpolated_vals, axis=0)
    print(f"Smoothing averaged line with {len(overall_vals)} points...")
    overall_smoothed = gaussian_filter1d(overall_vals, sigma=sigma, mode='nearest')
    print("Overall smoothing complete.")
    
    return common_x, overall_smoothed

# Generate plot with requested changes
def create_plot(line_nz_data, line_zero_data, all_nz_logs, all_nz_vals, closest, furthest, metric, suffix, sigma=400, max_lines=100):
    """Create plot with per-line smoothed signals, zero-density where applicable, and overall line."""
    print(f"Creating {metric} plot with per-line smoothed signal and overall line...")
    print(f"Input: {len(line_nz_data)} non-zero lines, {len(line_zero_data)} zero-density lines, {len(all_nz_logs)} total points")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    if len(all_nz_logs) > 0:
        if metric == 'Theta':
            print(f"Theta: Plotting up to {max_lines} zero-density lines...")
            for i, (logs, density) in enumerate(tqdm(line_zero_data[:max_lines], desc="Plotting Theta lines")):
                if len(logs) > 0:
                    print(f"Theta line {i + 1}: Smoothing {len(density)} points...")
                    smoothed = gaussian_filter1d(density, sigma=sigma, mode='nearest')
                    ax2.plot(logs, smoothed, color='red', ls='--', lw=0.5, alpha=0.8, label='Zero-Density' if i == 0 else None)
                    print(f"Theta line {i + 1}: Plotted")
            
            print("Theta: Computing overall zero-density line...")
            common_x, overall_smoothed = compute_overall_smoothed_line(line_zero_data, sigma)
            if len(common_x) > 0:
                ax2.plot(common_x, overall_smoothed, color='black', lw=2, alpha=0.8, label='Overall Zero-Density Average')
                print("Theta: Overall line plotted")
            
            ax2.legend(loc='upper right')
        else:  # Pi
            print("Pi: Plotting scatter points...")
            z_scores = np.clip((all_nz_vals - np.nanmean(all_nz_vals)) / np.nanstd(all_nz_vals), -5, 5)
            colors = plt.cm.coolwarm(plt.Normalize(-5, 5)(z_scores))
            ax1.scatter(all_nz_logs[closest], all_nz_vals[closest], c='black', s=15, alpha=0.7, edgecolors='none')
            ax1.scatter(all_nz_logs[furthest & ~closest], all_nz_vals[furthest & ~closest], c=colors[furthest & ~closest], 
                        s=15, alpha=0.7, edgecolors='black', linewidths=0.5)
            ax1.scatter(all_nz_logs[~closest & ~furthest], all_nz_vals[~closest & ~furthest], c=colors[~closest & ~furthest], 
                        s=15, alpha=0.7, edgecolors='none')
            print(f"Pi: Scatter plotted with {len(all_nz_logs)} points")
            
            print(f"Pi: Plotting up to {max_lines} non-zero lines...")
            for i, (logs, vals) in enumerate(tqdm(line_nz_data[:max_lines], desc="Plotting Pi lines")):
                if len(logs) > 0:
                    print(f"Pi line {i + 1}: Smoothing {len(vals)} points...")
                    smoothed = gaussian_filter1d(vals, sigma=sigma, mode='nearest')
                    ax1.plot(logs, smoothed, color='black', lw=0.5, alpha=1.0, label='Non-Zero' if i == 0 else None)
                    print(f"Pi line {i + 1}: Plotted")
            
            print("Pi: Computing overall non-zero line...")
            common_x, overall_smoothed = compute_overall_smoothed_line(line_nz_data, sigma)
            if len(common_x) > 0:
                ax1.plot(common_x, overall_smoothed, color='black', lw=2, alpha=0.8, label='Overall Non-Zero Average')
                print("Pi: Overall line plotted")
            
            ax1.legend(loc='upper left')
    else:
        print(f"No valid data for {metric} plot")
        plt.close(fig)
        return None
    
    print(f"Customizing {metric} plot...")
    ax1.set_title(f'{metric} vs. Log Distance from Edge', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Log10(Distance from Nearest Edge + 1)', fontsize=14)
    ax1.set_ylabel(f'{metric} Value', fontsize=14)
    ax2.set_ylabel('Smoothed % Zeros', fontsize=14, color='red')
    ax1.grid(True, ls='--', alpha=0.4)
    ax2.tick_params(axis='y', colors='red')
    
    plot_path = Path.home() / f'distance_plot_{suffix}.png'
    print(f"Saving {metric} plot to {plot_path}...")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"{metric} plot saved to: {plot_path}")
    return plot_path

# Main function
def main():
    print("Starting main execution...")
    file_path = 'per_site_output.falsta'
    (theta_labels, theta_data), (pi_labels, pi_data) = load_data(file_path)
    
    if not theta_data.size and not pi_data.size:
        print("No data to process. Exiting.")
        return
    
    print("Processing Theta data...")
    theta_nz, theta_zero, theta_logs, theta_vals, theta_close, theta_far = process_data(theta_data)
    print("Processing Pi data...")
    pi_nz, pi_zero, pi_logs, pi_vals, pi_close, pi_far = process_data(pi_data)
    
    print("Generating Theta plot...")
    theta_plot = create_plot(theta_nz, theta_zero, theta_logs, theta_vals, theta_close, theta_far, 'Theta', 'theta')
    print("Generating Pi plot...")
    pi_plot = create_plot(pi_nz, pi_zero, pi_logs, pi_vals, pi_close, pi_far, 'Pi', 'pi')
    
    print("Opening generated plots...")
    for plot in [theta_plot, pi_plot]:
        if plot:
            print(f"Opening {plot}...")
            if os.name == 'nt':
                os.startfile(plot)
            elif os.name == 'posix':
                os.system(f'open "{plot}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot}"')
    print("Execution completed.")

if __name__ == "__main__":
    main()
