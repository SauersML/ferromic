import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.ndimage import gaussian_filter1d
from numba import njit
from tqdm import tqdm
import time

# JIT-compiled function for log distances (no print inside njit)
@njit
def compute_log_distances(positions, sequence_length):
    """Calculate log10 distance from nearest edge."""
    log_dists = np.empty(len(positions), dtype=np.float32)
    for i in range(len(positions)):
        log_dists[i] = min(positions[i], sequence_length - 1 - positions[i])
    return np.log10(log_dists + 1)  # +1 to avoid log(0)

# Load data efficiently
def load_data(file_path):
    """Load theta and pi data from file."""
    print(f"INFO: Loading data from {file_path}")
    start_time = time.time()
    theta_labels, theta_data = [], []
    pi_labels, pi_data = [], []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        print(f"INFO: File read: {len(lines)} lines in {time.time() - start_time:.2f}s")
        for i in tqdm(range(len(lines) - 1), desc="Parsing lines", unit="line"):
            if 'filtered_theta' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                theta_labels.append(lines[i][1:].strip())
                theta_data.append(np.fromstring(values, sep=',', dtype=np.float32))
                print(f"DEBUG: Theta line {len(theta_labels)} loaded: {theta_labels[-1]}, {len(theta_data[-1])} values")
            elif 'filtered_pi' in lines[i]:
                values = lines[i + 1].strip().replace('NA', 'nan')
                pi_labels.append(lines[i][1:].strip())
                pi_data.append(np.fromstring(values, sep=',', dtype=np.float32))
                print(f"DEBUG: Pi line {len(pi_labels)} loaded: {pi_labels[-1]}, {len(pi_data[-1])} values")
    
    print(f"INFO: Loaded {len(theta_labels)} theta and {len(pi_labels)} pi lines in {time.time() - start_time:.2f}s")
    return (np.array(theta_labels, dtype=object), np.array(theta_data, dtype=object)), \
           (np.array(pi_labels, dtype=object), np.array(pi_data, dtype=object))

# Process data efficiently
def process_data(data_values):
    """Process sequences into non-zero and zero-density lines."""
    print(f"INFO: Processing {len(data_values)} sequences")
    start_time = time.time()
    line_nz_data, line_zero_data = [], []
    all_nz_logs, all_nz_vals = [], []
    all_closest, all_furthest = [], []
    
    for idx, values in enumerate(tqdm(data_values, desc="Processing sequences", unit="seq")):
        print(f"DEBUG: Sequence {idx + 1}: Length = {len(values)}")
        seq_len = len(values)
        positions = np.arange(seq_len, dtype=np.int32)
        print(f"DEBUG: Computing log distances for sequence {idx + 1} with length {seq_len}")
        log_dists = compute_log_distances(positions, seq_len)
        print(f"DEBUG: Log distances range: [{np.min(log_dists):.2f}, {np.max(log_dists):.2f}]")
        
        valid = ~np.isnan(values)
        nz = valid & (values != 0)
        zeros = valid & (values == 0)
        print(f"DEBUG: Sequence {idx + 1}: {np.sum(valid)} valid, {np.sum(nz)} non-zero, {np.sum(zeros)} zeros")
        
        if np.any(nz):
            nz_logs = log_dists[nz]
            nz_vals = values[nz]
            sort_idx = np.argsort(nz_logs)
            line_nz_data.append((nz_logs[sort_idx], nz_vals[sort_idx]))
            all_nz_logs.append(nz_logs)
            all_nz_vals.append(nz_vals)
            all_closest.append(nz_logs == np.min(nz_logs))
            all_furthest.append(nz_logs == np.max(nz_logs))
            print(f"DEBUG: Sequence {idx + 1}: Added {len(nz_logs)} non-zero points")
        else:
            line_nz_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            print(f"DEBUG: Sequence {idx + 1}: No non-zero data")
        
        if np.any(valid):
            valid_logs = log_dists[valid]
            valid_vals = values[valid]
            zero_density = (valid_vals == 0).astype(np.float32)
            sort_idx = np.argsort(valid_logs)
            line_zero_data.append((valid_logs[sort_idx], zero_density[sort_idx]))
            print(f"DEBUG: Sequence {idx + 1}: Added {len(valid_logs)} zero-density points")
        else:
            line_zero_data.append((np.array([], dtype=np.float32), np.array([], dtype=np.float32)))
            print(f"DEBUG: Sequence {idx + 1}: No valid data for zero-density")
    
    print(f"INFO: Concatenating non-zero data")
    all_nz_logs = np.concatenate(all_nz_logs) if all_nz_logs else np.array([], dtype=np.float32)
    all_nz_vals = np.concatenate(all_nz_vals) if all_nz_vals else np.array([], dtype=np.float32)
    all_closest = np.concatenate(all_closest) if all_closest else np.array([], dtype=bool)
    all_furthest = np.concatenate(all_furthest) if all_furthest else np.array([], dtype=bool)
    print(f"INFO: Processed {len(line_nz_data)} lines, {len(all_nz_logs)} non-zero points in {time.time() - start_time:.2f}s")
    
    return line_nz_data, line_zero_data, all_nz_logs, all_nz_vals, all_closest, all_furthest

# Optimized function to compute overall smoothed line
def compute_overall_smoothed_line(line_data, sigma=50):  # Reduced sigma for overall line
    """Compute overall smoothed line with NaN extrapolation and proper edge handling."""
    print(f"INFO: Starting overall line computation for {len(line_data)} lines")
    start_time = time.time()
    if not line_data or all(len(logs) == 0 for logs, _ in line_data):
        print("WARNING: No valid data found")
        return np.array([]), np.array([])
    
    print(f"DEBUG: Concatenating all log distances")
    all_logs = np.concatenate([logs for logs, _ in line_data if len(logs) > 0])
    common_x = np.linspace(np.min(all_logs), np.max(all_logs), 500)
    print(f"DEBUG: Common x-axis set: {len(common_x)} points, range [{np.min(all_logs):.2f}, {np.max(all_logs):.2f}]")
    
    print(f"DEBUG: Initializing interpolated values array: {len(line_data)} x 500")
    interpolated_vals = np.full((len(line_data), 500), np.nan, dtype=np.float32)
    for i, (logs, vals) in enumerate(tqdm(line_data, desc="Interpolating lines", unit="line")):
        if len(logs) > 0:
            print(f"DEBUG: Line {i + 1}: Interpolating {len(vals)} values onto {len(common_x)} points")
            interpolated_vals[i] = np.interp(common_x, logs, vals, left=np.nan, right=np.nan)
            print(f"DEBUG: Line {i + 1}: Interpolated, non-NaN count = {np.sum(~np.isnan(interpolated_vals[i]))}")
    
    print(f"DEBUG: Averaging {len(line_data)} interpolated lines")
    overall_vals = np.nanmean(interpolated_vals, axis=0)
    print(f"DEBUG: Overall_vals non-NaN count = {np.sum(~np.isnan(overall_vals))}, range = [{np.nanmin(overall_vals):.2f}, {np.nanmax(overall_vals):.2f}]")
    
    print(f"DEBUG: Smoothing averaged line with sigma={sigma} ({len(overall_vals)} points)")
    overall_smoothed = gaussian_filter1d(overall_vals, sigma=sigma, mode='constant', cval=np.nan)
    print(f"DEBUG: Overall_smoothed non-NaN count = {np.sum(~np.isnan(overall_smoothed))}, range = [{np.nanmin(overall_smoothed):.2f}, {np.nanmax(overall_smoothed):.2f}]")
    
    print(f"INFO: Overall line computed in {time.time() - start_time:.2f}s")
    return common_x, overall_smoothed

# Generate plot
def create_plot(line_nz_data, line_zero_data, all_nz_logs, all_nz_vals, closest, furthest, metric, suffix, sigma=400, max_lines=100):
    """Create plot with downsampled lines for efficiency."""
    print(f"\n=== Creating {metric} Plot ===")
    start_time = time.time()
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    
    if len(all_nz_logs) == 0:
        print("WARNING: No valid data to plot")
        plt.close(fig)
        return None
    
    print(f"INFO: Data stats: {len(line_nz_data)} non-zero lines, {len(line_zero_data)} zero-density lines, {len(all_nz_logs)} points")
    max_points = 10000  # Downsample threshold
    
    if metric == 'Theta':
        print(f"INFO: Plotting up to {max_lines} Theta zero-density lines")
        for i, (logs, density) in enumerate(tqdm(line_zero_data[:max_lines], desc="Plotting Theta lines", unit="line")):
            if len(logs) > 0:
                t0 = time.time()
                if len(density) > max_points:
                    idx = np.linspace(0, len(density) - 1, max_points, dtype=int)
                    logs_ds, density_ds = logs[idx], density[idx]
                    print(f"DEBUG: Theta line {i + 1}: Downsampled {len(density)} to {max_points} points")
                else:
                    logs_ds, density_ds = logs, density
                    print(f"DEBUG: Theta line {i + 1}: Using {len(density)} points (no downsampling)")
                smoothed = gaussian_filter1d(density_ds, sigma=sigma, mode='nearest')
                ax2.plot(logs_ds, smoothed, color='red', ls='--', lw=0.5, alpha=0.8, label='Zero-Density' if i == 0 else None)
                print(f"DEBUG: Theta line {i + 1}: Plotted in {time.time() - t0:.2f}s, range = [{np.min(smoothed):.2f}, {np.max(smoothed):.2f}]")
        
        print(f"INFO: Computing Theta overall zero-density line")
        common_x, overall_smoothed = compute_overall_smoothed_line(line_zero_data)
        if len(common_x) > 0:
            valid_idx = ~np.isnan(overall_smoothed)
            if np.any(valid_idx):
                ax2.plot(common_x[valid_idx], overall_smoothed[valid_idx], color='black', lw=2, alpha=0.8, label='Overall Zero-Density')
                print(f"DEBUG: Theta overall line plotted, {np.sum(valid_idx)} valid points, range = [{np.min(overall_smoothed[valid_idx]):.2f}, {np.max(overall_smoothed[valid_idx]):.2f}]")
            else:
                print(f"WARNING: Theta overall line not plotted - all values are NaN")
        ax2.legend(loc='upper right')
    else:  # Pi
        print(f"INFO: Plotting Pi scatter")
        z_scores = np.clip((all_nz_vals - np.nanmean(all_nz_vals)) / np.nanstd(all_nz_vals), -5, 5)
        colors = plt.cm.coolwarm(plt.Normalize(-5, 5)(z_scores))
        ax1.scatter(all_nz_logs[closest], all_nz_vals[closest], c='black', s=15, alpha=0.7, edgecolors='none')
        ax1.scatter(all_nz_logs[furthest & ~closest], all_nz_vals[furthest & ~closest], c=colors[furthest & ~closest], 
                    s=15, alpha=0.7, edgecolors='black', linewidths=0.5)
        ax1.scatter(all_nz_logs[~closest & ~furthest], all_nz_vals[~closest & ~furthest], c=colors[~closest & ~furthest], 
                    s=15, alpha=0.7, edgecolors='none')
        print(f"DEBUG: Pi scatter plotted: {len(all_nz_logs)} points")
        
        print(f"INFO: Plotting up to {max_lines} Pi non-zero lines")
        for i, (logs, vals) in enumerate(tqdm(line_nz_data[:max_lines], desc="Plotting Pi lines", unit="line")):
            if len(logs) > 0:
                t0 = time.time()
                if len(vals) > max_points:
                    idx = np.linspace(0, len(vals) - 1, max_points, dtype=int)
                    logs_ds, vals_ds = logs[idx], vals[idx]
                    print(f"DEBUG: Pi line {i + 1}: Downsampled {len(vals)} to {max_points} points")
                else:
                    logs_ds, vals_ds = logs, vals
                    print(f"DEBUG: Pi line {i + 1}: Using {len(vals)} points (no downsampling)")
                smoothed = gaussian_filter1d(vals_ds, sigma=sigma, mode='nearest')
                ax1.plot(logs_ds, smoothed, color='black', lw=0.5, alpha=1.0, label='Non-Zero' if i == 0 else None)
                print(f"DEBUG: Pi line {i + 1}: Plotted in {time.time() - t0:.2f}s, range = [{np.min(smoothed):.2f}, {np.max(smoothed):.2f}]")
        
        print(f"INFO: Computing Pi overall non-zero line")
        common_x, overall_smoothed = compute_overall_smoothed_line(line_nz_data)
        if len(common_x) > 0:
            valid_idx = ~np.isnan(overall_smoothed)
            if np.any(valid_idx):
                ax1.plot(common_x[valid_idx], overall_smoothed[valid_idx], color='black', lw=2, alpha=0.8, label='Overall Non-Zero')
                print(f"DEBUG: Pi overall line plotted, {np.sum(valid_idx)} valid points, range = [{np.min(overall_smoothed[valid_idx]):.2f}, {np.max(overall_smoothed[valid_idx]):.2f}]")
            else:
                print(f"WARNING: Pi overall line not plotted - all values are NaN")
        ax1.legend(loc='upper left')
    
    print(f"INFO: Customizing {metric} plot")
    ax1.set_title(f'{metric} vs. Log Distance from Edge', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Log10(Distance from Nearest Edge + 1)', fontsize=14)
    ax1.set_ylabel(f'{metric} Value', fontsize=14)
    ax2.set_ylabel('Smoothed % Zeros', fontsize=14, color='red')
    ax1.grid(True, ls='--', alpha=0.4)
    ax2.tick_params(axis='y', colors='red')
    
    plot_path = Path.home() / f'distance_plot_{suffix}.png'
    print(f"INFO: Saving plot to {plot_path}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"INFO: {metric} plot completed in {time.time() - start_time:.2f}s")
    return plot_path

# Main function
def main():
    print("=== Starting Execution ===")
    start_time = time.time()
    file_path = 'per_site_output.falsta'
    print(f"INFO: Loading data from {file_path}")
    (theta_labels, theta_data), (pi_labels, pi_data) = load_data(file_path)
    
    if not theta_data.size and not pi_data.size:
        print("WARNING: No data to process. Exiting.")
        return
    
    print(f"INFO: Processing Theta data")
    theta_nz, theta_zero, theta_logs, theta_vals, theta_close, theta_far = process_data(theta_data)
    print(f"INFO: Processing Pi data")
    pi_nz, pi_zero, pi_logs, pi_vals, pi_close, pi_far = process_data(pi_data)
    
    print(f"INFO: Generating Theta plot")
    theta_plot = create_plot(theta_nz, theta_zero, theta_logs, theta_vals, theta_close, theta_far, 'Theta', 'theta')
    print(f"INFO: Generating Pi plot")
    pi_plot = create_plot(pi_nz, pi_zero, pi_logs, pi_vals, pi_close, pi_far, 'Pi', 'pi')
    
    print(f"INFO: Opening plots")
    for plot in [theta_plot, pi_plot]:
        if plot:
            print(f"DEBUG: Opening {plot}")
            if os.name == 'nt':
                os.startfile(plot)
            elif os.name == 'posix':
                os.system(f'open "{plot}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot}"')
    print(f"INFO: Total execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
