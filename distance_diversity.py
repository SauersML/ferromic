import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from numba import njit
from matplotlib.colors import TwoSlopeNorm

# JIT-compiled function to calculate log distances from sequence edges
@njit
def compute_log_distances(positions, sequence_length):
    """Calculate the minimum log10 distance from either sequence edge for each position."""
    log_distances = np.empty(len(positions), dtype=np.float32)
    for i in range(len(positions)):
        log_distances[i] = min(positions[i], sequence_length - 1 - positions[i])
    return np.log10(log_distances + 1)  # Add 1 to avoid log(0)

# Load filtered theta and pi data from the input file
def load_filtered_measurements(file_path):
    """Read filtered theta and pi data from a binary file, handling large datasets efficiently."""
    print("Loading data from file")
    theta_labels, theta_data, pi_labels, pi_data = [], [], [], []
    buffer = b''
    last_header = None
    
    with open(file_path, 'rb') as file:
        buffer_size = 16 * 1024 * 1024  # 16MB buffer
        while True:
            chunk = file.read(buffer_size)
            if not chunk and not buffer:
                break
            buffer += chunk
            lines = buffer.split(b'\n')
            
            if chunk:  # Mid-file, last might be incomplete
                buffer = lines[-1]
                lines = lines[:-1]
            else:  # EOF, process everything
                buffer = b''
            
            i = 0
            while i < len(lines):
                if b'filtered' in lines[i]:
                    last_header = lines[i]
                    if i + 1 < len(lines):
                        values_line = lines[i + 1].decode('utf-8', errors='ignore')
                        value_array = np.fromstring(values_line.replace('NA', 'nan'), sep=',', dtype=np.float32)
                        header_text = last_header[1:].decode('utf-8', errors='ignore')
                        if 'theta' in header_text:
                            theta_labels.append(header_text)
                            theta_data.append(value_array)
                        elif 'pi' in header_text:
                            pi_labels.append(header_text)
                            pi_data.append(value_array)
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            
            if not chunk and last_header and buffer:
                values_line = buffer.decode('utf-8', errors='ignore')
                value_array = np.fromstring(values_line.replace('NA', 'nan'), sep=',', dtype=np.float32)
                header_text = last_header[1:].decode('utf-8', errors='ignore')
                if 'theta' in header_text:
                    theta_labels.append(header_text)
                    theta_data.append(value_array)
                elif 'pi' in header_text:
                    pi_labels.append(header_text)
                    pi_data.append(value_array)
    
    print(f"Loaded {len(theta_labels)} theta and {len(pi_labels)} pi filtered data lines")
    return (np.array(theta_labels, dtype=object), np.array(theta_data, dtype=object)), \
           (np.array(pi_labels, dtype=object), np.array(pi_data, dtype=object))

# Process measurements per sequence
def process_measurements(measurement_values):
    """Process measurements to get per-sequence data for lines and overall data for scatter points."""
    line_nz_data = []
    line_zero_data = []
    all_nz_log_distances_list = []
    all_nz_values_list = []
    is_closest_list = []
    is_furthest_list = []
    
    for value_array in measurement_values:
        seq_len = len(value_array)
        positions = np.arange(seq_len, dtype=np.int32)
        valid_mask = ~np.isnan(value_array)
        nz_mask = valid_mask & (value_array != 0)
        zero_mask = valid_mask & (value_array == 0)
        all_log_distances = compute_log_distances(positions, seq_len)
        
        # For zero-density (smoothed per line)
        zero_indicators = np.zeros(seq_len, dtype=np.float32)
        zero_indicators[zero_mask] = 1
        sort_indices_all = np.argsort(all_log_distances)
        sorted_all_log_distances = all_log_distances[sort_indices_all]
        sorted_zero_indicators = zero_indicators[sort_indices_all]
        line_zero_data.append((sorted_all_log_distances, sorted_zero_indicators))
        
        # For non-zero values (smoothed per line)
        if np.sum(nz_mask) > 0:
            nz_positions = positions[nz_mask]
            nz_log_distances = all_log_distances[nz_mask]
            nz_values = value_array[nz_mask]
            sort_indices_nz = np.argsort(nz_log_distances)
            sorted_nz_log_distances = nz_log_distances[sort_indices_nz]
            sorted_nz_values = nz_values[sort_indices_nz]
            line_nz_data.append((sorted_nz_log_distances, sorted_nz_values))
            
            # For scatter points: identify closest and furthest
            min_log_dist = np.min(nz_log_distances)
            max_log_dist = np.max(nz_log_distances)
            is_closest_for_sequence = (nz_log_distances == min_log_dist)
            is_furthest_for_sequence = (nz_log_distances == max_log_dist)
            all_nz_log_distances_list.append(nz_log_distances)
            all_nz_values_list.append(nz_values)
            is_closest_list.append(is_closest_for_sequence)
            is_furthest_list.append(is_furthest_for_sequence)
        else:
            line_nz_data.append((np.array([]), np.array([])))
    
    # Concatenate data for scatter points
    all_nz_log_distances = np.concatenate(all_nz_log_distances_list) if all_nz_log_distances_list else np.array([])
    all_nz_values = np.concatenate(all_nz_values_list) if all_nz_values_list else np.array([])
    is_closest = np.concatenate(is_closest_list) if is_closest_list else np.array([])
    is_furthest = np.concatenate(is_furthest_list) if is_furthest_list else np.array([])
    
    return line_nz_data, line_zero_data, all_nz_log_distances, all_nz_values, is_closest, is_furthest

# Create and save the scatter plot with per-line smoothed curves
def generate_scatter_plot_with_smoothed_curves(line_nz_data, line_zero_data, all_nz_log_distances, all_nz_values, is_closest, is_furthest, metric_name, file_suffix, sigma=50):
    """Generate a scatter plot with per-line Gaussian smoothed signal and zero-density curves."""
    print(f"Creating {metric_name} plot with per-line smoothed signal and zero-density")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='#f5f5f5')
    ax1.set_facecolor('#ffffff')
    ax2 = ax1.twinx()  # Secondary y-axis for zero-density
    
    if len(all_nz_log_distances) > 0:
        # Compute z-scores across all non-zero values
        z_scores = stats.zscore(all_nz_values, nan_policy='omit')
        # Define norm and colormap for z-score coloring
        norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
        z_score_colors = plt.cm.coolwarm(norm(z_scores))
        
        # Plot scatter points with special styling
        # Closest to edge: black (opaque)
        if np.any(is_closest):
            ax1.scatter(all_nz_log_distances[is_closest], all_nz_values[is_closest], c='black', s=15, alpha=0.7, edgecolors='none')
        # Furthest from edge: z-score color with black outline
        mask_furthest_not_closest = is_furthest & ~is_closest
        if np.any(mask_furthest_not_closest):
            ax1.scatter(all_nz_log_distances[mask_furthest_not_closest], all_nz_values[mask_furthest_not_closest],
                        c=z_score_colors[mask_furthest_not_closest], s=15, alpha=0.7, edgecolors='black', linewidths=0.5)
        # All other points: z-score color only
        mask_neither = ~is_closest & ~is_furthest
        if np.any(mask_neither):
            ax1.scatter(all_nz_log_distances[mask_neither], all_nz_values[mask_neither],
                        c=z_score_colors[mask_neither], s=15, alpha=0.7, edgecolors='none')
        
        # Plot per-line smoothed signal (non-zero values) and zero-density
        for (sorted_nz_log_distances, sorted_nz_values) in line_nz_data:
            if len(sorted_nz_log_distances) > 0:
                smoothed_nz_values = gaussian_filter1d(sorted_nz_values, sigma=sigma, mode='nearest')
                ax1.plot(sorted_nz_log_distances, smoothed_nz_values, color='black', linestyle='-', linewidth=0.5, alpha=1.0)
        
        for (sorted_all_log_distances, sorted_zero_indicators) in line_zero_data:
            if len(sorted_all_log_distances) > 0:
                smoothed_zero_density = gaussian_filter1d(sorted_zero_indicators, sigma=sigma, mode='nearest')
                ax2.plot(sorted_all_log_distances, smoothed_zero_density, color='red', linestyle='--', linewidth=0.5, alpha=0.8)
    else:
        print(f"No valid non-zero data to plot for {metric_name}")
        plt.close(fig)
        return None

    # Customize axes
    ax1.set_title(f'Log Distance from Edge vs. {metric_name} (Filtered Data)', 
                  fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax1.set_xlabel('Log10(Distance from Nearest Edge + 1)', size=14, color='#333333')
    ax1.set_ylabel(f'{metric_name} Value', size=14, color='#333333')
    ax2.set_ylabel('Zero-Density (Arbitrary Units)', size=14, color='#ff3333')
    ax1.grid(True, linestyle='--', alpha=0.4, color='#999999')
    ax1.tick_params(axis='both', which='major', labelsize=12, color='#666666')
    ax2.tick_params(axis='y', labelsize=12, colors='#ff3333')
    
    # Save plot without colorbar
    output_plot_path = Path.home() / f'distance_plot_{file_suffix}_per_line.png'
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight', facecolor='#f5f5f5')
    plt.close(fig)
    
    print(f"{metric_name} plot saved to: {output_plot_path}")
    return output_plot_path

# Main execution: Load data, process, and generate plots
input_file_path = 'per_site_output.falsta'
(theta_labels, theta_data), (pi_labels, pi_data) = load_filtered_measurements(input_file_path)

# Check if there's any data to plot
if not theta_labels.size and not pi_labels.size:
    print("No filtered data to plot. Exiting.")
    exit()

# Process theta and pi data
theta_line_nz, theta_line_zero, theta_all_nz_log, theta_all_nz_val, theta_is_closest, theta_is_furthest = process_measurements(theta_data)
pi_line_nz, pi_line_zero, pi_all_nz_log, pi_all_nz_val, pi_is_closest, pi_is_furthest = process_measurements(pi_data)

# Generate plots
theta_plot_path = generate_scatter_plot_with_smoothed_curves(
    theta_line_nz, theta_line_zero, theta_all_nz_log, theta_all_nz_val, theta_is_closest, theta_is_furthest, 'Theta', 'theta')
pi_plot_path = generate_scatter_plot_with_smoothed_curves(
    pi_line_nz, pi_line_zero, pi_all_nz_log, pi_all_nz_val, pi_is_closest, pi_is_furthest, 'Pi', 'pi')

# Open the generated plots based on OS
print("Opening plots")
for plot_path in [theta_plot_path, pi_plot_path]:
    if plot_path:
        if os.name == 'nt':  # Windows
            os.startfile(plot_path)
        elif os.name == 'posix':  # MacOS or Linux
            os.system(f'open "{plot_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot_path}"')

print("Plot generation complete")
