import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from numba import njit
from matplotlib.colors import hsv_to_rgb, TwoSlopeNorm

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

# Process each line separately, separating non-zero values and zero positions
def process_per_line_measurements(measurement_labels, measurement_values):
    """Process each line's data into distances and values for non-zeros and zero positions."""
    print("Processing data per line")
    line_log_distances_nonzero = []  # Log distances for non-zero values
    line_metric_values_nonzero = []  # Non-zero values
    line_zero_positions = []         # Original positions of zeros
    line_sequence_lengths = []       # Sequence length per line
    
    for value_array in measurement_values:
        valid_mask = ~np.isnan(value_array)
        positions = np.arange(len(value_array), dtype=np.int32)
        sequence_length = len(value_array)
        
        # Non-zero values (ignore zeros and NaNs)
        nonzero_mask = valid_mask & (value_array != 0)
        if np.sum(nonzero_mask) > 0:
            nonzero_positions = positions[nonzero_mask]
            nonzero_values = value_array[nonzero_mask]
            nonzero_distances = compute_log_distances(nonzero_positions, sequence_length)
            line_log_distances_nonzero.append(nonzero_distances)
            line_metric_values_nonzero.append(nonzero_values)
        else:
            line_log_distances_nonzero.append(np.array([], dtype=np.float32))
            line_metric_values_nonzero.append(np.array([], dtype=np.float32))
        
        # Zero positions (only where value == 0, ignore NaNs)
        zero_mask = valid_mask & (value_array == 0)
        if np.sum(zero_mask) > 0:
            zero_positions = positions[zero_mask]
            line_zero_positions.append(zero_positions)
        else:
            line_zero_positions.append(np.array([], dtype=np.int32))
        
        line_sequence_lengths.append(sequence_length)
    
    print(f"Processed {len(line_log_distances_nonzero)} lines with data")
    return line_log_distances_nonzero, line_metric_values_nonzero, line_zero_positions, line_sequence_lengths

# Create and save the scatter plot with smoothed signal and zero-density
def generate_scatter_plot_with_smoothed_curves(line_log_distances_nonzero, line_metric_values_nonzero, 
                                               line_zero_positions, line_sequence_lengths, metric_name, 
                                               base_hue, file_suffix, downsample_factor=0.1, sigma=50):
    """Generate a scatter plot with Gaussian-smoothed signal and zero-density curves per line."""
    print(f"Creating {metric_name} plot with smoothed signal and zero-density")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='#f5f5f5')
    ax1.set_facecolor('#ffffff')
    ax2 = ax1.twinx()  # Secondary y-axis for zero-density
    
    if line_metric_values_nonzero and any(len(v) > 0 for v in line_metric_values_nonzero):
        # Flatten non-zero data for scatter plotting
        valid_lines = [(d, v) for d, v in zip(line_log_distances_nonzero, line_metric_values_nonzero) if len(v) > 0]
        if valid_lines:
            all_log_distances = np.concatenate([d for d, _ in valid_lines])
            all_metric_values = np.concatenate([v for _, v in valid_lines])
            z_scores = stats.zscore(all_metric_values, nan_policy='omit')
            
            # Compute hues based on z-scores
            hue_variation_range = 0.2
            z_min, z_max = z_scores.min(), z_scores.max()
            if z_max > z_min:
                hue_normalized = (z_scores - z_min) / (z_max - z_min) * hue_variation_range - hue_variation_range/2
            else:
                hue_normalized = np.zeros_like(z_scores)
            point_hues = base_hue + hue_normalized
            hsv_colors = np.vstack((point_hues, np.full_like(point_hues, 0.8), np.full_like(point_hues, 0.8))).T
            scatter_colors = hsv_to_rgb(hsv_colors)
            
            # Downsample if too large
            if len(all_log_distances) > 1000000:
                num_points_to_keep = int(len(all_log_distances) * downsample_factor)
                sampled_indices = np.random.choice(len(all_log_distances), size=num_points_to_keep, replace=False)
                plot_distances = all_log_distances[sampled_indices]
                plot_values = all_metric_values[sampled_indices]
                plot_colors = scatter_colors[sampled_indices]
            else:
                plot_distances = all_log_distances
                plot_values = all_metric_values
                plot_colors = scatter_colors
            
            # Plot scatter points (non-zero values)
            ax1.scatter(plot_distances, plot_values, c=plot_colors, s=15, alpha=0.2, edgecolors='none')
        
        # Process each line for smoothed signal and zero-density
        for nz_distances, nz_values, zero_positions, seq_len in zip(line_log_distances_nonzero, 
                                                                   line_metric_values_nonzero, 
                                                                   line_zero_positions, 
                                                                   line_sequence_lengths):
            if len(nz_distances) > 10:  # Smoothed signal
                sort_indices = np.argsort(nz_distances)
                sorted_distances = nz_distances[sort_indices]
                sorted_values = nz_values[sort_indices]
                smoothed_signal = gaussian_filter1d(sorted_values, sigma=sigma, mode='nearest')
                ax1.plot(sorted_distances, smoothed_signal, color='black', linestyle='-', 
                         alpha=1.0, linewidth=1.0, label='Smoothed Signal' if metric_name == 'Theta' else '')
            
            if len(zero_positions) > 0:  # Zero-density
                all_positions = np.arange(seq_len, dtype=np.int32)
                zero_binary = np.zeros(seq_len, dtype=np.float32)
                zero_binary[zero_positions] = 1
                smoothed_density = gaussian_filter1d(zero_binary, sigma=sigma, mode='nearest')
                log_distances_all = compute_log_distances(all_positions, seq_len)
                ax2.plot(log_distances_all, smoothed_density, color='red', linestyle='--', 
                         alpha=0.8, linewidth=1.0, label='Zero-Density' if metric_name == 'Theta' else '')
        
        # Colorbar for scatter points
        colormap = 'Purples' if metric_name == 'Theta' else 'Greens'
        norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
        scalar_mappable = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax1, pad=0.01, aspect=30)
        colorbar.set_label(f'{metric_name} Z-score (SD)', size=12, weight='bold', color='#333333')
        colorbar.set_ticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor('#666666')
        colorbar.ax.tick_params(labelsize=10, color='#666666', width=0.5)
        colorbar.ax.yaxis.set_tick_params(pad=2)
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
    
    if metric_name == 'Theta':
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    # Save plot
    output_plot_path = Path.home() / f'distance_plot_{file_suffix}_smoothed.png'
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

# Process theta and pi data per line
theta_nz_distances, theta_nz_values, theta_zero_pos, theta_seq_lens = process_per_line_measurements(theta_labels, theta_data)
pi_nz_distances, pi_nz_values, pi_zero_pos, pi_seq_lens = process_per_line_measurements(pi_labels, pi_data)

# Generate plots with 10% downsampling factor
theta_plot_path = generate_scatter_plot_with_smoothed_curves(
    theta_nz_distances, theta_nz_values, theta_zero_pos, theta_seq_lens, 'Theta', 0.75, 'theta', downsample_factor=0.1)
pi_plot_path = generate_scatter_plot_with_smoothed_curves(
    pi_nz_distances, pi_nz_values, pi_zero_pos, pi_seq_lens, 'Pi', 0.33, 'pi', downsample_factor=0.1)

# Open the generated plots based on OS
print("Opening plots")
for plot_path in [theta_plot_path, pi_plot_path]:
    if plot_path:
        if os.name == 'nt':  # Windows
            os.startfile(plot_path)
        elif os.name == 'posix':  # MacOS or Linux
            os.system(f'open "{plot_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot_path}"')

print("Plot generation complete")
