import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy import stats
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
    theta_labels = []  # Headers for theta measurements
    theta_data = []    # Values for theta measurements
    pi_labels = []     # Headers for pi measurements
    pi_data = []       # Values for pi measurements
    
    with open(file_path, 'rb') as file:
        buffer_size = 16 * 1024 * 1024  # 16MB buffer for reading large files
        buffer = b''
        lines_processed = 0
        
        while True:
            chunk = file.read(buffer_size)
            if not chunk and not buffer:
                break  # End of file and buffer
            elif not chunk:
                lines = buffer.split(b'\n')
            else:
                buffer += chunk
                lines = buffer.split(b'\n')
                buffer = lines[-1]  # Keep incomplete line for next iteration
                lines = lines[:-1]  # Process all but the last line... double check
            
            i = 0
            while i < len(lines) - 1:
                header = lines[i]
                if b'filtered' in header:
                    header_text = header[1:].decode('utf-8', errors='ignore')
                    values_line = lines[i + 1].decode('utf-8', errors='ignore')
                    value_array = np.fromstring(values_line.replace('NA', 'nan'), sep=',', dtype=np.float32)
                    if 'theta' in header_text:
                        theta_labels.append(header_text)
                        theta_data.append(value_array)
                    elif 'pi' in header_text:
                        pi_labels.append(header_text)
                        pi_data.append(value_array)
                    lines_processed += 1
                i += 2
            
            # Handle any remaining line at EOF
            if not chunk and i < len(lines):
                header = lines[i]
                if b'filtered' in header and i + 1 < len(lines):
                    header_text = header[1:].decode('utf-8', errors='ignore')
                    values_line = lines[i + 1].decode('utf-8', errors='ignore')
                    value_array = np.fromstring(values_line.replace('NA', 'nan'), sep=',', dtype=np.float32)
                    if 'theta' in header_text:
                        theta_labels.append(header_text)
                        theta_data.append(value_array)
                    elif 'pi' in header_text:
                        pi_labels.append(header_text)
                        pi_data.append(value_array)
                    lines_processed += 1
    
    print(f"Loaded {len(theta_labels)} theta and {len(pi_labels)} pi filtered data lines (total processed: {lines_processed})")
    return (np.array(theta_labels, dtype=object), np.array(theta_data, dtype=object)), \
           (np.array(pi_labels, dtype=object), np.array(pi_data, dtype=object))

# Process each line separately, keeping data distinct
def process_per_line_measurements(measurement_labels, measurement_values):
    """Process each line's data into distances and values, keeping them separate for individual fits."""
    print("Processing data per line")
    line_log_distances = []
    line_metric_values = []
    
    for value_array in measurement_values:
        valid_mask = ~np.isnan(value_array)
        valid_count = np.sum(valid_mask)
        if valid_count > 0:
            valid_positions = np.arange(len(value_array), dtype=np.int32)[valid_mask]
            valid_measurements = value_array[valid_mask]
            computed_distances = compute_log_distances(valid_positions, len(value_array))
            line_log_distances.append(computed_distances)
            line_metric_values.append(valid_measurements)
    
    print(f"Processed {len(line_log_distances)} lines with valid data")
    return line_log_distances, line_metric_values

# Create and save the scatter plot with per-line best fits and smoothing
def generate_scatter_plot_with_per_line_fits(line_log_distances, line_metric_values, metric_name, base_hue, curve_color, file_suffix, downsample_factor=0.1):
    """Generate a scatter plot with a line of best fit for each individual line, smoothed over 200 positions."""
    print(f"Creating {metric_name} plot with per-line fits")
    plt.style.use('seaborn-v0_8-darkgrid')  # Set plot style
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#f5f5f5')  # Create figure and axis
    ax.set_facecolor('#ffffff')  # White plot background

    if line_metric_values:  # Check if there's any data
        # Flatten all data for scatter plotting and z-score calculation
        all_log_distances = np.concatenate(line_log_distances)
        all_metric_values = np.concatenate(line_metric_values)
        z_scores = stats.zscore(all_metric_values, nan_policy='omit')
        
        # Compute hues based on z-scores for point coloring
        hue_variation_range = 0.2  # ±0.1 around base hue
        z_min, z_max = z_scores.min(), z_scores.max()
        if z_max > z_min:
            hue_normalized = (z_scores - z_min) / (z_max - z_min) * hue_variation_range - hue_variation_range/2
        else:
            hue_normalized = np.zeros_like(z_scores)
        point_hues = base_hue + hue_normalized
        
        # Convert HSV to RGB for scatter point colors
        hsv_colors = np.vstack((point_hues, np.full_like(point_hues, 0.8), np.full_like(point_hues, 0.8))).T
        scatter_colors = hsv_to_rgb(hsv_colors)
        
        # Downsample data if too large, using a factor (e.g., 0.1 = 10% of points)
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
        
        # Plot all scatter points
        ax.scatter(plot_distances, plot_values, c=plot_colors, s=15, alpha=0.2, edgecolors='none')
        
        # Process each line for individual best fit and smoothing
        for distances, values in zip(line_log_distances, line_metric_values):
            if len(distances) > 10:  # Need enough points for a fit
                # Sort data for this line
                sort_indices = np.argsort(distances)
                sorted_distances = distances[sort_indices]
                sorted_values = values[sort_indices]
                
                # Compute line of best fit
                A = np.vstack([sorted_distances, np.ones(len(sorted_distances))]).T
                try:
                    coef, *_ = np.linalg.lstsq(A, sorted_values, rcond=None)
                    fit_values = coef[0] * sorted_distances + coef[1]
                    
                    # Smooth the fit over 200 positions (uniform kernel)
                    window_size = min(200, len(sorted_distances))  # Cap at data length
                    smoothed_values = np.convolve(fit_values, np.ones(window_size)/window_size, mode='valid')
                    smoothed_distances = sorted_distances[window_size-1:]  # Adjust x-axis to match smoothed y
                    
                    # Plot the smoothed fit
                    ax.plot(smoothed_distances, smoothed_values, color='black', linestyle='-', 
                            alpha=1.0, linewidth=1.0)
                except np.linalg.LinAlgError:
                    print(f"Warning: Could not fit line for a {metric_name} line due to numerical instability")
        
        # Set up colorbar with fixed SD range (-5 to +5)
        colormap = 'Purples' if metric_name == 'Theta' else 'Greens'
        norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)  # Center at 0, range ±5 SD
        scalar_mappable = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        scalar_mappable.set_array([])  # Required for colorbar
        colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.01, aspect=30)
        colorbar.set_label(f'{metric_name} Z-score (SD)', size=12, weight='bold', color='#333333')
        colorbar.set_ticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])  # Explicit SD ticks
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor('#666666')
        colorbar.ax.tick_params(labelsize=10, color='#666666', width=0.5)
        colorbar.ax.yaxis.set_tick_params(pad=2)
    else:
        print(f"No valid data to plot for {metric_name}")
        plt.close(fig)
        return None

    # Customize plot appearance
    ax.set_title(f'Log Distance from Edge vs. {metric_name} (Filtered Data)', 
                 fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax.set_xlabel('Log10(Distance from Nearest Edge + 1)', size=14, color='#333333')
    ax.set_ylabel(f'{metric_name} Value', size=14, color='#333333')
    ax.grid(True, linestyle='--', alpha=0.4, color='#999999')  # Light gridlines
    ax.tick_params(axis='both', which='major', labelsize=12, color='#666666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)

    # Save the plot to the home directory
    output_plot_path = Path.home() / f'distance_plot_{file_suffix}.png'
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
theta_line_distances, theta_line_values = process_per_line_measurements(theta_labels, theta_data)
pi_line_distances, pi_line_values = process_per_line_measurements(pi_labels, pi_data)

# Generate plots with 10% downsampling factor
theta_plot_path = generate_scatter_plot_with_per_line_fits(
    theta_line_distances, theta_line_values, 'Theta', 0.75, 'black', 'theta', downsample_factor=0.1)
pi_plot_path = generate_scatter_plot_with_per_line_fits(
    pi_line_distances, pi_line_values, 'Pi', 0.33, 'black', 'pi', downsample_factor=0.1)

# Open the generated plots based on OS
print("Opening plots")
for plot_path in [theta_plot_path, pi_plot_path]:
    if plot_path:
        if os.name == 'nt':  # Windows
            os.startfile(plot_path)
        elif os.name == 'posix':  # MacOS or Linux
            os.system(f'open "{plot_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot_path}"')

print("Plot generation complete")
