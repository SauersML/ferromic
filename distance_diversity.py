import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy import stats
from numba import njit

@njit
def calculate_distances(positions, sequence_length):
    """JIT-compiled distance calculation"""
    distances = np.empty(len(positions), dtype=np.float32)
    for i in range(len(positions)):
        distances[i] = min(positions[i], sequence_length - 1 - positions[i])
    return np.log10(distances + 1)

def load_filtered_data(file_path):
    print("Loading data")
    theta_headers = []
    theta_values = []
    pi_headers = []
    pi_values = []
    
    with open(file_path, 'rb') as f:
        buffer_size = 16 * 1024 * 1024  # 16MB buffer
        buffer = b''
        
        while True:
            chunk = f.read(buffer_size)
            if not chunk and not buffer:
                break
            buffer += chunk
            lines = buffer.split(b'\n')
            
            i = 0
            while i < len(lines) - 1:
                header = lines[i]
                if b'filtered' in header:
                    header_str = header[1:].decode('utf-8', errors='ignore')
                    value_line = lines[i + 1].decode('utf-8', errors='ignore')
                    value_array = np.fromstring(value_line.replace('NA', 'nan'), sep=',', dtype=np.float32)
                    
                    if 'theta' in header_str:
                        theta_headers.append(header_str)
                        theta_values.append(value_array)
                    elif 'pi' in header_str:
                        pi_headers.append(header_str)
                        pi_values.append(value_array)
                i += 2
            
            buffer = lines[-1] if i == len(lines) - 1 else b''
    
    print(f"Loaded {len(theta_headers)} theta and {len(pi_headers)} pi filtered data lines")
    return (np.array(theta_headers, dtype=object), np.array(theta_values, dtype=object)), \
           (np.array(pi_headers, dtype=object), np.array(pi_values, dtype=object))

def process_measurements_combined(headers, values):
    print("Processing data into combined arrays")
    total_points = sum(len(v[~np.isnan(v)]) for v in values)
    distances_array = np.empty(total_points, dtype=np.float32)
    values_array = np.empty(total_points, dtype=np.float32)
    current_idx = 0
    
    for value_array in values:
        valid_mask = ~np.isnan(value_array)
        valid_count = np.sum(valid_mask)
        if valid_count > 0:
            valid_positions = np.arange(len(value_array), dtype=np.int32)[valid_mask]
            valid_vals = value_array[valid_mask]
            log_dists = calculate_distances(valid_positions, len(value_array))
            distances_array[current_idx:current_idx + valid_count] = log_dists
            values_array[current_idx:current_idx + valid_count] = valid_vals
            current_idx += valid_count
    
    distances_array = distances_array[:current_idx]
    values_array = values_array[:current_idx]
    
    print(f"Processed arrays (distances: {distances_array.shape}, values: {values_array.shape})")
    return distances_array, values_array

def create_measurement_plot_vectorized(distances, values, metric_name, color_map, fit_line_color, file_suffix):
    print(f"Creating {metric_name} plot (vectorized)")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='#f5f5f5')
    ax.set_facecolor('#ffffff')

    if values.size > 0:
        # Single scatter plot with all points
        z_scores = stats.zscore(values, nan_policy='omit')
        scatter = ax.scatter(distances, values, c=z_scores, cmap=color_map, 
                           s=15, alpha=0.2, edgecolors='none')
        
        # Robust single linear fit
        unique_dists = np.unique(distances)
        if unique_dists.size > 5:  # Require some unique points for stability
            try:
                coef = np.polyfit(distances, values, 1)  # no cov
                fit_line = np.poly1d(coef)
                # Use min/max of distances for cleaner line
                x_range = np.array([distances.min(), distances.max()])
                ax.plot(x_range, fit_line(x_range), color=fit_line_color, 
                       alpha=0.3, linewidth=0.5)
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: Could not fit line for {metric_name}: {str(e)}")
        
        # Enhanced colorbar
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
        colorbar.set_label(f'{metric_name} Z-score', size=12, weight='bold', color='#333333')
        colorbar.outline.set_linewidth(0.5)
        colorbar.outline.set_edgecolor('#666666')
        colorbar.ax.tick_params(labelsize=10, color='#666666', width=0.5)
        colorbar.ax.yaxis.set_tick_params(pad=2)
    else:
        print(f"No valid data to plot for {metric_name}")
        plt.close(fig)
        return None

    # Customize plot
    ax.set_title(f'Log Distance from Edge vs. {metric_name} (Filtered Data)', 
                fontsize=16, fontweight='bold', color='#333333', pad=20)
    ax.set_xlabel('Log10(Distance from Nearest Edge + 1)', size=14, color='#333333')
    ax.set_ylabel('Value', size=14, color='#333333')
    ax.grid(True, linestyle='--', alpha=0.4, color='#999999')
    ax.tick_params(axis='both', which='major', labelsize=12, color='#666666')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(0.5)

    # Save plot
    plot_path = Path.home() / f'distance_plot_{file_suffix}.png'
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#f5f5f5')
    plt.close(fig)
    
    print(f"{metric_name} plot saved to: {plot_path}")
    return plot_path

# Load and process data
input_file = 'per_site_output.falsta'
(theta_headers, theta_values), (pi_headers, pi_values) = load_filtered_data(input_file)

if not theta_headers.size and not pi_headers.size:
    print("No filtered data to plot. Exiting.")
    exit()

# Process measurements into single arrays
theta_distances, theta_vals = process_measurements_combined(theta_headers, theta_values)
pi_distances, pi_vals = process_measurements_combined(pi_headers, pi_values)

# Create two plots with all points
theta_plot_path = create_measurement_plot_vectorized(theta_distances, theta_vals, 'Theta', 'Purples', 'purple', 'theta')
pi_plot_path = create_measurement_plot_vectorized(pi_distances, pi_vals, 'Pi', 'Greens', 'green', 'pi')

# Open plots
print("Opening plots")
for plot_path in [theta_plot_path, pi_plot_path]:
    if plot_path:
        if os.name == 'nt':
            os.startfile(plot_path)
        elif os.name == 'posix':
            os.system(f'open "{plot_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot_path}"')

print("Plot generation complete")
