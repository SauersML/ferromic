import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import time

# Load data efficiently, only points within 10K from either end
def load_data(file_path, max_dist=10000, max_sequences=500):
    """Efficiently load theta and pi data within 10K from either end."""
    print(f"INFO: Loading data from {file_path} (max {max_sequences} sequences, {max_dist} from ends)")
    start_time = time.time()
    theta_dists, theta_vals = [], []
    pi_dists, pi_vals = [], []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        total_lines = len(lines)
        print(f"INFO: File read: {total_lines} lines in {time.time() - start_time:.2f}s")
        
        seq_count = 0
        i = 0
        with tqdm(total=min(total_lines - 1, max_sequences * 2), desc="Parsing lines") as pbar:
            while i < total_lines - 1 and seq_count < max_sequences:
                if 'filtered_theta' in lines[i] or 'filtered_pi' in lines[i]:
                    values = np.fromstring(lines[i + 1].strip().replace('NA', 'nan'), sep=',', dtype=np.float32)
                    seq_len = len(values)
                    positions = np.arange(seq_len, dtype=np.float32)
                    dists = np.minimum(positions, seq_len - 1 - positions)
                    mask = dists <= max_dist
                    
                    if 'filtered_theta' in lines[i]:
                        theta_dists.extend(dists[mask])
                        theta_vals.extend(values[mask])
                    else:
                        pi_dists.extend(dists[mask])
                        pi_vals.extend(values[mask])
                    
                    seq_count += 1
                    pbar.update(2)  # Update for header + data line
                i += 1
    
    # Convert to arrays only once
    theta_dists = np.array(theta_dists, dtype=np.float32)
    theta_vals = np.array(theta_vals, dtype=np.float32)
    pi_dists = np.array(pi_dists, dtype=np.float32)
    pi_vals = np.array(pi_vals, dtype=np.float32)
    
    print(f"INFO: Loaded {len(theta_dists)} theta points, {len(pi_dists)} pi points in {time.time() - start_time:.2f}s")
    return (theta_dists, theta_vals), (pi_dists, pi_vals)

def create_plots(theta_dists, theta_vals, pi_dists, pi_vals):
    plt.style.use('seaborn-v0_8-white')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Colors and styles
    scatter_color = '#6D8299'  # Soft blue-gray
    bg_color = '#F8F9FA'      # Light gray background
    
    # Theta Plot
    valid_theta = ~np.isnan(theta_vals)
    if np.any(valid_theta):
        ax1.scatter(theta_dists[valid_theta], theta_vals[valid_theta], c=scatter_color, s=15, alpha=0.6, edgecolors='none')
    
    ax1.set_title('Theta vs. Distance from Edge (0-10K)', fontsize=18, fontweight='bold', pad=15)
    ax1.set_ylabel('Theta Value', fontsize=16, labelpad=10)
    ax1.set_facecolor(bg_color)
    ax1.tick_params(axis='both', labelsize=14)
    
    # Pi Plot
    nz_pi = pi_vals != 0
    if np.any(nz_pi):
        ax2.scatter(pi_dists[nz_pi], pi_vals[nz_pi], c=scatter_color, s=15, alpha=0.6, edgecolors='none')
    
    ax2.set_title('Pi vs. Distance from Edge (0-10K)', fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('Distance from Nearest Edge', fontsize=16, labelpad=10)
    ax2.set_ylabel('Pi Value (Non-Zero)', fontsize=16, labelpad=10)
    ax2.set_facecolor(bg_color)
    ax2.tick_params(axis='both', labelsize=14)
    
    # Common settings
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10000)
    
    fig.suptitle('Association with Distance from Edge', fontsize=22, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = Path.home() / 'distance_plots_10K_beautiful.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"INFO: Plots saved to {plot_path}")
    return plot_path

# Main function
def main():
    print("=== Starting Execution ===")
    start_time = time.time()
    file_path = 'per_site_diversity_output.falsta'
    (theta_dists, theta_vals), (pi_dists, pi_vals) = load_data(file_path)
    
    if not theta_dists.size and not pi_dists.size:
        print("WARNING: No data to process. Exiting.")
        return
    
    plot_path = create_plots(theta_dists, theta_vals, pi_dists, pi_vals)
    
    if plot_path:
        if os.name == 'nt':
            os.startfile(plot_path)
        elif os.name == 'posix':
            os.system(f'open "{plot_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{plot_path}"')
    print(f"INFO: Total execution time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
