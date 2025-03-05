import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy import stats

def load_data(file_path):
    print("Loading data")
    data_lines = []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        data_lines = [(lines[i][1:], lines[i + 1]) 
                     for i in range(0, len(lines) - 1, 2) 
                     if 'filtered' in lines[i]]
    print(f"Loaded {len(data_lines)} filtered data lines")
    return data_lines

def process_data(data_lines):
    print("Processing data")
    theta_dist, theta_val = [], []
    pi_dist, pi_val = [], []
    
    for label, values in data_lines:
        val_array = np.fromstring(values.replace('NA', 'nan'), sep=',')
        mask = (~np.isnan(val_array)) & (val_array != 0)
        dist = np.minimum(np.arange(len(val_array)), len(val_array) - 1 - np.arange(len(val_array)))
        
        if 'theta' in label:
            theta_dist.append(dist[mask])
            theta_val.append(val_array[mask])
        elif 'pi' in label:
            pi_dist.append(dist[mask])
            pi_val.append(val_array[mask])
    
    theta_dist = np.concatenate(theta_dist) if theta_dist else np.array([])
    theta_val = np.concatenate(theta_val) if theta_val else np.array([])
    pi_dist = np.concatenate(pi_dist) if pi_dist else np.array([])
    pi_val = np.concatenate(pi_val) if pi_val else np.array([])
    
    print(f"Processed {len(theta_val)} theta points and {len(pi_val)} pi points")
    return theta_dist, theta_val, pi_dist, pi_val

# Set up plot
print("Configuring plot")
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(12, 7), facecolor='#f5f5f5')
ax = plt.gca()
ax.set_facecolor('#ffffff')

# Load and process data
file_path = 'per_site_output.falsta'
data_lines = load_data(file_path)
if not data_lines:
    print("No filtered data to plot. Exiting.")
    exit()

theta_dist, theta_val, pi_dist, pi_val = process_data(data_lines)

# Calculate z-normalized colors
print("Calculating colors")
theta_colors = stats.zscore(theta_val) if len(theta_val) > 0 else np.array([])
pi_colors = stats.zscore(pi_val) if len(pi_val) > 0 else np.array([])

# Create scatter plots
print("Generating scatter plots")
theta_scatter = plt.scatter(np.log10(theta_dist + 1), theta_val,
                          c=theta_colors, cmap='Purples',
                          s=20, alpha=0.3, edgecolors='none',
                          label='Theta') if len(theta_val) > 0 else None

pi_scatter = plt.scatter(np.log10(pi_dist + 1), pi_val,
                        c=pi_colors, cmap='Greens',
                        s=20, alpha=0.3, edgecolors='none',
                        label='Pi') if len(pi_val) > 0 else None

# Add colorbars
print("Adding colorbars")
if theta_scatter:
    theta_cbar = plt.colorbar(theta_scatter, ax=ax, pad=0.01)
    theta_cbar.set_label('Theta Z-score', size=12, color='#333333')
    theta_cbar.ax.tick_params(labelsize=10)

if pi_scatter:
    pi_cbar = plt.colorbar(pi_scatter, ax=ax, pad=0.05)
    pi_cbar.set_label('Pi Z-score', size=12, color='#333333')
    pi_cbar.ax.tick_params(labelsize=10)

# Customize plot
print("Customizing plot")
plt.title('Log Distance from Edge vs. Value (Filtered Data)', 
         pad=20, size=16, fontweight='bold', color='#333333')
plt.xlabel('Log10(Distance from Nearest Edge + 1)', size=14, color='#333333')
plt.ylabel('Value', size=14, color='#333333')
plt.legend(fontsize=12)

ax.grid(True, linestyle='--', alpha=0.4, color='#999999')
ax.tick_params(axis='both', which='major', labelsize=12, color='#666666')
for spine in ax.spines.values():
    spine.set_edgecolor('#cccccc')
    spine.set_linewidth(0.5)

# Adjust layout
print("Adjusting layout")
plt.tight_layout()

# Save and open plot
print("Saving plot")
output_path = Path.home() / 'distance_plot_optimized.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f5f5f5')
print(f"Plot saved to: {output_path}")

print("Opening plot")
if os.name == 'nt':
    os.startfile(output_path)
elif os.name == 'posix':
    os.system(f'open "{output_path}"' if 'darwin' in os.sys.platform else f'xdg-open "{output_path}"')

print("Plot generation complete")
