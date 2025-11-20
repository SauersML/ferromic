import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def extract_superpopulation(haplotype):
    """Extract superpopulation from haplotype name (e.g., EUR from EUR_GBR_HG00096_L)"""
    parts = haplotype.split('_')
    if len(parts) >= 1:
        return parts[0]
    return "Unknown"

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """Create a plot of the covariance confidence ellipse of x and y."""
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    if x.size <= 2:
        return None

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of
    # the variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_pca_for_chromosome(df, chromosome, output_folder, superpop_colors):
    """Create and save a PCA plot for a single chromosome"""
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Plot each superpopulation
    for sp, color in superpop_colors.items():
        subset = df[df['Superpopulation'] == sp]
        if not subset.empty:
            plt.scatter(subset['PC1'], subset['PC2'], 
                       label=f"{sp} (n={len(subset)})", 
                       color=color, alpha=0.7)
            
            # Add confidence ellipse if enough points
            if len(subset) > 2:
                confidence_ellipse(subset['PC1'], subset['PC2'], 
                                  ax, n_std=2.0, edgecolor=color, linestyle='--', 
                                  linewidth=1.5, alpha=0.6)
    
    plt.title(f"PCA Plot - Chromosome {chromosome}", fontsize=16)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(loc='best', fontsize=12)
    
    # Save the plot
    output_file = os.path.join(output_folder, f"pca_plot_chr_{chromosome}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def main():
    # Folder containing PCA results
    pca_folder = "pca"
    # Output folder for plots
    output_folder = "pca_plots"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PCA files
    pca_files = glob.glob(os.path.join(pca_folder, "pca_chr_*.tsv"))
    
    if not pca_files:
        print(f"No PCA files found in {pca_folder}")
        return
    
    print(f"Found {len(pca_files)} PCA files")
    
    # Collect all unique superpopulations across all files
    all_superpops = set()
    for file_path in pca_files:
        try:
            df = pd.read_csv(file_path, sep='\t')
            df['Superpopulation'] = df['Haplotype'].apply(extract_superpopulation)
            all_superpops.update(df['Superpopulation'].unique())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Detected superpopulations: {', '.join(sorted(all_superpops))}")
    
    # Create a colormap for superpopulations
    colormap = plt.cm.get_cmap('tab10', len(all_superpops))
    superpop_colors = {sp: colormap(i) for i, sp in enumerate(sorted(all_superpops))}
    
    # Process each chromosome file
    output_files = []
    for file_path in sorted(pca_files):
        try:
            # Extract chromosome name from filename
            chr_name = os.path.basename(file_path).replace("pca_chr_", "").replace(".tsv", "")
            print(f"Processing chromosome {chr_name}...")
            
            # Read PCA data
            df = pd.read_csv(file_path, sep='\t')
            
            # Extract superpopulation from haplotype
            df['Superpopulation'] = df['Haplotype'].apply(extract_superpopulation)
            
            # Create and save the plot
            output_file = plot_pca_for_chromosome(df, chr_name, output_folder, superpop_colors)
            output_files.append(output_file)
            
            print(f"  Plot saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create a combined figure showing all chromosomes
    print("Creating combined plot for all chromosomes...")
    
    # Determine grid size
    n_files = len(pca_files)
    grid_size = int(np.ceil(np.sqrt(n_files)))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 4*grid_size))
    if grid_size > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for the loop below
    
    # Process each chromosome file for the combined plot
    for i, file_path in enumerate(sorted(pca_files)):
        try:
            # Extract chromosome name from filename
            chr_name = os.path.basename(file_path).replace("pca_chr_", "").replace(".tsv", "")
            
            # Read PCA data
            df = pd.read_csv(file_path, sep='\t')
            df['Superpopulation'] = df['Haplotype'].apply(extract_superpopulation)
            
            # Plot on the current axis
            ax = axes[i]
            
            # Plot each superpopulation
            for sp, color in superpop_colors.items():
                subset = df[df['Superpopulation'] == sp]
                if not subset.empty:
                    ax.scatter(subset['PC1'], subset['PC2'], color=color, alpha=0.7, s=15)
                    
                    # Add confidence ellipse if enough points
                    if len(subset) > 2:
                        confidence_ellipse(subset['PC1'], subset['PC2'], 
                                          ax, n_std=2.0, edgecolor=color, linestyle='--', 
                                          linewidth=1.0, alpha=0.6)
            
            ax.set_title(f"Chr {chr_name}", fontsize=16)
            ax.set_xlabel("PC1", fontsize=13)
            ax.set_ylabel("PC2", fontsize=13)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
        except Exception as e:
            print(f"Error adding {file_path} to combined plot: {e}")
            if i < len(axes):
                axes[i].text(0.5, 0.5, f"Error: {chr_name}", ha='center', va='center')
    
    # Remove unused subplots
    for j in range(i+1, grid_size*grid_size):
        if j < len(axes):
            fig.delaxes(axes[j])
    
    # Add a legend to the figure
    handles, labels = [], []
    for sp in sorted(all_superpops):
        color = superpop_colors[sp]
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
        labels.append(sp)
    
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=min(len(all_superpops), 5),
        bbox_to_anchor=(0.5, 0.02),
        fontsize=12,
    )
    
    plt.suptitle("PCA Plots for All Chromosomes (PC1 vs PC2)", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save combined plot
    combined_output = os.path.join(output_folder, "pca_plot_all_chromosomes.png")
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plot saved to {combined_output}")
    print(f"PCA visualization complete. All plots saved to {output_folder}")

if __name__ == "__main__":
    main()
