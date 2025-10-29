"""
PheWAS Category Summary Visualization
Creates a heatmap showing GBJ enrichment (outlines) and GLS directionality (arrows)
for inversion variants across disease categories.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from pathlib import Path
from urllib.parse import quote

# Set font globally
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 20,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Configuration
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/"

category_files = [
    "phewas v2 - chr10-79542902-INV-674513_category_summary.tsv",
    "phewas v2 - chr12-46897663-INV-16289_category_summary.tsv",
    "phewas v2 - chr17-45585160-INV-706887_category_summary.tsv",
    "phewas v2 - chr21-13992018-INV-65632_category_summary.tsv",
    "phewas v2 - chr4-33098029-INV-7075_category_summary.tsv",
    "phewas v2 - chr6-141867315-INV-29159_category_summary.tsv",
    "phewas v2 - chr6-76111919-INV-44661_category_summary.tsv",
    "phewas v2 - chr7-57835189-INV-284465_category_summary.tsv",
]

# Custom category order (organ systems → pathological)
CATEGORY_ORDER = [
    'Cardiovascular',
    'Respiratory',
    'Gastrointestinal',
    'Genitourinary',
    'Endocrine/Metab',
    'Blood/Immune',
    'Neurological',
    'Mental',
    'Sense organs',
    'Muscloskeletal',
    'Dermatological',
    'Symptoms',
    'Infections',
    'Neoplasms',
    'Congenital'
]

def download_file(filename):
    """Download a single file from GitHub"""
    encoded_filename = quote(filename)
    url = BASE_URL + encoded_filename
    local_path = data_dir / filename
    
    response = requests.get(url)
    response.raise_for_status()
    
    with open(local_path, 'w') as f:
        f.write(response.text)
    
    return local_path

def download_all_files():
    """Download all required files from GitHub"""
    print("Downloading files...\n")
    
    # Download coordinate mapping file
    print("  balanced_recurrence_results.tsv...", end=" ")
    download_file("balanced_recurrence_results.tsv")
    print("done")
    
    # Download category summary files
    for filename in category_files:
        print(f"  {filename}...", end=" ")
        download_file(filename)
        print("done")

def load_coordinate_mapping():
    """Load inversion ID to coordinate mapping"""
    mapping_file = data_dir / "balanced_recurrence_results.tsv"
    df = pd.read_csv(mapping_file, sep='\t')
    
    # Create mapping dictionary: Inversion_ID -> (Chromosome, Start, End)
    mapping = {}
    for _, row in df.iterrows():
        inv_id = row['Inversion_ID']
        mapping[inv_id] = {
            'chromosome': row['Chromosome'],
            'start': int(row['Start']),
            'end': int(row['End'])
        }
    
    return mapping

def extract_inversion_id(filename):
    """Extract Inversion_ID from filename"""
    # Format: "phewas v2 - chr6-141867315-INV-29159_category_summary.tsv"
    parts = filename.replace("phewas v2 - ", "").replace("_category_summary.tsv", "")
    return parts

def format_variant_label(chrom, start, end):
    """Format variant label as chr:start-end with comma separators"""
    return f"{chrom}:{start:,}-{end:,}"

def load_all_data():
    """Load all category summary files into a combined dataframe"""
    # Load coordinate mapping
    coord_map = load_coordinate_mapping()
    
    all_data = []
    
    for filename in category_files:
        local_path = data_dir / filename
        df = pd.read_csv(local_path, sep='\t')
        
        # Extract inversion ID and get coordinates
        inv_id = extract_inversion_id(filename)
        
        if inv_id in coord_map:
            coords = coord_map[inv_id]
            chrom = coords['chromosome']
            start = coords['start']
            end = coords['end']
            
            # Add columns
            df['inversion_id'] = inv_id
            df['chromosome'] = chrom
            df['start'] = start
            df['end'] = end
            df['variant_label'] = format_variant_label(chrom, start, end)
            
            # For sorting: extract chromosome number
            chrom_num = int(chrom.replace('chr', '')) if chrom.replace('chr', '').isdigit() else 99
            df['chrom_num'] = chrom_num
            
            all_data.append(df)
        else:
            print(f"Warning: {inv_id} not found in coordinate mapping")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['-log10_P_GBJ'] = -np.log10(combined['P_GBJ'])
    
    return combined

def create_phewas_heatmap(df, output_file='phewas_category_heatmap.pdf'):
    """Create the main visualization"""
    
    # Sort variants by chromosome and position
    variant_info = df.groupby('variant_label').agg({
        'chrom_num': 'first',
        'start': 'first'
    }).reset_index()
    variant_info = variant_info.sort_values(['chrom_num', 'start'])
    variants = variant_info['variant_label'].tolist()
    
    # Filter categories to only those present
    categories = [cat for cat in CATEGORY_ORDER if cat in df['Category'].unique()]
    
    n_variants = len(variants)
    n_categories = len(categories)
    
    # Create matrices for visualization
    p_gbj_matrix = np.ones((n_variants, n_categories))
    q_gbj_matrix = np.ones((n_variants, n_categories))
    q_gls_matrix = np.ones((n_variants, n_categories))
    direction_matrix = np.full((n_variants, n_categories), '', dtype=object)
    
    for i, variant in enumerate(variants):
        for j, category in enumerate(categories):
            row = df[(df['variant_label'] == variant) & (df['Category'] == category)]
            if len(row) > 0:
                row = row.iloc[0]
                p_gbj_matrix[i, j] = row['P_GBJ']
                q_gbj_matrix[i, j] = row['Q_GBJ']
                q_gls_matrix[i, j] = row['Q_GLS']
                direction_matrix[i, j] = row['Direction']
    
    # Convert to -log10(P) for background gradient
    log_p_matrix = -np.log10(p_gbj_matrix)
    
    # Clip gradient at P=0.05 (-log10(0.05) = 1.301)
    log_p_clipped = np.clip(log_p_matrix, 0, 1.301)
    
    # Calculate figure size to make cells square
    # Each cell should be square, so width/height ratio = n_categories/n_variants
    cell_size = 1.2  # inches per cell
    fig_width = n_categories * cell_size + 5  # extra space for labels and colorbar
    fig_height = n_variants * cell_size + 2  # extra space for labels
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create gridspec for main plot and colorbar
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.05)
    
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    
    # Background gradient (white to light gray)
    background_norm = log_p_clipped / 1.301
    
    # Create custom colormap: white to light gray
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('custom', ['white', '#d0d0d0'])
    
    # Plot background
    im = ax.imshow(background_norm, cmap=cmap, aspect='equal', 
                   extent=[0, n_categories, n_variants, 0],
                   vmin=0, vmax=1, zorder=1)
    
    # Add grid lines
    for i in range(n_variants + 1):
        ax.axhline(i, color='white', linewidth=2, zorder=2)
    for j in range(n_categories + 1):
        ax.axvline(j, color='white', linewidth=2, zorder=2)
    
    # Add bold outlines for Q_GBJ < 0.05
    for i in range(n_variants):
        for j in range(n_categories):
            if q_gbj_matrix[i, j] < 0.05:
                rect = Rectangle((j, i), 1, 1, 
                               linewidth=4, 
                               edgecolor='black', 
                               facecolor='none',
                               zorder=3)
                ax.add_patch(rect)
    
    # Add arrows for Q_GLS < 0.05
    for i in range(n_variants):
        for j in range(n_categories):
            if q_gls_matrix[i, j] < 0.05:
                if direction_matrix[i, j] == 'increase':
                    symbol = '▲'
                    color = '#d62728'  # Red
                elif direction_matrix[i, j] == 'decrease':
                    symbol = '▼'
                    color = '#1f77b4'  # Blue
                else:
                    continue
                
                ax.text(j + 0.5, i + 0.5, symbol,
                       ha='center', va='center',
                       fontsize=36,
                       color=color, zorder=4)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_categories) + 0.5)
    ax.set_yticks(np.arange(n_variants) + 0.5)
    
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=24)
    ax.set_yticklabels(variants, fontsize=24)
    
    # Remove tick marks
    ax.tick_params(which='both', length=0)
    
    # Labels (no y-axis label, no title)
    ax.set_xlabel('Disease Category', fontsize=28, labelpad=15)
    
    # Add colorbar for p-value gradient
    # Create a colorbar with custom ticks
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('P-value (GBJ)', fontsize=24, labelpad=20)
    
    # Set custom tick positions and labels
    tick_positions = [0, 0.5, 1.0]
    tick_labels = ['1.0', '0.22', '≤0.05']
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels, fontsize=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='black', linewidth=3,
                      label='GBJ enrichment (Q < 0.05)'),
        mpatches.Patch(facecolor='#d62728', edgecolor='none',
                      label='▲ Increased risk (GLS Q < 0.05)'),
        mpatches.Patch(facecolor='#1f77b4', edgecolor='none',
                      label='▼ Decreased risk (GLS Q < 0.05)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', 
             bbox_to_anchor=(1.15, 1), fontsize=20,
             frameon=True, fancybox=False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure as PDF
    plt.savefig(output_file, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved: {output_file}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  GBJ significant (Q<0.05): {(q_gbj_matrix < 0.05).sum()}")
    print(f"  GLS significant (Q<0.05): {(q_gls_matrix < 0.05).sum()}")
    print(f"  Both significant: {((q_gbj_matrix < 0.05) & (q_gls_matrix < 0.05)).sum()}")
    
    increase_count = ((q_gls_matrix < 0.05) & (direction_matrix == 'increase')).sum()
    decrease_count = ((q_gls_matrix < 0.05) & (direction_matrix == 'decrease')).sum()
    print(f"  Risk-increasing: {increase_count}")
    print(f"  Risk-decreasing: {decrease_count}")
    
    return fig, ax

def main():
    """Main execution"""
    # Download all files
    download_all_files()
    
    print("\nLoading data...")
    df = load_all_data()
    
    print(f"Loaded {len(df)} variant-category combinations")
    
    # Create visualization
    print("\nCreating visualization...")
    fig, ax = create_phewas_heatmap(df)
    
    print("\nDone!")
    
    plt.show()

if __name__ == "__main__":
    main()
