import requests
import re
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


def benjamini_hochberg(p_values):
    """Calculates Benjamini-Hochberg FDR q-values."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    
    # Sort indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Calculate q = (p * n) / rank
    ranks = np.arange(1, n + 1)
    q_values = (sorted_p * n) / ranks
    
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])
        
    # Cap at 1.0
    q_values = np.minimum(q_values, 1.0)
    
    # Map back to original order
    original_order_q = np.zeros_like(q_values)
    original_order_q[sorted_indices] = q_values
    
    return original_order_q

def get_recurrence_status(val):
    """Maps the '0_single_1_recur_consensus' column to a string label."""
    if pd.isna(val):
        return "Other"
    if val == 1:
        return "Recurrent"
    elif val == 0:
        return "Single-Event"
    return "Other"

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def main():
    # -----------------------------------------------------
    # 1. DOWNLOAD & PARSE LOG FILE (METRICS)
    # -----------------------------------------------------
    log_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/imputation/linked.log"
    print(f"Downloading Metrics Log: {log_url}...")
    
    try:
        resp_log = requests.get(log_url)
        resp_log.raise_for_status()
    except Exception as e:
        print(f"Error downloading logs: {e}")
        return

    # Regex to capture metrics
    regex_pattern = re.compile(
        r"\[\[(?P<id>.*?)\]\s+METRICS:\s+"
        r"r2=(?P<r2>[\d\.\-]+)\s+"
        r"rmse=(?P<rmse>[\d\.\-]+)\s+"
        r"p=(?P<p>[\de\.\-]+)\s+"
        r"ncomp=(?P<ncomp>\d+)\s+"
        r"snps=(?P<snps>\d+)"
    )

    data_rows = []
    for line in resp_log.text.splitlines():
        match = regex_pattern.search(line)
        if match:
            entry = match.groupdict()
            data_rows.append({
                "id": entry["id"],
                "unbiased_pearson_r2": float(entry["r2"]),
                "unbiased_rmse": float(entry["rmse"]),
                "model_p_value": float(entry["p"]),
                "best_n_components": int(entry["ncomp"]),
                "num_snps_in_model": int(entry["snps"])
            })

    df_metrics = pd.DataFrame(data_rows)
    # Keep last entry for duplicate IDs in the log
    df_metrics = df_metrics.drop_duplicates(subset=['id'], keep='last')
    print(f"Parsed {len(df_metrics)} unique models from log.")

    # -----------------------------------------------------
    # 2. DOWNLOAD & PARSE PROPERTIES FILE (METADATA)
    # -----------------------------------------------------
    prop_url = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/inv_properties.tsv"
    print(f"Downloading Properties TSV: {prop_url}...")
    
    try:
        resp_prop = requests.get(prop_url)
        resp_prop.raise_for_status()
    except Exception as e:
        print(f"Error downloading properties: {e}")
        return
    
    df_props = pd.read_csv(io.StringIO(resp_prop.text), sep='\t')
    
    # -----------------------------------------------------
    # 3. MERGE DATA
    # -----------------------------------------------------
    print("Merging metrics with metadata...")
    df_merged = pd.merge(
        df_metrics, 
        df_props[['OrigID', 'Chromosome', 'Start', 'End', '0_single_1_recur_consensus']], 
        left_on='id', 
        right_on='OrigID', 
        how='inner'
    )

    # -----------------------------------------------------
    # 4. CALCULATIONS & FORMATTING
    # -----------------------------------------------------
    # Calculate FDR
    df_merged['p_fdr_bh'] = benjamini_hochberg(df_merged['model_p_value'].values)
    
    # Format Coordinates: chr:start-end
    df_merged['coords'] = (
        df_merged['Chromosome'].astype(str) + ":" + 
        df_merged['Start'].astype(str) + "-" + 
        df_merged['End'].astype(str)
    )

    # Set Status labels
    df_merged['status'] = df_merged['0_single_1_recur_consensus'].apply(get_recurrence_status)

    # Sort by R2 descending
    df_merged = df_merged.sort_values(by='unbiased_pearson_r2', ascending=False).reset_index(drop=True)

    # Save to file
    df_merged.to_csv("imputation_results_merged.tsv", sep='\t', index=False)
    print("Merged data saved to 'imputation_results_merged.tsv'.")

    # -----------------------------------------------------
    # 5. PLOTTING
    # -----------------------------------------------------
    print("Generating High-Res Plot...")

    # --- STYLE SETTINGS ---
    sns.set_style("whitegrid", {'axes.grid' : False}) 
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 20,           
        'axes.labelsize': 32,      
        'xtick.labelsize': 15,  
        'ytick.labelsize': 22,     
        'legend.fontsize': 24,
        'legend.title_fontsize': 26,
        'figure.titlesize': 34
    })

    # Set up figure size: EXTRA WIDE (50 inches wide, 14 inches high)
    fig, ax = plt.subplots(figsize=(50, 14))

    # Define Colors
    color_map = {
        "Recurrent": "#E74C3C",    # Flat Red
        "Single-Event": "#3498DB", # Flat Blue
        "Other": "#95A5A6"         # Flat Gray
    }
    bar_colors = [color_map[s] for s in df_merged['status']]

    # Draw Bars
    bars = ax.bar(
        x=df_merged['coords'],
        height=df_merged['unbiased_pearson_r2'],
        color=bar_colors,
        width=0.8,
        edgecolor='none', 
        zorder=3
    )

    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray', zorder=0)

    ax.axhline(y=0.5, color='#2C3E50', linewidth=5, linestyle='--', alpha=0.9, zorder=4)

    # Formatting Axes
    ax.set_ylabel(r'$r^2$', labelpad=25, fontweight='bold') 
    ax.set_xlabel("") 
    
    plt.xticks(rotation=45, ha='right')
    
    sns.despine(left=False, bottom=False, top=True, right=True)

    # Custom Legend
    legend_patches = [
        mpatches.Patch(color=color_map['Recurrent'], label='Recurrent'),
        mpatches.Patch(color=color_map['Single-Event'], label='Single-Event'),
        mpatches.Patch(color=color_map['Other'], label='Other/NA')
    ]
    
    ax.legend(
        handles=legend_patches, 
        title="Inversion Status", 
        loc="upper right", 
        frameon=True, 
        framealpha=0.95,
        edgecolor='white',
        shadow=True
    )

    # Tight layout prevents cutting off the tilted labels
    plt.tight_layout()

    # Save as PDF
    out_pdf = "inversion_r2_sorted.pdf"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    
    print(f"Plot saved successfully to '{out_pdf}'")

if __name__ == "__main__":
    main()
