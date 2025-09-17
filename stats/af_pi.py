import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import logging
import sys
import io # For handling potential BOM
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
INV_INFO_FILE = "inv_info.tsv"
STATS_FILE = "output.csv"

# Columns to use (Reduced)
INV_COLS = ['Chromosome', 'Start', 'End', 'Inverted_AF']
STATS_COLS = ['chr', 'region_start', 'region_end', '0_pi_filtered', '1_pi_filtered']

# Renamed columns for clarity
INV_RENAME = {'Chromosome': 'chr', 'Start': 'start_inv', 'End': 'end_inv', 'Inverted_AF': 'inv_af'}
STATS_RENAME = {'region_start': 'start_stats', 'region_end': 'end_stats', '0_pi_filtered': 'pi0_filtered', '1_pi_filtered': 'pi1_filtered'}

COORDINATE_TOLERANCE = 1

# Output Plot filename
COMBINED_PLOT_FILENAME = "scatter_af_vs_pi_combined.png"

# Styling
PLOT_STYLE = "white"
SCATTER_COLOR_DIRECT = "#377eb8" # Blue
SCATTER_COLOR_INVERTED = "#ff7f00" # Orange
SCATTER_ALPHA = 0.6
SCATTER_SIZE = 40
SCATTER_EDGECOLOR = 'k'
SCATTER_LINEWIDTH = 0.5
LOESS_COLOR = 'darkred'
LOESS_LINEWIDTH = 2.0
LOESS_LINESTYLE = '--'
FIGURE_SIZE = (12, 6) # Width, Height for side-by-side plots
TITLE_FONTSIZE = 18
SUBPLOT_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 15
CORR_TEXT_FONTSIZE = 12


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('correlation_analysis')

# --- Helper Functions ---

def normalize_chromosome(chrom):
    """Converts chromosome names to a consistent 'chr...' format (lowercase)."""
    if pd.isna(chrom):
        return None
    chrom_str = str(chrom).strip().lower()
    if chrom_str.startswith('chr'):
        return 'chr' + chrom_str[3:].replace('_','')
    elif chrom_str.isdigit() or chrom_str in ('x', 'y', 'm', 'mt'):
        return f"chr{chrom_str}"
    else:
        logger.debug(f"Unexpected chromosome format encountered: '{chrom}'. Returning as is.")
        return chrom_str

def format_p_value(p_val):
    """Formats p-value for display."""
    if p_val < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p_val:.3f}"

def add_correlation_text(ax, rho, pval, n_pairs):
    """Adds formatted correlation text to an axes object."""
    if rho is not None and pval is not None and n_pairs is not None:
        p_text = format_p_value(pval)
        corr_text = f"Spearman $\\rho$ = {rho:.3f}\n{p_text}\n(N = {n_pairs})"
        ax.text(0.95, 0.95, corr_text, transform=ax.transAxes, fontsize=CORR_TEXT_FONTSIZE,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='lightgrey'))

def plot_subplot(ax, df, x_col, y_col, title, y_axis_label, scatter_color):
    """Helper function to plot one subplot (scatter + LOESS)."""
    df_filtered = df[[x_col, y_col]].dropna()
    n_pairs = len(df_filtered)

    if n_pairs < 3:
        logger.warning(f"Skipping subplot '{title}': Insufficient data points ({n_pairs}).")
        ax.text(0.5, 0.5, f"Insufficient data (N={n_pairs})",
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlabel(x_col, fontsize=AXIS_LABEL_FONTSIZE) # Set labels even if empty
        ax.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
        return None, None, n_pairs # Indicate insufficient data

    # Calculate correlation
    try:
        rho, pval = spearmanr(df_filtered[x_col], df_filtered[y_col])
        logger.info(f"Correlation for '{title}': Rho={rho:.4f}, P={pval:.4g}, N={n_pairs}")
    except Exception as e:
        logger.error(f"Error calculating Spearman correlation for {title}: {e}")
        rho, pval = None, None

    # Scatter plot
    sns.scatterplot(data=df_filtered, x=x_col, y=y_col, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
                    edgecolor=SCATTER_EDGECOLOR, linewidth=SCATTER_LINEWIDTH, color=scatter_color, ax=ax)

    # LOESS curve
    try:
        sns.regplot(data=df_filtered, x=x_col, y=y_col,
                    scatter=False, lowess=True,
                    line_kws={'color': LOESS_COLOR, 'lw': LOESS_LINEWIDTH, 'linestyle': LOESS_LINESTYLE},
                    ax=ax)
    except Exception as e:
        logger.warning(f"Could not generate LOESS curve for '{title}': {e}")

    # Styling and Labels
    ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE, pad=10)
    ax.set_xlabel("Inversion Allele Frequency", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_axis_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=AXIS_LABEL_FONTSIZE - 2)

    # Add correlation text
    add_correlation_text(ax, rho, pval, n_pairs)

    return rho, pval, n_pairs


def create_combined_plot(df, filename):
    """Creates and saves a combined scatter plot with two subplots."""
    logger.info(f"Creating combined scatter plot with LOESS...")
    sns.set_theme(style=PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=False) # Use separate Y axes if scales differ

    # --- Subplot 1: AF vs Pi (Direct Orientation) ---
    plot_subplot(axes[0], df, 'inv_af', 'pi0_filtered',
                 'Direct Orientation',
                 'Nucleotide Diversity (π)',
                 SCATTER_COLOR_DIRECT)

    # --- Subplot 2: AF vs Pi (Inverted Orientation) ---
    plot_subplot(axes[1], df, 'inv_af', 'pi1_filtered',
                 'Inverted Orientation',
                 'Nucleotide Diversity (π)',
                 SCATTER_COLOR_INVERTED)

    # --- Overall Figure Styling ---
    fig.suptitle('Inversion Allele Frequency vs. Nucleotide Diversity (π)', fontsize=TITLE_FONTSIZE, y=0.98) # Adjust y to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

    # --- Save ---
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved combined plot to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to save combined plot '{filename}': {e}")
    plt.close(fig) # Close the figure to free memory


# --- Main Execution ---
def main():
    logger.info("Starting simplified correlation analysis (AF vs Pi) with combined LOESS plot...")

    # --- Load Inversion Info ---
    try:
        # BOM Handling and loading logic
        try:
            with open(INV_INFO_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('\ufeff'): content = content[1:]
            df_inv = pd.read_csv(io.StringIO(content), sep='\t', usecols=INV_COLS)
        except UnicodeDecodeError:
             logger.warning(f"UTF-8 decoding failed for {INV_INFO_FILE}. Trying latin-1.")
             with open(INV_INFO_FILE, 'r', encoding='latin-1') as f: content = f.read()
             df_inv = pd.read_csv(io.StringIO(content), sep='\t', usecols=INV_COLS)

        logger.info(f"Loaded {len(df_inv)} rows from {INV_INFO_FILE}")
        df_inv = df_inv.rename(columns=INV_RENAME)
        # Ensure columns exist
        required_inv_cols = list(INV_RENAME.values())
        if not all(col in df_inv.columns for col in required_inv_cols):
             missing = [k for k,v in INV_RENAME.items() if v not in df_inv.columns]
             raise ValueError(f"Missing required columns after rename in {INV_INFO_FILE}: {missing}")

        # Data cleaning and type conversion
        df_inv['chr'] = df_inv['chr'].apply(normalize_chromosome)
        df_inv['start_inv'] = pd.to_numeric(df_inv['start_inv'], errors='coerce').astype('Int64')
        df_inv['end_inv'] = pd.to_numeric(df_inv['end_inv'], errors='coerce').astype('Int64')
        df_inv['inv_af'] = pd.to_numeric(df_inv['inv_af'], errors='coerce')

        # Drop rows missing essential coordinates or chromosome
        initial_inv_count = len(df_inv)
        df_inv = df_inv.dropna(subset=['chr', 'start_inv', 'end_inv'])
        logger.info(f"Kept {len(df_inv)} inversion rows after removing rows with missing chr/coordinates (out of {initial_inv_count}).")

    except FileNotFoundError: logger.critical(f"Error: Input file not found: {INV_INFO_FILE}"); sys.exit(1)
    except ValueError as e: logger.critical(f"Error processing {INV_INFO_FILE}: {e}"); sys.exit(1)
    except Exception as e: logger.critical(f"An unexpected error occurred loading {INV_INFO_FILE}: {e}"); sys.exit(1)

    # --- Load Stats Info ---
    try:
        df_stats = pd.read_csv(STATS_FILE, usecols=STATS_COLS, low_memory=False)
        logger.info(f"Loaded {len(df_stats)} rows from {STATS_FILE}")
        df_stats = df_stats.rename(columns=STATS_RENAME)
        # Ensure columns exist
        required_stats_cols = list(STATS_RENAME.values())
        if not all(col in df_stats.columns for col in required_stats_cols):
             missing = [k for k,v in STATS_RENAME.items() if v not in df_stats.columns]
             raise ValueError(f"Missing required columns after rename in {STATS_FILE}: {missing}")

        # Data cleaning and type conversion
        df_stats['chr'] = df_stats['chr'].apply(normalize_chromosome)
        df_stats['start_stats'] = pd.to_numeric(df_stats['start_stats'], errors='coerce').astype('Int64')
        df_stats['end_stats'] = pd.to_numeric(df_stats['end_stats'], errors='coerce').astype('Int64')
        df_stats['pi0_filtered'] = pd.to_numeric(df_stats['pi0_filtered'], errors='coerce')
        df_stats['pi1_filtered'] = pd.to_numeric(df_stats['pi1_filtered'], errors='coerce')

        # Drop rows missing essential coordinates or chromosome
        initial_stats_count = len(df_stats)
        df_stats = df_stats.dropna(subset=['chr', 'start_stats', 'end_stats'])
        logger.info(f"Kept {len(df_stats)} stats rows after removing rows with missing chr/coordinates (out of {initial_stats_count}).")

    except FileNotFoundError: logger.critical(f"Error: Input file not found: {STATS_FILE}"); sys.exit(1)
    except ValueError as e: logger.critical(f"Error processing {STATS_FILE}: {e}"); sys.exit(1)
    except Exception as e: logger.critical(f"An unexpected error occurred loading {STATS_FILE}: {e}"); sys.exit(1)


    # --- Merge DataFrames ---
    logger.info("--- Merging Inversion and Stats Data ---")
    # Merge on chromosome first
    merged_inv_stats = pd.merge(df_inv, df_stats, on='chr', how='inner')
    logger.info(f"Initial merge on chr: {len(merged_inv_stats)} potential pairs.")

    # Filter by coordinate tolerance
    filter_coords_stats = (
        abs(merged_inv_stats['start_inv'] - merged_inv_stats['start_stats']) <= COORDINATE_TOLERANCE
    ) & (
        abs(merged_inv_stats['end_inv'] - merged_inv_stats['end_stats']) <= COORDINATE_TOLERANCE
    )
    merged_inv_stats_filtered = merged_inv_stats[filter_coords_stats].copy()
    logger.info(f"Filtered by coordinate tolerance: {len(merged_inv_stats_filtered)} pairs.")

    # Handle duplicates: Keep one stats record per inversion
    if not merged_inv_stats_filtered.empty:
        inversion_id_cols = ['chr', 'start_inv', 'end_inv']
        # Select the non-key columns needed for correlation/plotting
        cols_to_select_within_group = ['inv_af', 'pi0_filtered', 'pi1_filtered']
        # Group by inversion, take first matched stats, reset index
        df_final = merged_inv_stats_filtered.groupby(
            inversion_id_cols,
            observed=True,
            dropna=False # Keep group ID even if values are NaN
        )[cols_to_select_within_group].first().reset_index()
        logger.info(f"Deduplicated: {len(df_final)} unique inversions with matched stats.")
    else:
        logger.warning("No matching regions found between Inversions and Stats after coordinate filtering.")
        df_final = pd.DataFrame(columns=['chr', 'start_inv', 'end_inv', 'inv_af', 'pi0_filtered', 'pi1_filtered'])


    # --- Generate Combined Plot ---
    if df_final.empty:
         logger.warning("Final merged dataframe is empty. No plot can be generated.")
    else:
        create_combined_plot(df_final, COMBINED_PLOT_FILENAME)

    logger.info("--- Analysis complete ---")


if __name__ == "__main__":
    main()
