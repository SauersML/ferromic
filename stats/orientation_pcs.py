import pandas as pd
import numpy as np
import os
import glob
import re
import logging
import sys
import time
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationError

# =====================================================================
# Configuration
# =====================================================================
GENOTYPE_FILE = 'variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv'
PCA_FOLDER = "pca"
N_PCS: int = 5  # Number of Principal Components to use in the analysis
MIN_HAPS_PER_GROUP: int = 10 # Minimum number of both direct (0) and inverted (1) haplotypes needed to run analysis for an inversion
OUTPUT_RESULTS_CSV = 'inversion_stratification_results.csv'

# Columns to identify samples in GENOTYPE_FILE (adjust if needed)
# Assuming sample IDs start after the 'categ' column
FIRST_SAMPLE_COL_INDEX: int = 8 # Index of the first sample column (0-based)

# Valid genotype strings
VALID_GENOTYPES = {'0|0', '0|1', '1|0', '1|1'}

# =====================================================================
# Logging Setup
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Optional: Add FileHandler if you want logs saved to a file
        # logging.FileHandler('stratification_analysis.log')
    ]
)
logger = logging.getLogger('stratification_analysis')

# =====================================================================
# Helper Functions
# =====================================================================

def normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome name to 'chrN' or 'chrX' format."""
    chrom = str(chrom).strip()
    # Strip any existing 'chr' or 'chr_' prefix first
    if chrom.startswith('chr_'):
        chrom = chrom[4:]
    elif chrom.startswith('chr'):
        chrom = chrom[3:]
    # Add 'chr' prefix back if it's missing
    if not chrom.startswith('chr'):
        return f"chr{chrom}"
    return chrom # Already in correct format or was corrected

def harmonize_pca_haplotype_name(pca_hap_name: str) -> Optional[str]:
    """
    Converts PCA haplotype name (e.g., 'EUR_GBR_HG00096_L')
    to harmonized format (e.g., 'HG00096_L') by stripping population prefixes.
    ASSUMPTION: Sample ID is the second-to-last part before _L/_R suffix.
    """
    parts = pca_hap_name.split('_')
    if len(parts) < 2:
        # logger.warning(f"Cannot parse PCA haplotype name format: {pca_hap_name}")
        return None # Cannot reliably extract sample ID and suffix

    sample_id = parts[-2]
    hap_suffix = parts[-1]

    if hap_suffix not in ('L', 'R'):
        # logger.warning(f"Unexpected suffix in PCA haplotype name: {pca_hap_name}")
        return None # Suffix should be L or R

    # Basic check if sample_id looks like common formats (e.g., HG..., NA...)
    # This is optional but can help catch errors
    # if not (sample_id.startswith('HG') or sample_id.startswith('NA')):
        # logger.debug(f"Potentially unusual sample ID extracted: {sample_id} from {pca_hap_name}")

    return f"{sample_id}_{hap_suffix}"

def load_pca_data(pca_folder: str, n_pcs: int) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """
    Loads PCA data for all chromosomes, harmonizes haplotype names.

    Args:
        pca_folder: Path to the folder containing PCA files.
        n_pcs: Number of principal components to load.

    Returns:
        A nested dictionary: {chromosome: {harmonized_hap_id: [PC1, ..., PCn]}}
        Returns None if the folder doesn't exist or no files are loaded.
    """
    logger.info(f"Loading PCA data for {n_pcs} PCs from '{pca_folder}'...")
    if not os.path.isdir(pca_folder):
        logger.error(f"PCA folder not found: {pca_folder}")
        return None

    pca_data: Dict[str, Dict[str, List[float]]] = {}
    pca_files = glob.glob(os.path.join(pca_folder, "pca_chr_*.tsv"))

    if not pca_files:
        logger.warning(f"No PCA files found matching 'pca_chr_*.tsv' in {pca_folder}.")
        return None

    pc_cols_to_load = [f"PC{i+1}" for i in range(n_pcs)]
    total_haplotypes_loaded = 0
    missing_hap_names = set()

    for pca_file in pca_files:
        try:
            base_name = os.path.basename(pca_file)
            match = re.search(r'pca_chr_(\w+)\.tsv', base_name)
            if not match:
                logger.warning(f"Could not determine chromosome from PCA filename: {base_name}")
                continue
            chrom_num = match.group(1)
            chrom = normalize_chromosome(chrom_num)

            df = pd.read_csv(pca_file, sep='\t')

            # Verify required columns
            if 'Haplotype' not in df.columns:
                 logger.warning(f"Column 'Haplotype' not found in {pca_file}. Skipping.")
                 continue
            missing_pcs = [pc for pc in pc_cols_to_load if pc not in df.columns]
            if missing_pcs:
                logger.error(f"Missing required PC columns in {pca_file}: {missing_pcs}. Cannot use this file for {n_pcs} PCs.")
                continue # Skip file if requested PCs aren't present

            pca_data[chrom] = {}
            file_hap_count = 0
            for _, row in df.iterrows():
                pca_hap_name = row['Haplotype']
                harmonized_hap_id = harmonize_pca_haplotype_name(pca_hap_name)

                if harmonized_hap_id:
                    # Store PC values as a list of floats, handle potential non-numeric data
                    try:
                        pc_values = [float(row[pc]) for pc in pc_cols_to_load]
                        pca_data[chrom][harmonized_hap_id] = pc_values
                        file_hap_count += 1
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not convert PC values to float for haplotype {pca_hap_name} in {pca_file}: {e}")
                else:
                     missing_hap_names.add(pca_hap_name)


            logger.info(f"  Loaded {file_hap_count} haplotypes for {chrom} from {base_name}")
            total_haplotypes_loaded += file_hap_count

        except FileNotFoundError:
             logger.error(f"PCA file processed during loop but now not found: {pca_file}") # Should not happen if glob worked
        except pd.errors.EmptyDataError:
             logger.warning(f"PCA file is empty: {pca_file}")
        except Exception as e:
            logger.error(f"Failed to process PCA file {pca_file}: {e}", exc_info=True)

    if not pca_data:
        logger.warning("No PCA data was successfully loaded.")
        return None
    if missing_hap_names:
         logger.warning(f"Could not harmonize or parse format for {len(missing_hap_names)} unique PCA haplotype names (example: {next(iter(missing_hap_names))}).")

    logger.info(f"Finished loading PCA data for {len(pca_data)} chromosomes, {total_haplotypes_loaded} total haplotype entries.")
    return pca_data


def load_genotype_data(genotype_file: str, first_sample_col_idx: int) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Loads inversion genotype data, identifies sample columns, normalizes chromosomes.

    Args:
        genotype_file: Path to the inversion genotype TSV file.
        first_sample_col_idx: Index of the first column containing sample genotypes.

    Returns:
        A tuple: (DataFrame containing genotype data, List of sample ID column names)
        Returns None if file loading fails or format is unexpected.
    """
    logger.info(f"Loading inversion genotype data from '{genotype_file}'...")
    try:
        geno_df = pd.read_csv(genotype_file, sep='\t')
        logger.info(f"Loaded {len(geno_df)} inversions.")

        if 'seqnames' not in geno_df.columns:
             logger.error("Required column 'seqnames' not found in genotype file.")
             return None

        # Normalize chromosome names
        geno_df['chrom_normalized'] = geno_df['seqnames'].apply(normalize_chromosome)

        # Identify sample columns
        if first_sample_col_idx >= len(geno_df.columns):
             logger.error(f"First sample column index ({first_sample_col_idx}) is out of bounds.")
             return None
        sample_id_cols = geno_df.columns[first_sample_col_idx:].tolist()
        if not sample_id_cols:
             logger.error("Could not identify any sample ID columns.")
             return None
        logger.info(f"Identified {len(sample_id_cols)} sample columns (starting from '{sample_id_cols[0]}').")

        return geno_df, sample_id_cols

    except FileNotFoundError:
        logger.error(f"Genotype file not found: {genotype_file}")
        return None
    except Exception as e:
        logger.error(f"Error loading genotype data: {e}", exc_info=True)
        return None

# =====================================================================
# Main Analysis Logic
# =====================================================================

def run_stratification_analysis(
    geno_df: pd.DataFrame,
    sample_id_cols: List[str],
    pca_data: Dict[str, Dict[str, List[float]]],
    n_pcs: int,
    min_haps_per_group: int
) -> pd.DataFrame:
    """
    Performs stratification analysis for each inversion.

    Args:
        geno_df: DataFrame with inversion genotypes.
        sample_id_cols: List of column names for samples.
        pca_data: Loaded PCA data.
        n_pcs: Number of PCs to use.
        min_haps_per_group: Minimum haplotypes required per group (0 and 1).

    Returns:
        DataFrame containing analysis results for each inversion.
    """
    results = []
    pc_col_names = [f"PC{i+1}" for i in range(n_pcs)]
    missing_hap_pca_warnings = {} # Track warnings per inversion

    # Use tqdm for progress bar if available
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(geno_df.iterrows(), total=len(geno_df), desc="Analyzing Inversions")
    except ImportError:
        logger.info("tqdm not found, progress bar disabled.")
        iterator = geno_df.iterrows()


    for index, inversion in iterator:
        chrom = inversion['chrom_normalized']
        inv_id = f"{chrom}:{inversion['start']}-{inversion['end']}" # Unique ID for the inversion row
        # Use original ID if available and seems unique, otherwise use coordinates
        if 'orig_ID' in inversion and pd.notna(inversion['orig_ID']):
             inv_id = inversion['orig_ID']


        result_row = {
            'InversionID': inv_id,
            'Chromosome': chrom,
            'Start': inversion['start'],
            'End': inversion['end'],
            'DirectCount': 0,
            'InvertedCount': 0,
            'ModelFitStatus': 'Not Run',
            'LLR_PValue': np.nan,
            'NumPCsUsed': n_pcs,
            'ReasonSkipped': None
        }
        missing_hap_pca_warnings[inv_id] = set() # Track missing haplotypes for this inversion


        # Check if PCA data exists for this chromosome
        if chrom not in pca_data:
            result_row['ReasonSkipped'] = f"No PCA data for chromosome {chrom}"
            results.append(result_row)
            continue

        chrom_pca_data = pca_data[chrom]
        haplotype_pc_list = [] # Collect PC data for haplotypes of this inversion

        # --- Collect PC data for each valid haplotype ---
        for sample_id in sample_id_cols:
            genotype_str = str(inversion.get(sample_id, '')).strip()

            # Filter invalid genotypes
            if genotype_str not in VALID_GENOTYPES:
                continue # Skip this sample for this inversion

            try:
                state_L = int(genotype_str[0]) # 0 or 1
                state_R = int(genotype_str[2]) # 0 or 1
            except (IndexError, ValueError):
                 logger.warning(f"Could not parse valid genotype '{genotype_str}' for sample {sample_id}, inv {inv_id}. Skipping.")
                 continue # Should not happen if VALID_GENOTYPES check passed

            # Construct harmonized haplotype IDs
            hap_id_L = f"{sample_id}_L"
            hap_id_R = f"{sample_id}_R"

            # Process Left Haplotype
            if hap_id_L in chrom_pca_data:
                pcs = chrom_pca_data[hap_id_L]
                if len(pcs) == n_pcs: # Ensure correct number of PCs loaded
                    hap_info = {'HaplotypeState': state_L}
                    hap_info.update({pc_col_names[i]: pcs[i] for i in range(n_pcs)})
                    haplotype_pc_list.append(hap_info)
                else:
                    missing_hap_pca_warnings[inv_id].add(f"{hap_id_L} (Expected {n_pcs} PCs, got {len(pcs)})")
            else:
                missing_hap_pca_warnings[inv_id].add(hap_id_L)

            # Process Right Haplotype
            if hap_id_R in chrom_pca_data:
                pcs = chrom_pca_data[hap_id_R]
                if len(pcs) == n_pcs:
                    hap_info = {'HaplotypeState': state_R}
                    hap_info.update({pc_col_names[i]: pcs[i] for i in range(n_pcs)})
                    haplotype_pc_list.append(hap_info)
                else:
                     missing_hap_pca_warnings[inv_id].add(f"{hap_id_R} (Expected {n_pcs} PCs, got {len(pcs)})")
            else:
                 missing_hap_pca_warnings[inv_id].add(hap_id_R)

        # --- Check counts and run analysis ---
        if not haplotype_pc_list:
            result_row['ReasonSkipped'] = "No valid haplotypes with PCA data found"
            results.append(result_row)
            continue

        inv_hap_df = pd.DataFrame(haplotype_pc_list)
        direct_count = (inv_hap_df['HaplotypeState'] == 0).sum()
        inverted_count = (inv_hap_df['HaplotypeState'] == 1).sum()

        result_row['DirectCount'] = direct_count
        result_row['InvertedCount'] = inverted_count

        if direct_count < min_haps_per_group or inverted_count < min_haps_per_group:
            result_row['ReasonSkipped'] = f"Insufficient haplotypes (Dir={direct_count}, Inv={inverted_count}, Min={min_haps_per_group})"
            results.append(result_row)
            continue

        # --- Fit Logistic Regression Model ---
        try:
            # Check for perfect separation before fitting
            # A simple check: are all direct haplotypes on one side of a PC and all inverted on the other?
            # This isn't exhaustive but catches simple cases.
            perfect_separation_flag = False
            for pc in pc_col_names:
                if (inv_hap_df.loc[inv_hap_df['HaplotypeState']==0, pc].max() < inv_hap_df.loc[inv_hap_df['HaplotypeState']==1, pc].min()) or \
                   (inv_hap_df.loc[inv_hap_df['HaplotypeState']==1, pc].max() < inv_hap_df.loc[inv_hap_df['HaplotypeState']==0, pc].min()):
                    perfect_separation_flag = True
                    break

            if perfect_separation_flag:
                 raise PerfectSeparationError("Potential perfect separation detected based on at least one PC.")


            formula = f"HaplotypeState ~ {' + '.join(pc_col_names)}"
            # Add intercept explicitly for clarity, though it's default
            inv_hap_df['Intercept'] = 1.0
            # Use Logit; endog = dependent var (HaplotypeState), exog = independent vars (PCs + Intercept)
            logit_model = sm.Logit(inv_hap_df['HaplotypeState'], inv_hap_df[['Intercept'] + pc_col_names])
            # Increase maxiter if convergence issues arise
            logit_result = logit_model.fit(disp=0, maxiter=100) # disp=0 hides optimization output

            result_row['ModelFitStatus'] = 'Success'
            result_row['LLR_PValue'] = logit_result.llr_pvalue # Likelihood ratio test p-value

        except PerfectSeparationError as pse:
             result_row['ModelFitStatus'] = 'Failed (Perfect Separation)'
             result_row['ReasonSkipped'] = f"Perfect separation detected - groups likely completely separable by PCs. {pse}"
             logger.warning(f"Perfect separation detected for inversion {inv_id}. Cannot fit model reliably.")
        except np.linalg.LinAlgError:
             result_row['ModelFitStatus'] = 'Failed (Singular Matrix)'
             result_row['ReasonSkipped'] = "Singular matrix - PCs might be highly collinear for this subset."
             logger.warning(f"Singular matrix error for inversion {inv_id}.")
        except Exception as e:
            result_row['ModelFitStatus'] = 'Failed (Other Error)'
            result_row['ReasonSkipped'] = f"Logistic regression error: {str(e)[:100]}"
            logger.error(f"Error fitting model for inversion {inv_id}: {e}", exc_info=False) # Set exc_info=True for full traceback if needed

        results.append(result_row)

    # Log missing haplotype warnings aggregated
    total_missing_warnings = 0
    for inv_id, missing_haps in missing_hap_pca_warnings.items():
         if missing_haps:
              count = len(missing_haps)
              total_missing_warnings += count
              # Log only once per inversion ID if many are missing
              logger.warning(f"For inversion '{inv_id}', PCA data was missing for {count} unique haplotype IDs (e.g., {next(iter(missing_haps))}).")
    if total_missing_warnings > 0:
         logger.warning(f"Total missing PCA data warnings across all inversions: {total_missing_warnings}")


    return pd.DataFrame(results)


# =====================================================================
# Main Execution Block
# =====================================================================

if __name__ == "__main__":
    main_start_time = time.time()
    logger.info("--- Starting Inversion Stratification Analysis ---")

    # --- 1. Load PCA Data ---
    pca_data = load_pca_data(PCA_FOLDER, N_PCS)
    if pca_data is None:
        logger.error("Failed to load PCA data. Exiting.")
        sys.exit(1)

    # --- 2. Load Genotype Data ---
    loaded_geno = load_genotype_data(GENOTYPE_FILE, FIRST_SAMPLE_COL_INDEX)
    if loaded_geno is None:
        logger.error("Failed to load genotype data. Exiting.")
        sys.exit(1)
    geno_df, sample_id_cols = loaded_geno

    # --- 3. Run Analysis ---
    logger.info("Starting analysis loop for each inversion...")
    results_df = run_stratification_analysis(
        geno_df, sample_id_cols, pca_data, N_PCS, MIN_HAPS_PER_GROUP
    )

    # --- 4. Multiple Testing Correction ---
    logger.info("Applying multiple testing correction (Benjamini-Hochberg FDR)...")
    valid_pvals = results_df['LLR_PValue'].dropna()
    if not valid_pvals.empty:
        reject, pvals_corrected, _, _ = multipletests(valid_pvals, method='fdr_bh', alpha=0.05)
        # Add corrected p-values back to the DataFrame, aligning by index
        results_df['LLR_PValue_Corrected'] = np.nan
        results_df.loc[valid_pvals.index, 'LLR_PValue_Corrected'] = pvals_corrected
    else:
        logger.warning("No valid p-values found to correct.")
        results_df['LLR_PValue_Corrected'] = np.nan

    # --- 5. Save Results ---
    logger.info(f"Saving results to '{OUTPUT_RESULTS_CSV}'...")
    try:
        # Sort results for easier interpretation (e.g., by corrected p-value)
        results_df_sorted = results_df.sort_values(by=['LLR_PValue_Corrected', 'LLR_PValue'], na_position='last')
        results_df_sorted.to_csv(OUTPUT_RESULTS_CSV, index=False, float_format='%.6g')
        logger.info("Results saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)

    # --- 6. Summarize Output ---
    num_processed = len(results_df)
    num_tested = results_df['ModelFitStatus'].eq('Success').sum()
    num_skipped_data = results_df['ReasonSkipped'].str.contains('Insufficient|No valid', na=False).sum()
    num_skipped_pca = results_df['ReasonSkipped'].str.contains('No PCA', na=False).sum()
    num_failed_model = results_df['ModelFitStatus'].str.contains('Failed', na=False).sum()
    num_significant_raw = (results_df['LLR_PValue'] < 0.05).sum()
    num_significant_corrected = (results_df['LLR_PValue_Corrected'] < 0.05).sum()

    logger.info("\n--- Analysis Summary ---")
    logger.info(f"Inversions Processed: {num_processed}")
    logger.info(f"Inversions Tested (Sufficient Data & PCA): {num_tested}")
    logger.info(f"  - Skipped (Insufficient Haplotypes): {num_skipped_data}")
    logger.info(f"  - Skipped (Missing Chrom PCA): {num_skipped_pca}")
    logger.info(f"  - Failed (Model Fitting Error): {num_failed_model}")
    logger.info(f"Significant Stratification (Raw p < 0.05): {num_significant_raw}")
    logger.info(f"Significant Stratification (FDR corrected p < 0.05): {num_significant_corrected}")

    if num_significant_corrected > 0:
         logger.info("\n--- Inversions with Significant Stratification (FDR < 0.05) ---")
         sig_results = results_df[results_df['LLR_PValue_Corrected'] < 0.05].sort_values('LLR_PValue_Corrected')
         # Print relevant columns
         cols_to_print = ['InversionID', 'Chromosome', 'DirectCount', 'InvertedCount', 'LLR_PValue', 'LLR_PValue_Corrected']
         logger.info(sig_results[cols_to_print].to_string(index=False, float_format="%.4g"))

    # --- 6a. Detailed Skipped Inversion Report ---
    skipped_df = results_df[results_df['ReasonSkipped'].notna()].copy()
    if not skipped_df.empty:
        logger.info("\n--- Details for Skipped or Failed Inversions ---")
        # Add counts to the report for context when skipping due to insufficient data
        skipped_df['Details'] = skipped_df.apply(
            lambda row: f"{row['ReasonSkipped']} (Dir={row['DirectCount']}, Inv={row['InvertedCount']})"
            if 'Insufficient' in str(row['ReasonSkipped']) else row['ReasonSkipped'],
            axis=1
        )
        # Sort for better readability
        skipped_df_sorted = skipped_df.sort_values(by=['Chromosome', 'Details'])
        logger.info(f"{'InversionID':<40} {'Chromosome':<10} {'Reason Skipped / Details'}")
        logger.info("-" * 100)
        for index, row in skipped_df_sorted.iterrows():
            # So long IDs or reasons don't break formatting too much
            inv_id_str = str(row['InversionID'])[:40]
            reason_str = str(row['Details']) # Use the combined details string
            logger.info(f"{inv_id_str:<40} {row['Chromosome']:<10} {reason_str}")

    main_end_time = time.time()
    logger.info(f"--- Analysis finished in {main_end_time - main_start_time:.2f} seconds ---")
