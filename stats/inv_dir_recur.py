import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import glob
import re
import logging
import sys
import time
from typing import Dict, List, Tuple, Optional

# =====================================================================
# Configuration
# =====================================================================
# --- Input File Paths ---
# Contains pi values per orientation (e.g., 0_pi_filtered, 1_pi_filtered)
OUTPUT_PI_PATH = 'output.csv'
# Contains recurrence info per region (e.g., 0_single_1_recur)
INV_INFO_PATH = 'inv_info.csv'
# Contains phased genotypes (e.g., 0|1) for inversions per sample
GENOTYPE_FILE = 'variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv'
# Folder containing PCA results per chromosome
PCA_FOLDER = "pca"

# --- Parameters ---
N_PCS: int = 5  # Number of Principal Components to use
# Index of the first column containing sample genotypes in GENOTYPE_FILE
FIRST_SAMPLE_COL_INDEX: int = 8
COORDINATE_TOLERANCE: int = 1 # Allowable difference in start/end coordinates for matching
# Minimum number of haplotypes required per group (Direct/Inverted for an inversion)
# to calculate standard deviation reliably and potentially for model stability.
# Set to 2 for std calculation, may need > N_PCS for model.
MIN_HAPS_FOR_STD: int = 2

# --- Output ---
OUTPUT_DIR = 'analysis_results_with_pcs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
MERGED_WIDE_DATA_PATH = os.path.join(OUTPUT_DIR, 'merged_pi_recur_geno_data_wide.csv')
HAPLOTYPE_DETAILS_PATH = os.path.join(OUTPUT_DIR, 'haplotype_pc_details.csv')
AGGREGATED_PCS_PATH = os.path.join(OUTPUT_DIR, 'aggregated_pc_stats.csv')
FINAL_LONG_DATA_PATH = os.path.join(OUTPUT_DIR, 'final_modeling_data_long.csv')
MODEL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'lmm_with_pcs_summary.txt')
BOXPLOT_PATH = os.path.join(OUTPUT_DIR, 'pi_boxplot_grouped.png')
INTERACTION_PLOT_PATH = os.path.join(OUTPUT_DIR, 'pi_interaction_plot.png')

# =====================================================================
# Logging Setup
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'analysis_log.txt'))
    ]
)
logger = logging.getLogger('lmm_analysis')

# Suppress specific warnings if needed (e.g., from statsmodels)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================================
# Helper Functions (Adapted from provided examples)
# =====================================================================

def normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome name to 'chrN' or 'chrX'/'chrY' format."""
    chrom = str(chrom).strip().lower()
    if chrom.startswith('chr'):
        return chrom # Already has prefix
    else:
        return f"chr{chrom}" # Add prefix

def harmonize_pca_haplotype_name(pca_hap_name: str) -> Optional[str]:
    """
    Converts PCA haplotype name (e.g., 'EUR_GBR_HG00096_L')
    to harmonized format (e.g., 'HG00096_L').
    ASSUMPTION: Sample ID is the second-to-last part before _L/_R suffix.
    """
    parts = str(pca_hap_name).split('_')
    if len(parts) < 2:
        return None
    sample_id = parts[-2]
    hap_suffix = parts[-1]
    if hap_suffix not in ('L', 'R'):
        return None
    return f"{sample_id}_{hap_suffix}"

def load_pca_data(pca_folder: str, n_pcs: int) -> Optional[Dict[str, Dict[str, List[float]]]]:
    """Loads PCA data for all chromosomes, harmonizes haplotype names."""
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
    unharmonized_count = 0

    for pca_file in pca_files:
        try:
            base_name = os.path.basename(pca_file)
            match = re.search(r'pca_chr_([\w]+)\.tsv', base_name, re.IGNORECASE) # Handle chrX etc.
            if not match:
                logger.warning(f"Could not determine chromosome from PCA filename: {base_name}")
                continue
            chrom_num = match.group(1)
            chrom = normalize_chromosome(chrom_num)

            df = pd.read_csv(pca_file, sep='\t')

            if 'Haplotype' not in df.columns:
                 logger.warning(f"Column 'Haplotype' not found in {pca_file}. Skipping.")
                 continue
            missing_pcs = [pc for pc in pc_cols_to_load if pc not in df.columns]
            if missing_pcs:
                logger.error(f"Missing required PC columns in {pca_file}: {missing_pcs}. Cannot use this file.")
                continue

            pca_data[chrom] = {}
            file_hap_count = 0
            for _, row in df.iterrows():
                pca_hap_name = row['Haplotype']
                harmonized_hap_id = harmonize_pca_haplotype_name(pca_hap_name)

                if harmonized_hap_id:
                    try:
                        pc_values = [float(row[pc]) for pc in pc_cols_to_load]
                        # Check for NaNs in PC values
                        if any(np.isnan(pc_values)):
                            # logger.debug(f"NaN PC value found for haplotype {pca_hap_name} in {pca_file}. Skipping haplotype.")
                            continue
                        pca_data[chrom][harmonized_hap_id] = pc_values
                        file_hap_count += 1
                    except (ValueError, TypeError):
                        # logger.debug(f"Could not convert PC values for {pca_hap_name} in {pca_file}. Skipping.")
                        continue # Skip if PC values aren't numeric
                else:
                    unharmonized_count +=1

            logger.info(f"  Loaded {file_hap_count} valid haplotypes for {chrom} from {base_name}")
            total_haplotypes_loaded += file_hap_count

        except Exception as e:
            logger.error(f"Failed to process PCA file {pca_file}: {e}", exc_info=True)

    if not pca_data:
        logger.error("No PCA data was successfully loaded.")
        return None
    if unharmonized_count > 0:
        logger.warning(f"Could not harmonize or parse format for {unharmonized_count} PCA haplotype entries.")

    logger.info(f"Finished loading PCA data for {len(pca_data)} chromosomes, {total_haplotypes_loaded} total valid haplotype entries.")
    return pca_data

# =====================================================================
# Main Script Logic
# =====================================================================
logger.info("--- Starting LMM Analysis with PC Control ---")
main_start_time = time.time()

# --- Step 1: Load All Input Data ---
logger.info("--- Step 1: Loading Input Data ---")
try:
    output_df = pd.read_csv(OUTPUT_PI_PATH)
    logger.info(f"Loaded {len(output_df)} rows from Pi data: {OUTPUT_PI_PATH}")

    inv_info_df = pd.read_csv(INV_INFO_PATH)
    logger.info(f"Loaded {len(inv_info_df)} rows from Inversion Info: {INV_INFO_PATH}")

    geno_df = pd.read_csv(GENOTYPE_FILE, sep='\t')
    logger.info(f"Loaded {len(geno_df)} inversions from Genotype file: {GENOTYPE_FILE}")

    # Load PCA data using helper function
    pca_data = load_pca_data(PCA_FOLDER, N_PCS)
    if pca_data is None:
        raise FileNotFoundError("PCA data loading failed.")

except FileNotFoundError as e:
    logger.error(f"ERROR: Cannot find input file: {e}. Exiting.")
    sys.exit(1)
except Exception as e:
    logger.error(f"ERROR: Failed during initial data loading: {e}", exc_info=True)
    sys.exit(1)

# --- Step 2: Prepare and Standardize Data ---
logger.info("--- Step 2: Preparing and Standardizing Data ---")

# A. Standardize Pi Data ('output.csv')
pi_value_cols = []
coord_cols_pi = []
chr_col_pi = None
# Find required columns dynamically (assuming structure from first script)
if 'chr' in output_df.columns: chr_col_pi = 'chr'
elif 'Chr' in output_df.columns: chr_col_pi = 'Chr' # Add variations if needed
else: raise ValueError(f"Chromosome column not found in {OUTPUT_PI_PATH}")

if 'region_start' in output_df.columns: coord_cols_pi.append('region_start')
elif 'Start' in output_df.columns: coord_cols_pi.append('Start')
else: raise ValueError(f"Start coordinate column not found in {OUTPUT_PI_PATH}")

if 'region_end' in output_df.columns: coord_cols_pi.append('region_end')
elif 'End' in output_df.columns: coord_cols_pi.append('End')
else: raise ValueError(f"End coordinate column not found in {OUTPUT_PI_PATH}")

if '0_pi_filtered' in output_df.columns: pi_value_cols.append('0_pi_filtered')
else: raise ValueError(f"Direct Pi column ('0_pi_filtered') not found in {OUTPUT_PI_PATH}")
if '1_pi_filtered' in output_df.columns: pi_value_cols.append('1_pi_filtered')
else: raise ValueError(f"Inverted Pi column ('1_pi_filtered') not found in {OUTPUT_PI_PATH}")

# Rename columns to standard internal names
pi_col_rename = {
    chr_col_pi: 'chr',
    coord_cols_pi[0]: 'region_start',
    coord_cols_pi[1]: 'region_end',
    pi_value_cols[0]: 'pi_direct',
    pi_value_cols[1]: 'pi_inverted'
}
output_df = output_df.rename(columns=pi_col_rename)
output_df['chr'] = output_df['chr'].apply(normalize_chromosome)
output_df['region_start'] = pd.to_numeric(output_df['region_start'], errors='coerce').astype('Int64')
output_df['region_end'] = pd.to_numeric(output_df['region_end'], errors='coerce').astype('Int64')
output_df.dropna(subset=['chr', 'region_start', 'region_end'], inplace=True)


# B. Standardize Inversion Info Data ('inv_info.csv')
recur_col = None
coord_cols_inv = []
chr_col_inv = None
# Find columns dynamically
if 'Chromosome' in inv_info_df.columns: chr_col_inv = 'Chromosome'
elif 'chr' in inv_info_df.columns: chr_col_inv = 'chr'
else: raise ValueError(f"Chromosome column not found in {INV_INFO_PATH}")

if 'Start' in inv_info_df.columns: coord_cols_inv.append('Start')
elif 'region_start' in inv_info_df.columns: coord_cols_inv.append('region_start')
else: raise ValueError(f"Start coordinate column not found in {INV_INFO_PATH}")

if 'End' in inv_info_df.columns: coord_cols_inv.append('End')
elif 'region_end' in inv_info_df.columns: coord_cols_inv.append('region_end')
else: raise ValueError(f"End coordinate column not found in {INV_INFO_PATH}")

if '0_single_1_recur' in inv_info_df.columns: recur_col = '0_single_1_recur'
elif 'RecurrenceCode' in inv_info_df.columns: recur_col = 'RecurrenceCode'
else: raise ValueError(f"Recurrence column not found in {INV_INFO_PATH}")

inv_info_rename = {
    chr_col_inv: 'chr',
    coord_cols_inv[0]: 'region_start',
    coord_cols_inv[1]: 'region_end',
    recur_col: 'RecurrenceCode'
}
inv_info_df = inv_info_df.rename(columns=inv_info_rename)
inv_info_df['chr'] = inv_info_df['chr'].apply(normalize_chromosome)
inv_info_df['region_start'] = pd.to_numeric(inv_info_df['region_start'], errors='coerce').astype('Int64')
inv_info_df['region_end'] = pd.to_numeric(inv_info_df['region_end'], errors='coerce').astype('Int64')
inv_info_df.dropna(subset=['chr', 'region_start', 'region_end', 'RecurrenceCode'], inplace=True)
inv_info_df['RecurrenceCode'] = inv_info_df['RecurrenceCode'].astype(int)

# C. Standardize Genotype Data
if 'seqnames' not in geno_df.columns: raise ValueError("'seqnames' column missing from genotype file.")
if 'start' not in geno_df.columns: raise ValueError("'start' column missing from genotype file.")
if 'end' not in geno_df.columns: raise ValueError("'end' column missing from genotype file.")

geno_df = geno_df.rename(columns={'seqnames': 'chr_geno', 'start': 'start_geno', 'end': 'end_geno'})
geno_df['chr_geno'] = geno_df['chr_geno'].apply(normalize_chromosome)
geno_df['start_geno'] = pd.to_numeric(geno_df['start_geno'], errors='coerce').astype('Int64')
geno_df['end_geno'] = pd.to_numeric(geno_df['end_geno'], errors='coerce').astype('Int64')
geno_df.dropna(subset=['chr_geno', 'start_geno', 'end_geno'], inplace=True)
# Create a unique ID for each inversion row in the genotype file
geno_df['InversionRegionID_geno'] = geno_df['chr_geno'] + ':' + \
                                    geno_df['start_geno'].astype(str) + '-' + \
                                    geno_df['end_geno'].astype(str)
# Check for duplicate IDs - implies same region listed multiple times
if geno_df['InversionRegionID_geno'].duplicated().any():
    logger.warning(f"Duplicate inversion region IDs found in genotype file. Check for redundant entries.")
    # Optional: Keep only first occurrence if needed, but investigation is better
    # geno_df = geno_df.drop_duplicates(subset='InversionRegionID_geno', keep='first')

# Identify sample columns
if FIRST_SAMPLE_COL_INDEX >= len(geno_df.columns):
     raise ValueError(f"FIRST_SAMPLE_COL_INDEX ({FIRST_SAMPLE_COL_INDEX}) is out of bounds.")
sample_id_cols = geno_df.columns[FIRST_SAMPLE_COL_INDEX:].tolist()
if not sample_id_cols:
     raise ValueError("Could not identify any sample ID columns in genotype file.")
logger.info(f"Identified {len(sample_id_cols)} sample columns in genotype file (starting from '{sample_id_cols[0]}').")

VALID_GENOTYPES = {'0|0', '0|1', '1|0', '1|1'}
logger.info("Data standardization complete.")


# --- Step 3: Extract Haplotype-Level PC Data ---
logger.info("--- Step 3: Extracting Haplotype PC Data ---")
haplotype_pc_records = []
missing_pca_hap_count = 0
pc_col_names = [f"PC{i+1}" for i in range(N_PCS)]

for index, inversion_row in geno_df.iterrows():
    chrom = inversion_row['chr_geno']
    inv_id_geno = inversion_row['InversionRegionID_geno']

    if chrom not in pca_data:
        # logger.debug(f"No PCA data found for chromosome {chrom}, skipping inversion {inv_id_geno}")
        continue # Skip if no PCA data for the whole chromosome

    chrom_pca_data = pca_data[chrom]

    for sample_id in sample_id_cols:
        genotype_str = str(inversion_row.get(sample_id, '')).strip()

        if genotype_str not in VALID_GENOTYPES:
            continue # Skip invalid or missing genotypes

        try:
            state_L = int(genotype_str[0]) # 0 or 1
            state_R = int(genotype_str[2]) # 0 or 1
        except (IndexError, ValueError):
             logger.warning(f"Could not parse genotype '{genotype_str}' for sample {sample_id}, inv {inv_id_geno}. Skipping.")
             continue

        hap_id_L = f"{sample_id}_L"
        hap_id_R = f"{sample_id}_R"

        # Process Left Haplotype
        if hap_id_L in chrom_pca_data:
            pcs = chrom_pca_data[hap_id_L] # PC values are already checked for NaN during loading
            hap_info = {'InversionRegionID_geno': inv_id_geno, 'HaplotypeState': state_L}
            hap_info.update({pc_col_names[i]: pcs[i] for i in range(N_PCS)})
            haplotype_pc_records.append(hap_info)
        else:
            missing_pca_hap_count += 1

        # Process Right Haplotype
        if hap_id_R in chrom_pca_data:
            pcs = chrom_pca_data[hap_id_R]
            hap_info = {'InversionRegionID_geno': inv_id_geno, 'HaplotypeState': state_R}
            hap_info.update({pc_col_names[i]: pcs[i] for i in range(N_PCS)})
            haplotype_pc_records.append(hap_info)
        else:
            missing_pca_hap_count += 1

if not haplotype_pc_records:
     logger.error("No valid haplotypes with corresponding PCA data were found across all inversions. Cannot proceed.")
     sys.exit(1)

haplotype_details_df = pd.DataFrame(haplotype_pc_records)
logger.info(f"Extracted PC data for {len(haplotype_details_df)} haplotypes.")
if missing_pca_hap_count > 0:
    logger.warning(f"PCA data was not found for {missing_pca_hap_count} haplotype instances (sample-inversion pairs).")
haplotype_details_df.to_csv(HAPLOTYPE_DETAILS_PATH, index=False)
logger.info(f"Haplotype PC details saved to {HAPLOTYPE_DETAILS_PATH}")

# --- Step 4: Calculate Average and Std Dev of PCs per Group ---
logger.info("--- Step 4: Aggregating PC Statistics per Group ---")

# Define aggregation functions: mean and std
agg_funcs = {pc: ['mean', 'std'] for pc in pc_col_names}

# Group by inversion region (from genotype file) and haplotype state (0 or 1)
grouped_pcs = haplotype_details_df.groupby(['InversionRegionID_geno', 'HaplotypeState'])

# Check group sizes before calculating std dev
group_sizes = grouped_pcs.size()
groups_too_small_for_std = group_sizes[group_sizes < MIN_HAPS_FOR_STD].index
if not groups_too_small_for_std.empty:
    logger.warning(f"{len(groups_too_small_for_std)} groups have fewer than {MIN_HAPS_FOR_STD} haplotypes. "
                   f"Standard deviation will be NaN/0 for these groups.")

# Apply aggregation
agg_pc_df = grouped_pcs.agg(agg_funcs)

# Flatten MultiIndex columns (e.g., ('PC1', 'mean') -> 'PC1_mean')
agg_pc_df.columns = ['_'.join(col).strip() for col in agg_pc_df.columns.values]

# Rename columns for clarity (AvgPCn, StdPCn)
rename_dict = {}
for i in range(N_PCS):
    pc = f"PC{i+1}"
    rename_dict[f"{pc}_mean"] = f"AvgPC{i+1}"
    rename_dict[f"{pc}_std"] = f"StdPC{i+1}"
agg_pc_df = agg_pc_df.rename(columns=rename_dict)

# Reset index to make InversionRegionID_geno and HaplotypeState columns
agg_pc_df = agg_pc_df.reset_index()

# Handle NaN standard deviations for groups with < MIN_HAPS_FOR_STD (typically size 1)
# Fill NaN std devs with 0, as variance/std dev is 0 for a single point.
std_cols = [f"StdPC{i+1}" for i in range(N_PCS)]
agg_pc_df[std_cols] = agg_pc_df[std_cols].fillna(0)

logger.info(f"Calculated mean and standard deviation for PCs for {len(agg_pc_df)} groups.")
agg_pc_df.to_csv(AGGREGATED_PCS_PATH, index=False)
logger.info(f"Aggregated PC stats saved to {AGGREGATED_PCS_PATH}")


# --- Step 5: Merge Pi, Recurrence, and Link to Genotype Regions ---
logger.info("--- Step 5: Merging Pi, Recurrence, and Linking to Genotype Regions ---")

# A. Merge Pi and Recurrence data based on coordinates (+/- tolerance)
merged_pi_recur_df = pd.merge(
    output_df.add_suffix('_pi'),
    inv_info_df.add_suffix('_inv'),
    left_on='chr_pi',
    right_on='chr_inv',
    how='inner' # Keep only regions present in both
)

# Apply coordinate tolerance filter
coord_match_mask = (
    (abs(merged_pi_recur_df['region_start_pi'] - merged_pi_recur_df['region_start_inv']) <= COORDINATE_TOLERANCE) &
    (abs(merged_pi_recur_df['region_end_pi'] - merged_pi_recur_df['region_end_inv']) <= COORDINATE_TOLERANCE)
)
merged_pi_recur_df = merged_pi_recur_df[coord_match_mask].copy()

# Check for ambiguous matches (one pi/recur region matching multiple based on tolerance)
# Use the Pi region as the primary key for checking ambiguity
merged_pi_recur_df['pi_coords'] = merged_pi_recur_df['chr_pi'] + ':' + merged_pi_recur_df['region_start_pi'].astype(str) + '-' + merged_pi_recur_df['region_end_pi'].astype(str)
merged_pi_recur_df['match_count'] = merged_pi_recur_df.groupby('pi_coords')['pi_coords'].transform('count')

ambiguous_matches = merged_pi_recur_df[merged_pi_recur_df['match_count'] > 1]
if not ambiguous_matches.empty:
    logger.warning(f"Found {ambiguous_matches['pi_coords'].nunique()} Pi/Recurrence regions ambiguously matching based on coordinate tolerance. Removing them.")
    merged_pi_recur_df = merged_pi_recur_df[merged_pi_recur_df['match_count'] == 1].copy()

# Select and rename columns for clarity
merged_pi_recur_df = merged_pi_recur_df[[
    'chr_pi', 'region_start_pi', 'region_end_pi', 'pi_direct_pi', 'pi_inverted_pi', 'RecurrenceCode_inv'
]].rename(columns={
    'chr_pi': 'chr',
    'region_start_pi': 'region_start',
    'region_end_pi': 'region_end',
    'pi_direct_pi': 'pi_direct',
    'pi_inverted_pi': 'pi_inverted',
    'RecurrenceCode_inv': 'RecurrenceCode'
})
logger.info(f"Merged Pi and Recurrence data: {len(merged_pi_recur_df)} unique regions found.")

# B. Link Merged Pi/Recurrence data to Genotype File Regions
# We need the InversionRegionID_geno associated with the Pi/Recurrence data
# Merge based on coordinates (+/- tolerance) again, this time with geno_df
# Select only coordinate columns and the ID from geno_df to avoid large merge
geno_coords_df = geno_df[['chr_geno', 'start_geno', 'end_geno', 'InversionRegionID_geno']].drop_duplicates()

combined_data_wide = pd.merge(
    merged_pi_recur_df,
    geno_coords_df,
    left_on='chr',
    right_on='chr_geno',
    how='inner' # Keep only regions present in both Pi/Recur and Genotype file
)

# Apply coordinate tolerance filter between Pi/Recur and Genotype regions
coord_match_mask_final = (
    (abs(combined_data_wide['region_start'] - combined_data_wide['start_geno']) <= COORDINATE_TOLERANCE) &
    (abs(combined_data_wide['region_end'] - combined_data_wide['end_geno']) <= COORDINATE_TOLERANCE)
)
combined_data_wide = combined_data_wide[coord_match_mask_final].copy()

# Check for ambiguous matches again (one Pi/Recur region matching multiple Genotype regions)
combined_data_wide['pi_coords_again'] = combined_data_wide['chr'] + ':' + combined_data_wide['region_start'].astype(str) + '-' + combined_data_wide['region_end'].astype(str)
combined_data_wide['match_count_final'] = combined_data_wide.groupby('pi_coords_again')['pi_coords_again'].transform('count')

ambiguous_matches_final = combined_data_wide[combined_data_wide['match_count_final'] > 1]
if not ambiguous_matches_final.empty:
    logger.warning(f"Found {ambiguous_matches_final['pi_coords_again'].nunique()} Pi/Recurrence regions ambiguously matching Genotype file regions. Removing them.")
    combined_data_wide = combined_data_wide[combined_data_wide['match_count_final'] == 1].copy()

# Select final columns needed before reshaping
combined_data_wide = combined_data_wide[[
    'InversionRegionID_geno', 'pi_direct', 'pi_inverted', 'RecurrenceCode'
]].drop_duplicates(subset=['InversionRegionID_geno']) # Ensure one row per geno region ID

logger.info(f"Successfully linked Pi/Recurrence data to {len(combined_data_wide)} unique Genotype file regions.")
combined_data_wide.to_csv(MERGED_WIDE_DATA_PATH, index=False)


# --- Step 6: Reshape to Long Format and Merge Aggregated PCs ---
logger.info("--- Step 6: Reshaping Data and Merging PC Statistics ---")

# Melt the Pi data to long format
direct_df = combined_data_wide[['InversionRegionID_geno', 'RecurrenceCode', 'pi_direct']].copy()
direct_df['Orientation'] = 'Direct'
direct_df = direct_df.rename(columns={'pi_direct': 'PiValue'})
direct_df['HaplotypeState'] = 0 # Add state for merging with PCs

inverted_df = combined_data_wide[['InversionRegionID_geno', 'RecurrenceCode', 'pi_inverted']].copy()
inverted_df['Orientation'] = 'Inverted'
inverted_df = inverted_df.rename(columns={'pi_inverted': 'PiValue'})
inverted_df['HaplotypeState'] = 1 # Add state for merging with PCs

data_long = pd.concat([direct_df, inverted_df], ignore_index=True)

# Merge aggregated PC stats (mean and std dev) into the long dataframe
data_long = pd.merge(
    data_long,
    agg_pc_df,
    on=['InversionRegionID_geno', 'HaplotypeState'],
    how='left' # Keep all Pi measurements, even if PC data was missing for that group
)

# Drop the temporary HaplotypeState column
data_long = data_long.drop(columns=['HaplotypeState'])

# Create final Recurrence categorical column
recurrence_map = {0: 'Single-event', 1: 'Recurrent'}
data_long['Recurrence'] = data_long['RecurrenceCode'].map(recurrence_map)
data_long = data_long.drop(columns=['RecurrenceCode'])


# --- Step 7: Final Data Cleaning for Modeling ---
logger.info("--- Step 7: Final Data Cleaning for Modeling ---")

# A. Handle missing Pi Values
initial_rows = len(data_long)
data_long['PiValue'] = data_long['PiValue'].replace([np.inf, -np.inf], np.nan)
data_long.dropna(subset=['PiValue'], inplace=True)
rows_removed_pi = initial_rows - len(data_long)
if rows_removed_pi > 0:
    logger.warning(f"Removed {rows_removed_pi} rows with missing or infinite PiValue.")

# B. Handle missing PC Stats (Avg or Std)
# These would be missing if a group had NO haplotypes with PCA data found in Step 3/4
pc_stat_cols = [f"AvgPC{i+1}" for i in range(N_PCS)] + [f"StdPC{i+1}" for i in range(N_PCS)]
initial_rows = len(data_long)
data_long.dropna(subset=pc_stat_cols, inplace=True)
rows_removed_pcs = initial_rows - len(data_long)
if rows_removed_pcs > 0:
    logger.warning(f"Removed {rows_removed_pcs} rows missing aggregated PC statistics (likely due to missing haplotype PCA data).")

# C. Check for missing Recurrence/Orientation (should not happen if logic is correct)
if data_long['Recurrence'].isnull().any() or data_long['Orientation'].isnull().any():
     logger.error("Found unexpected missing values in Recurrence or Orientation columns after processing.")
     # Handle or exit if needed

# D. Convert to Categorical for model
data_long['Orientation'] = pd.Categorical(data_long['Orientation'], categories=['Direct', 'Inverted'], ordered=False)
data_long['Recurrence'] = pd.Categorical(data_long['Recurrence'], categories=['Single-event', 'Recurrent'], ordered=False)

# Log final dataset size
final_obs = len(data_long)
final_regions = data_long['InversionRegionID_geno'].nunique()
logger.info(f"Final dataset for modeling contains {final_obs} observations across {final_regions} unique inversion regions.")

if final_obs == 0:
    logger.error("No data remaining after cleaning. Cannot fit model. Check intermediate files and logs.")
    sys.exit(1)

data_long.to_csv(FINAL_LONG_DATA_PATH, index=False)
logger.info(f"Final long-format data for modeling saved to {FINAL_LONG_DATA_PATH}")


# --- Step 8: Fit Linear Mixed-Effects Model (LMM) ---
logger.info("--- Step 8: Fitting Linear Mixed-Effects Model ---")

# Check minimum group sizes for model stability
group_counts = data_long.groupby(['Orientation', 'Recurrence']).size()
logger.info("Data counts per Orientation/Recurrence group:")
logger.info(group_counts)
if (group_counts < N_PCS + 2).any(): # Rule of thumb: need more data points than predictors per group
    logger.warning("Some groups have very few data points relative to the number of predictors (PCs). Model estimation might be unstable or fail.")

# Define the model formula including interaction and PC controls (Mean + StdDev)
pc_terms = [f"AvgPC{i+1}" for i in range(N_PCS)] + [f"StdPC{i+1}" for i in range(N_PCS)]
model_formula = (f"PiValue ~ C(Orientation, Treatment('Direct')) * C(Recurrence, Treatment('Single-event')) + "
                 f"{' + '.join(pc_terms)}")

logger.info(f"Using LMM formula: {model_formula}")

try:
    # Use REML (Restricted Maximum Likelihood) for variance components estimation, common for LMM
    mixed_model = smf.mixedlm(model_formula, data_long, groups=data_long["InversionRegionID_geno"])
    result = mixed_model.fit(reml=True, method=["lbfgs"]) # Try L-BFGS optimizer first
    logger.info("Model fitting successful.")

except np.linalg.LinAlgError:
    logger.warning("Singular matrix error during initial fit. Trying alternative optimizer (CG)...")
    try:
        result = mixed_model.fit(reml=True, method=["cg"]) # Conjugate Gradient
        logger.info("Model fitting successful with CG optimizer.")
    except Exception as e_cg:
        logger.error(f"Model fitting failed even with CG optimizer: {e_cg}", exc_info=True)
        result = None # Ensure result is None if fitting failed completely
except Exception as e:
    logger.error(f"ERROR: Model fitting failed: {e}", exc_info=True)
    result = None # Ensure result is None if fitting failed

# --- Step 9: Output Results and Visualizations ---
logger.info("--- Step 9: Saving Results and Generating Visualizations ---")

if result:
    # Save Model Summary
    logger.info("Saving model summary...")
    try:
        with open(MODEL_SUMMARY_PATH, 'w') as f:
            f.write("Linear Mixed Model Regression Results (REML)\n")
            f.write("=============================================\n")
            f.write(f"Model Formula: {model_formula}\n")
            f.write(f"Grouping Variable: InversionRegionID_geno (N={final_regions})\n")
            f.write(f"Number of Observations: {final_obs}\n")
            f.write("Data Counts per Group:\n")
            f.write(group_counts.to_string())
            f.write("\n=============================================\n")
            f.write(result.summary().as_text())
        logger.info(f"Model summary saved to {MODEL_SUMMARY_PATH}")
        print("\n--- Mixed Effects Model Results ---")
        print(result.summary())
    except Exception as e:
        logger.error(f"Failed to save model summary: {e}")

    # Generate Visualizations
    logger.info("Generating visualizations...")
    try:
        # Set seaborn style
        sns.set_theme(style="whitegrid", palette="muted")
        color_palette_orient = sns.color_palette("Set2", n_colors=2)
        color_palette_recur = sns.color_palette("viridis", n_colors=2)
        
        # Boxplot with semi-transparent boxes for clarity
        plt.figure(figsize=(9, 7))
        
        # Create the boxplot with custom properties
        ax_box = sns.boxplot(x='Recurrence', y='PiValue', hue='Orientation', data=data_long,
                         palette=color_palette_orient, showfliers=False, 
                         hue_order=['Direct', 'Inverted'],
                         order=['Single-event', 'Recurrent'])
        
        # Make the boxes semi-transparent for better clarity
        for patch in ax_box.patches:
            # Get current color
            current_color = patch.get_facecolor()
            # Set transparency to 40% (0.4) - visible but subtle
            patch.set_facecolor((*current_color[:3], 0.4))
            # Make the edge lines a bit thicker for better visibility
            patch.set_linewidth(1.5)
        
        # Add the strip plot on top for data points
        sns.stripplot(x='Recurrence', y='PiValue', hue='Orientation', data=data_long,
                      palette=color_palette_orient, dodge=True, size=3, alpha=0.7, jitter=True,
                      legend=False, hue_order=['Direct', 'Inverted'], order=['Single-event', 'Recurrent'])
        
        # Set titles and labels
        ax_box.set_title('Nucleotide Diversity (π) by Inversion Type and Orientation', fontsize=16, pad=15)
        ax_box.set_xlabel('Inversion Recurrence Type', fontsize=12)
        ax_box.set_ylabel('Nucleotide Diversity (π)', fontsize=12)
        ax_box.tick_params(axis='both', which='major', labelsize=10)
        ax_box.legend(title='Orientation', title_fontsize='11', fontsize='10', loc='upper right')
        
        plt.tight_layout()
        plt.savefig(BOXPLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Boxplot with semi-transparent boxes saved to {BOXPLOT_PATH}")

        # Interaction Plot
        plt.figure(figsize=(8, 6))
        ax_int = sns.pointplot(x='Orientation', y='PiValue', hue='Recurrence', data=data_long,
                               palette=color_palette_recur, markers=["o", "^"], linestyles=["-", "--"],
                               dodge=0.1, errorbar=('ci', 95), capsize=.1,
                               hue_order=['Single-event', 'Recurrent'], order=['Direct', 'Inverted'])
        ax_int.set_title('Interaction Plot: Mean π by Orientation and Recurrence', fontsize=16, pad=15)
        ax_int.set_xlabel('Haplotype Orientation', fontsize=12)
        ax_int.set_ylabel('Estimated Mean Nucleotide Diversity (π) [95% CI]', fontsize=12)
        ax_int.tick_params(axis='both', which='major', labelsize=10)
        ax_int.legend(title='Recurrence Type', title_fontsize='11', fontsize='10', loc='best')
        ax_int.grid(True, axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(INTERACTION_PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Interaction plot saved to {INTERACTION_PLOT_PATH}")

    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}", exc_info=True)
else:
    logger.error("Model fitting failed, cannot generate summary or plots.")


main_end_time = time.time()
logger.info(f"\n--- Analysis Complete ---")
logger.info(f"Total execution time: {main_end_time - main_start_time:.2f} seconds")
logger.info(f"Results and logs saved in directory: {OUTPUT_DIR}")
