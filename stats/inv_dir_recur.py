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
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.stats import wilcoxon, mannwhitneyu

# =====================================================================
# Configuration
# =====================================================================
# --- Input File Paths ---
# Contains pi values per orientation (e.g., 0_pi_filtered, 1_pi_filtered)
OUTPUT_PI_PATH = 'output.csv'
# Contains recurrence info per region (e.g., 0_single_1_recur)
INV_INFO_PATH = 'inv_info.tsv'
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
# Helper Functions
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

    inv_info_df = pd.read_csv(INV_INFO_PATH, sep='\t')
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


# B. Standardize Inversion Info Data ('inv_info.tsv')
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
]].drop_duplicates(subset=['InversionRegionID_geno']) # inner=None,one row per geno region ID

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
        result = None # inner=None,result is None if fitting failed completely
except Exception as e:
    logger.error(f"ERROR: Model fitting failed: {e}", exc_info=True)
    result = None # inner=None,result is None if fitting failed




# =====================================================================
# --- Step 9a: Performing Direct Comparisons for Plot Annotation ---
# =====================================================================
# This section calculates p-values for specific comparisons to be added
# as annotations to the violin plot. It uses Wilcoxon signed-rank test
# for paired data (Direct vs. Inverted within groups) and Mann-Whitney U
# test for independent groups (Overall Single-event vs. Recurrent).

logger.info("--- Step 9a: Performing Direct Comparisons for Plot Annotation ---")

# Import necessary statistical functions
from scipy.stats import wilcoxon, mannwhitneyu

# --- Helper Function for P-value Formatting ---
def format_p_value_plotting(p: Optional[float], min_n_for_ns: int = 5, n_obs: int = 0) -> str:
    """Formats p-values for plot annotations.

    Args:
        p: The calculated p-value.
        min_n_for_ns: Minimum observations required to report 'n.s.' instead of 'n/a'.
        n_obs: Actual number of observations/pairs used in the test.

    Returns:
        Formatted p-value string (e.g., "p < 0.001", "p = 0.04", "n.s.", "n/a").
    """
    if pd.isna(p):
        # Distinguish between test not run due to insufficient data vs. other error
        if n_obs < min_n_for_ns:
             return "n/a" # Not applicable / not enough data
        else:
             return "error" # Test failed for other reason
    elif p < 0.001:
        return "p < 0.001"
    elif p < 0.05:
        # Use general format 'g' which handles decimals and switches to scientific notation
        # .2g aims for 2 significant digits, good for p-values like 0.043 or 0.0012
        return f"p = {p:.2g}"
    else:
        # Only report non-significant if we had enough data to potentially find significance
        if n_obs >= min_n_for_ns:
            return "n.s." # Non-significant
        else:
            return "n/a" # Not enough data to claim non-significance meaningfully

# Initialize p-value variables
pval_wilcoxon_single = np.nan
pval_wilcoxon_recurrent = np.nan
pval_mw_overall = np.nan
n_pairs_single = 0
n_pairs_recurrent = 0
n_obs_single_overall = 0
n_obs_recurrent_overall = 0
MIN_N_PAIRS_WILCOXON = 5 # Minimum number of pairs for Wilcoxon test reporting
MIN_N_SAMPLES_MWU = 5   # Minimum number of samples per group for MWU test reporting


# --- 1. Paired Tests (Wilcoxon: Direct vs. Inverted) ---
logger.info("Calculating Wilcoxon p-values (Direct vs. Inverted)...")
# We need the 'paired_data' DataFrame which pivots data_long.
# Create it here to ensure it's available and has the required structure.
try:
    # Pivot the final cleaned data (data_long)
    paired_data_for_stats = data_long.pivot_table(
        index=['InversionRegionID_geno', 'Recurrence'],
        columns='Orientation',
        values='PiValue',
        observed=False # Keep consistency with plotting if observed=False used there
    ).reset_index()

    # IMPORTANT: Only include pairs where BOTH Direct and Inverted Pi are non-missing
    paired_data_for_stats = paired_data_for_stats.dropna(subset=['Direct', 'Inverted'])

    if paired_data_for_stats.empty:
        logger.warning("No complete pairs (Direct and Inverted) found after pivoting. Cannot run Wilcoxon tests.")
    else:
        # Subset for Single-event
        paired_single = paired_data_for_stats[paired_data_for_stats['Recurrence'] == 'Single-event']
        n_pairs_single = len(paired_single)
        if n_pairs_single >= MIN_N_PAIRS_WILCOXON:
            try:
                # Wilcoxon test requires differences; identical pairs are often dropped.
                # alternative='two-sided' is the default, but good to be explicit.
                # zero_method='wilcox' handles zero differences by default (drops them).
                # Consider 'pratt' or 'zsplit' if zeros are numerous and meaningful.
                diff = paired_single['Direct'] - paired_single['Inverted']
                # Check if all differences are zero (causes ValueError)
                if np.all(np.isclose(diff, 0)):
                     logger.warning(f"Wilcoxon (Single-event): All {n_pairs_single} pairs have zero difference. Setting p=1.0")
                     pval_wilcoxon_single = 1.0
                else:
                    stat_ws, pval_wilcoxon_single = wilcoxon(paired_single['Direct'], paired_single['Inverted'],
                                                             zero_method='wilcox', alternative='two-sided')
                    logger.info(f"  Wilcoxon (Single-event, N={n_pairs_single} pairs): p = {pval_wilcoxon_single:.4g}")

            except ValueError as e:
                logger.warning(f"Wilcoxon test failed for Single-event (N={n_pairs_single} pairs): {e}")
        else:
            logger.warning(f"Skipping Wilcoxon for Single-event: Insufficient pairs ({n_pairs_single} < {MIN_N_PAIRS_WILCOXON})")

        # Subset for Recurrent
        paired_recurrent = paired_data_for_stats[paired_data_for_stats['Recurrence'] == 'Recurrent']
        n_pairs_recurrent = len(paired_recurrent)
        if n_pairs_recurrent >= MIN_N_PAIRS_WILCOXON:
            try:
                diff = paired_recurrent['Direct'] - paired_recurrent['Inverted']
                if np.all(np.isclose(diff, 0)):
                     logger.warning(f"Wilcoxon (Recurrent): All {n_pairs_recurrent} pairs have zero difference. Setting p=1.0")
                     pval_wilcoxon_recurrent = 1.0
                else:
                    stat_wr, pval_wilcoxon_recurrent = wilcoxon(paired_recurrent['Direct'], paired_recurrent['Inverted'],
                                                                zero_method='wilcox', alternative='two-sided')
                    logger.info(f"  Wilcoxon (Recurrent, N={n_pairs_recurrent} pairs): p = {pval_wilcoxon_recurrent:.4g}")
            except ValueError as e:
                 logger.warning(f"Wilcoxon test failed for Recurrent (N={n_pairs_recurrent} pairs): {e}")
        else:
            logger.warning(f"Skipping Wilcoxon for Recurrent: Insufficient pairs ({n_pairs_recurrent} < {MIN_N_PAIRS_WILCOXON})")

except Exception as e:
    logger.error(f"Failed during paired data preparation or Wilcoxon tests: {e}", exc_info=True)

# --- 3. Store Formatted P-values for Plotting ---
# These variables will be used later in the plotting code
pval_str_w_single = format_p_value_plotting(pval_wilcoxon_single, MIN_N_PAIRS_WILCOXON, n_pairs_single)
pval_str_w_recurrent = format_p_value_plotting(pval_wilcoxon_recurrent, MIN_N_PAIRS_WILCOXON, n_pairs_recurrent)

logger.info(f"Formatted p-values for plot annotation:")
logger.info(f"  Single (D vs I): {pval_str_w_single}")
logger.info(f"  Recurrent (D vs I): {pval_str_w_recurrent}")

# =====================================================================


# --- Step 9: Output Results and Visualizations ---
logger.info("--- Step 9: Saving Results and Generating Visualizations ---")

# Define file paths for the new plots
VIOLIN_PLOT_PATH = os.path.join(OUTPUT_DIR, 'pi_violin_plot_grouped_paired.png')
INTERACTION_PLOT_WITH_DATA_PATH = os.path.join(OUTPUT_DIR, 'pi_interaction_plot_with_data.png')

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

    # Print raw nucleotide diversity values per group
    print("\n--- Nucleotide Diversity by Group (Raw Values) ---")
    try:
        # Use observed=False for categorical grouping consistency if needed
        group_stats = data_long.groupby(['Orientation', 'Recurrence'], observed=False)['PiValue'].agg(['median', 'mean', 'std', 'count'])
        print(group_stats)
    except Exception as e:
        logger.error(f"Failed to calculate group stats: {e}")
        group_stats = None # it's None if calculation fails

    # Print median values in scientific notation
    print("\n--- Group Median Values (Scientific Notation) ---")
    if group_stats is not None:
        for idx, row in group_stats.iterrows():
            if isinstance(idx, tuple) and len(idx) == 2:
                 print(f"{idx[0]}/{idx[1]}: median π = {row['median']:.6e} (n={int(row['count'])})")
            else:
                 print(f"Index: {idx}, Data: median π = {row['median']:.6e} (n={int(row['count'])})")
    else:
        print("Group stats calculation failed, cannot print medians.")

    
    def safe_divide(numerator, denominator):
        # Returns NaN for division by zero or NaN inputs. Essential for Ratio of Aggregates.
        if pd.isna(numerator) or pd.isna(denominator):
            return np.nan
        if np.isclose(denominator, 0):
             return np.nan
        return numerator / denominator
    
    print("\n--- Fold Difference Calculations ---")
    
    # --- Preparations ---
    # inner=None,group_stats (with medians) exists from earlier code block
    if 'group_stats' not in locals() or group_stats is None:
        logger.error("Dependency Error: 'group_stats' DataFrame not found. Cannot calculate Ratio of Medians.")
        # Fallback: Create group_stats if absolutely necessary, but indicates a potential flow issue
        group_stats = data_long.groupby(['Orientation', 'Recurrence'], observed=False)['PiValue'].agg(['median', 'mean', 'std', 'count'])
    
    # inner=None,paired_data exists (created from pivot for violin plot or explicitly here)
    if 'paired_data' not in locals() or not isinstance(paired_data, pd.DataFrame) or paired_data.empty:
         # This line previously logged a warning. Removed per instruction.
         paired_data = data_long.pivot_table(index=['InversionRegionID_geno', 'Recurrence'], columns='Orientation', values='PiValue', observed=False).reset_index()
         paired_data = paired_data.dropna(subset=['Direct', 'Inverted']) # Still need pairs with BOTH values non-NaN
    
    # --- 1. Fold Difference of Group Aggregates (Ratio of Aggregates) ---
    # Uses safe_divide (NaN for 0 denominator) for both mean and median ratios.
    print("\n1. Fold Difference of Group Aggregates (Direct Aggregate / Inverted Aggregate):")
    
    # 1a. Ratio of MEANS
    print("  - Ratio of Means:")
    overall_means_df = data_long.groupby('Orientation', observed=False)['PiValue'].agg(['mean', 'count'])
    overall_direct_mean = overall_means_df.loc['Direct', 'mean'] if 'Direct' in overall_means_df.index else np.nan
    overall_inverted_mean = overall_means_df.loc['Inverted', 'mean'] if 'Inverted' in overall_means_df.index else np.nan
    overall_direct_n = int(overall_means_df.loc['Direct', 'count']) if 'Direct' in overall_means_df.index else 0
    overall_inverted_n = int(overall_means_df.loc['Inverted', 'count']) if 'Inverted' in overall_means_df.index else 0
    overall_fold_diff_means = safe_divide(overall_direct_mean, overall_inverted_mean)
    print(f"    Overall: {overall_fold_diff_means:.4f} (n_Dir={overall_direct_n}, n_Inv={overall_inverted_n})")
    
    group_means_df = data_long.groupby(['Orientation', 'Recurrence'], observed=False)['PiValue'].agg(['mean', 'count'])
    dir_sing_mean = group_means_df.loc[('Direct', 'Single-event'), 'mean'] if ('Direct', 'Single-event') in group_means_df.index else np.nan
    inv_sing_mean = group_means_df.loc[('Inverted', 'Single-event'), 'mean'] if ('Inverted', 'Single-event') in group_means_df.index else np.nan
    dir_rec_mean = group_means_df.loc[('Direct', 'Recurrent'), 'mean'] if ('Direct', 'Recurrent') in group_means_df.index else np.nan
    inv_rec_mean = group_means_df.loc[('Inverted', 'Recurrent'), 'mean'] if ('Inverted', 'Recurrent') in group_means_df.index else np.nan
    sing_n_direct = int(group_means_df.loc[('Direct', 'Single-event'), 'count']) if ('Direct', 'Single-event') in group_means_df.index else 0
    sing_n_inverted = int(group_means_df.loc[('Inverted', 'Single-event'), 'count']) if ('Inverted', 'Single-event') in group_means_df.index else 0
    rec_n_direct = int(group_means_df.loc[('Direct', 'Recurrent'), 'count']) if ('Direct', 'Recurrent') in group_means_df.index else 0
    rec_n_inverted = int(group_means_df.loc[('Inverted', 'Recurrent'), 'count']) if ('Inverted', 'Recurrent') in group_means_df.index else 0
    sing_fold_diff_means = safe_divide(dir_sing_mean, inv_sing_mean)
    rec_fold_diff_means = safe_divide(dir_rec_mean, inv_rec_mean)
    print(f"    Single-event: {sing_fold_diff_means:.4f} (n_Dir={sing_n_direct}, n_Inv={sing_n_inverted})")
    print(f"    Recurrent: {rec_fold_diff_means:.4f} (n_Dir={rec_n_direct}, n_Inv={rec_n_inverted})")
    
    # 1b. Ratio of MEDIANS
    print("  - Ratio of Medians:")
    overall_medians_df = data_long.groupby('Orientation', observed=False)['PiValue'].agg(['median', 'count'])
    overall_direct_median = overall_medians_df.loc['Direct', 'median'] if 'Direct' in overall_medians_df.index else np.nan
    overall_inverted_median = overall_medians_df.loc['Inverted', 'median'] if 'Inverted' in overall_medians_df.index else np.nan
    overall_fold_diff_medians = safe_divide(overall_direct_median, overall_inverted_median)
    print(f"    Overall: {overall_fold_diff_medians:.4f} (n_Dir={overall_direct_n}, n_Inv={overall_inverted_n})")
    
    if 'group_stats' in locals() and group_stats is not None:
        dir_sing_median = group_stats.loc[('Direct', 'Single-event'), 'median'] if ('Direct', 'Single-event') in group_stats.index else np.nan
        inv_sing_median = group_stats.loc[('Inverted', 'Single-event'), 'median'] if ('Inverted', 'Single-event') in group_stats.index else np.nan
        dir_rec_median = group_stats.loc[('Direct', 'Recurrent'), 'median'] if ('Direct', 'Recurrent') in group_stats.index else np.nan
        inv_rec_median = group_stats.loc[('Inverted', 'Recurrent'), 'median'] if ('Inverted', 'Recurrent') in group_stats.index else np.nan
        sing_fold_diff_medians = safe_divide(dir_sing_median, inv_sing_median)
        rec_fold_diff_medians = safe_divide(dir_rec_median, inv_rec_median)
        print(f"    Single-event: {sing_fold_diff_medians:.4f} (n_Dir={sing_n_direct}, n_Inv={sing_n_inverted})")
        print(f"    Recurrent: {rec_fold_diff_medians:.4f} (n_Dir={rec_n_direct}, n_Inv={rec_n_inverted})")
    else:
        print("    Could not calculate Ratio of Medians for subgroups ('group_stats' missing).")
    
    
    # --- 2. Aggregate of Individual Paired Fold Differences (Aggregate of Ratios) ---
    print("\n2. Aggregate of Individual Paired Fold Differences (Aggregate[Direct π / Inverted π]):")
    
    # Prepare base DataFrame for ratio calculation (all pairs where Direct/Inverted are not NaN)
    calc_df_ratios = paired_data.copy()
    calc_df_ratios = calc_df_ratios.rename(columns={'Direct': 'pi_direct', 'Inverted': 'pi_inverted'})
    calc_df_ratios = calc_df_ratios.dropna(subset=['pi_direct', 'pi_inverted']) # inner=None,both values exist
    
    # Calculate ratios, allowing division by zero (results in inf/nan)
    # Suppress division warnings temporarily for this calculation block if desired
    with np.errstate(divide='ignore', invalid='ignore'):
        calc_df_ratios['paired_fold_diff'] = calc_df_ratios['pi_direct'] / calc_df_ratios['pi_inverted']
    # Replace potential -inf with inf for consistency if needed, although median usually handles them ok
    # calc_df_ratios['paired_fold_diff'] = calc_df_ratios['paired_fold_diff'].replace(-np.inf, np.inf)
    
    
    # 2a. MEAN of Ratios (Filter out non-finite ratios AFTER calculation)
    print("  - Mean of Ratios (excludes pairs with Inverted π=0):")
    # Filter AFTER calculating ratio: keep only finite ratios for the mean
    finite_ratios_mask = np.isfinite(calc_df_ratios['paired_fold_diff'])
    calc_df_valid_mean = calc_df_ratios[finite_ratios_mask]
    n_mean_pairs = len(calc_df_valid_mean)
    
    if n_mean_pairs > 0:
        mean_overall_paired_fd = calc_df_valid_mean['paired_fold_diff'].mean()
        print(f"    Overall: {mean_overall_paired_fd:.4f} (n={n_mean_pairs} valid pairs)")
    
        mean_paired_fd_by_recurrence = calc_df_valid_mean.groupby('Recurrence', observed=False)['paired_fold_diff'].agg(['mean', 'count'])
        for idx, row in mean_paired_fd_by_recurrence.iterrows():
            print(f"    {idx}: {row['mean']:.4f} (n={int(row['count'])} valid pairs)")
    else:
        print("    No pairs with finite ratios found for Mean of Ratios calculation.")
    
    # 2b. MEDIAN of Ratios (Includes non-finite ratios in calculation, median often ignores NaN/inf)
    print("  - Median of Ratios (includes pairs with Inverted π=0):")
    n_median_pairs = len(calc_df_ratios) # Total pairs where ratio could be calculated (incl. inf/nan)
    
    if n_median_pairs > 0:
        # Calculate median directly on the potentially non-finite ratios series
        # pandas median typically ignores NaN, treatement of Inf varies but often ignored too
        median_overall_paired_fd = calc_df_ratios['paired_fold_diff'].median()
        print(f"    Overall: {median_overall_paired_fd:.4f} (n={n_median_pairs} pairs considered)")
    
        # Group by recurrence and calculate median on potentially non-finite ratios
        median_paired_fd_by_recurrence = calc_df_ratios.groupby('Recurrence', observed=False)['paired_fold_diff'].agg(['median', 'count'])
        for idx, row in median_paired_fd_by_recurrence.iterrows():
            print(f"    {idx}: {row['median']:.4f} (n={int(row['count'])} pairs considered)")
    else:
         print("    No pairs found to calculate Median of Ratios.")
    # Generate Visualizations
    logger.info("Generating visualizations...")
    try:
        # Set cleaner style and define specific, non-orange palettes
        sns.set_style("ticks") # Cleaner background than whitegrid
        orient_palette = {'Direct': '#0072B2', # A medium blue
                          'Inverted': '#009E73'} # A teal/green
        # Keep viridis for recurrence, it's generally well-perceived
        recur_palette = {'Single-event': sns.color_palette("viridis", n_colors=2)[0],
                         'Recurrent': sns.color_palette("viridis", n_colors=2)[1]}
        recur_markers = {'Single-event': 'o', 'Recurrent': 'X'}
        recur_lines = {'Single-event': '-', 'Recurrent': ':'}

        # Define file paths for the new plots (should be already defined earlier)
        VIOLIN_PLOT_PATH = os.path.join(OUTPUT_DIR, 'pi_violin_plot_grouped_paired_with_pvals.png') # Modified name
        INTERACTION_PLOT_WITH_DATA_PATH = os.path.join(OUTPUT_DIR, 'pi_interaction_plot_with_data.png')


        # --- Violin Plot with Paired Lines and P-value Annotations ---
        logger.info("Generating Violin Plot with Paired Lines and P-value Annotations...")
        fig_viol, ax_viol = plt.subplots(figsize=(11, 7)) # Slightly adjusted size

        # 1. Prepare data for pairing lines (uses data_long)
        # It's good practice to ensure the paired_data used for lines is the same basis
        # as the one used for Wilcoxon tests (paired_data_for_stats). Re-use if possible or recalculate.
        # Assuming 'paired_data_for_stats' from Step 9a is the definitive source for pairs:
        if 'paired_data_for_stats' in locals() and not paired_data_for_stats.empty:
             plot_paired_data = paired_data_for_stats.copy()
             logger.info(f"Using {len(plot_paired_data)} pairs for plotting lines.")
        else:
            # Fallback if Step 9a failed or didn't create it, but warn
            logger.warning("Recreating paired data for plotting lines; ensure consistency with stats if Step 9a ran.")
            plot_paired_data = data_long.pivot_table(index=['InversionRegionID_geno', 'Recurrence'], columns='Orientation', values='PiValue', observed=False).reset_index()
            plot_paired_data = plot_paired_data.dropna(subset=['Direct', 'Inverted'])


        # --- Robust L2FC Calculation ---
        # Apply to the data being used for plotting lines
        if not plot_paired_data.empty:
            valid_direct = plot_paired_data['Direct'] > 0
            valid_inverted = plot_paired_data['Inverted'] > 0
            valid_both = valid_direct & valid_inverted
            plot_paired_data['L2FC'] = np.nan
            # Ensure division by zero results in NaN or Inf, log2 handles Inf appropriately
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = plot_paired_data.loc[valid_both, 'Direct'] / plot_paired_data.loc[valid_both, 'Inverted']
            plot_paired_data.loc[valid_both, 'L2FC'] = np.log2(ratio.replace([np.inf, -np.inf], np.nan)) # Convert Inf ratio to NaN before log2
        else:
            plot_paired_data['L2FC'] = np.nan # Add column even if empty
        # --- End Robust L2FC Calculation ---

        # 2. Define coordinates for pairing lines (same as before)
        recurrence_categories = ['Single-event', 'Recurrent']
        orientation_categories = ['Direct', 'Inverted']
        recurrence_map_pos = {cat: i for i, cat in enumerate(recurrence_categories)}

        if not plot_paired_data.empty:
            plot_paired_data['x_recurrence_num'] = plot_paired_data['Recurrence'].map(recurrence_map_pos).astype(float)
        else:
             plot_paired_data['x_recurrence_num'] = np.nan # Add column even if empty

        n_hues = len(orientation_categories)
        violin_width = 0.8 # Width of the violins
        dodge_sep = 0.02   # Separation between violins of different hues
        total_dodge_width = violin_width + dodge_sep # Total width covered by dodged elements
        # Calculate offsets relative to the center of the recurrence category position
        orient_offsets = {'Direct': -total_dodge_width / 4, 'Inverted': total_dodge_width / 4}

        if not plot_paired_data.empty:
            plot_paired_data['x_direct'] = plot_paired_data['x_recurrence_num'] + orient_offsets['Direct']
            plot_paired_data['x_inverted'] = plot_paired_data['x_recurrence_num'] + orient_offsets['Inverted']
        else:
            plot_paired_data['x_direct'] = np.nan
            plot_paired_data['x_inverted'] = np.nan

        # 3. Set up colormap for L2FC lines (same as before)
        l2fc_values = plot_paired_data['L2FC'].dropna()
        if not l2fc_values.empty:
            vmin, vmax = l2fc_values.min(), l2fc_values.max()
            # Handle cases where all L2FC are 0 or very close
            if np.isclose(vmin, 0) and np.isclose(vmax, 0):
                 max_abs = 1.0 # Default range if no variation
            elif pd.isna(vmin) or pd.isna(vmax):
                 max_abs = 1.0 # Default if calculation resulted in NaN
            else:
                 max_abs = max(abs(vmin), abs(vmax), 1e-9) # Ensure max_abs is not zero
            norm = mcolors.Normalize(vmin=-max_abs, vmax=max_abs)
        else:
            norm = mcolors.Normalize(vmin=-1, vmax=1) # Default norm if no L2FC values
        cmap = cm.coolwarm
        scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        # --- Plotting Elements ---
        # 4. Add the transparent strip plot (drawn first)
        sns.stripplot(x='Recurrence', y='PiValue', hue='Orientation', data=data_long,
                      palette=orient_palette, dodge=True, size=3.0, alpha=0.35,
                      jitter=0.1, legend=False, hue_order=orientation_categories, order=recurrence_categories,
                      ax=ax_viol, zorder=5)

        # 5. Create the main Violin plot (drawn over strip plot)
        sns.violinplot(x='Recurrence', y='PiValue', hue='Orientation', data=data_long,
                       palette=orient_palette, hue_order=orientation_categories, order=recurrence_categories,
                       inner=None, # No box/whisker/points inside violin
                       linewidth=1.2,
                       width=violin_width, cut=0, dodge=dodge_sep, # Control width and dodging
                       scale='width', # Violins have same max width regardless of N
                       alpha=0.2, # Make violins transparent
                       ax=ax_viol, zorder=10) # Draw above stripplot

        # 6. Add Median Lines inside Violins
        median_values = data_long.groupby(['Recurrence', 'Orientation'], observed=False)['PiValue'].median()
        median_line_width_on_plot = 0.15 # Horizontal width of median line
        median_line_color = 'k'; median_line_style = '-'; median_line_lw = 1.5
        median_line_zorder = 12 # Above violin fill, below pairing lines
        for group_index, median_val in median_values.items():
            if pd.notna(median_val): # Only plot if median exists
                recurrence_cat, orientation_cat = group_index
                x_center = recurrence_map_pos[recurrence_cat] + orient_offsets[orientation_cat]
                xmin = x_center - median_line_width_on_plot / 2
                xmax = x_center + median_line_width_on_plot / 2
                ax_viol.hlines(y=median_val, xmin=xmin, xmax=xmax, color=median_line_color,
                               linestyle=median_line_style, linewidth=median_line_lw,
                               zorder=median_line_zorder, alpha=0.7) # Slightly transparent

        # 7. Draw the Pairing Lines (drawn over violins)
        if not plot_paired_data.empty:
            line_alpha = 0.7; line_lw = 0.9
            for _, row in plot_paired_data.iterrows():
                l2fc_val = row['L2FC']
                # Check if L2FC is valid, and if coordinates are valid
                if pd.notna(l2fc_val) and pd.notna(row['x_direct']) and pd.notna(row['x_inverted']):
                    line_color = scalar_mappable.to_rgba(l2fc_val)
                    ax_viol.plot([row['x_direct'], row['x_inverted']], [row['Direct'], row['Inverted']],
                                 color=line_color, alpha=line_alpha, lw=line_lw, zorder=15) # Draw last

        # --- Add P-value Annotations (NEW PART) ---
        # Use the formatted p-value strings calculated in Step 9a:
        # pval_str_w_single, pval_str_w_recurrent, pval_str_mw_overall

        # Define bracket heights - adjust factors as needed
        bracket_height_factor = 1.05 # How much higher than max data point
        bracket_spacing_factor = 1.08 # Multiplier to space stacked brackets
        bracket_line_width = 1.0
        bracket_tick_height = 0.02 # Relative height of vertical ticks on brackets
        annotation_fontsize = 8

        y_max_overall = data_long['PiValue'].max() if not data_long['PiValue'].empty else 0
        current_max_y_for_brackets = y_max_overall # Start tracking highest point needed

        # Annotation 1: Direct vs Inverted (Single-event)
        if pval_str_w_single != "error": # Only plot if test ran successfully
            y_max_single_group = data_long[data_long['Recurrence'] == 'Single-event']['PiValue'].max()
            y_bracket1 = y_max_single_group * bracket_height_factor
            x1_s = recurrence_map_pos['Single-event'] + orient_offsets['Direct']
            x2_s = recurrence_map_pos['Single-event'] + orient_offsets['Inverted']
            y_tick1 = y_bracket1 * bracket_tick_height # Calculate tick height based on bracket y
            # Draw bracket line
            ax_viol.plot([x1_s, x1_s, x2_s, x2_s], [y_bracket1, y_bracket1 + y_tick1, y_bracket1 + y_tick1, y_bracket1],
                         lw=bracket_line_width, c='k')
            # Add text slightly above bracket ticks
            ax_viol.text((x1_s + x2_s) / 2, y_bracket1 + y_tick1 * 1.1, pval_str_w_single,
                         ha='center', va='bottom', color='k', fontsize=annotation_fontsize)
            current_max_y_for_brackets = max(current_max_y_for_brackets, y_bracket1 + y_tick1 * 2) # Update max Y needed

        # Annotation 2: Direct vs Inverted (Recurrent)
        if pval_str_w_recurrent != "error":
            y_max_recurrent_group = data_long[data_long['Recurrence'] == 'Recurrent']['PiValue'].max()
            # Ensure this bracket is above the first one if groups overlap in x
            y_bracket2 = max(y_max_recurrent_group * bracket_height_factor, current_max_y_for_brackets * bracket_spacing_factor * 0.8) # Place relative to current max
            x1_r = recurrence_map_pos['Recurrent'] + orient_offsets['Direct']
            x2_r = recurrence_map_pos['Recurrent'] + orient_offsets['Inverted']
            y_tick2 = y_bracket2 * bracket_tick_height
            ax_viol.plot([x1_r, x1_r, x2_r, x2_r], [y_bracket2, y_bracket2 + y_tick2, y_bracket2 + y_tick2, y_bracket2],
                         lw=bracket_line_width, c='k')
            ax_viol.text((x1_r + x2_r) / 2, y_bracket2 + y_tick2 * 1.1, pval_str_w_recurrent,
                         ha='center', va='bottom', color='k', fontsize=annotation_fontsize)
            current_max_y_for_brackets = max(current_max_y_for_brackets, y_bracket2 + y_tick2 * 2)

        # Adjust y-limits to make space for annotations
        current_ylim = ax_viol.get_ylim()
        new_ylim_top = max(current_ylim[1], current_max_y_for_brackets * 1.05) # Add a little padding above highest annotation
        ax_viol.set_ylim(current_ylim[0], new_ylim_top)

        # --- End P-value Annotations ---

        # 8. Add Colorbar (Smaller)
        cbar = fig_viol.colorbar(scalar_mappable, ax=ax_viol, pad=0.02, aspect=25, shrink=0.65)
        cbar.set_label('Log2 (π Direct / π Inverted)', rotation=270, labelpad=18, fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        cbar.outline.set_visible(False)

        # 9. Set titles, labels, and aesthetics
        title_text = "Nucleotide Diversity (π) by Inversion Type and Orientation"
        ax_viol.set_title(title_text, fontsize=18, pad=20) # Adjusted pad for potential top annotation
        ax_viol.set_xlabel('Inversion Recurrence Type', fontsize=16)
        ax_viol.set_ylabel('Nucleotide Diversity (π)', fontsize=16)
        ax_viol.tick_params(axis='both', which='major', labelsize=13, length=4)
        ax_viol.set_xticks(range(len(recurrence_categories)))
        ax_viol.set_xticklabels(recurrence_categories, fontsize=13)

        # Remove top/right spines for a cleaner presentation
        sns.despine(ax=ax_viol, offset=5)

        # Handle legend (ensure it doesn't overlap colorbar)
        handles, labels = ax_viol.get_legend_handles_labels()
        # Filter to get only handles corresponding to orientation
        # Need to be careful here as violinplot might return complex handles
        # Simplest way is often to create proxy artists
        from matplotlib.patches import Patch
        orient_legend_handles = [Patch(facecolor=orient_palette[label], alpha=0.6, label=label) for label in orientation_categories] # Use alpha similar to violins
        orient_legend_labels = orientation_categories

        ax_viol.legend(
            orient_legend_handles,
            orient_legend_labels,
            title='Haplotype Orientation',
            title_fontsize=13,
            fontsize=12,
            loc='upper left',
            bbox_to_anchor=(1.04, 1),
            frameon=False,
        )  # Adjusted anchor slightly right

        # Adjust layout AFTER placing elements like legend/colorbar/annotations
        # May need to adjust 'top' in rect to accommodate annotations
        fig_viol.tight_layout(rect=[0.02, 0.02, 0.88, 0.93]) # Adjusted top boundary slightly

        plt.savefig(VIOLIN_PLOT_PATH, dpi=300, bbox_inches='tight')
        plt.close(fig_viol)
        logger.info(f"Violin plot with pairing lines and p-value annotations saved to {VIOLIN_PLOT_PATH}")


        # --- Interaction Plot with Raw Data Points (No changes needed here) ---
        logger.info("Generating Interaction Plot...")
        fig_int, ax_int = plt.subplots(figsize=(7, 5.5))
        point_dodge = 0.15

        # 1. Plot transparent raw data points first
        sns.stripplot(x='Orientation', y='PiValue', hue='Recurrence', data=data_long,
                      palette=recur_palette, hue_order=['Single-event', 'Recurrent'], order=['Direct', 'Inverted'],
                      dodge=point_dodge, size=3.5, alpha=0.3,
                      jitter=0.1, legend=False,
                      ax=ax_int, zorder=1)

        # 2. Plot the interaction plot (means and CIs) on top
        sns.pointplot(x='Orientation', y='PiValue', hue='Recurrence', data=data_long,
                      palette=recur_palette,
                      markers=[recur_markers[cat] for cat in ['Single-event', 'Recurrent']],
                      linestyles=[recur_lines[cat] for cat in ['Single-event', 'Recurrent']],
                      hue_order=['Single-event', 'Recurrent'], order=['Direct', 'Inverted'],
                      dodge=point_dodge, errorbar=('ci', 95), capsize=.08,
                      linewidth=1.5,
                      ax=ax_int, zorder=10)

        # 3. Set titles, labels, and aesthetics
        title_text_int = "Interaction Plot: Mean Nucleotide Diversity (π)"
        caption_text_int = "Lines: Group Means ± 95% CI. Points: Raw Data per Inversion/Orientation."
        ax_int.set_title(title_text_int, fontsize=17, pad=20)
        fig_int.text(0.5, 0.95, caption_text_int, ha="center", va="bottom", fontsize=11, alpha=0.8, wrap=True)

        ax_int.set_xlabel('Haplotype Orientation', fontsize=15)
        ax_int.set_ylabel('Mean Nucleotide Diversity (π) [95% CI]', fontsize=15)
        ax_int.tick_params(axis='both', which='major', labelsize=13, length=4)
        sns.despine(ax=ax_int, offset=5)

        handles, labels = ax_int.get_legend_handles_labels()
        num_recur_cats = len(recur_palette)
        ax_int.legend(
            handles[:num_recur_cats],
            labels[:num_recur_cats],
            title='Recurrence Type',
            title_fontsize=13,
            fontsize=12,
            loc='best',
            frameon=False,
        )

        fig_int.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])

        plt.savefig(INTERACTION_PLOT_WITH_DATA_PATH, dpi=300, bbox_inches='tight')
        plt.close(fig_int)
        logger.info(f"Interaction plot with raw data points saved to {INTERACTION_PLOT_WITH_DATA_PATH}")
        logger.info("Interaction Plot Details: Shows group means +/- 95% CI over raw data points.")

    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}", exc_info=True)
else:
    logger.error("Model fitting failed, cannot generate summary or plots.")

main_end_time = time.time()
logger.info(f"\n--- Analysis Complete ---")
logger.info(f"Total execution time: {main_end_time - main_start_time:.2f} seconds")
logger.info(f"Results and logs saved in directory: {OUTPUT_DIR}")
