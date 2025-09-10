import os
import sys
import time
import gc

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_DIR = "impute"
GENOTYPE_DIR = "genotype_matrices"
PLINK_PREFIX = "subset"
OUTPUT_FILE = "imputed_inversion_dosages.tsv"
MISSING_VALUE_CODE = -127

# Only models in this set will be processed. All others will be skipped.
TARGET_INVERSIONS = {
    "chr3-195680867-INV-272256",
    "chr3-195749464-INV-230745",
    "chr6-76111919-INV-44661",
    "chr12-46897663-INV-16289",
    "chr6-141867315-INV-29159",
    "chr3-131969892-INV-7927",
    "chr6-167181003-INV-209976",
    "chr11-71571191-INV-6980",
    "chr9-102565835-INV-4446",
    "chr4-33098029-INV-7075",
    "chr7-57835189-INV-284465",
    "chr10-46135869-INV-77646",
    "chr11-24263185-INV-392",
    "chr13-79822252-INV-17591",
    "chr1-60775308-INV-5023",
    "chr6-130527042-INV-4267",
    "chr13-48199211-INV-7451",
    "chr21-13992018-INV-65632",
    "chr8-7301025-INV-5297356",
    "chr9-30951702-INV-5595",
    "chr17-45585160-INV-706887",
    "chr12-131333944-INV-289865",
    "chr7-70955928-INV-18020",
    "chr16-28471894-INV-165758",
    "chr7-65219158-INV-312667",
    "chr10-79542902-INV-674513",
    "chr1-13084312-INV-62181",
    "chr10-37102555-INV-11157",
    "chr4-40233409-INV-2010",
    "chr2-138246733-INV-5010",
}

def main():
    """
    Main function to run the inference pipeline.
    """
    print("--- Starting Memory-Efficient Imputation Inference Pipeline ---")
    start_time = time.time()

    # --- 1. Pre-flight Checks ---
    if not os.path.isdir(MODEL_DIR):
        print(f"[FATAL] Model directory not found: '{MODEL_DIR}'")
        sys.exit(1)
    if not os.path.isdir(GENOTYPE_DIR):
        print(f"[FATAL] Genotype matrix directory not found: '{GENOTYPE_DIR}'")
        sys.exit(1)

    fam_path = f"{PLINK_PREFIX}.fam"
    if not os.path.exists(fam_path):
        print(f"[FATAL] PLINK .fam file not found: '{fam_path}'. Cannot get sample IDs.")
        sys.exit(1)

    # --- 2. Load Sample IDs ---
    print(f"Loading sample IDs from {fam_path}...")
    try:
        fam_df = pd.read_csv(fam_path, sep=r'\s+', header=None, usecols=[1], dtype=str)
        sample_ids = fam_df[1].tolist()
        print(f"Successfully loaded {len(sample_ids)} sample IDs.")
    except Exception as e:
        print(f"[FATAL] Could not read sample IDs from .fam file. Error: {e}")
        sys.exit(1)

    # --- 3. Identify and Filter Models to Process ---
    # ... (this section is correct and unchanged)
    try:
        all_available_models = sorted([
            f.replace(".genotypes.npy", "")
            for f in os.listdir(GENOTYPE_DIR)
            if f.endswith(".genotypes.npy")
        ])
    except FileNotFoundError:
        all_available_models = []
    if not all_available_models:
        print("[FATAL] No '.genotypes.npy' files found in the genotype directory. Nothing to process.")
        sys.exit(1)
    print(f"Found {len(all_available_models)} total staged genotype matrices.")
    models_to_process = [m for m in all_available_models if m in TARGET_INVERSIONS]
    if not models_to_process:
        print("[FATAL] None of the available models are in the target list. Nothing to process.")
        sys.exit(1)
    print(f"After filtering, {len(models_to_process)} models will be processed.")

    # --- 4. Initialize final output file ---
    print(f"Initializing output file: {OUTPUT_FILE}")
    results_df = pd.DataFrame(index=sample_ids)
    results_df.index.name = "SampleID"
    results_df.to_csv(OUTPUT_FILE, sep='\t')

    # --- 5. Iterate, Impute, Predict, and Append ---
    for model_name in tqdm(models_to_process, desc="Predicting Inversions", unit="model"):
        print(f"\n--- Processing: {model_name} ---")

        model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
        matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")
        
        predicted_dosages = None

        try:
            model = joblib.load(model_path)
            X_inference = np.load(matrix_path, mmap_mode='r')
            n_samples, n_snps = X_inference.shape
            print(f"  - Samples: {n_samples}, SNPs in model: {n_snps}")
            if n_samples != len(sample_ids):
                print(f"[ERROR] Sample count mismatch! FAM has {len(sample_ids)} but matrix has {n_samples}. Skipping.")
                continue

            if X_inference.size == 0:
                print(f"  - [WARN] Genotype matrix is empty. Skipping.")
                continue
            
            missing_count = np.sum(X_inference == MISSING_VALUE_CODE)
            percent_missing = (missing_count / X_inference.size) * 100
            print(f"  - Missing data: {missing_count} / {X_inference.size} ({percent_missing:.2f}%)")

            # Using the original loop for imputation as it's faster for this data shape
            X_imputed = X_inference.astype(np.float32, copy=True)
            print("  - Imputing missing values with column means...")
            for j in range(n_snps):
                column_data = X_imputed[:, j]
                valid_mask = (column_data != MISSING_VALUE_CODE)
                if np.any(valid_mask):
                    col_mean = np.mean(column_data[valid_mask])
                else:
                    col_mean = 1.0
                column_data[~valid_mask] = col_mean
            
            print("  - Running prediction...")
            predicted_dosages = model.predict(X_imputed)

        except FileNotFoundError:
            print(f"  - [ERROR] Could not find matching model file: {model_path}. Skipping.")
            continue
        except Exception as e:
            print(f"  - [ERROR] An unexpected error occurred while processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        finally:
            if 'model' in locals(): del model
            if 'X_inference' in locals(): del X_inference
            if 'X_imputed' in locals(): del X_imputed
            gc.collect()

        # Step E: Append results directly to the file
        if predicted_dosages is not None:
            print(f"  - Appending results to {OUTPUT_FILE}...")
            
            # Force the index column to be read as a string.
            # This ensures the index dtypes match for a robust assignment.
            current_results = pd.read_csv(
                OUTPUT_FILE, 
                sep='\t', 
                index_col="SampleID",
                dtype={"SampleID": str} # This prevents pandas from inferring it as numeric
            )
            
            current_results[model_name] = pd.Series(predicted_dosages, index=sample_ids)
            
            current_results.to_csv(OUTPUT_FILE, sep='\t', float_format='%.4f')
            print("  - Done appending.")

    end_time = time.time()
    print("\n--- Pipeline Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Final output file is ready at '{os.path.abspath(OUTPUT_FILE)}'")


if __name__ == "__main__":
    main()
