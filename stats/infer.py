import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_DIR = "impute"
GENOTYPE_DIR = "genotype_matrices"
PLINK_PREFIX = "subset"  # Used to get sample IDs from the .fam file
OUTPUT_FILE = "imputed_inversion_dosages.tsv"
MISSING_VALUE_CODE = -127


def main():
    """
    Main function to run the inference pipeline.
    """
    print("--- Starting Imputation Inference Pipeline ---")
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

    # --- 3. Identify Models to Process ---
    # We use the generated .npy files as the source of truth for which models are usable.
    try:
        model_names = sorted([
            f.replace(".genotypes.npy", "")
            for f in os.listdir(GENOTYPE_DIR)
            if f.endswith(".genotypes.npy")
        ])
    except FileNotFoundError:
        model_names = []
        
    if not model_names:
        print("[FATAL] No '.genotypes.npy' files found in the genotype directory. Nothing to process.")
        sys.exit(1)

    print(f"Found {len(model_names)} staged genotype matrices to process.")

    # --- 4. Iterate, Impute, and Predict ---
    predictions_dict = {}
    
    for model_name in tqdm(model_names, desc="Predicting Inversions", unit="model"):
        print(f"\n--- Processing: {model_name} ---")

        model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
        matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")

        try:
            # Step A: Load the model and the genotype matrix
            model = joblib.load(model_path)
            # Use mmap_mode='r' for memory efficiency with large files
            X_inference = np.load(matrix_path, mmap_mode='r')

            n_samples, n_snps = X_inference.shape
            print(f"  - Samples: {n_samples}, SNPs in model: {n_snps}")
            
            if n_samples != len(sample_ids):
                print(f"[ERROR] Sample count mismatch! FAM has {len(sample_ids)} but matrix has {n_samples}. Skipping.")
                continue

            # Step B: Calculate missingness percentage
            if X_inference.size == 0:
                print(f"  - [WARN] Genotype matrix is empty. Skipping.")
                continue
            
            missing_mask = (X_inference == MISSING_VALUE_CODE)
            missing_count = np.sum(missing_mask)
            percent_missing = (missing_count / X_inference.size) * 100
            print(f"  - Missing data: {missing_count} / {X_inference.size} ({percent_missing:.2f}%)")

            # Step C: Handle missing values via mean imputation (CRITICAL STEP)
            X_imputed = X_inference.copy().astype(np.float32) # Work on a float copy
            
            # Calculate mean for each column, ignoring missing values
            # This is a robust way to do it column-by-column
            print("  - Imputing missing values with column means...")
            for j in range(n_snps):
                column_data = X_imputed[:, j]
                valid_data = column_data[column_data != MISSING_VALUE_CODE]
                
                if valid_data.size > 0:
                    col_mean = np.mean(valid_data)
                else:
                    # Edge case: if an entire SNP column is missing, default to 1.0
                    col_mean = 1.0 
                
                # Replace missing values in this column with its calculated mean
                column_data[column_data == MISSING_VALUE_CODE] = col_mean
                X_imputed[:, j] = column_data

            # Step D: Run prediction
            print("  - Running prediction...")
            predicted_dosages = model.predict(X_imputed)

            # Step E: Store results
            predictions_dict[model_name] = predicted_dosages

        except FileNotFoundError:
            print(f"  - [ERROR] Could not find matching model file: {model_path}. Skipping.")
        except Exception as e:
            print(f"  - [ERROR] An unexpected error occurred while processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # --- 5. Assemble and Save Final Results ---
    if not predictions_dict:
        print("\nNo predictions were successfully generated. Exiting.")
        sys.exit(1)

    print(f"\n--- Assembling Final Results for {len(predictions_dict)} Models ---")
    
    # Create the final DataFrame
    results_df = pd.DataFrame(predictions_dict)
    results_df.index = sample_ids
    results_df.index.name = "SampleID"
    
    # Save to a tab-separated file
    print(f"Saving final imputed dosages to: {OUTPUT_FILE}")
    results_df.to_csv(OUTPUT_FILE, sep='\t', float_format='%.4f')

    end_time = time.time()
    print("\n--- Pipeline Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Final output file is ready at '{os.path.abspath(OUTPUT_FILE)}'")


if __name__ == "__main__":
    main()
