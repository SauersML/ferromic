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
    "chr2-138246733-INV-5010",
    "chr7-54220528-INV-101153",
    "chr6-76111919-INV-44661",
    "chr12-46897663-INV-16289",
    "chr1-197787661-INV-1197",
    "chr6-167181003-INV-209976",
    "chr10-55007454-INV-5370",
    "chr7-40839738-INV-1134",
    "chr4-87925789-INV-11799",
    "chr3-195680867-INV-272256",
    "chr3-162827167-INV-3077",
    "chr3-195749464-INV-230745",
    "chr9-30951702-INV-5595",
    "chr16-28471894-INV-165758",
    "chr10-79542902-INV-674513",
    "chr11-55662740-INV-3952",
    "chr7-70955928-INV-18020",
    "chr14-60604531-INV-8718",
    "chr10-37102555-INV-11157",
    "chr12-71546144-INV-1652",
    "chr6-141867315-INV-29159",
    "chr4-187948402-INV-8158",
    "chr11-66251212-INV-1252",
    "chr3-131969892-INV-7927",
    "chr11-310146-INV-10302",
    "chr4-33098029-INV-7075",
    "chr5-64470929-INV-4190",
    "chr11-24263185-INV-392",
    "chr6-130527042-INV-4267",
    "chr9-123976301-INV-18001",
    "chr15-30618104-INV-1535102",
    "chr8-7301025-INV-5297356",
    "chr4-40233409-INV-2010",
    "chr15-84373376-INV-43322",
    "chr13-48199211-INV-7451",
    "chr11-50136371-INV-206505",
    "chr5-64466170-INV-15635",
    "chr10-46135869-INV-77646",
    "chr7-65219158-INV-312667",
    "chr2-95800192-INV-224213",
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

    # Apply the hardcoded filter
    models_to_process = [m for m in all_available_models if m in TARGET_INVERSIONS]
    
    if not models_to_process:
        print("[FATAL] None of the available models are in the target list. Nothing to process.")
        sys.exit(1)
        
    print(f"After filtering, {len(models_to_process)} models will be processed.")
    
    # --- 4. MEMORY: Initialize final output file and write header ---
    # We will append columns to this file instead of holding results in memory.
    print(f"Initializing output file: {OUTPUT_FILE}")
    # Start with a DataFrame containing only the sample IDs
    results_df = pd.DataFrame(index=sample_ids)
    results_df.index.name = "SampleID"
    # Write the initial file with just the index
    results_df.to_csv(OUTPUT_FILE, sep='\t')

    # --- 5. Iterate, Impute, Predict, and Append ---
    for model_name in tqdm(models_to_process, desc="Predicting Inversions", unit="model"):
        print(f"\n--- Processing: {model_name} ---")

        model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
        matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")
        
        predicted_dosages = None # Ensure variable is defined

        try:
            # Step A: Load model and memory-mapped matrix
            model = joblib.load(model_path)
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
            
            missing_count = np.sum(X_inference == MISSING_VALUE_CODE)
            percent_missing = (missing_count / X_inference.size) * 100
            print(f"  - Missing data: {missing_count} / {X_inference.size} ({percent_missing:.2f}%)")

            # Step C: Handle missing values with a memory-efficient approach
            # We create the float copy ONLY for this loop iteration
            X_imputed = X_inference.astype(np.float32, copy=True)
            
            print("  - Imputing missing values with column means...")
            for j in range(n_snps):
                column_data = X_imputed[:, j]
                valid_mask = (column_data != MISSING_VALUE_CODE)
                
                if np.any(valid_mask):
                    col_mean = np.mean(column_data[valid_mask])
                else:
                    col_mean = 1.0 # Default if entire column is missing
                
                column_data[~valid_mask] = col_mean
            
            # Step D: Run prediction
            print("  - Running prediction...")
            predicted_dosages = model.predict(X_imputed)

        except FileNotFoundError:
            print(f"  - [ERROR] Could not find matching model file: {model_path}. Skipping.")
            continue # Skip to the next model
        except Exception as e:
            print(f"  - [ERROR] An unexpected error occurred while processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue # Skip to the next model
            
        finally:
            # MEMORY: Explicitly clean up large objects from this iteration
            del model, X_inference
            if 'X_imputed' in locals(): del X_imputed
            gc.collect()

        # Step E: MEMORY - Append results directly to the file
        if predicted_dosages is not None:
            print(f"  - Appending results to {OUTPUT_FILE}...")
            # Reload the current results, add the new column, and save back
            current_results = pd.read_csv(OUTPUT_FILE, sep='\t', index_col="SampleID")
            current_results[model_name] = predicted_dosages
            current_results.to_csv(OUTPUT_FILE, sep='\t', float_format='%.4f')
            print("  - Done appending.")

    end_time = time.time()
    print("\n--- Pipeline Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Final output file is ready at '{os.path.abspath(OUTPUT_FILE)}'")


if __name__ == "__main__":
    main()
