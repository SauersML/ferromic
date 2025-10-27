import os
import sys
import time
import gc
import math
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict

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

# Conservative overhead per parallel worker in bytes to account for model objects,
# Python process overhead, and temporary arrays created by scikit-learn.
PER_WORKER_OVERHEAD_BYTES = 512 * 1024 * 1024

# Safety factor to ensure parallelization happens only when memory is comfortably sufficient.
MEMORY_SAFETY_FACTOR = 2.0

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


def _read_mem_available_bytes() -> Optional[int]:
    """
    Returns available system memory in bytes using /proc/meminfo when present.
    Falls back to psutil if available. Returns None if neither method is available.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    # Value is in kB.
                    return int(parts[1]) * 1024
    except Exception:
        pass
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Writes a DataFrame to `path` atomically by writing to a temporary file, fsyncing,
    and replacing the target path.
    """
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", newline="") as f:
        df.to_csv(f, sep="\t", float_format="%.4f")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def _robust_read_results(path: str, sample_ids: List[str]) -> pd.DataFrame:
    """
    Reads the output TSV robustly, tolerating truncated lines and ensuring correct index ordering and dtype.
    """
    if not os.path.exists(path):
        df = pd.DataFrame(index=sample_ids)
        df.index.name = "SampleID"
        _atomic_to_csv(df, path)
        return df

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            index_col="SampleID",
            dtype={"SampleID": str},
            engine="python",
            on_bad_lines="skip",
        )
    except Exception:
        df = pd.DataFrame(index=sample_ids)
        df.index.name = "SampleID"
        _atomic_to_csv(df, path)
        return df

    # Ensure index alignment and order
    df = df.reindex(sample_ids)
    df.index.name = "SampleID"
    return df


def _ensure_output_initialized(path: str, sample_ids: List[str]) -> pd.DataFrame:
    """
    Ensures the output file exists, has the correct index, and returns the DataFrame view.
    """
    df = _robust_read_results(path, sample_ids)
    expected_index = pd.Index(sample_ids, name="SampleID")
    if len(df.index) != len(expected_index) or not df.index.equals(expected_index):
        df = pd.DataFrame(index=sample_ids)
        df.index.name = "SampleID"
        _atomic_to_csv(df, path)
    return df


def _estimate_job_memory_bytes(matrix_path: str) -> Optional[int]:
    """
    Estimates memory in bytes required by a single worker for a given genotype matrix.
    The major cost is the float32 imputed copy of the memmapped matrix.
    Returns None if the matrix cannot be inspected.
    """
    try:
        x = np.load(matrix_path, mmap_mode="r")
        n_samples, n_snps = x.shape
        # Float32 copy for imputation.
        imputed_bytes = int(n_samples) * int(n_snps) * 4
        # Add conservative overhead.
        return imputed_bytes + PER_WORKER_OVERHEAD_BYTES
    except Exception:
        return None


def _decide_num_workers(pending_models: List[str]) -> int:
    """
    Decides how many workers to use based on empirical available memory and the
    maximum per-job memory footprint across the pending models.
    """
    if not pending_models:
        return 0

    avail = _read_mem_available_bytes()
    if avail is None:
        # Without a reliable memory reading, disable parallelization.
        return 1

    max_job_bytes = 0
    for m in pending_models:
        matrix_path = os.path.join(GENOTYPE_DIR, f"{m}.genotypes.npy")
        est = _estimate_job_memory_bytes(matrix_path)
        if est is None:
            # If any matrix cannot be inspected, be conservative and go sequential.
            return 1
        max_job_bytes = max(max_job_bytes, est)

    if max_job_bytes <= 0:
        return 1

    safe_avail = int(avail / MEMORY_SAFETY_FACTOR)
    if safe_avail < max_job_bytes:
        return 1

    max_concurrency_by_mem = max(1, safe_avail // max_job_bytes)
    max_concurrency_by_cpu = max(1, os.cpu_count() or 1)
    return int(max(1, min(max_concurrency_by_mem, max_concurrency_by_cpu)))


def _impute_and_predict(model_name: str, expected_sample_count: int) -> Dict[str, object]:
    """
    Worker function. Loads the model and genotype matrix, imputes missing values with column means,
    and runs prediction. Returns a dictionary with keys:
      - "model_name": str
      - "dosages": np.ndarray (float32) on success
      - "error": str on failure
      - "skipped": bool when skipped due to validation
      - "n_samples": int
      - "n_snps": int
      - "missing_count": int
    """
    result: Dict[str, object] = {
        "model_name": model_name,
        "skipped": False,
        "error": "",
        "dosages": None,
        "n_samples": 0,
        "n_snps": 0,
        "missing_count": 0,
    }

    model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
    matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")

    predicted_dosages: Optional[np.ndarray] = None

    try:
        model = joblib.load(model_path)
        X_inference = np.load(matrix_path, mmap_mode="r")
        n_samples, n_snps = X_inference.shape
        result["n_samples"] = int(n_samples)
        result["n_snps"] = int(n_snps)

        if n_samples != expected_sample_count:
            result["skipped"] = True
            result["error"] = f"Sample count mismatch. Expected {expected_sample_count}, found {n_samples}."
            return result

        if X_inference.size == 0:
            result["skipped"] = True
            result["error"] = "Empty genotype matrix."
            return result

        missing_count = int(np.sum(X_inference == MISSING_VALUE_CODE))
        result["missing_count"] = missing_count

        # Imputation with column means on a float32 copy.
        X_imputed = X_inference.astype(np.float32, copy=True)
        for j in range(n_snps):
            column_data = X_imputed[:, j]
            valid_mask = column_data != MISSING_VALUE_CODE
            if np.any(valid_mask):
                col_mean = float(np.mean(column_data[valid_mask]))
            else:
                col_mean = 1.0
            column_data[~valid_mask] = col_mean

        predicted_dosages = model.predict(X_imputed)
        # Normalize dtype to float32 to reduce memory in parent process.
        if isinstance(predicted_dosages, np.ndarray) and predicted_dosages.dtype != np.float32:
            predicted_dosages = predicted_dosages.astype(np.float32, copy=False)

        result["dosages"] = predicted_dosages

    except FileNotFoundError as e:
        result["error"] = f"File not found: {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"
    finally:
        try:
            del predicted_dosages
        except Exception:
            pass
        try:
            del X_inference  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            del X_imputed  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            del model  # type: ignore[name-defined]
        except Exception:
            pass
        gc.collect()

    return result


def main() -> None:
    """
    Launches the imputation pipeline with robust, restartable, and crash-safe behavior.
    Parallelization is enabled only after confirming that available memory comfortably supports it.
    Writes to the output file are atomic and performed from the main process only.
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
        fam_df = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[1], dtype=str)
        sample_ids: List[str] = fam_df[1].tolist()
        print(f"Successfully loaded {len(sample_ids)} sample IDs.")
    except Exception as e:
        print(f"[FATAL] Could not read sample IDs from .fam file. Error: {e}")
        sys.exit(1)

    # --- 3. Identify and Filter Models to Process ---
    try:
        all_available_models = sorted(
            [
                f.replace(".genotypes.npy", "")
                for f in os.listdir(GENOTYPE_DIR)
                if f.endswith(".genotypes.npy")
            ]
        )
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

    # --- 4. Initialize or recover output file, and determine already completed models ---
    print(f"Initializing output file: {OUTPUT_FILE}")
    current_results = _ensure_output_initialized(OUTPUT_FILE, sample_ids)

    completed_models = [c for c in current_results.columns if c in TARGET_INVERSIONS]
    if completed_models:
        before = len(models_to_process)
        models_to_process = [m for m in models_to_process if m not in completed_models]
        print(f"Resuming; {len(completed_models)} models already in output. {before - len(models_to_process)} will be skipped.")
    print(f"{len(models_to_process)} models remain to process.")

    if not models_to_process:
        end_time = time.time()
        print("\n--- Pipeline Complete ---")
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        print(f"Final output file is ready at '{os.path.abspath(OUTPUT_FILE)}'")
        return

    # --- 5. Decide parallelization based on empirical memory availability ---
    num_workers = _decide_num_workers(models_to_process)
    if num_workers <= 1:
        print("Parallelization disabled due to memory constraints or insufficient information. Running sequentially.")
    else:
        print(f"Parallelization enabled with {num_workers} workers based on available memory.")

    # --- 6. Execute imputation + prediction ---
    # Results are written by the main process only, atomically, after each model finishes,
    # enabling safe restarts. When parallel, worker processes only compute and return dosages.
    pending = models_to_process

    if num_workers <= 1:
        iterator = tqdm(pending, desc="Predicting Inversions", unit="model")
        for model_name in iterator:
            print(f"\n--- Processing: {model_name} ---")
            res = _impute_and_predict(model_name, expected_sample_count=len(sample_ids))

            if res.get("skipped", False):
                print(f"  - Skipped: {res.get('error')}")
                continue
            if res.get("error"):
                print(f"  - [ERROR] {res.get('error')}")
                continue

            dosages = res["dosages"]
            if not isinstance(dosages, np.ndarray):
                print("  - [ERROR] Invalid dosages returned.")
                continue

            # Robust read to tolerate any prior partial write.
            current_results = _robust_read_results(OUTPUT_FILE, sample_ids)

            if model_name in current_results.columns:
                print("  - Column already present; skipping write.")
                continue

            current_results[model_name] = pd.Series(dosages, index=sample_ids, dtype=np.float32)
            _atomic_to_csv(current_results, OUTPUT_FILE)
            print(f"  - Appended results for {model_name}.")
            gc.collect()
    else:
        # Stream results as they complete to keep memory bounded and ensure frequent durable progress.
        with mp.Pool(processes=num_workers) as pool:
            for res in tqdm(pool.imap_unordered(
                func=_impute_and_predict,
                iterable=[(m, len(sample_ids)) for m in pending],
            ), total=len(pending), desc="Predicting Inversions", unit="model"):
                # The worker signature expects two arguments; wrap to unpack when using pool.imap_unordered.
                if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], str):
                    # Compatibility guard if pool passes tuples through unchanged.
                    res = _impute_and_predict(res[0], expected_sample_count=res[1])

                model_name = res.get("model_name", "")
                if not model_name:
                    print("  - [ERROR] Missing model name in result.")
                    continue

                print(f"\n--- Processing: {model_name} ---")
                if res.get("skipped", False):
                    print(f"  - Skipped: {res.get('error')}")
                    continue
                if res.get("error"):
                    print(f"  - [ERROR] {res.get('error')}")
                    continue

                dosages = res["dosages"]
                if not isinstance(dosages, np.ndarray):
                    print("  - [ERROR] Invalid dosages returned.")
                    continue

                current_results = _robust_read_results(OUTPUT_FILE, sample_ids)
                if model_name in current_results.columns:
                    print("  - Column already present; skipping write.")
                    continue

                current_results[model_name] = pd.Series(dosages, index=sample_ids, dtype=np.float32)
                _atomic_to_csv(current_results, OUTPUT_FILE)
                print(f"  - Appended results for {model_name}.")
                gc.collect()

    end_time = time.time()
    print("\n--- Pipeline Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print(f"Final output file is ready at '{os.path.abspath(OUTPUT_FILE)}'")


if __name__ == "__main__":
    main()
