import os
import sys
import shutil
import glob
import gc
import json
import multiprocessing as mp
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import List, Set, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_DIR = "impute"
_DEFAULT_MODEL_SOURCE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "models")
)
MODEL_SOURCE_DIR = os.getenv(
    "MODEL_SOURCE_DIR",
    _DEFAULT_MODEL_SOURCE if os.path.isdir(_DEFAULT_MODEL_SOURCE) else "",
)
MODEL_MANIFEST_URL = os.getenv(
    "MODEL_MANIFEST_URL",
    "https://api.github.com/repos/SauersML/ferromic/contents/data/models",
)
ANCESTRY_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
GENOTYPE_DIR = "genotype_matrices"
PLINK_PREFIX = "subset"
OUTPUT_FILE = "imputed_inversion_dosages.tsv"
TEMP_RESULT_DIR = "temp_dosages" 
MISSING_VALUE_CODE = -127
BATCH_SIZE = 10000       # Process 10k samples at a time
FULL_LOAD_THRESHOLD = 250 * 1024 * 1024  # If matrix < 250MB, load fully into RAM for speed

# The specific inversions we want
TARGET_INVERSIONS = {
    "chr3-195680867-INV-272256", "chr3-195749464-INV-230745", "chr6-76111919-INV-44661",
    "chr12-46897663-INV-16289", "chr6-141867315-INV-29159", "chr3-131969892-INV-7927",
    "chr6-167181003-INV-209976", "chr11-71571191-INV-6980", "chr9-102565835-INV-4446",
    "chr4-33098029-INV-7075", "chr7-57835189-INV-284465", "chr10-46135869-INV-77646",
    "chr11-24263185-INV-392", "chr13-79822252-INV-17591", "chr1-60775308-INV-5023",
    "chr6-130527042-INV-4267", "chr13-48199211-INV-7451", "chr21-13992018-INV-65632",
    "chr8-7301025-INV-5297356", "chr9-30951702-INV-5595", "chr17-45585160-INV-706887",
    "chr12-131333944-INV-289865", "chr7-70955928-INV-18020", "chr16-28471894-INV-165758",
    "chr7-65219158-INV-312667", "chr10-79542902-INV-674513", "chr1-13084312-INV-62181",
    "chr10-37102555-INV-11157", "chr4-40233409-INV-2010", "chr2-138246733-INV-5010",
}

# --- UTILITIES ---

def _download_url_to_path(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = f"{dest_path}.tmp"
    req = Request(url, headers={"User-Agent": "ferromic-infer/1.0"})
    with urlopen(req, timeout=300) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp_path, dest_path)

def _load_model_manifest(url: str) -> Dict[str, str]:
    if not url:
        return {}
    try:
        req = Request(url, headers={"User-Agent": "ferromic-infer/1.0"})
        with urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except Exception as exc:
        print(f"[FATAL] Unable to fetch model manifest: {exc}")
        sys.exit(1)
    mapping: Dict[str, str] = {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        for entry in payload:
            name = entry.get("name", "")
            download_url = entry.get("download_url")
            if name.endswith(".model.joblib") and download_url:
                mapping[name[:-13]] = download_url
        if mapping:
            return mapping

    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if s and s.endswith(".model.joblib"):
            model_name = os.path.basename(urlparse(s).path)[:-13]
            mapping[model_name] = s
    return mapping

def _build_local_model_manifest(source_dir: str) -> Dict[str, str]:
    if not source_dir or not os.path.isdir(source_dir):
        return {}
    mapping: Dict[str, str] = {}
    for path in glob.glob(os.path.join(source_dir, "*.model.joblib")):
        model_name = os.path.basename(path)[:-13]
        mapping[model_name] = path
    return mapping

def _ensure_models_available(models: List[str]) -> None:
    missing = [m for m in models if not os.path.exists(os.path.join(MODEL_DIR, f"{m}.model.joblib"))]
    if missing:
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_manifest = _build_local_model_manifest(MODEL_SOURCE_DIR)
        remote_manifest: Dict[str, str] = {}
        source_desc = "GitHub" if MODEL_MANIFEST_URL else "local"
        print(f"Ensuring {len(missing)} models from {source_desc} data/models...")

        for model_name in missing:
            dest_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
            source_path = local_manifest.get(model_name)
            if source_path and os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                continue

            if MODEL_MANIFEST_URL:
                if not remote_manifest:
                    remote_manifest = _load_model_manifest(MODEL_MANIFEST_URL)
                url = remote_manifest.get(model_name)
                if url:
                    _download_url_to_path(url, dest_path)
                    continue

            print(f"[FATAL] Model {model_name} is missing locally and no remote entry exists.")
            sys.exit(1)

# --- ANCESTRY LOGIC ---

def load_ancestry_map(sample_ids: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Downloads ancestry TSV, maps strings to ints, and aligns to the sample_ids order.
    Returns:
        ancestry_indices: np.array (int8) of shape (N_samples,)
        code_map: Dict mapping 'eur' -> 0, etc.
    """
    print(f"Loading Ancestry Metadata from {ANCESTRY_URI}...")
    storage_opts = {"project": os.environ.get("GOOGLE_PROJECT"), "requester_pays": True}
    
    try:
        # Load only necessary columns
        df = pd.read_csv(
            ANCESTRY_URI, 
            sep="\t", 
            usecols=["research_id", "ancestry_pred"], 
            dtype=str,
            storage_options=storage_opts
        )
    except Exception as e:
        sys.exit(f"[FATAL] Failed to load ancestry data: {e}")

    # Create Integer Mapping
    # Standard 1000G superpopulations + others found in AoU
    # We assign codes 0-6. Code 7 will be 'Unknown/Global'.
    mapping = {'eur': 0, 'afr': 1, 'amr': 2, 'eas': 3, 'sas': 4, 'mid': 5, 'oth': 6}
    unknown_code = 7
    
    # Map DataFrame
    # Normalize strings just in case
    df['ancestry_pred'] = df['ancestry_pred'].str.lower().str.strip()
    df['code'] = df['ancestry_pred'].map(mapping).fillna(unknown_code).astype(np.int8)
    
    # Create Lookup Dictionary
    lookup = dict(zip(df['research_id'], df['code']))
    
    # Align to provided sample_ids (from FAM file)
    print("Aligning ancestry to genotype samples...")
    aligned_codes = []
    missing_count = 0
    
    for sid in sample_ids:
        if sid in lookup:
            aligned_codes.append(lookup[sid])
        else:
            aligned_codes.append(unknown_code)
            missing_count += 1
            
    if missing_count > 0:
        print(f"  [WARN] {missing_count} samples in FAM file missing from Ancestry TSV. Assigned 'Unknown' ({unknown_code}).")
        
    return np.array(aligned_codes, dtype=np.int8), mapping

def compute_ancestry_means(X: np.ndarray, 
                          ancestry_indices: np.ndarray, 
                          n_codes: int) -> np.ndarray:
    """
    Vectorized calculation of mean dosage per ancestry group.
    
    Args:
        X: Genotype matrix (samples x snps), includes -127 missing.
        ancestry_indices: Array of ancestry codes for each sample.
        n_codes: Number of valid ancestry codes (0..6). Code 7 is global fallback.
    
    Returns:
        mean_matrix: (n_codes + 1, n_snps) array. 
                     Rows 0-6 are specific means. 
                     Row 7 is the Global Mean (all valid data).
    """
    n_samples, n_snps = X.shape
    # +1 for the Global/Unknown row
    sums = np.zeros((n_codes + 1, n_snps), dtype=np.float64)
    counts = np.zeros((n_codes + 1, n_snps), dtype=np.float64)
    
    # Determine chunking strategy to avoid massive RAM spikes during boolean masking
    # if X is huge. Since we passed 'X' which might be in RAM or mmap.
    chunk_size = 50000
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        
        # Slicing mmap returns array, efficient
        X_chunk = X[start:end]
        anc_chunk = ancestry_indices[start:end]
        
        # Identify valid data (not missing)
        valid_mask = (X_chunk != MISSING_VALUE_CODE)
        
        # 1. Accumulate Global Stats (Row 7)
        # Sum of valid entries in this chunk per column
        # np.where(valid_mask, X_chunk, 0) replaces -127 with 0 for summation safety
        safe_X = np.where(valid_mask, X_chunk, 0).astype(np.float64)
        
        # Add to global totals
        sums[n_codes] += np.sum(safe_X, axis=0)
        counts[n_codes] += np.sum(valid_mask, axis=0)
        
        # 2. Accumulate Per-Ancestry Stats
        for code in range(n_codes):
            # Which rows in this chunk belong to this ancestry?
            # Creating mask (chunk_size,)
            row_mask = (anc_chunk == code)
            
            # Combine with valid data mask
            # valid_mask[row_mask] selects only valid entries for this ancestry
            # But we need to be careful with broadcasting.
            
            # Efficient approach: Filter X_chunk first
            if not np.any(row_mask):
                continue
                
            # Subset of safe_X for this ancestry
            X_anc = safe_X[row_mask]
            mask_anc = valid_mask[row_mask]
            
            sums[code] += np.sum(X_anc, axis=0)
            counts[code] += np.sum(mask_anc, axis=0)

    # Calculate Means
    with np.errstate(divide='ignore', invalid='ignore'):
        means = sums / counts
        
    # Handling Edge Cases
    
    # 1. The Global Vector (Row 7)
    global_mean = means[n_codes]
    # If global count is 0 for a SNP (entire column missing), default to 0 dosage (reference)
    global_mean = np.nan_to_num(global_mean, nan=0.0)
    means[n_codes] = global_mean
    
    # 2. Specific Ancestries
    # If a specific ancestry is NaN (no data for that SNP), fill with Global Mean
    for code in range(n_codes):
        # Find indices where this ancestry has no data
        nan_mask = np.isnan(means[code])
        if np.any(nan_mask):
            means[code, nan_mask] = global_mean[nan_mask]
            
    return means.astype(np.float32)

# --- INFERENCE WORKER LOGIC ---

def _process_model_batched(args):
    """
    Worker function. 
    1. Loads matrix (RAM optimization if small).
    2. Computes ancestry-specific means dynamically from the input data.
    3. Imputes missing values using these means.
    4. Predicts.
    """
    model_name, expected_count, ancestry_indices, n_ancestry_codes = args
    out_file = os.path.join(TEMP_RESULT_DIR, f"{model_name}.npy")
    
    # Resume logic
    if os.path.exists(out_file):
        try:
            prev = np.load(out_file)
            if len(prev) == expected_count:
                return {"model": model_name, "status": "skipped_temp_exists"}
        except:
            pass 

    res = {"model": model_name, "status": "ok", "error": None}
    
    try:
        # Load Model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
        clf = joblib.load(model_path)
        
        # Load Matrix
        matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")
        
        # Check size for RAM optimization
        file_size = os.path.getsize(matrix_path)
        
        if file_size < FULL_LOAD_THRESHOLD:
            # FAST PATH: Load fully into RAM
            # Using mmap first to load, then array() to copy to RAM
            X_mmap = np.load(matrix_path, mmap_mode="r")
            X_full = np.array(X_mmap, copy=True)
            del X_mmap # Close file handle
            
            n_samples, n_snps = X_full.shape
        else:
            # SLOW PATH: Keep on disk
            X_full = np.load(matrix_path, mmap_mode="r")
            n_samples, n_snps = X_full.shape

        if n_samples != expected_count:
            return {"model": model_name, "status": "error", "error": f"Sample mismatch: {n_samples} vs {expected_count}"}

        # --- STEP 1: Compute Dynamic Priors ---
        # We ignore the saved model mean to fix the "flipped allele" bug.
        # We calculate means from X_full itself, stratified by ancestry.
        
        ancestry_means = compute_ancestry_means(X_full, ancestry_indices, n_ancestry_codes)
        # ancestry_means shape: (8, n_snps), where row 7 is Global/Unknown
        
        # --- STEP 2: Batched Inference ---
        batch_predictions = []
        
        for i in range(0, n_samples, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_samples)
            
            # Ensure we have a RAM copy for this batch to modify
            # If X_full is already RAM, this is a slice copy. 
            # If X_full is mmap, this reads from disk.
            X_batch = X_full[i:end].astype(np.float32, copy=True)
            batch_anc = ancestry_indices[i:end]
            
            # Find missing values
            missing_mask = (X_batch == MISSING_VALUE_CODE)
            
            if np.any(missing_mask):
                # Ancestry-Aware Imputation
                # Construct a matrix of fill values matching the batch shape
                # batch_anc contains codes 0..7. 
                # ancestry_means has rows 0..7.
                # Broadcasting magic:
                fill_values = ancestry_means[batch_anc] # Shape: (batch_size, n_snps)
                
                # Apply fill
                X_batch[missing_mask] = fill_values[missing_mask]
            
            # Predict
            preds = clf.predict(X_batch)
            batch_predictions.append(preds.astype(np.float32))
            
            del X_batch, missing_mask, preds
        
        # Concatenate and Save
        full_result = np.concatenate(batch_predictions)
        np.save(out_file, full_result)
        
        del clf, X_full, full_result
        gc.collect()
        
    except Exception as e:
        res["status"] = "error"
        res["error"] = str(e)
        import traceback
        traceback.print_exc()
        
    return res

# --- MAIN ORCHESTRATOR ---

def main():
    print("--- Starting Ancestry-Aware, Robust Imputation Pipeline ---")
    
    # 1. Setup Dirs
    os.makedirs(TEMP_RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Load Sample List
    fam_path = f"{PLINK_PREFIX}.fam"
    if not os.path.exists(fam_path):
        sys.exit(f"FATAL: {fam_path} not found.")
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[1], dtype=str)
    sample_ids = fam[1].tolist()
    n_samples = len(sample_ids)
    print(f"Target Sample Count: {n_samples}")

    # 3. Load Ancestry Data (Global Setup)
    # This returns an int8 array aligned to sample_ids, and a map dict
    ancestry_indices, anc_map = load_ancestry_map(sample_ids)
    # Number of specific codes (excluding unknown). 
    # 'unknown' is mapped to len(anc_map) which is 7.
    # The codes in ancestry_indices range from 0 to 7.
    # Our compute_means function expects n_codes=7 (0..6) + 1 fallback.
    n_ancestry_codes = len(anc_map) 

    # 4. Check existing Output File
    valid_done_models = verify_existing_output(OUTPUT_FILE, n_samples)
    print(f"Found {len(valid_done_models)} valid models in existing output.")

    # 5. Determine Work List
    avail_files = [f for f in os.listdir(GENOTYPE_DIR) if f.endswith(".genotypes.npy")]
    avail_models = set(f.replace(".genotypes.npy", "") for f in avail_files)
    
    todo_models = [m for m in TARGET_INVERSIONS if m in avail_models and m not in valid_done_models]
    
    print(f"Total Targets: {len(TARGET_INVERSIONS)}")
    print(f"Available Genotypes: {len(avail_models)}")
    print(f"Models Remaining to Compute: {len(todo_models)}")

    if not todo_models:
        print("All models are complete and valid. Nothing to do.")
        return

    # 6. Ensure Model Files exist
    _ensure_models_available(todo_models)

    # 7. Run Inference (Multiprocessed)
    n_workers = min(8, os.cpu_count())
    print(f"Running inference with {n_workers} workers (Batch size: {BATCH_SIZE})...")
    
    # Pass ancestry array to workers. 
    # It is read-only and relatively small (~400KB), so pickling overhead is negligible.
    pool_args = [(m, n_samples, ancestry_indices, n_ancestry_codes) for m in todo_models]
    
    successful_temp_models = []

    with mp.Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_model_batched, pool_args), total=len(todo_models)):
            if res["status"] == "error":
                print(f"\n[ERROR] Model {res['model']} failed: {res['error']}")
            else:
                successful_temp_models.append(res["model"])

    # 8. Merge Phase
    print("--- Stitching results ---")
    
    if os.path.exists(OUTPUT_FILE) and len(valid_done_models) > 0:
        print("Loading existing base file...")
        df = pd.read_csv(OUTPUT_FILE, sep="\t", index_col="SampleID", usecols=["SampleID"] + list(valid_done_models))
        df = df.reindex(sample_ids)
    else:
        print("Creating new dataframe...")
        df = pd.DataFrame(index=sample_ids)
        df.index.name = "SampleID"

    newly_added_count = 0
    for m in successful_temp_models:
        npy_path = os.path.join(TEMP_RESULT_DIR, f"{m}.npy")
        if not os.path.exists(npy_path):
            continue
            
        try:
            data = np.load(npy_path)
            if len(data) != n_samples:
                print(f"Skipping merge for {m}: length mismatch")
                continue
                
            df[m] = data
            newly_added_count += 1
        except Exception as e:
            print(f"Error merging {m}: {e}")

    # 9. Atomic Write
    if newly_added_count > 0:
        print(f"Writing final file with {len(df.columns)} models...")
        tmp_name = OUTPUT_FILE + ".tmp"
        df.to_csv(tmp_name, sep="\t", float_format="%.4f")
        os.replace(tmp_name, OUTPUT_FILE)
        
        print("Cleaning up temp files...")
        shutil.rmtree(TEMP_RESULT_DIR)
        print("Done.")
    else:
        print("No new data to merge.")

if __name__ == "__main__":
    main()
