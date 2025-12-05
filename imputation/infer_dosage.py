import os
import sys
import time
import shutil
import glob
import gc
import json
import multiprocessing as mp
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from typing import List, Set, Dict

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
GENOTYPE_DIR = "genotype_matrices"
PLINK_PREFIX = "subset"
OUTPUT_FILE = "imputed_inversion_dosages.tsv"
TEMP_RESULT_DIR = "temp_dosages" 
MISSING_VALUE_CODE = -127
BATCH_SIZE = 10000       # Process 10k samples at a time to keep RAM flat
MEAN_SUBSET_SIZE = 50000 # Use 50k samples to estimate column means (extremely fast)

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

    # Try JSON first (GitHub contents API)
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

    # Fallback to newline-delimited manifest text
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

            print(
                f"[FATAL] Model {model_name} is missing locally and no remote entry exists at {MODEL_MANIFEST_URL}."
            )
            sys.exit(1)

# --- VALIDATION LOGIC ---

def verify_existing_output(output_path: str, expected_samples: int) -> Set[str]:
    """
    Reads the output file in chunks.
    Returns a set of 'Complete' models (valid row count, zero NaNs).
    """
    if not os.path.exists(output_path):
        return set()

    print("--- Verifying existing output file integrity... ---")
    valid_models = set()
    try:
        # Read header only
        header = pd.read_csv(output_path, sep="\t", index_col=0, nrows=0)
        # Deduplicate in case the header unexpectedly contains duplicate model columns
        potential_models = list(dict.fromkeys(c for c in header.columns if c in TARGET_INVERSIONS))
        
        if not potential_models:
            return set()

        # Initialize trackers
        nan_counts = {m: 0 for m in potential_models}
        row_count = 0
        
        # Read in chunks to save RAM
        chunk_iter = pd.read_csv(output_path, sep="\t", index_col=0, usecols=["SampleID"] + potential_models, chunksize=50000)
        
        for chunk in chunk_iter:
            row_count += len(chunk)
            # Count NaNs per column in this chunk
            chunk_nans = chunk.isna().sum()
            for m in potential_models:
                nan_counts[m] += chunk_nans[m]
        
        # Final Check
        for m in potential_models:
            if row_count == expected_samples and nan_counts[m] == 0:
                valid_models.add(m)
            else:
                print(f"  [WARN] Model {m} is incomplete (Rows: {row_count}/{expected_samples}, NaNs: {nan_counts[m]}). Will re-run.")
                
    except Exception as e:
        print(f"  [WARN] Error reading output file ({e}). Assuming all need re-running.")
        return set()

    return valid_models

# --- INFERENCE WORKER LOGIC ---

def _process_model_batched(args):
    """
    Worker function. Loads model, loads matrix via mmap, processes in batches.
    Saves result to temp file. Returns metadata.
    """
    model_name, expected_count = args
    out_file = os.path.join(TEMP_RESULT_DIR, f"{model_name}.npy")
    
    # If temp file exists and is correct size, skip (resume logic)
    if os.path.exists(out_file):
        try:
            prev = np.load(out_file)
            if len(prev) == expected_count:
                return {"model": model_name, "status": "skipped_temp_exists"}
        except:
            pass # corrupted, re-run

    res = {"model": model_name, "status": "ok", "error": None}
    
    try:
        # Load Model
        model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
        clf = joblib.load(model_path)
        pls_model = clf.named_steps.get("pls") if hasattr(clf, "named_steps") else clf
        model_means = getattr(pls_model, "_x_mean", None)
        if model_means is None:
            return {"model": model_name, "status": "error", "error": "Model missing training mean (_x_mean)."}
        model_means = np.asarray(model_means, dtype=np.float32)
        
        # Load Matrix (MMAP - Zero RAM)
        matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")
        X_mmap = np.load(matrix_path, mmap_mode="r")
        n_samples, n_snps = X_mmap.shape

        if model_means.shape[0] != n_snps:
            return {"model": model_name, "status": "error", "error": f"Model mean length {model_means.shape[0]} != genotype columns {n_snps}"}
        
        if n_samples != expected_count:
            return {"model": model_name, "status": "error", "error": f"Sample mismatch: {n_samples} vs {expected_count}"}

        # 1. Batched Inference using training means for imputation
        batch_predictions = []
        
        for i in range(0, n_samples, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_samples)
            
            # Load small chunk into RAM
            X_batch = X_mmap[i:end].astype(np.float32, copy=True)
            
            # Impute (Vectorized on the batch)
            missing_mask = (X_batch == MISSING_VALUE_CODE)
            if np.any(missing_mask):
                # Advanced indexing to fill
                rows, cols = np.where(missing_mask)
                X_batch[rows, cols] = model_means[cols]
            
            # Predict
            preds = clf.predict(X_batch)
            batch_predictions.append(preds.astype(np.float32))
            
            # Free RAM
            del X_batch, missing_mask, preds
        
        # Concatenate and Save
        full_result = np.concatenate(batch_predictions)
        np.save(out_file, full_result)
        
        del clf, X_mmap, full_result
        gc.collect()
        
    except Exception as e:
        res["status"] = "error"
        res["error"] = str(e)
        
    return res

# --- MAIN ORCHESTRATOR ---

def main():
    print("--- Starting Robust, Low-Memory Imputation Pipeline ---")
    
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

    # 3. Check existing Output File
    valid_done_models = verify_existing_output(OUTPUT_FILE, n_samples)
    print(f"Found {len(valid_done_models)} valid models in existing output.")

    # 4. Determine Work List
    avail_files = [f for f in os.listdir(GENOTYPE_DIR) if f.endswith(".genotypes.npy")]
    avail_models = set(f.replace(".genotypes.npy", "") for f in avail_files)
    
    # We want: Targets that exist as genotype files AND are not already valid in output
    todo_models = [m for m in TARGET_INVERSIONS if m in avail_models and m not in valid_done_models]
    
    print(f"Total Targets: {len(TARGET_INVERSIONS)}")
    print(f"Available Genotypes: {len(avail_models)}")
    print(f"Models Remaining to Compute: {len(todo_models)}")

    if not todo_models:
        print("All models are complete and valid. Nothing to do.")
        return

    # 5. Ensure Model Files exist
    _ensure_models_available(todo_models)

    # 6. Run Inference (Multiprocessed)
    # Using 4 workers is usually safe with batching. If 4 crashes, lower to 2.
    n_workers = min(4, os.cpu_count())
    print(f"Running inference with {n_workers} workers (Batch size: {BATCH_SIZE})...")
    
    pool_args = [(m, n_samples) for m in todo_models]
    
    # We track which models successfully created a temp file
    successful_temp_models = []

    with mp.Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_model_batched, pool_args), total=len(todo_models)):
            if res["status"] == "error":
                print(f"\n[ERROR] Model {res['model']} failed: {res['error']}")
            else:
                successful_temp_models.append(res["model"])

    # 7. Merge Phase (Stitching)
    print("--- Stitching results ---")
    
    # Load base dataframe (or create new)
    if os.path.exists(OUTPUT_FILE) and len(valid_done_models) > 0:
        print("Loading existing base file...")
        # Only read index and valid columns to keep it clean
        df = pd.read_csv(OUTPUT_FILE, sep="\t", index_col="SampleID", usecols=["SampleID"] + list(valid_done_models))
        # Enforce correct index order
        df = df.reindex(sample_ids)
    else:
        print("Creating new dataframe...")
        df = pd.DataFrame(index=sample_ids)
        df.index.name = "SampleID"

    # Add new columns from temp files
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

    # 8. Atomic Write
    if newly_added_count > 0:
        print(f"Writing final file with {len(df.columns)} models...")
        tmp_name = OUTPUT_FILE + ".tmp"
        df.to_csv(tmp_name, sep="\t", float_format="%.4f")
        os.replace(tmp_name, OUTPUT_FILE)
        
        # Cleanup
        print("Cleaning up temp files...")
        shutil.rmtree(TEMP_RESULT_DIR)
        print("Done.")
    else:
        print("No new data to merge.")

if __name__ == "__main__":
    main()
