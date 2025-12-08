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

# --- MODEL SOURCE CONFIGURATION ---
# Set MODEL_SOURCE to "github" or "s3" to switch between remote sources
MODEL_SOURCE = "s3"

_MANIFEST_URLS = {
    "github": "https://api.github.com/repos/SauersML/ferromic/contents/data/models",
    "s3": "https://sharedspace.s3.msi.umn.edu/public_internet/final_imputation_models.manifest.txt",
}

if MODEL_SOURCE not in _MANIFEST_URLS:
    print(f"[WARN] Invalid MODEL_SOURCE '{MODEL_SOURCE}'. Valid options: {list(_MANIFEST_URLS.keys())}")
    print(f"[WARN] Defaulting to 's3'")
    MODEL_SOURCE = "s3"

MODEL_MANIFEST_URL = os.getenv("MODEL_MANIFEST_URL", _MANIFEST_URLS[MODEL_SOURCE])

print(f"[CONFIG] Model source: {MODEL_SOURCE.upper()} → {MODEL_MANIFEST_URL}")

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
        header = pd.read_csv(output_path, sep="\t", index_col=0, nrows=0)
        potential_models = list(dict.fromkeys(c for c in header.columns if c in TARGET_INVERSIONS))
        
        if not potential_models:
            return set()

        nan_counts = {m: 0 for m in potential_models}
        row_count = 0
        
        chunk_iter = pd.read_csv(output_path, sep="\t", index_col=0, usecols=["SampleID"] + potential_models, chunksize=50000)
        
        for chunk in chunk_iter:
            row_count += len(chunk)
            chunk_nans = chunk.isna().sum()
            for m in potential_models:
                nan_counts[m] += chunk_nans[m]
        
        for m in potential_models:
            if row_count == expected_samples and nan_counts[m] == 0:
                valid_models.add(m)
            else:
                print(f"  [WARN] Model {m} is incomplete (Rows: {row_count}/{expected_samples}, NaNs: {nan_counts[m]}). Will re-run.")
                
    except Exception as e:
        print(f"  [WARN] Error reading output file ({e}). Assuming all need re-running.")
        return set()

    return valid_models

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
    mapping = {'eur': 0, 'afr': 1, 'amr': 2, 'eas': 3, 'sas': 4, 'mid': 5, 'oth': 6}
    unknown_code = 7
    
    df['ancestry_pred'] = df['ancestry_pred'].str.lower().str.strip()
    df['code'] = df['ancestry_pred'].map(mapping).fillna(unknown_code).astype(np.int8)
    
    lookup = dict(zip(df['research_id'], df['code']))
    
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
    """
    n_samples, n_snps = X.shape
    sums = np.zeros((n_codes + 1, n_snps), dtype=np.float64)
    counts = np.zeros((n_codes + 1, n_snps), dtype=np.float64)
    
    chunk_size = 50000
    
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        X_chunk = X[start:end]
        anc_chunk = ancestry_indices[start:end]
        valid_mask = (X_chunk != MISSING_VALUE_CODE)
        
        safe_X = np.where(valid_mask, X_chunk, 0).astype(np.float64)
        sums[n_codes] += np.sum(safe_X, axis=0)
        counts[n_codes] += np.sum(valid_mask, axis=0)
        
        for code in range(n_codes):
            row_mask = (anc_chunk == code)
            if not np.any(row_mask):
                continue
            
            X_anc = safe_X[row_mask]
            mask_anc = valid_mask[row_mask]
            
            sums[code] += np.sum(X_anc, axis=0)
            counts[code] += np.sum(mask_anc, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        means = sums / counts
        
    global_mean = means[n_codes]
    global_mean = np.nan_to_num(global_mean, nan=0.0)
    means[n_codes] = global_mean
    
    for code in range(n_codes):
        nan_mask = np.isnan(means[code])
        if np.any(nan_mask):
            means[code, nan_mask] = global_mean[nan_mask]
            
    return means.astype(np.float32)

# --- DIAGNOSTICS & METADATA ---

def load_snp_metadata(model_name: str) -> List[str]:
    """Attempts to load the .snps.json file to get SNP IDs."""
    # Try Model Dir
    p1 = os.path.join(MODEL_DIR, f"{model_name}.snps.json")
    if os.path.exists(p1):
        try:
            with open(p1, 'r') as f:
                return [d.get('id', '?') for d in json.load(f)]
        except:
            pass
            
    # Try Source Dir (if local)
    p2 = os.path.join(MODEL_SOURCE_DIR, f"{model_name}.snps.json")
    if os.path.exists(p2):
        try:
            with open(p2, 'r') as f:
                return [d.get('id', '?') for d in json.load(f)]
        except:
            pass
            
    return []

def log_snp_diagnostics(X, anc_indices, model_name, snp_ids, inv_anc_map):
    """
    Calculates and prints detailed missingness stats to stdout.
    """
    n_samples, n_snps = X.shape
    is_missing = (X == MISSING_VALUE_CODE)
    
    # Global
    missing_counts = np.sum(is_missing, axis=0)
    missing_pcts = 100 * missing_counts / n_samples
    avg_miss = np.mean(missing_pcts)
    
    # Worst SNP
    if n_snps > 0:
        worst_idx = np.argmax(missing_pcts)
        worst_snp = snp_ids[worst_idx] if (snp_ids and len(snp_ids) > worst_idx) else f"Col_{worst_idx}"
        worst_val = missing_pcts[worst_idx]
    else:
        worst_snp = "N/A"
        worst_val = 0.0
    
    lines = []
    lines.append(f"\n[DIAGNOSTICS] Model: {model_name}")
    lines.append(f"  > Global Missingness: {avg_miss:.2f}% (Worst SNP: {worst_snp} @ {worst_val:.1f}%)")
    
    # Per Ancestry
    lines.append(f"  > By Ancestry:")
    # codes 0..6
    sorted_codes = sorted([c for c in inv_anc_map.keys() if isinstance(c, int)])
    
    for code in sorted_codes:
        name = inv_anc_map[code].upper()
        mask = (anc_indices == code)
        count = np.sum(mask)
        if count == 0:
            continue
        
        # Slicing bool array is fast
        sub_missing = is_missing[mask]
        
        # Calculate avg missingness across all SNPs for this ancestry
        total_miss = np.sum(sub_missing)
        total_slots = sub_missing.size
        if total_slots == 0:
             anc_miss_pct = 0.0
        else:
             anc_miss_pct = 100 * total_miss / total_slots
        
        # Check for specific bad SNPs (>10% missing)
        snp_miss_pcts = 100 * np.sum(sub_missing, axis=0) / count
        bad_snps_count = np.sum(snp_miss_pcts > 10.0)
        
        warning = ""
        if bad_snps_count > 0:
            warning = f" [WARNING: {bad_snps_count} SNPs > 10% missing]"
        
        lines.append(f"    - {name:<4} (N={count:<6,}): {anc_miss_pct:.2f}% avg missing{warning}")
    
    # Flush to stdout as one block
    print("\n".join(lines), flush=True)

# --- INFERENCE WORKER LOGIC ---

def _process_model_batched(args):
    """
    Worker function. 
    1. Loads matrix.
    2. *PRINTS DIAGNOSTICS*.
    3. Computes ancestry-specific means.
    4. Imputes.
    5. Predicts.
    """
    model_name, expected_count, ancestry_indices, n_ancestry_codes, inv_anc_map = args
    out_file = os.path.join(TEMP_RESULT_DIR, f"{model_name}.npy")
    
    if os.path.exists(out_file):
        try:
            prev = np.load(out_file)
            if len(prev) == expected_count:
                return {"model": model_name, "status": "skipped_temp_exists"}
        except:
            pass 

    res = {"model": model_name, "status": "ok", "error": None}
    
    try:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.model.joblib")
        clf = joblib.load(model_path)
        
        matrix_path = os.path.join(GENOTYPE_DIR, f"{model_name}.genotypes.npy")
        file_size = os.path.getsize(matrix_path)
        
        if file_size < FULL_LOAD_THRESHOLD:
            X_mmap = np.load(matrix_path, mmap_mode="r")
            X_full = np.array(X_mmap, copy=True)
            del X_mmap
            n_samples, n_snps = X_full.shape
        else:
            X_full = np.load(matrix_path, mmap_mode="r")
            n_samples, n_snps = X_full.shape

        if n_samples != expected_count:
            return {"model": model_name, "status": "error", "error": f"Sample mismatch: {n_samples} vs {expected_count}"}

        # --- DIAGNOSTICS STEP ---
        snp_ids = load_snp_metadata(model_name)
        log_snp_diagnostics(X_full, ancestry_indices, model_name, snp_ids, inv_anc_map)

        # --- STEP 1: Compute Dynamic Priors ---
        ancestry_means = compute_ancestry_means(X_full, ancestry_indices, n_ancestry_codes)
        
        # --- STEP 2: Batched Inference ---
        batch_predictions = []
        
        for i in range(0, n_samples, BATCH_SIZE):
            end = min(i + BATCH_SIZE, n_samples)
            
            X_batch = X_full[i:end].astype(np.float32, copy=True)
            batch_anc = ancestry_indices[i:end]
            
            missing_mask = (X_batch == MISSING_VALUE_CODE)
            
            if np.any(missing_mask):
                fill_values = ancestry_means[batch_anc]
                X_batch[missing_mask] = fill_values[missing_mask]
            
            preds = clf.predict(X_batch)
            batch_predictions.append(preds.astype(np.float32))
            
            del X_batch, missing_mask, preds
        
        full_result = np.concatenate(batch_predictions)
        np.save(out_file, full_result)
        
        del clf, X_full, full_result
        gc.collect()
        
    except Exception as e:
        res["status"] = "error"
        res["error"] = str(e)
        # import traceback
        # traceback.print_exc()
        
    return res

# --- MAIN ORCHESTRATOR ---

def main():
    print("--- Starting Ancestry-Aware, Robust Imputation Pipeline ---")
    
    os.makedirs(TEMP_RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    fam_path = f"{PLINK_PREFIX}.fam"
    if not os.path.exists(fam_path):
        sys.exit(f"FATAL: {fam_path} not found.")
    fam = pd.read_csv(fam_path, sep=r"\s+", header=None, usecols=[1], dtype=str)
    sample_ids = fam[1].tolist()
    n_samples = len(sample_ids)
    print(f"Target Sample Count: {n_samples}")

    # Load Ancestry
    ancestry_indices, anc_map = load_ancestry_map(sample_ids)
    n_ancestry_codes = len(anc_map) 
    
    # Invert map for diagnostics
    inv_anc_map = {v: k for k, v in anc_map.items()}
    # Add Unknown fallback
    inv_anc_map[7] = "unk"

    valid_done_models = verify_existing_output(OUTPUT_FILE, n_samples)
    print(f"Found {len(valid_done_models)} valid models in existing output.")

    avail_files = [f for f in os.listdir(GENOTYPE_DIR) if f.endswith(".genotypes.npy")]
    avail_models = set(f.replace(".genotypes.npy", "") for f in avail_files)
    
    todo_models = [m for m in TARGET_INVERSIONS if m in avail_models and m not in valid_done_models]
    
    print(f"Total Targets: {len(TARGET_INVERSIONS)}")
    print(f"Available Genotypes: {len(avail_models)}")
    print(f"Models Remaining to Compute: {len(todo_models)}")

    if not todo_models:
        print("All models are complete and valid. Nothing to do.")
        return

    _ensure_models_available(todo_models)

    n_workers = min(8, os.cpu_count())
    print(f"Running inference with {n_workers} workers (Batch size: {BATCH_SIZE})...")
    
    # Pack arguments, including the inverted map for diagnostics
    pool_args = [(m, n_samples, ancestry_indices, n_ancestry_codes, inv_anc_map) for m in todo_models]
    
    successful_temp_models = []

    with mp.Pool(n_workers) as pool:
        for res in tqdm(pool.imap_unordered(_process_model_batched, pool_args), total=len(todo_models)):
            if res["status"] == "error":
                print(f"\n[ERROR] Model {res['model']} failed: {res['error']}")
            else:
                successful_temp_models.append(res["model"])

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
