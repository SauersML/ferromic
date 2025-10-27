import os
import json
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.lib.format import open_memmap

# Fast PLINK reader (Rust backend)
# pip install bed_reader
from bed_reader import open_bed

# Remote fetch
import requests
from urllib.parse import urlparse

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PLINK_PREFIX = os.getenv("PLINK_PREFIX", "subset")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "genotype_matrices")

BLOCK_SNPS   = int(os.getenv("BLOCK_SNPS", "4096"))   # SNP columns to read per block
NUM_THREADS  = os.getenv("NUM_THREADS", "auto")      # "auto" or integer string

# Remote manifest of models/SNPs JSON
MANIFEST_URL = os.getenv(
    "MANIFEST_URL",
    "https://sharedspace.s3.msi.umn.edu/public_internet/final_imputation_models.manifest.txt",
)
# ------------------------------------------------------------

# Write int8 missing in PLINK .bed
MISSING_INT8 = np.int8(-127)

# A small record for the models we keep
ModelSpec = namedtuple("ModelSpec", ["name", "ncols", "col_ids", "col_effects"])

def parse_num_threads(val: str) -> Optional[int]:
    if val is None or val == "" or val.lower() == "auto":
        return None
    try:
        n = int(val)
        return n if n > 0 else None
    except Exception:
        return None

def chrom_aliases(ch: str) -> Iterable[str]:
    """Return chromosome aliases: with/without 'chr', numeric/sex synonyms."""
    s = str(ch).strip()
    if s == "":
        return [s]

    if s.lower().startswith("chr"):
        base = s[3:]
        prefixed = s
    else:
        base = s
        prefixed = "chr" + s

    base_up = base.upper()
    out = {base, prefixed}

    if base_up in {"X", "23"}:
        out.update({"X", "23", "chrX", "chr23"})
    elif base_up in {"Y", "24"}:
        out.update({"Y", "24", "chrY", "chr24"})
    elif base_up in {"XY", "25"}:
        out.update({"XY", "25", "chrXY", "chr25"})
    elif base_up in {"MT", "M", "26"}:
        out.update({"MT", "M", "26", "chrMT", "chrM", "chr26"})
    else:
        try:
            n = int(base_up)
            out.update({str(n), "chr" + str(n)})
        except ValueError:
            out.update({base_up, "chr" + base_up})
    return out

def load_bed_meta(prefix: str):
    bed_path = prefix + ".bed"
    # Open once and keep open for all reads
    bed = open_bed(bed_path)

    n_samples, n_snps = bed.shape
    chrom = np.asarray(bed.chromosome, dtype=str)
    pos   = np.asarray(bed.bp_position, dtype=np.int64)
    a1    = np.asarray(bed.allele_1, dtype=str)  # dosage counts A1 (default)
    a2    = np.asarray(bed.allele_2, dtype=str)

    # Uppercase alleles for robust matching
    a1 = np.char.upper(a1)
    a2 = np.char.upper(a2)

    print(f"BED: samples={n_samples:,}, variants={n_snps:,}")
    print(f"BLOCK_SNPS={BLOCK_SNPS}, NUM_THREADS={NUM_THREADS}")

    # Build ID -> list of indices (fixes multi-allelic duplicates)
    id_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for v in range(n_snps):
        c = chrom[v]
        p = pos[v]
        for c_alias in chrom_aliases(c):
            key = f"{c_alias}:{p}"
            id_to_idxs[key].append(v)

    return bed, n_samples, n_snps, chrom, pos, a1, a2, id_to_idxs

def _list_snps_json_urls_from_manifest(manifest_url: str) -> List[str]:
    """Fetch manifest and return all .snps.json URLs, sorted by basename for stable order."""
    r = requests.get(manifest_url, timeout=60)
    text = r.text
    urls = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith(".snps.json"):
            urls.append(s)
    # Stable order by basename
    urls_sorted = sorted(urls, key=lambda u: os.path.basename(urlparse(u).path))
    print(f"Found {len(urls_sorted)} model files in manifest.")
    return urls_sorted

def load_models_and_build_router_from_manifest(manifest_url: str,
                                               id_to_idxs: Dict[str, List[int]],
                                               a1: np.ndarray,
                                               a2: np.ndarray) -> Tuple[List[ModelSpec], Dict[int, List[Tuple[int,int,bool]]], int, int]:
    """
    Returns:
      - models: list of ModelSpec
      - routes_for_variant: dict v -> list of (model_index, dest_col, flip)
      - skipped_models: count with fetch/schema/JSON problems
      - total_requested_cols: sum of cols across usable models
    """
    urls = _list_snps_json_urls_from_manifest(manifest_url)

    models: List[ModelSpec] = []
    skipped = 0
    total_cols = 0

    # Temporary per-model selected mapping: col_j -> (v_idx, flip) or (None, False) if missing
    selected_per_model: List[List[Tuple[Optional[int], bool]]] = []

    for url in tqdm(urls, desc="Indexing models", unit="model", leave=False):
        base = os.path.basename(urlparse(url).path)
        if not base.endswith(".snps.json"):
            continue
        name = base[:-10]

        try:
            resp = requests.get(url, timeout=120)
            rows = resp.json()
            df = pd.DataFrame(rows)
        except Exception as e:
            print(f"[WARN] skip {name}: cannot fetch/parse JSON ({e})")
            skipped += 1
            continue

        # Validate schema
        if not {"id", "effect_allele"}.issubset(df.columns):
            print(f"[WARN] skip {name}: requires columns ['id','effect_allele']")
            skipped += 1
            continue

        # Normalize
        df["id"] = df["id"].astype(str).str.strip()
        df["effect_allele"] = df["effect_allele"].astype(str).str.strip().str.upper()
        col_ids = df["id"].tolist()
        col_eff = df["effect_allele"].tolist()

        ncols = len(col_ids)
        total_cols += ncols

        # Resolve each column to a BIM index with duplicate-aware matching
        chosen: List[Tuple[Optional[int], bool]] = []
        for id_str, eff in zip(col_ids, col_eff):
            cand_idxs: List[int] = []
            try:
                chrom_str, pos_str = id_str.split(":")
            except ValueError:
                print(f"[WARN] {name}: malformed id '{id_str}'")
                chosen.append((None, False))
                continue

            for alias in chrom_aliases(chrom_str):
                key = f"{alias}:{pos_str}"
                lst = id_to_idxs.get(key)
                if lst:
                    cand_idxs.extend(lst)

            if not cand_idxs:
                chosen.append((None, False))
                continue

            a1_matches = [v for v in cand_idxs if a1[v] == eff]
            a2_matches = [v for v in cand_idxs if a2[v] == eff]

            if a1_matches:
                v_idx = a1_matches[0]
                if len(a1_matches) > 1:
                    print(f"[WARN ambiguous] {name}: multiple A1 matches for '{id_str}'='{eff}', picked index {v_idx}")
                chosen.append((v_idx, False))
            elif a2_matches:
                v_idx = a2_matches[0]
                if len(a2_matches) > 1:
                    print(f"[WARN ambiguous] {name}: multiple A2 matches for '{id_str}'='{eff}', picked index {v_idx}")
                chosen.append((v_idx, True))
            else:
                chosen.append((None, False))
                v0 = cand_idxs[0]
                print(f"[WARN] {name}: effect '{eff}' not in {{A1='{a1[v0]}', A2='{a2[v0]}'}} at '{id_str}'")

        models.append(ModelSpec(name=name, ncols=ncols, col_ids=col_ids, col_effects=col_eff))
        selected_per_model.append(chosen)

    usable_models = len(models)
    if skipped:
        print(f"[INFO] skipped {skipped} model files (fetch/schema/JSON problems)")
    print(f"Usable models: {usable_models}")
    print(f"Total requested columns (across usable models): {total_cols:,}")

    # Build global router: v -> list of (model_idx, dest_col, flip)
    routes_for_variant: Dict[int, List[Tuple[int, int, bool]]] = defaultdict(list)
    for m_idx, chosen in enumerate(selected_per_model):
        for j, (v_idx, flip) in enumerate(chosen):
            if v_idx is not None:
                routes_for_variant[v_idx].append((m_idx, j, flip))

    return models, routes_for_variant, skipped, total_cols

def safe_flip_int8(col: np.ndarray) -> np.ndarray:
    """Flip counts (2 - x) where x>=0; keep missing (-127) as-is."""
    out = col.copy()
    mask = out >= 0
    out[mask] = np.int8(2 - out[mask].astype(np.int16))  # widen then back to avoid overflow
    return out

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- BED metadata & ID index (duplicate-aware) ----
    bed, n_samples, n_snps, chrom, pos, a1, a2, id_to_idxs = load_bed_meta(PLINK_PREFIX)

    # ---- Models & router (REMOTE manifest) ----
    models, routes_for_variant, skipped, total_cols = load_models_and_build_router_from_manifest(
        MANIFEST_URL, id_to_idxs, a1, a2
    )
    if not models:
        print("No usable models. Exiting.")
        return

    # Unique SNP indices we actually need to read
    needed_variants = sorted(routes_for_variant.keys())
    print(f"Unique SNPs to read (drives I/O): {len(needed_variants):,}")

    # ---- Prepare outputs (memmaps prefilled with missing) ----
    model_memmaps: List[np.memmap] = []
    for ms in models:
        out_path = os.path.join(OUTPUT_DIR, f"{ms.name}.genotypes.npy")
        mm = open_memmap(out_path, mode="w+", dtype=np.int8, shape=(n_samples, ms.ncols), fortran_order=True)
        mm[:] = MISSING_INT8  # prefill as missing
        model_memmaps.append(mm)

    # ---- Stream the BED once in SNP blocks ----
    num_threads = parse_num_threads(NUM_THREADS)
    needed_set = set(needed_variants)
    processed_needed = 0

    pbar = tqdm(total=len(needed_variants), desc="Scanning BED", unit="snp", dynamic_ncols=True)

    for start in range(0, n_snps, BLOCK_SNPS):
        end = min(start + BLOCK_SNPS, n_snps)
        block_width = end - start

        # Read block: rows=samples, cols=SNPs in [start:end)
        block = bed.read(index=np.s_[:, start:end], dtype="int8", order="F", num_threads=num_threads)

        # Determine which local columns are needed
        needed_local: List[int] = [j for j in range(block_width) if (start + j) in needed_set]
        if not needed_local:
            continue

        # Precompute flip need per column
        flip_need = {}
        for j in needed_local:
            v_idx = start + j
            flip_needed = any(flip for (_m, _c, flip) in routes_for_variant[v_idx])
            flip_need[j] = flip_needed

        # Route columns
        for j in needed_local:
            v_idx = start + j
            assignments = routes_for_variant[v_idx]

            col = block[:, j]  # int8 column
            col_flip = None
            if flip_need[j]:
                col_flip = safe_flip_int8(col)

            for (m_idx, dest_col, flip) in assignments:
                if flip:
                    model_memmaps[m_idx][:, dest_col] = col_flip
                else:
                    model_memmaps[m_idx][:, dest_col] = col

            processed_needed += 1
            pbar.update(1)

    pbar.close()

    # ---- Finalize ----
    for mm in model_memmaps:
        mm.flush()

    print("Done.")

if __name__ == "__main__":
    main()
