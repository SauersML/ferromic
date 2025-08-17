import os
import sys
import json
import time
import numpy as np
import pandas as pd

import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from pandas_plink import read_plink1_bin
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
PLINK_PREFIX = "subset"
MODEL_DIR = "impute"
OUTPUT_DIR = "genotype_matrices"

# Tuning knobs (override with env vars if needed)
VARIANT_CHUNK = int(os.getenv("VARIANT_CHUNK", "100000"))           # columns per dask chunk
ARRAY_CHUNK_SIZE_HINT = os.getenv("DASK_ARRAY_CHUNK_SIZE", "256 MiB")
PERSIST_BED = bool(int(os.getenv("PERSIST_BED", "0")))              # 1 to persist bed in RAM
PRINT_BIM_HEAD = int(os.getenv("PRINT_BIM_HEAD", "3"))              # show first N BIM rows


# ============================================================
# DEBUG HELPERS
# ============================================================
def dbg(msg):
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def show_chunks(da_arr, name="array"):
    try:
        ch = da_arr.chunks
        dbg(f"{name}: ndim={da_arr.ndim}, shape={da_arr.shape}, dtype={da_arr.dtype}")
        if ch is not None and isinstance(ch, tuple):
            for ax, c in enumerate(ch):
                sizes = list(c[:5])
                more = " ..." if len(c) > 5 else ""
                dbg(f"{name}: axis {ax} chunk_sizes (first 5): {sizes}{more}")
        else:
            dbg(f"{name}: chunks=<unknown or scalar>")
    except Exception as e:
        dbg(f"show_chunks({name}) failed: {e}")


def meminfo():
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        head = "".join(lines[:5]).strip()
        dbg("Meminfo (first lines):\n" + head)
    except Exception as e:
        dbg(f"meminfo failed: {e}")


# ============================================================
# CHROMOSOME NORMALIZATION
# ============================================================
def chrom_aliases(ch: str):
    """
    Return a set of equivalent chromosome name aliases for robust matching.
    Handles:
      - with/without 'chr' prefix (chr1 <-> 1)
      - X <-> 23, Y <-> 24, XY <-> 25, MT/M <-> 26
    """
    s = str(ch).strip()
    if s == "":
        return {s}

    # strip 'chr' prefix if present
    if s.lower().startswith("chr"):
        base = s[3:]
        prefixed = s
    else:
        base = s
        prefixed = "chr" + s

    base_up = base.upper()

    # start with both prefixed/non-prefixed forms we observed
    aliases = {base, prefixed}

    # Map sex/mito numeric codes used by some tools
    if base_up in {"X", "23"}:
        aliases.update({"X", "23", "chrX", "chr23"})
    elif base_up in {"Y", "24"}:
        aliases.update({"Y", "24", "chrY", "chr24"})
    elif base_up in {"XY", "25"}:
        aliases.update({"XY", "25", "chrXY", "chr25"})
    elif base_up in {"MT", "M", "26"}:
        aliases.update({"MT", "M", "26", "chrMT", "chrM", "chr26"})
    else:
        # autosomes: normalize possible zero-padded numbers (e.g., '01')
        try:
            n = int(base_up)
            aliases.update({str(n), "chr" + str(n)})
        except ValueError:
            # non-numeric weirdness: just ensure both prefixed and base forms
            aliases.update({base_up, "chr" + base_up})

    return aliases


# ============================================================
# CORE BUILDERS
# ============================================================
def build_bim_and_mappings(G):
    """
    Build:
      - bim DataFrame with: chrom,pos,a0,a1,i,id (id is the PLINK-provided form)
      - key_to_i: mapping "id|allele" for MANY aliases (chr/nochr and sex/mito synonyms)
      - a0_arr, a1_arr: numpy arrays aligned by variant index
    """
    dbg("Building BIM and lookup mappings...")

    chrom = np.asarray(G.chrom.values).astype(str)
    pos = np.asarray(G.pos.values)

    if not np.issubdtype(pos.dtype, np.integer):
        dbg(f"Converting pos dtype {pos.dtype} -> int64")
        pos = pos.astype(np.int64)

    a0 = np.asarray(G.a0.values).astype(str)
    a1 = np.asarray(G.a1.values).astype(str)

    n_variants = int(G.shape[1])
    idx = np.arange(n_variants, dtype=np.int64)

    bim = pd.DataFrame({
        "chrom": chrom,
        "pos": pos,
        "a0": a0,
        "a1": a1,
        "i": idx
    })
    bim["id"] = bim["chrom"] + ":" + bim["pos"].astype(str)

    if PRINT_BIM_HEAD > 0:
        dbg("BIM head:\n" + bim.head(PRINT_BIM_HEAD).to_string(index=False))

    # Build a Python dict for speed and to control first-wins deduping
    key_to_i_dict = {}

    # Create keys for BOTH a0 and a1 for ALL reasonable ID aliases
    t0 = time.perf_counter()
    for row in bim.itertuples(index=False):
        ch = row.chrom
        p = row.pos
        i = int(row.i)
        a0v = row.a0
        a1v = row.a1

        for ch_alias in chrom_aliases(ch):
            base_id = f"{ch_alias}:{p}"
            k0 = f"{base_id}|{a0v}"
            k1 = f"{base_id}|{a1v}"

            if k0 not in key_to_i_dict:
                key_to_i_dict[k0] = i
            if k1 not in key_to_i_dict:
                key_to_i_dict[k1] = i
    t1 = time.perf_counter()

    key_to_i = pd.Series(key_to_i_dict)
    dbg(f"BIM rows: {len(bim)}")
    dbg(f"key_to_i unique keys: {len(key_to_i)}  (built in {t1 - t0:.2f}s)")

    a0_arr = bim["a0"].to_numpy()
    a1_arr = bim["a1"].to_numpy()
    return bim, key_to_i, a0_arr, a1_arr


def normalize_model_df(model_df, model_name):
    """Ensure required columns exist and normalize strings."""
    if "id" not in model_df.columns:
        raise ValueError(f"Model '{model_name}' missing 'id'.")
    if "effect_allele" not in model_df.columns:
        raise ValueError(f"Model '{model_name}' missing 'effect_allele'.")

    model_df = model_df.set_index("id", drop=True)
    model_df.index = model_df.index.astype(str).str.strip()
    model_df["effect_allele"] = model_df["effect_allele"].astype(str).str.strip()
    return model_df


def indices_for_model(model_df: pd.DataFrame, key_to_i: pd.Series, model_name: str) -> np.ndarray:
    """
    Resolve variant column indices using composite key 'id|effect_allele',
    trying multiple chromosome aliases where needed.
    """
    # Fast path: try direct map first
    keys_direct = pd.Series(
        model_df.index.astype(str) + "|" + model_df["effect_allele"].astype(str).to_numpy(),
        index=model_df.index
    )
    idx_direct = keys_direct.map(key_to_i)

    if not idx_direct.isna().any():
        idx_np = idx_direct.to_numpy(dtype=np.int64)
        dbg(f"[{model_name}] resolved via direct id match: {len(idx_np)} indices")
        return idx_np

    # Slow path: for missing entries, try aliasing the chromosome name
    dbg(f"[{model_name}] some IDs did not match directly; trying chromosome aliases...")
    resolved = []
    misses = []

    # Pre-split model ids into chrom and pos (assume 'chrom:pos')
    # Be robust to weird inputs by skipping malformed ids
    for id_str, eff_allele in zip(model_df.index.tolist(), model_df["effect_allele"].tolist()):
        try:
            chrom_str, pos_str = id_str.split(":")
        except ValueError:
            misses.append(f"{id_str}|{eff_allele}")
            continue

        found = False
        for ch_alias in chrom_aliases(chrom_str):
            key = f"{ch_alias}:{pos_str}|{eff_allele}"
            if key in key_to_i.index:
                resolved.append(key_to_i.loc[key])
                found = True
                break
        if not found:
            misses.append(f"{id_str}|{eff_allele}")

    if misses:
        dbg(f"[{model_name}] missing after aliasing: {len(misses)} / {len(model_df)}. "
            f"Examples: {misses[:10]}")
        raise ValueError(
            f"Model '{model_name}': {len(misses)} SNPs not found even after aliasing. "
            f"Examples: {misses[:10]}"
        )

    idx_np = np.asarray(resolved, dtype=np.int64)
    dbg(f"[{model_name}] resolved via aliasing: {len(idx_np)} indices")
    return idx_np


def compute_model_matrix(
    bed_da: da.core.Array,
    idx: np.ndarray,
    model_effect_alleles: np.ndarray,
    a0_arr: np.ndarray,
    a1_arr: np.ndarray,
    model_name: str
) -> da.core.Array:
    """
    Build a lazy dask array for the model with allele flips applied (bed counts a1).
    """
    dbg(f"[{model_name}] selecting columns: min={idx.min()} max={idx.max()} n={len(idx)}")
    sel = bed_da[:, idx].astype(np.int8)
    show_chunks(sel, name=f"sel_preflip[{model_name}]")

    a0_sel = a0_arr[idx]
    a1_sel = a1_arr[idx]
    eff = model_effect_alleles.astype(str)

    flip_mask = (eff == a0_sel)
    valid_mask = flip_mask | (eff == a1_sel)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        bad_pos = np.where(~valid_mask)[0][:5]
        dbg(f"[{model_name}] INVALID effect alleles at positions {bad_pos.tolist()}")
        raise ValueError(f"Model '{model_name}': {n_invalid} SNPs have effect_allele not in (a0, a1).")

    mask_da = da.from_array(flip_mask, chunks=(sel.chunks[1]))
    sel = da.where(mask_da[None, :], 2 - sel, sel).astype(np.int8)
    show_chunks(sel, name=f"sel_postflip[{model_name}]")
    return sel


# ============================================================
# MAIN
# ============================================================
def main():
    print("--- Genotype Matrix Preparation ---", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dask.config.set({"array.chunk-size": ARRAY_CHUNK_SIZE_HINT})
    dbg(f"Dask array chunk-size hint: {ARRAY_CHUNK_SIZE_HINT}")
    dbg(f"VARIANT_CHUNK: {VARIANT_CHUNK}")
    dbg(f"Python: {sys.version.split()[0]}  PID: {os.getpid()}")
    dbg(f"CPU count: {os.cpu_count()}")
    meminfo()

    # Load PLINK xarray/dask
    print(f"Loading PLINK fileset: '{PLINK_PREFIX}.*'", flush=True)
    G = read_plink1_bin(f"{PLINK_PREFIX}.bed", verbose=True)

    # Lazy dask array (do NOT use .values)
    bed = G.data

    # Rechunk variants axis for fast column slicing
    dbg("Rechunking genotype array along variant axis...")
    bed = bed.rechunk({1: VARIANT_CHUNK})
    show_chunks(bed, name="bed")

    if PERSIST_BED:
        dbg("Persisting bed array in RAM (optional).")
        with ProgressBar():
            bed = bed.persist()
        show_chunks(bed, name="bed_persisted")

    n_samples, n_variants = G.shape
    print(f"Loaded metadata for {n_samples:,} samples and {n_variants:,} variants.", flush=True)

    # BIM + mappings (with aliases)
    bim, key_to_i, a0_arr, a1_arr = build_bim_and_mappings(G)

    # Discover models
    model_json_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".snps.json")])
    if not model_json_files:
        raise FileNotFoundError(f"No '.snps.json' model files found in '{MODEL_DIR}'.")
    print(f"Found {len(model_json_files)} models.", flush=True)

    # Sequential to avoid I/O thrash on one disk
    for json_filename in tqdm(model_json_files, desc="Models", unit="model"):
        model_name = json_filename[:-10]  # strip .snps.json
        json_path = os.path.join(MODEL_DIR, json_filename)
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}.genotypes.npy")

        dbg(f"[{model_name}] reading model file: {json_path}")
        with open(json_path, "r") as f:
            model_snps_list = json.load(f)

        model_df = pd.DataFrame(model_snps_list)
        dbg(f"[{model_name}] raw rows: {len(model_df)}")
        model_df = normalize_model_df(model_df, model_name)
        dbg(f"[{model_name}] normalized rows: {len(model_df)}")
        dbg(f"[{model_name}] head:\n{model_df.head(5).to_string()}")

        # Resolve indices (try direct, then aliasing)
        idx = indices_for_model(model_df, key_to_i, model_name)

        # Build lazy computation (with flip)
        dbg(f"[{model_name}] building dask graph for compute")
        sel_da = compute_model_matrix(
            bed_da=bed,
            idx=idx,
            model_effect_alleles=model_df["effect_allele"].to_numpy(),
            a0_arr=a0_arr,
            a1_arr=a1_arr,
            model_name=model_name
        )

        # Compute with visible progress bar
        start = time.perf_counter()
        dbg(f"[{model_name}] compute() start; target shape: [samples x {len(idx)}]")
        with ProgressBar():
            arr = sel_da.compute()
        elapsed = time.perf_counter() - start
        dbg(f"[{model_name}] compute() done in {elapsed:.2f}s; result shape={arr.shape}, dtype={arr.dtype}")

        # Save
        np.save(output_path, arr)
        dbg(f"[{model_name}] saved -> {output_path}")

    print("--- Process Complete ---", flush=True)
    print(f"Saved genotype matrices in './{OUTPUT_DIR}/'.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        dbg(f"FATAL: {type(e).__name__}: {e}")
        raise
