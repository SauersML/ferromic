"""
Batch extraction of best tagging SNPs with BH-corrected q-values.

This script filters inversion regions from ``data/inv_properties.tsv`` where
``0_single_1_recur_consensus`` is 0 or 1, selects the strongest tagging SNP per
region using the existing tagging SNP utilities, collects their selection
P-values (``P_X``) and coefficients (``S``), and applies Benjamini–Hochberg
correction across the set.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure repository root is importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_best_tagging_snp import (
    TaggingSNPResult,
    ensure_selection_data,
    load_tagging_snps,
    sanitize_region,
    select_top_tags,
)
import scripts.extract_best_tagging_snp as ebts

# Patching module-level constants to redirect inputs/outputs to data/
ebts.OUTPUT_DIR = Path("data")
ebts.SELECTION_DIR = ebts.OUTPUT_DIR / "selection_data"
ebts.SELECTION_TSV_PATH = ebts.SELECTION_DIR / ebts.SELECTION_TSV_NAME


DEFAULT_INV_PATH = Path("data/inv_properties.tsv")
DEFAULT_TAGGING_PATHS = (
    Path("data/tagging_snps.tsv"),
    Path("data/tagging-snps-report/tagging_snps.tsv"),
    Path("tagging_snps.tsv"),
)


def log(message: str) -> None:
    print(message, flush=True)


@dataclass
class RegionRecord:
    region: str
    consensus: float
    best: Optional[TaggingSNPResult] = None
    p_x: Optional[float] = None
    s: Optional[float] = None
    reasons: list[str] = field(default_factory=list)


def find_tagging_table() -> Path:
    for candidate in DEFAULT_TAGGING_PATHS:
        if candidate.exists():
            log(f"[tagging] Found tagging SNPs at {candidate}")
            return candidate

    raise FileNotFoundError(
        "tagging_snps.tsv not found. Provide --tagging-snps or place the file in data/."
    )


def bh_qvalues(pvals: pd.Series) -> pd.Series:
    """Benjamini–Hochberg FDR correction on a Series of p-values."""
    arr = pd.to_numeric(pvals, errors="coerce").to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)

    valid_idx = np.where(~np.isnan(arr))[0]
    if valid_idx.size == 0:
        return pd.Series(out, index=pvals.index)

    p_valid = arr[valid_idx]
    order = np.argsort(p_valid)
    ranks = np.arange(1, len(p_valid) + 1)
    bh = p_valid[order] * len(p_valid) / ranks
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.minimum(bh, 1.0)

    out_valid = np.empty_like(p_valid)
    out_valid[order] = bh
    out[valid_idx] = out_valid
    return pd.Series(out, index=pvals.index)


def iter_regions(inv_path: Path) -> list[tuple[str, float]]:
    log(f"[regions] Loading inversion properties from {inv_path}")
    df = pd.read_csv(inv_path, sep="\t")
    log(f"[regions] Loaded {len(df)} rows total")
    mask = df["0_single_1_recur_consensus"].isin([0, 1])
    filtered = df[mask]
    log(f"[regions] Filtered to {len(filtered)} rows with consensus in {{0,1}}")
    regions: list[tuple[str, float]] = []
    for _, row in filtered.iterrows():
        chrom = str(row["Chromosome"])
        start = int(row["Start"])
        end = int(row["End"])
        consensus = float(row["0_single_1_recur_consensus"])
        regions.append((f"{chrom}:{start}-{end}", consensus))
    return regions


def compute_best_tags(tag_df: pd.DataFrame, regions: list[tuple[str, float]], workers: int) -> list[RegionRecord]:
    records: list[RegionRecord] = []
    
    def _process(region_consensus: tuple[str, float]) -> RegionRecord:
        region, consensus = region_consensus
        # log(f"[process] Start {region}") # reduce verbosity
        try:
            top_results, _ = select_top_tags(region, tag_df, top_n=1)
            if not top_results:
                return RegionRecord(region=region, consensus=consensus, reasons=["No tagging SNPs found"])
            best = top_results[0]
            return RegionRecord(region=region, consensus=consensus, best=best)
        except Exception as exc:
            log(f"[warn] Failed {region}: {exc}")
            return RegionRecord(region=region, consensus=consensus, reasons=[f"Error: {exc}"])

    log(f"[process] Using {workers} worker threads for top-tag selection")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_process, rc): rc[0] for rc in regions}
        for future in concurrent.futures.as_completed(future_map):
            try:
                records.append(future.result())
            except Exception as exc:
                # Should be caught inside _process, but just in case
                reg_txt = future_map[future]
                log(f"[error] Unexpected failure for {reg_txt}: {exc}")
                
    return records


def load_selection_subset(keys_df: pd.DataFrame, selection_path: Path, *, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream the selection file and keep only rows matching requested keys."""
    if keys_df.empty:
        return pd.DataFrame(columns=["CHROM_norm", "POS", "P_X", "S"])

    log(f"[selection] Streaming selection file {selection_path} in chunks of {chunksize}")
    matches: list[pd.DataFrame] = []
    key_cols = ["chrom_norm", "position_hg37"]
    total_rows = 0
    found = 0
    for idx, chunk in enumerate(
        pd.read_csv(
            selection_path,
            sep="\t",
            comment="#",
            usecols=["CHROM", "POS", "P_X", "S"],
            chunksize=chunksize,
        )
    ):
        total_rows += len(chunk)
        chunk["CHROM_norm"] = chunk["CHROM"].astype(str).str.removeprefix("chr").str.removesuffix(".0")
        merged = chunk.merge(
            keys_df,
            left_on=["CHROM_norm", "POS"],
            right_on=key_cols,
            how="inner",
        )
        if not merged.empty:
            found += len(merged)
            matches.append(merged[["CHROM_norm", "POS", "P_X", "S"]])
        if (idx + 1) % 5 == 0:
            log(f"[selection] Processed ~{total_rows:,} rows; found {found} matches so far")

    if matches:
        out = pd.concat(matches, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["CHROM_norm", "POS", "P_X", "S"])

    out = out.drop_duplicates(subset=["CHROM_norm", "POS"], keep="first")
    log(f"[selection] Finished streaming; matched {len(out)} rows for {len(keys_df)} requested positions")
    return out


def process_regions(inv_path: Path, tagging_path: Path, *, workers: Optional[int] = None) -> pd.DataFrame:
    log("[load] Reading tagging SNP table")
    tag_df = load_tagging_snps(tagging_path)
    log(f"[load] Tagging SNPs loaded: {len(tag_df)} rows")

    regions = iter_regions(inv_path)
    if not regions:
        raise RuntimeError("No regions found with consensus 0 or 1")

    workers = workers or min(8, os.cpu_count() or 4)
    records = compute_best_tags(tag_df, regions, workers)
    if not records:
        log("[error] No regions processed successfully; exiting")
        return pd.DataFrame()

    # Annotate exclusion reasons
    for r in records:
        if r.best is None:
            # Already has a reason (e.g. "No tagging data found")
            continue
        
        # 1. Haplotype counts
        try:
            n_dir = float(r.best.row.get("direct_group_size", 0))
            n_inv = float(r.best.row.get("inverted_group_size", 0))
            if n_dir < 3 or n_inv < 3:
                r.reasons.append("Low haplotype count (<3)")
        except (ValueError, TypeError):
             r.reasons.append("Malformed group size data")

        # 2. r^2
        r2 = r.best.abs_correlation ** 2
        if r2 < 0.5:
            r.reasons.append("Low r^2 (<0.5)")
        
        # 3. hg37 coord
        if r.best.position_hg37 is None:
            r.reasons.append("No hg37 coordinate")

    # Build lookup keys for selection stats (even for excluded ones, if they have coords)
    key_rows = []
    for r in records:
        if r.best is None or r.best.position_hg37 is None:
            continue
        key_rows.append(
            {
                "chrom_norm": str(r.best.chromosome_hg37).lstrip("chr").removesuffix(".0"),
                "position_hg37": int(r.best.position_hg37),
            }
        )
    
    keys_df = pd.DataFrame(key_rows).drop_duplicates()
    log(f"[selection] Need selection stats for {len(keys_df)} unique hg37 positions")

    # Ensure selection file exists, then stream only needed rows
    selection_path = ensure_selection_data()
    
    # --- DIAGNOSTICS ---
    try:
        size_mb = selection_path.stat().st_size / (1024 * 1024)
        log(f"[debug] Selection file: {selection_path} ({size_mb:.2f} MB)")
        with open(selection_path, "r") as f:
            head = [next(f).strip() for _ in range(5)]
        log(f"[debug] Selection file head: {head}")
        log(f"[debug] Keys DF head:\n{keys_df[['chrom_norm', 'position_hg37']].head().to_string()}")
    except Exception as e:
        log(f"[debug] Diagnostic failed: {e}")
    # -------------------

    subset = load_selection_subset(keys_df, selection_path)
    lookup = {
        (row["CHROM_norm"], int(row["POS"])): (row.get("P_X"), row.get("S"))
        for _, row in subset.iterrows()
    }
    log(f"[debug] Lookup table size: {len(lookup)}")
    if len(lookup) > 0:
        log(f"[debug] Lookup sample keys: {list(lookup.keys())[:5]}")

    # Attach selection values
    lookup_failures_logged = 0
    for r in records:
        if r.best is None or r.best.position_hg37 is None:
            continue
            
        # Fix: Ensure key matches the normalized format used in keys_df (no .0 suffix)
        key = (
            str(r.best.chromosome_hg37).lstrip("chr").removesuffix(".0"),
            int(r.best.position_hg37),
        )
        vals = lookup.get(key)
        
        if vals is None:
            # Expanded diagnostics for missing keys
            if lookup_failures_logged < 10:
                log(f"[debug] Lookup failed for key: {key}")
                # Check if pos exists with different chrom
                candidates = [k for k in lookup if k[1] == key[1]]
                if candidates:
                    log(f"[debug]   -> Position {key[1]} found with other chroms: {candidates}")
                else:
                    log(f"[debug]   -> Position {key[1]} not found in lookup table at all.")
                lookup_failures_logged += 1
                
            r.reasons.append(f"Selection stats missing for key {key}")
            continue
            
        p_x_val, s_val = vals
        try:
            r.p_x = float(p_x_val)
        except Exception:
            pass # Remains None
            
        try:
            r.s = float(s_val)
        except Exception:
            pass # Remains None
            
        if r.p_x is None:
             r.reasons.append("Selection stats missing (P_X is NaN)")


    order_map = {region: idx for idx, (region, _) in enumerate(regions)}
    
    # Construct DataFrame
    data_list = []
    for r in records:
        row_dict = {
            "region": r.region,
            "sanitized_region": sanitize_region(r.region),
            "consensus": r.consensus,
            "p_x": r.p_x,
            "S": r.s,
            "exclusion_reasons": "; ".join(sorted(set(r.reasons))) if r.reasons else ""
        }
        if r.best:
            row_dict.update({
                "inversion_region": r.best.inversion_region,
                "correlation_r": r.best.correlation,
                "abs_r": r.best.abs_correlation,
                "chromosome_hg37": r.best.chromosome_hg37,
                "position_hg37": r.best.position_hg37,
            })
        else:
            # Fill missing fields with None or NaN
            row_dict.update({
                "inversion_region": None,
                "correlation_r": None,
                "abs_r": None,
                "chromosome_hg37": None,
                "position_hg37": None,
            })
        data_list.append(row_dict)

    df = pd.DataFrame(data_list)
    df["order"] = df["region"].map(order_map)
    df = df.sort_values("order").drop(columns=["order"])
    
    # Calculate Q-values ONLY for records with NO exclusion reasons
    pass_mask = df["exclusion_reasons"] == ""
    log(f"[fdr] Applying BH correction to {pass_mask.sum()} passing regions out of {len(df)}")
    
    df.loc[pass_mask, "q_value"] = bh_qvalues(df.loc[pass_mask, "p_x"])
    # Ensure q_value is NaN where not passed (it should be by default if column init was NaN, but let's be safe)
    # If column didn't exist, the assignment created it with NaNs elsewhere.
    
    return df


def main() -> int:
    log("[cli] Starting batch best-tagging SNP extraction with default settings")
    tagging_path = find_tagging_table()
    df = process_regions(DEFAULT_INV_PATH, tagging_path, workers=None)
    out_path = Path("data/best_tagging_snps_qvalues.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    if not df.empty:
        print(df[["region", "S", "p_x", "q_value", "exclusion_reasons"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
