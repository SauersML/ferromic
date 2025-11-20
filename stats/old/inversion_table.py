#!/usr/bin/env python3
from __future__ import annotations
import sys, math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# ====================== Paths ======================
INV_FILE   = Path("inv_info.tsv")          # required
SUMMARY    = Path("output.csv")            # required
OUT_TSV    = Path("per_inversion_raw.tsv") # output (one row per inversion)

# ====================== Logging ======================
def dbg(msg: str) -> None:
    print(f"[DEBUG] {msg}", file=sys.stdout, flush=True)

def warn(msg: str) -> None:
    print(f"[WARN]  {msg}", file=sys.stdout, flush=True)

def err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ====================== Helpers ======================
def norm_chr(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("chr_"): s = s[4:]
    elif s.startswith("chr"): s = s[3:]
    return f"chr{s}"

def region_id(chr_: str, start: int, end: int) -> str:
    return f"{chr_}:{start}-{end}"

def parse_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def parse_freq(x) -> float:
    """
    Parse frequency values from CSV/TSV into floats in [0,1].
    Accepts '0.37', '37%', 'NA', ''.
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return math.nan
    s = str(x).strip()
    if s == "" or s.lower() == "na":
        return math.nan
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    except Exception:
        return math.nan

# ====================== Loaders ======================
def load_inversions(inv_path: Path) -> pd.DataFrame:
    if not inv_path.is_file():
        raise FileNotFoundError(f"Required inversion file not found: {inv_path}")
    dbg("Loading inversion mapping from inv_info.tsv ...")
    df = pd.read_csv(inv_path, sep=None, engine="python", dtype=str)
    dbg(f"inv_info.tsv columns: {list(df.columns)}")

    need = ["Chromosome", "Start", "End", "0_single_1_recur_consensus", "Inverted_AF"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"inv_info.tsv missing required columns: {miss}")

    out = pd.DataFrame({
        "chr": df["Chromosome"].map(norm_chr),
        "start": df["Start"].map(parse_int),
        "end": df["End"].map(parse_int),
        "_cat": pd.to_numeric(df["0_single_1_recur_consensus"], errors="coerce"),
        "Inverted_AF": df["Inverted_AF"].map(parse_freq),
        "OrigID": df.get("OrigID", pd.Series([None]*len(df), dtype="string")).astype("string"),
    })

    before = len(out)
    out = out[out["chr"].notna() & out["start"].notna() & out["end"].notna()]
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)
    dropped = before - len(out)

    def lab(v):
        if pd.isna(v): return "uncategorized"
        return "recurrent" if int(v) == 1 else ("single-event" if int(v) == 0 else "uncategorized")
    out["recurrence"] = out["_cat"].map(lab)

    # Strict duplicate detection on exact coordinates
    dup_counts = out.groupby(["chr","start","end"]).size()
    dups = dup_counts[dup_counts > 1]
    if len(dups) > 0:
        examples = out.merge(dups.rename("n"), on=["chr","start","end"])
        err("Duplicate exact inversion coordinates detected in inv_info.tsv (this is a hard error).")
        err("Examples of duplicates:\n" + str(examples.head(10)))
        raise RuntimeError("Duplicate exact inversion coordinates in inversion table.")

    # Filter to categorized rows
    cts_all = out["recurrence"].value_counts().to_dict()
    out = out[out["recurrence"].isin(["recurrent","single-event"])].copy()
    dropped_uncat = cts_all.get("uncategorized", 0)
    dbg(
        f"Inversions loaded (valid rows): {len(out)} (dropped {dropped}; excluded {dropped_uncat} uncategorized); "
        f"Recurrent={cts_all.get('recurrent',0)}, Single-event={cts_all.get('single-event',0)}"
    )

    # Inversion ID: prefer OrigID if present, otherwise region id
    out["inversion_id"] = out.apply(
        lambda r: (r["OrigID"].strip() if isinstance(r["OrigID"], str) and r["OrigID"].strip()
                   else region_id(r["chr"], r["start"], r["end"])),
        axis=1
    )
    return out[["chr","start","end","recurrence","Inverted_AF","inversion_id"]]

def load_summary(sum_path: Path) -> pd.DataFrame:
    if not sum_path.is_file():
        raise FileNotFoundError(f"Required summary file not found: {sum_path}")
    dbg("Loading per-region summary from output.csv ...")
    df = pd.read_csv(sum_path, dtype=str)
    dbg(f"output.csv columns: {list(df.columns)}")

    need = ["chr","region_start","region_end",
            "0_pi_filtered","1_pi_filtered",
            "hudson_fst_hap_group_0v1",
            "inversion_freq_no_filter","inversion_freq_filter"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"output.csv missing required columns: {miss}")

    out = pd.DataFrame({
        "chr": df["chr"].map(norm_chr),
        "start": pd.to_numeric(df["region_start"], errors="coerce"),
        "end": pd.to_numeric(df["region_end"], errors="coerce"),
        "pi_direct_filtered": pd.to_numeric(df["0_pi_filtered"], errors="coerce"),
        "pi_inverted_filtered": pd.to_numeric(df["1_pi_filtered"], errors="coerce"),
        "hudson_fst_region": pd.to_numeric(df["hudson_fst_hap_group_0v1"], errors="coerce"),
        "inversion_freq_no_filter": df["inversion_freq_no_filter"].map(parse_freq),
        "inversion_freq_filter": df["inversion_freq_filter"].map(parse_freq),
    })

    before = len(out)
    out = out[out["chr"].notna() & out["start"].notna() & out["end"].notna()]
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)
    kept = len(out)
    dbg(f"output.csv rows retained: {kept} (dropped {before-kept} with missing keys)")
    dbg("First 3 normalized rows from output.csv:\n" + str(out[["chr","start","end"]].head(3).to_string(index=False)))
    return out

# ====================== Matching (STRICT; ±1 on SUMMARY side) ======================
def match_summary_to_each_inversion(inv_df: pd.DataFrame, sum_df: pd.DataFrame) -> pd.DataFrame:
    dbg("Building summary index and matching each inversion (±1 bp on summary side) ...")

    # Fast lookup for exact keys
    sum_index: Dict[Tuple[str,int,int], int] = {}
    for i, r in enumerate(sum_df.itertuples(index=False)):
        sum_index[(r.chr, int(r.start), int(r.end))] = i
    dbg(f"Summary exact-key index size: {len(sum_index)}")

    # For each inversion, look for 9 candidate (±1,±1) keys in summary
    rows = []
    for inv in inv_df.itertuples(index=False):
        c, s, e = inv.chr, int(inv.start), int(inv.end)
        candidates: List[int] = []
        for ds in (-1, 0, 1):
            for de in (-1, 0, 1):
                idx = sum_index.get((c, s + ds, e + de))
                if idx is not None:
                    candidates.append(idx)
        if len(candidates) == 0:
            err(f"No summary match found for inversion {region_id(c,s,e)} under STRICT ±1 rule.")
            raise RuntimeError("Missing summary row for an inversion.")
        if len(candidates) > 1:
            sample = "\n".join(
                f"  - summary={sum_df.iloc[idx]['chr']}:{sum_df.iloc[idx]['start']}-{sum_df.iloc[idx]['end']}"
                for idx in candidates[:10]
            )
            err(f"{region_id(c,s,e)} matched multiple summary rows (hard error):\n{sample}")
            raise RuntimeError("Ambiguous inversion→summary match (>1 candidate).")

        rr = sum_df.iloc[candidates[0]]
        rows.append({
            "chr": c,
            "start": s,
            "end": e,
            "pi_direct_filtered": rr["pi_direct_filtered"],
            "pi_inverted_filtered": rr["pi_inverted_filtered"],
            "hudson_fst_region": rr["hudson_fst_region"],
            "inversion_freq_no_filter": rr["inversion_freq_no_filter"],
            "inversion_freq_filter": rr["inversion_freq_filter"],
        })

    matched = pd.DataFrame(rows)
    dbg(f"Matched inversions → summary rows: {len(matched)} (expected {len(inv_df)})")
    return matched

# ====================== Correlations & Mismatches ======================
def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").astype(float)
    b = pd.to_numeric(y, errors="coerce").astype(float)
    m = a.notna() & b.notna()
    if m.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])

def equal_exact(a, b) -> bool:
    """
    Exact match test per request:
    - If both are NaN -> equal.
    - If both parse as floats -> require exact numeric equality.
    - Else compare trimmed strings for exact equality.
    """
    a_is_nan = isinstance(a, float) and math.isnan(a)
    b_is_nan = isinstance(b, float) and math.isnan(b)
    if a_is_nan and b_is_nan:
        return True
    try:
        fa = float(a)
        fb = float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return fa == fb
    except Exception:
        sa = str(a).strip()
        sb = str(b).strip()
        return sa == sb

# ====================== Main ======================
def main():
    inv = load_inversions(INV_FILE)
    summ = load_summary(SUMMARY)

    matched_sum = match_summary_to_each_inversion(inv, summ)

    # Join to per-inversion table
    out = inv.merge(matched_sum, on=["chr","start","end"], how="inner", validate="one_to_one")
    if len(out) != len(inv):
        err(f"Joined rows = {len(out)}, expected {len(inv)} from inv_info.tsv")
        raise RuntimeError("Join did not yield 1 row per inversion.")

    # Build human-readable TSV (capitalize first word; no abbreviations except ID)
    out["Region ID"] = out.apply(lambda r: region_id(r["chr"], int(r["start"]), int(r["end"])), axis=1)
    out_hr = pd.DataFrame({
        "Inversion ID": out["inversion_id"],
        "Chromosome": out["chr"],
        "Start position": out["start"],
        "End position": out["end"],
        "Recurrence status": out["recurrence"],
        "Inversion allele frequency": out["Inverted_AF"],
        "Nucleotide diversity (direct haplotypes)": out["pi_direct_filtered"],
        "Nucleotide diversity (inverted haplotypes)": out["pi_inverted_filtered"],
        "Hudson FST": out["hudson_fst_region"],
        "Region ID": out["Region ID"],
    })

    # Write TSV (EXCLUDES the two inversion_freq_* columns)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    out_hr.to_csv(OUT_TSV, sep="\t", index=False, na_rep="NA")
    dbg(f"Wrote per-inversion raw table → {OUT_TSV.resolve()}")

    # Correlation checks (printed only)
    c1 = pearson_corr(out["Inverted_AF"], out["inversion_freq_no_filter"])
    c2 = pearson_corr(out["Inverted_AF"], out["inversion_freq_filter"])
    print("\nCorrelation checks (Pearson, pairwise complete observations):")
    print(f"- Inversion allele frequency (inv_info.tsv) vs inversion frequency without filter (output.csv): {c1 if not math.isnan(c1) else float('nan'):.6f}")
    print(f"- Inversion allele frequency (inv_info.tsv) vs inversion frequency with filter    (output.csv): {c2 if not math.isnan(c2) else float('nan'):.6f}")

    # Exact mismatch report (printed only)
    mismatches = []
    for r in out.itertuples(index=False):
        a = r.Inverted_AF
        nf = r.inversion_freq_no_filter
        ff = r.inversion_freq_filter
        if equal_exact(a, nf) and equal_exact(a, ff):
            continue
        mismatches.append({
            "Inversion ID": r.inversion_id,
            "Region ID": region_id(r.chr, int(r.start), int(r.end)),
            "Inversion allele frequency (inv_info.tsv)": a,
            "Inversion frequency without filter (output.csv)": nf,
            "Inversion frequency with filter (output.csv)": ff,
        })

    if not mismatches:
        print("\nAll inversions have exact matching frequencies across the three sources (exact comparison).")
    else:
        print("\nFrequency mismatches (any where inv_info.tsv differs from output.csv):")
        dfm = pd.DataFrame(mismatches, columns=[
            "Inversion ID","Region ID",
            "Inversion allele frequency (inv_info.tsv)",
            "Inversion frequency without filter (output.csv)",
            "Inversion frequency with filter (output.csv)",
        ])
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 100):
            print(dfm.to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err(str(e))
        sys.exit(2)
