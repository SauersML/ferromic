from __future__ import annotations
import sys, os, re, math, statistics as stats
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# ---------- Files ----------
INV_FILE   = Path("inv_info.tsv")                 # required
SUMMARY    = Path("output.csv")                   # required
PI_FALSTA  = Path("per_site_diversity_output.falsta")
FST_FALSTA = Path("per_site_fst_output.falsta")

# ---------- Debug printing ----------
def dbg(msg: str): print(f"[DEBUG] {msg}", file=sys.stdout, flush=True)
def warn(msg: str): print(f"[WARN]  {msg}", file=sys.stdout, flush=True)
def err(msg: str): print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ---------- Basic helpers ----------
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

def mean_median_sd(xs: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    vals = [float(v) for v in xs if pd.notna(v)]
    if not vals:
        return (math.nan, math.nan, math.nan, 0)
    mu = float(np.mean(vals))
    med = float(np.median(vals))
    sd = float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0
    return (mu, med, sd, len(vals))

# ---------- 1) Load inversion mapping (STRICT, crash on duplicates) ----------
def load_inversions(inv_path: Path) -> pd.DataFrame:
    if not inv_path.is_file():
        raise FileNotFoundError(f"Required inversion file not found: {inv_path}")
    dbg("Loading inversion mapping from inv_info.tsv ...")
    df = pd.read_csv(inv_path, sep=None, engine="python", dtype=str)
    dbg(f"inv_info.tsv columns: {list(df.columns)}")

    need = ["Chromosome", "Start", "End", "0_single_1_recur_consensus"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"inv_info.tsv missing required columns: {miss}")

    # Normalize
    df["_chr"]   = df["Chromosome"].map(norm_chr)
    df["_start"] = df["Start"].map(parse_int)
    df["_end"]   = df["End"].map(parse_int)
    df["_cat"]   = pd.to_numeric(df["0_single_1_recur_consensus"], errors="coerce")

    # Drop rows with invalid coords
    before = len(df)
    df = df[df["_chr"].notna() & df["_start"].notna() & df["_end"].notna()]
    df["_start"] = df["_start"].astype(int)
    df["_end"]   = df["_end"].astype(int)
    dropped = before - len(df)

    # Category to label
    def lab(v):
        if pd.isna(v): return "uncategorized"
        return "recurrent" if int(v) == 1 else ("single-event" if int(v) == 0 else "uncategorized")
    df["_grp"] = df["_cat"].map(lab)

    # Strict duplicate detection on exact coordinates
    dup_counts = df.groupby(["_chr","_start","_end"]).size()
    dups = dup_counts[dup_counts > 1]
    if len(dups) > 0:
        # Show a few offending rows to make diagnosis easy.
        examples = df.merge(dups.rename("n"), on=["_chr","_start","_end"])
        err("Duplicate exact inversion coordinates detected in inv_info.tsv (this is a hard error).")
        err("Examples of duplicates:\n" + str(examples.head(10)))
        raise RuntimeError("Duplicate exact inversion coordinates in inversion table.")

    # Drop uncategorized rows prior to downstream analyses
    cts_all = Counter(df["_grp"])
    df = df[df["_grp"].isin(["recurrent", "single-event"])]
    dropped_uncat = cts_all.get("uncategorized", 0)

    # Summaries
    cts = Counter(df["_grp"])
    dbg(
        f"Inversions loaded (valid rows): {len(df)} (dropped {dropped}; excluded {dropped_uncat} uncategorized); "
        f"Recurrent={cts.get('recurrent',0)}, Single-event={cts.get('single-event',0)}"
    )

    return df[["_chr","_start","_end","_grp"]].rename(columns={"_chr":"chr","_start":"start","_end":"end","_grp":"grp"})

# ---------- 2) Load summary table (region-level metrics) ----------
def load_summary(sum_path: Path) -> pd.DataFrame:
    if not sum_path.is_file():
        raise FileNotFoundError(f"Required summary file not found: {sum_path}")
    dbg("Loading per-region summary from output.csv ...")
    df = pd.read_csv(sum_path, dtype=str)
    dbg(f"output.csv columns: {list(df.columns)}")

    need = ["chr","region_start","region_end",
            "0_pi_filtered","1_pi_filtered",
            "0_num_hap_filter","1_num_hap_filter",
            "hudson_fst_hap_group_0v1"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise RuntimeError(f"output.csv missing required columns: {miss}")

    out = pd.DataFrame({
        "chr": df["chr"].map(norm_chr),
        "start": pd.to_numeric(df["region_start"], errors="coerce"),
        "end": pd.to_numeric(df["region_end"], errors="coerce"),
        "pi0_f": pd.to_numeric(df["0_pi_filtered"], errors="coerce"),
        "pi1_f": pd.to_numeric(df["1_pi_filtered"], errors="coerce"),
        "n0": pd.to_numeric(df["0_num_hap_filter"], errors="coerce"),
        "n1": pd.to_numeric(df["1_num_hap_filter"], errors="coerce"),
        "fst_region_csv": pd.to_numeric(df["hudson_fst_hap_group_0v1"], errors="coerce"),
    })
    before = len(out)
    out = out[out["chr"].notna() & out["start"].notna() & out["end"].notna()]
    out["start"] = out["start"].astype(int)
    out["end"]   = out["end"].astype(int)
    kept = len(out)
    dbg(f"output.csv rows retained: {kept} (dropped {before-kept} with missing keys)")
    dbg("First 3 normalized rows from output.csv:\n" + str(out[["chr","start","end"]].head(3).to_string(index=False)))
    return out

# ---------- 3) Strict region↔inversion matching (±1 on region side; crash on >1 match) ----------
def match_regions(summary_df: pd.DataFrame, inv_df: pd.DataFrame) -> pd.DataFrame:
    dbg("Building ±1 bp candidate keys and performing strict match ...")
    # Build map of exact inversion coords -> group (no duplicates by construction)
    inv_index: Dict[Tuple[str,int,int], str] = {(r.chr, int(r.start), int(r.end)): r.grp for r in inv_df.itertuples(index=False)}

    # For debug: a quick hash of index size
    dbg(f"Inversion exact-key index size: {len(inv_index)}")

    # For each region create 9 candidate keys, look up in exact inv_index
    matches = []
    cand_total = 0
    for r in summary_df.itertuples(index=False):
        c, s, e = r.chr, int(r.start), int(r.end)
        region_key = (c, s, e)
        candidates = []
        for ds in (-1, 0, 1):
            for de in (-1, 0, 1):
                cand_total += 1
                key = (c, s + ds, e + de)
                if key in inv_index:
                    candidates.append((key, inv_index[key]))
        if len(candidates) == 0:
            # not matched → drop silently
            continue
        # STRICT: if more than one match (even if same group), crash.
        if len(candidates) > 1:
            msg = [f"{region_id(c,s,e)} matched multiple inversion rows (this is a hard error):"]
            for (kc, kg) in candidates[:10]:
                msg.append(f"  - inv={kc[0]}:{kc[1]}-{kc[2]}  group={kg}")
            err("\n".join(msg))
            raise RuntimeError("Ambiguous region→inversion match (>1 candidate).")
        (ikey, grp) = candidates[0]
        matches.append((c, s, e, grp))

    dbg(f"Candidate rows created: {cand_total} for {len(summary_df)} regions")
    dbg(f"Matched unique regions: {len(matches)}")
    if not matches:
        raise RuntimeError("No regions matched inversions under strict rule.")

    out = pd.DataFrame(matches, columns=["chr","start","end","recurrence"])
    cts = Counter(out["recurrence"])
    dbg(f"Matched recurrence counts → Recurrent={cts.get('recurrent',0)}, Single-event={cts.get('single-event',0)}")

    # Join back onto summary metrics
    merged = (summary_df
              .merge(out, on=["chr","start","end"], how="inner", validate="one_to_one"))
    dbg("First 5 matched region_ids:\n" + str(pd.DataFrame({
        "region_id": [region_id(r.chr, r.start, r.end) for r in merged.itertuples(index=False)],
        "Recurrence": merged["recurrence"]
    }).head(5).to_string(index=False)))
    return merged

# ---------- 4) Parse per-site arrays ----------
_RE_PI = re.compile(r">.*?filtered_pi.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)_group_(0|1)", re.IGNORECASE)
_RE_FH = re.compile(r">.*?hudson_pairwise_fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)", re.IGNORECASE)

def parse_pi_falsta(pi_path: Path) -> Dict[Tuple[str,int,int,str], np.ndarray]:
    if not pi_path.is_file():
        warn(f"π FALSTA not found: {pi_path} → flank π will be NA")
        return {}
    dbg("Parsing filtered per-site π with explicit group from per_site_diversity_output.falsta ...")
    pi_map: Dict[Tuple[str,int,int,str], np.ndarray] = {}
    headers = 0
    matched = 0
    badlen = 0
    got_chr = set()
    sample_headers = []
    with pi_path.open("r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for line in fh:
            line=line.rstrip("\n")
            if not line: continue
            if line[0] == ">":
                headers += 1
                header = line
                if len(sample_headers) < 8:
                    sample_headers.append(line)
                continue
            if header is None: continue
            m = _RE_PI.search(header)
            if not m:
                header = None
                continue
            chr_ = norm_chr(m.group(1)); s = int(m.group(2)); e = int(m.group(3))
            grp = "direct" if m.group(4) == "0" else "inverted"
            vals = np.fromstring(line.strip().replace("NA","nan"), sep=",", dtype=float)
            exp = e - s + 1
            if vals.size != exp:
                badlen += 1
                header = None
                continue
            pi_map[(chr_, s, e, grp)] = vals
            matched += 1
            got_chr.add(chr_)
            header = None
    dbg(f"π headers seen: {headers}, matched(filtered+grouped): {matched}, bad-length: {badlen}")
    dbg(f"π intervals by orientation: {Counter([k[3] for k in pi_map.keys()])}")
    dbg("Sample π headers:\n  " + "\n  ".join(sample_headers[:8]) if sample_headers else "  (none)")
    dbg(f"π chromosomes loaded: {len(got_chr)}; example: {list(got_chr)[:5]}")
    return pi_map

def parse_fst_falsta(fst_path: Path) -> Dict[Tuple[str,int,int], np.ndarray]:
    if not fst_path.is_file():
        warn(f"FST FALSTA not found: {fst_path} → flank FST will be NA")
        return {}
    dbg("Scanning Hudson per-site FST from per_site_fst_output.falsta ...")
    fst_map: Dict[Tuple[str,int,int], np.ndarray] = {}
    headers = 0
    hudson_like = 0
    badlen = 0
    got_chr = set()
    sample_headers = []
    token_tally = Counter()
    with fst_path.open("r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for line in fh:
            line=line.rstrip("\n")
            if not line: continue
            if line[0] == ">":
                headers += 1
                header = line
                if len(sample_headers) < 10:
                    sample_headers.append(line)
                continue
            if header is None: continue
            m = _RE_FH.search(header)
            if not m:
                header = None
                continue
            # Just accept; do NOT try to detect "filtered" tokens
            hudson_like += 1
            if "hudson" in header.lower(): token_tally["hudson"] += 1
            if "pairwise" in header.lower(): token_tally["pairwise"] += 1
            chr_ = norm_chr(m.group(1)); s = int(m.group(2)); e = int(m.group(3))
            vals = np.fromstring(line.strip().replace("NA","nan"), sep=",", dtype=float)
            exp = e - s + 1
            if vals.size != exp:
                badlen += 1
                header = None
                continue
            fst_map[(chr_, s, e)] = vals
            got_chr.add(chr_)
            header = None
    dbg(f"Total headers seen in FST file: {headers}")
    dbg(f"Hudson-like headers captured: {hudson_like}, bad-length: {badlen}")
    dbg(f"Token tallies among Hudson headers: {dict(token_tally)}")
    dbg("Sample Hudson headers (lowercased):\n  " + "\n  ".join([h.lower() for h in sample_headers[:10]]) if sample_headers else "  (none)")
    dbg(f"FST chromosomes loaded: {len(got_chr)}; example: {list(got_chr)[:5]}")
    return fst_map

# ---------- 5) Flank windows (inside region) ----------
def flank_union(start: int, end: int, flank_bp: int = 10_000) -> Tuple[int,int,int,int,Tuple[int,int]]:
    L = end - start + 1
    # Left: [start, start+flank-1], Right: [end-flank+1, end]
    Ls, Le = start, min(end, start + flank_bp - 1)
    Rs, Re = max(start, end - flank_bp + 1), end
    # Union
    Us, Ue = min(Ls, Rs), max(Le, Re)
    # If L < 2*flank, the union may be the whole region; that's fine.
    return Ls, Le, Rs, Re, (Us, Ue)

def slice_per_site(vals: np.ndarray, start: int, end: int, ws: int, we: int) -> np.ndarray:
    # vals corresponds to [start..end] inclusive
    off0 = ws - start
    off1 = we - start
    if off0 < 0 or off1 >= vals.size:
        # Out-of-bounds means no overlap
        return np.array([], dtype=float)
    return vals[off0:off1+1]

# ---------- 6) Main ----------
def main():
    # 1) Load inputs
    inv = load_inversions(INV_FILE)
    summ = load_summary(SUMMARY)

    # 2) Strict match
    matched = match_regions(summ, inv)
    matched = matched[matched["recurrence"].isin(["recurrent","single-event"])].copy()

    # 3) Parse per-site arrays (π + FST)
    pi_map  = parse_pi_falsta(PI_FALSTA)
    fst_map = parse_fst_falsta(FST_FALSTA)

    # 4) Compute per-region flank means (no thresholds)
    dbg("=== COVERAGE DIAGNOSTICS (INSIDE-REGION FLANKS; NO FILTER THRESHOLDS) ===")
    rows = []
    # Track Ns
    have = {
        "flank_pi_dir": Counter(),
        "flank_pi_inv": Counter(),
        "flank_fst": Counter(),
        "region_fst_csv": Counter(),
    }

    # Optional quick N-size debug from CSV
    n0_series = matched["n0"].fillna(np.nan).astype(float).tolist()
    n1_series = matched["n1"].fillna(np.nan).astype(float).tolist()
    def nstats(vs):
        vs2 = [x for x in vs if not math.isnan(x)]
        if not vs2: return (0, math.nan, math.nan, math.nan, 0)
        return (len(vs2), float(np.median(vs2)), float(np.mean(vs2)), float(np.min(vs2)), float(np.max(vs2)))
    n0N, n0med, n0mean, n0min, n0max = nstats(n0_series)
    n1N, n1med, n1mean, n1min, n1max = nstats(n1_series)
    dbg(f"[N-size] direct : N={n0N}, median={n0med}, mean={n0mean}, min={n0min}, max={n0max}; missing={len(n0_series)-n0N}")
    dbg(f"[N-size] inverted: N={n1N}, median={n1med}, mean={n1mean}, min={n1min}, max={n1max}; missing={len(n1_series)-n1N}")

    spot = 0
    for r in matched.itertuples(index=False):
        chr_, s, e = r.chr, int(r.start), int(r.end)
        L = e - s + 1
        rid = region_id(chr_, s, e)
        rec = r.recurrence

        # region FST from CSV ONLY (no fallback!)
        fst_csv = r.fst_region_csv if pd.notna(r.fst_region_csv) else math.nan
        if pd.notna(fst_csv):
            have["region_fst_csv"][rec] += 1

        # flank geometry
        Ls, Le, Rs, Re, (Us, Ue) = flank_union(s, e, flank_bp=10_000)
        # π flanks (direct/inverted)
        def flank_pi(grp: str) -> float:
            arr = pi_map.get((chr_, s, e, grp), None)
            if arr is None: return math.nan
            segL = slice_per_site(arr, s, e, Ls, Le)
            segR = slice_per_site(arr, s, e, Rs, Re)
            # union via simple concat then unique indices is overkill; the L/R can overlap, but
            # using both slices and concatenating will double-count the overlap. Instead, slice union directly:
            segU = slice_per_site(arr, s, e, Us, Ue)
            if segU.size == 0: return math.nan
            finite = segU[np.isfinite(segU)]
            if finite.size == 0: return math.nan
            return float(np.mean(finite))

        pi_dir = flank_pi("direct")
        pi_inv = flank_pi("inverted")
        if pd.notna(pi_dir): have["flank_pi_dir"][rec] += 1
        if pd.notna(pi_inv): have["flank_pi_inv"][rec] += 1

        # FST flank (per-site, no thresholds)
        def flank_fst_mean() -> float:
            arr = fst_map.get((chr_, s, e), None)
            if arr is None: return math.nan
            segU = slice_per_site(arr, s, e, Us, Ue)
            if segU.size == 0: return math.nan
            finite = segU[np.isfinite(segU)]
            if finite.size == 0: return math.nan
            return float(np.mean(finite))
        fst_flank = flank_fst_mean()
        if pd.notna(fst_flank): have["flank_fst"][rec] += 1

        # Save row
        rows.append({
            "region_id": rid, "rec": rec, "L": L,
            "pi_dir_region": r.pi0_f, "pi_inv_region": r.pi1_f,
            "fst_region_csv": fst_csv,
            "flank_pi_dir": pi_dir, "flank_pi_inv": pi_inv,
            "flank_fst": fst_flank
        })

        # Spot-check verbose for first ~12
        if spot < 12:
            cov_fst = "NA"
            if (chr_, s, e) in fst_map:
                # simple fraction finite in union
                arr = fst_map[(chr_, s, e)]
                segU = slice_per_site(arr, s, e, Us, Ue)
                cov_fst = f"{np.isfinite(segU).sum()}/{segU.size}" if segU.size else "0/0"
            dbg(f"  {rid:>23} rec={rec:<10} L={L:<7} "
                f"Lflank={Ls}-{Le}  Rflank={Rs}-{Re}  U={Us}-{Ue}  "
                f"CSV={fst_csv if pd.notna(fst_csv) else 'nan':<10}  flankFSTfinite={cov_fst}")
            spot += 1

    # 5) Aggregate by category
    tab = pd.DataFrame(rows)

    def agg_print(label: str, values: List[float], extraN: Optional[int]=None):
        mu, med, sd, N = mean_median_sd(values)
        N_print = N if extraN is None else extraN
        print(f"- {label}: {mu:.6f}, {med:.6f} ({sd if not math.isnan(sd) else float('nan'):.6f}), N={N_print}")

    # π region (already filtered in CSV)
    by = tab.groupby("rec", dropna=False)
    # Sample sizes (from CSV)
    for rec in ["recurrent", "single-event"]:
        grp = matched[matched["recurrence"] == rec]
        n0 = grp["n0"].astype(float).tolist()
        n1 = grp["n1"].astype(float).tolist()
        print(f"\nDirect haplotypes ({rec} region)")
        mu, med, sd, N = mean_median_sd(n0)
        print(f"- Sample size: {int(np.nansum(n0)) if N else 0}, {med:.1f} ({sd:.1f}), N={N}")
        agg_print("Nucleotide diversity (π)", grp["pi0_f"].astype(float).tolist(), extraN=len(grp))

        # flanks π (direct)
        dtab = tab[tab["rec"] == rec]
        agg_print("10 kb flanking π (inside region, direct)", dtab["flank_pi_dir"].tolist())

        print(f"\nInverted haplotypes ({rec} region)")
        mu, med, sd, N = mean_median_sd(n1)
        print(f"- Sample size: {int(np.nansum(n1)) if N else 0}, {med:.1f} ({sd:.1f}), N={N}")
        agg_print("Nucleotide diversity (π)", grp["pi1_f"].astype(float).tolist(), extraN=len(grp))
        agg_print("10 kb flanking π (inside region, inverted)", dtab["flank_pi_inv"].tolist())

    # Region FST (CSV ONLY)
    print("\nFST (Hudson; region means; CSV-only, strict like example)")
    for rec in ["recurrent", "single-event"]:
        vals = tab.loc[tab["rec"] == rec, "fst_region_csv"].tolist()
        mu, med, sd, N = mean_median_sd(vals)
        print(f"- {rec.capitalize()} regions: {mu:.6f}, {med:.6f} ({sd if not math.isnan(sd) else float('nan'):.6f}), N={N}")

    # Flank FST (per-site, no thresholds)
    print("\n10 kb flanking FST (Hudson; per-site inside region; NO thresholds)")
    for rec in ["recurrent", "single-event"]:
        vals = tab.loc[tab["rec"] == rec, "flank_fst"].tolist()
        mu, med, sd, N = mean_median_sd(vals)
        print(f"- {rec.capitalize()} regions: {mu:.6f}, {med:.6f} ({sd if not math.isnan(sd) else float('nan'):.6f}), N={N}")

    # Final availability snapshot
    dbg("Availability snapshot after parsing & aggregation (strict, no thresholds):")
    dbg(f"  π chroms: {len(set([k[0] for k in pi_map.keys()])) if pi_map else 0}; "
        f"FST chroms: {len(set([k[0] for k in fst_map.keys()])) if fst_map else 0}")
    dbg("Column availability in matched table:")
    dbg(f"  size cols present: n0={matched['n0'].notna().sum()>0}, n1={matched['n1'].notna().sum()>0}")
    dbg(f"  π cols present: pi0_f={matched['pi0_f'].notna().sum()>0}, pi1_f={matched['pi1_f'].notna().sum()>0}")
    dbg("=== COVERAGE COUNTS (no filters applied) ===")
    dbg(f"  flank_pi_dir_have: {dict(have['flank_pi_dir'])}")
    dbg(f"  flank_pi_inv_have: {dict(have['flank_pi_inv'])}")
    dbg(f"  flank_fst_have:    {dict(have['flank_fst'])}")
    dbg(f"  region_fst_csv_have: {dict(have['region_fst_csv'])}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err(str(e))
        sys.exit(2)
