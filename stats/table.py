import sys
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# --------------------------- CONSTANTS ----------------------------

INVINFO_TSV = "inv_info.tsv"
OUTPUT_CSV = "output.csv"
DIVERSITY_FALSTA = "per_site_diversity_output.falsta"
FST_FALSTA = "per_site_fst_output.falsta"

FLANK_BP = 10_000  # 10 kb on each side

# Regexes (case-insensitive)
RE_PI = re.compile(
    r"^>.*?filtered_pi.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+).*?_group_([01])\b",
    re.IGNORECASE,
)
# Broad Hudson header catcher (we'll classify tokens afterwards)
RE_HUDSON_ANY = re.compile(
    r"^>.*?hudson.*?pairwise.*?fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)",
    re.IGNORECASE,
)

# --------------------------- HELPERS ------------------------------

def debug(msg: str):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

def warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)

def err(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

def norm_chr(val: str) -> str:
    s = str(val).strip()
    s_low = s.lower()
    if s_low.startswith("chr_"):
        s = s[4:]
    elif s_low.startswith("chr"):
        s = s[3:]
    return f"chr{s}"

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def sample_sd(a: np.ndarray) -> float:
    if a.size <= 1:
        return np.nan
    return float(np.std(a, ddof=1))

def fmt_stat(median: Optional[float], mean: Optional[float], sd: Optional[float],
             n: int, kind: str) -> str:
    if n == 0 or median is None or mean is None or (np.isnan(median) and np.isnan(mean)):
        return "NA, NA (NA), N=0"
    if kind == "size":
        med_txt  = "NA" if median is None or np.isnan(median) else f"{int(round(median))}"
        mean_txt = "NA" if mean   is None or np.isnan(mean)   else f"{mean:.1f}"
        sd_txt   = "NA" if sd     is None or np.isnan(sd)     else f"{sd:.1f}"
    else:
        med_txt  = "NA" if median is None or np.isnan(median) else f"{median:.6f}"
        mean_txt = "NA" if mean   is None or np.isnan(mean)   else f"{mean:.6f}"
        sd_txt   = "NA" if sd     is None or np.isnan(sd)     else f"{sd:.6f}"
    return f"{med_txt}, {mean_txt} ({sd_txt}), N={n}"

def describe_vector(values: List[float], kind: str, n_override: Optional[int] = None) -> str:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        n = 0 if n_override is None else int(n_override)
        return f"NA, NA (NA), N={n}"
    med  = float(np.median(arr))
    mean = float(np.mean(arr))
    sd   = sample_sd(arr)
    n    = int(arr.size if n_override is None else n_override)
    return fmt_stat(med, mean, sd, n, kind)

# --------------------- LOADING & MATCHING -------------------------

def load_invinfo() -> pd.DataFrame:
    debug(f"Loading inversion mapping from {INVINFO_TSV} ...")
    inv = pd.read_csv(INVINFO_TSV, sep="\t", engine="python", dtype=str)
    inv.columns = [c.strip() for c in inv.columns]
    debug(f"{INVINFO_TSV} columns: {list(inv.columns)}")

    need_cols = {"Chromosome", "Start", "End"}
    if not need_cols.issubset(inv.columns):
        raise RuntimeError(f"{INVINFO_TSV} must contain columns: {sorted(need_cols)}")

    recur_col = None
    for c in ["0_single_1_recur_consensus", "0_single_1_recur"]:
        if c in inv.columns:
            recur_col = c
            break
    if recur_col is None:
        raise RuntimeError("Missing recurrence column: need '0_single_1_recur_consensus' or '0_single_1_recur'")

    out = pd.DataFrame({
        "chr_std": inv["Chromosome"].map(norm_chr),
        "Start":   to_num(inv["Start"]),
        "End":     to_num(inv["End"]),
        "rec_flag": to_num(inv[recur_col]),
    })
    before = out.shape[0]
    out = out.dropna(subset=["chr_std","Start","End","rec_flag"]).copy()
    out["Start"] = out["Start"].astype(int)
    out["End"]   = out["End"].astype(int)
    out["Recurrence"] = out["rec_flag"].map({0:"Single-event", 1:"Recurrent"})
    out = out.dropna(subset=["Recurrence"]).drop_duplicates(subset=["chr_std","Start","End"])
    after = out.shape[0]
    counts = Counter(out["Recurrence"])
    debug(f"Inversions loaded (valid rows): {after} (dropped {before-after}); Recurrent={counts.get('Recurrent',0)}, Single-event={counts.get('Single-event',0)}")
    return out[["chr_std","Start","End","Recurrence"]].reset_index(drop=True)

def load_output() -> pd.DataFrame:
    debug(f"Loading per-region summary from {OUTPUT_CSV} ...")
    df = pd.read_csv(OUTPUT_CSV, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    debug(f"{OUTPUT_CSV} columns: {list(df.columns)}")

    need = {"chr","region_start","region_end"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{OUTPUT_CSV} must contain columns: {sorted(need)}")

    keep_cols = {
        "chr","region_start","region_end",
        "0_pi_filtered","1_pi_filtered","0_pi","1_pi",
        "0_num_hap_filter","1_num_hap_filter","0_num_hap_no_filter","1_num_hap_no_filter",
        "hudson_fst_hap_group_0v1"  # for sanity cross-check only
    }
    cols_present = [c for c in df.columns if c in keep_cols]
    missing = sorted(list(keep_cols.difference(set(cols_present))))
    if missing:
        debug(f"Note: output.csv missing optional cols (OK): {missing}")

    df = df[cols_present].copy()
    df["chr_std"] = df["chr"].map(norm_chr)
    df["region_start"] = to_num(df["region_start"]).astype("Int64")
    df["region_end"]   = to_num(df["region_end"]).astype("Int64")
    before = df.shape[0]
    df = df.dropna(subset=["chr_std","region_start","region_end"]).copy()
    df["region_start"] = df["region_start"].astype(int)
    df["region_end"]   = df["region_end"].astype(int)

    for c in ["0_pi_filtered","1_pi_filtered","0_pi","1_pi",
              "0_num_hap_filter","1_num_hap_filter","0_num_hap_no_filter","1_num_hap_no_filter",
              "hudson_fst_hap_group_0v1"]:
        if c in df.columns:
            df[c] = to_num(df[c])

    after = df.shape[0]
    debug(f"{OUTPUT_CSV} rows retained: {after} (dropped {before-after} with missing keys)")
    debug("First 3 normalized rows from output.csv:")
    debug(df[["chr_std","region_start","region_end"]].head(3).to_string(index=False))
    return df.reset_index(drop=True)

def strict_match(df_out: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    debug("Building ±1 bp candidate keys and performing strict match ...")
    df_small = df_out[["chr_std","region_start","region_end"]].copy()

    cands = []
    for ds in (-1, 0, 1):
        for de in (-1, 0, 1):
            tmp = df_small.copy()
            tmp["Start"] = tmp["region_start"] + ds
            tmp["End"]   = tmp["region_end"]   + de
            tmp["match_priority"] = abs(ds) + abs(de)  # 0 (exact), 1, 2
            cands.append(tmp)
    cand = pd.concat(cands, ignore_index=True)
    debug(f"Candidate rows created: {cand.shape[0]} for {df_small.shape[0]} regions")

    merged = cand.merge(inv, on=["chr_std","Start","End"], how="inner")  # keep only real overlaps
    debug(f"Candidate matches against inv_info: {merged.shape[0]}")

    if merged.empty:
        raise RuntimeError("No regions matched inv_info under ±1 bp tolerance.")

    key = ["chr_std","region_start","region_end"]

    # Step 1: keep rows at minimal match_priority per key
    min_mp = (merged.groupby(key)["match_priority"].min().reset_index().rename(columns={"match_priority":"min_mp"}))
    best = merged.merge(min_mp, on=key, how="inner")
    best = best[best["match_priority"] == best["min_mp"]].copy()

    # Step 2: enforce single best row per key
    counts = best.groupby(key).size().reset_index(name="n_best")
    n_ambig = int((counts["n_best"] != 1).sum())
    if n_ambig > 0:
        debug(f"Ambiguous-at-best-priority regions: {n_ambig} → dropping them")
    best = best.merge(counts, on=key, how="left")
    best = best[best["n_best"] == 1].copy()
    best.drop(columns=["n_best","min_mp"], inplace=True)

    if best.empty:
        raise RuntimeError("After strict selection, no regions remained.")

    debug(f"Matched unique regions: {best.shape[0]}")

    out = best.merge(df_out, on=["chr_std","region_start","region_end"], how="left", suffixes=("",""))
    if not {"chr_std","region_start","region_end","Recurrence"}.issubset(out.columns):
        missing = {"chr_std","region_start","region_end","Recurrence"} - set(out.columns)
        raise RuntimeError(f"Strict match lost key columns unexpectedly: {sorted(missing)}")

    out["region_id"] = (
        out["chr_std"].astype(str) + ":" +
        out["region_start"].astype(int).astype(str) + "-" +
        out["region_end"].astype(int).astype(str)
    )

    n_rec   = int((out["Recurrence"] == "Recurrent").sum())
    n_single= int((out["Recurrence"] == "Single-event").sum())
    debug(f"Matched recurrence counts → Recurrent={n_rec}, Single-event={n_single}")
    debug("First 5 matched region_ids:")
    debug(out[["region_id","Recurrence"]].head(5).to_string(index=False))
    return out

# ------------------------ FALSTA PARSING ---------------------------

class IntervalRec:
    __slots__ = ("start","end","data","header")
    def __init__(self, start: int, end: int, data: np.ndarray, header: str):
        self.start = int(start)
        self.end   = int(end)
        self.data  = data  # np.ndarray length == end-start+1
        self.header= header

def parse_falsta_pi() -> Dict[str, Dict[str, List[IntervalRec]]]:
    debug(f"Parsing filtered per-site π with explicit group from {DIVERSITY_FALSTA} ...")
    store: Dict[str, Dict[str, List[IntervalRec]]] = {}
    n_headers = n_matched = n_badlen = 0
    n_by_orient = Counter()
    sample_headers = []

    with open(DIVERSITY_FALSTA, "r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line[0] == ">":
                n_headers += 1
                header = line
                if len(sample_headers) < 8:
                    sample_headers.append(header.strip()[:260])
                continue
            if header is None:
                continue
            m = RE_PI.search(header)
            if not m:
                header = None
                continue
            chrom = norm_chr(m.group(1))
            start = int(m.group(2))
            end   = int(m.group(3))
            gid   = m.group(4)  # '0' or '1'
            orient = "direct" if gid == "0" else "inverted"

            arr = np.fromstring(line.strip().replace("NA","nan"), sep=",", dtype=np.float64)
            exp = end - start + 1
            if arr.size != exp:
                n_badlen += 1
                header = None
                continue
            store.setdefault(chrom, {"direct":[],"inverted":[]})
            store[chrom][orient].append(IntervalRec(start, end, arr, header))
            n_matched += 1
            n_by_orient[orient] += 1
            header = None

    for chrom in store:
        for orient in ("direct","inverted"):
            store[chrom][orient].sort(key=lambda r: r.start)

    debug(f"π headers seen: {n_headers}, matched(filtered+grouped): {n_matched}, bad-length: {n_badlen}")
    debug(f"π intervals by orientation: {dict(n_by_orient)}")
    if sample_headers:
        debug("Sample π headers:")
        for h in sample_headers:
            debug(f"  {h}")
    debug(f"π chromosomes loaded: {len(store)}; example: {list(store.keys())[:5]}")
    return store

def parse_falsta_fst_filtered_intent() -> Tuple[Dict[str, List[IntervalRec]], Dict[str, int], Dict[str, int], bool]:
    """
    Parse Hudson pairwise FST per-site series and **prefer filtered**.
    If no header contains 'filtered', assume the only Hudson series is already filtered (logged).
    Returns:
      - store: dict chr -> [IntervalRec]
      - counts_all_chr: per-chrom counts (all Hudson)
      - counts_filtered_chr: per-chrom counts (filtered-only subset actually used)
      - assumed_filtered: True iff no explicit 'filtered' token was found and we used all Hudson anyway
    """
    debug(f"Scanning Hudson per-site FST from {FST_FALSTA} ...")
    all_store: Dict[str, List[IntervalRec]] = defaultdict(list)
    filtered_store: Dict[str, List[IntervalRec]] = defaultdict(list)

    n_headers = 0
    n_hudson  = 0
    n_badlen  = 0
    token_tally = Counter()
    sample_hudson_headers = []

    with open(FST_FALSTA, "r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                n_headers += 1
                header = line
                continue
            if header is None:
                continue

            m = RE_HUDSON_ANY.search(header)
            if not m:
                header = None
                continue

            # Token classification (for debugging)
            hlow = header.lower()
            has_filtered = ("filtered" in hlow) or ("mask_filtered" in hlow) or ("filtered_mask" in hlow)
            has_wc = ("weir" in hlow and "cockerham" in hlow) or ("wc" in hlow)
            has_hap = ("hap_group" in hlow)
            has_pair = ("pairwise" in hlow)
            for tok, present in [
                ("hudson", True),
                ("pairwise", has_pair),
                ("filtered", has_filtered),
                ("wc", has_wc),
                ("hap_group", has_hap),
            ]:
                if present:
                    token_tally[tok] += 1

            # It's a Hudson pairwise FST series
            chrom = norm_chr(m.group(1))
            start = int(m.group(2))
            end   = int(m.group(3))
            arr = np.fromstring(line.strip().replace("NA","nan"), sep=",", dtype=np.float64)
            exp = end - start + 1
            if arr.size != exp:
                n_badlen += 1
                header = None
                continue

            rec = IntervalRec(start, end, arr, header)
            all_store[chrom].append(rec)
            n_hudson += 1
            if has_filtered:
                filtered_store[chrom].append(rec)

            if len(sample_hudson_headers) < 12:
                sample_hudson_headers.append((hlow[:280], has_filtered))
            header = None

    counts_all_chr = {c: len(v) for c, v in all_store.items()}
    counts_filt_chr = {c: len(v) for c, v in filtered_store.items()}

    for chrom in all_store:
        all_store[chrom].sort(key=lambda r: r.start)
    for chrom in filtered_store:
        filtered_store[chrom].sort(key=lambda r: r.start)

    debug(f"Total headers seen in FST file: {n_headers}")
    debug(f"Hudson-like headers captured: {n_hudson}, bad-length: {n_badlen}")
    debug(f"Token tallies among Hudson headers: {dict(token_tally)}")
    if sample_hudson_headers:
        debug("Sample Hudson headers (lowercased) with filtered-flag:")
        for h, fl in sample_hudson_headers[:10]:
            debug(f"  filtered={fl} :: {h}")

    explicit_filtered_found = sum(counts_filt_chr.values()) > 0
    if explicit_filtered_found:
        debug(f"Using EXPLICIT filtered Hudson series; chromosomes with filtered intervals: {len(counts_filt_chr)}")
        # summarize first few chromosomes
        for c in list(counts_filt_chr.keys())[:6]:
            debug(f"  chr {c}: {counts_filt_chr[c]} filtered intervals (all={counts_all_chr.get(c,0)})")
        return filtered_store, counts_all_chr, counts_filt_chr, False

    # No filtered token anywhere — fallback assumption:
    if n_hudson == 0:
        warn("No Hudson pairwise FST headers matched at all. All FST stats will be NA.")
        return {}, counts_all_chr, counts_filt_chr, False

    warn("No 'filtered' token found in any Hudson FST headers. "
         "Assuming the *only* Hudson series present is ALREADY filtered. "
         "Proceeding with ALL Hudson intervals (documented below).")
    debug(f"Chromosomes with Hudson intervals (assumed filtered): {len(counts_all_chr)}")
    for c in list(counts_all_chr.keys())[:10]:
        debug(f"  chr {c}: {counts_all_chr[c]} intervals")
    return all_store, counts_all_chr, counts_filt_chr, True

def window_sum_count(wstart: int, wend: int, intervals: List[IntervalRec]) -> Tuple[float, int]:
    if wend < wstart:
        return 0.0, 0
    total_sum = 0.0
    total_n   = 0
    for rec in intervals:
        if rec.end < wstart:
            continue
        if rec.start > wend:
            break
        s = max(wstart, rec.start)
        e = min(wend, rec.end)
        if s > e:
            continue
        i0 = s - rec.start
        i1 = e - rec.start + 1
        seg = rec.data[i0:i1]
        if seg.size == 0:
            continue
        mask = np.isfinite(seg)
        if not np.any(mask):
            continue
        vals = seg[mask]
        total_sum += float(np.sum(vals))
        total_n   += int(vals.size)
    return total_sum, total_n

def window_mean(wstart: int, wend: int, intervals: List[IntervalRec]) -> float:
    s, n = window_sum_count(wstart, wend, intervals)
    return np.nan if n == 0 else (s / n)

# ------------------------- MAIN PIPELINE ---------------------------

def main():
    # 1) Load + strict match
    inv = load_invinfo()
    out = load_output()
    matched = strict_match(out, inv)

    # 2) Parse per-site series
    pi_intervals  = parse_falsta_pi()
    fst_intervals, fst_counts_all_chr, fst_counts_filt_chr, assumed_filtered = parse_falsta_fst_filtered_intent()

    debug("Availability snapshot after parsing:")
    debug(f"  π chroms: {len(pi_intervals)}; FST chroms (used): {len(fst_intervals)}; assumed_filtered={assumed_filtered}")

    # 2b) QUICK SANITY: if we have any region-level FST in CSV, compare with per-site-derived means
    if "hudson_fst_hap_group_0v1" in matched.columns and len(fst_intervals) > 0:
        # compute per-region per-site FST means for first ~10 matched regions to compare
        debug("Sanity cross-check: per-site FST mean vs output.csv hudson_fst_hap_group_0v1 (first 10 regions)")
        check_rows = matched.head(10)
        for _, r in check_rows.iterrows():
            chrom = str(r["chr_std"])
            rs = int(r["region_start"]); re_ = int(r["region_end"])
            csv_val = r.get("hudson_fst_hap_group_0v1", np.nan)
            psite_val = np.nan
            if chrom in fst_intervals:
                psite_val = window_mean(rs, re_, fst_intervals[chrom])
            debug(f"  {chrom}:{rs}-{re_}  CSV={csv_val}  per-site={psite_val}")

    # 3) Aggregation containers
    cat = {
        "direct_recurrent":   {"size": [], "pi": [], "flank_pi": []},
        "inverted_recurrent": {"size": [], "pi": [], "flank_pi": []},
        "direct_single":      {"size": [], "pi": [], "flank_pi": []},
        "inverted_single":    {"size": [], "pi": [], "flank_pi": []},
    }
    fst_cat       = {"recurrent": [], "single": []}
    fst_flank_cat = {"recurrent": [], "single": []}
    total_count   = {"recurrent": 0, "single": 0}

    coverage = {
        "have_pi_dir": Counter(), "have_pi_inv": Counter(),
        "have_size_dir": Counter(), "have_size_inv": Counter(),
        "have_flank_pi_dir": Counter(), "have_flank_pi_inv": Counter(),
        "have_region_fst": Counter(), "have_flank_fst": Counter(),
        "miss_chr_fst": Counter(), "miss_chr_pi": Counter(),
        "zero_overlap_region_fst": Counter(), "zero_overlap_flank_fst": Counter(),
    }

    has_0_size_f  = "0_num_hap_filter" in matched.columns
    has_1_size_f  = "1_num_hap_filter" in matched.columns
    has_0_size_nf = "0_num_hap_no_filter" in matched.columns
    has_1_size_nf = "1_num_hap_no_filter" in matched.columns
    has_0_pi_f    = "0_pi_filtered" in matched.columns
    has_1_pi_f    = "1_pi_filtered" in matched.columns
    has_0_pi_un   = "0_pi" in matched.columns
    has_1_pi_un   = "1_pi" in matched.columns

    debug("Column availability in matched table:")
    debug(f"  size cols: 0_f={has_0_size_f}, 1_f={has_1_size_f}, 0_nf={has_0_size_nf}, 1_nf={has_1_size_nf}")
    debug(f"  π cols:    0_pi_f={has_0_pi_f}, 1_pi_f={has_1_pi_f}, 0_pi={has_0_pi_un}, 1_pi={has_1_pi_un}")

    # 4) Iterate regions
    spot_examples = []  # collect a few for print-out
    fst_overlap_counter = Counter()
    fst_flank_overlap_counter = Counter()

    for i, r in matched.iterrows():
        chrom = str(r["chr_std"])
        rs = int(r["region_start"]); re_ = int(r["region_end"])
        if rs > re_: rs, re_ = re_, rs
        rec_label = "recurrent" if str(r["Recurrence"]) == "Recurrent" else "single"
        total_count[rec_label] += 1

        # --- Sample size
        size_dir = np.nan
        size_inv = np.nan
        if has_0_size_f and pd.notna(r.get("0_num_hap_filter", np.nan)):
            size_dir = float(r["0_num_hap_filter"])
        elif has_0_size_nf and pd.notna(r.get("0_num_hap_no_filter", np.nan)):
            size_dir = float(r["0_num_hap_no_filter"])
        if has_1_size_f and pd.notna(r.get("1_num_hap_filter", np.nan)):
            size_inv = float(r["1_num_hap_filter"])
        elif has_1_size_nf and pd.notna(r.get("1_num_hap_no_filter", np.nan)):
            size_inv = float(r["1_num_hap_no_filter"])
        if np.isfinite(size_dir): coverage["have_size_dir"][rec_label] += 1
        if np.isfinite(size_inv): coverage["have_size_inv"][rec_label] += 1

        # --- π (prefer filtered)
        pi_dir = np.nan
        pi_inv = np.nan
        if has_0_pi_f and pd.notna(r.get("0_pi_filtered", np.nan)):
            pi_dir = float(r["0_pi_filtered"])
        elif has_0_pi_un and pd.notna(r.get("0_pi", np.nan)):
            pi_dir = float(r["0_pi"])
        if has_1_pi_f and pd.notna(r.get("1_pi_filtered", np.nan)):
            pi_inv = float(r["1_pi_filtered"])
        elif has_1_pi_un and pd.notna(r.get("1_pi", np.nan)):
            pi_inv = float(r["1_pi"])
        if np.isfinite(pi_dir): coverage["have_pi_dir"][rec_label] += 1
        if np.isfinite(pi_inv): coverage["have_pi_inv"][rec_label] += 1

        # --- Region-level Hudson FST (per-site)
        region_fst = np.nan
        if chrom in fst_intervals:
            s, n = window_sum_count(rs, re_, fst_intervals[chrom])
            if n > 0:
                region_fst = s / n
                coverage["have_region_fst"][rec_label] += 1
                fst_overlap_counter[rec_label] += n
            else:
                coverage["zero_overlap_region_fst"][rec_label] += 1
        else:
            coverage["miss_chr_fst"][rec_label] += 1

        # --- 10 kb flanking π (direct & inverted) from filtered per-site π
        left_s, left_e = max(1, rs - FLANK_BP), rs - 1
        right_s, right_e = re_ + 1, re_ + FLANK_BP

        flank_pi_dir = np.nan
        flank_pi_inv = np.nan
        if chrom in pi_intervals:
            # direct
            sumv, cnt = 0.0, 0
            if left_e >= left_s:
                s, n = window_sum_count(left_s, left_e, pi_intervals[chrom]["direct"])
                sumv += s; cnt += n
            if right_e >= right_s:
                s, n = window_sum_count(right_s, right_e, pi_intervals[chrom]["direct"])
                sumv += s; cnt += n
            if cnt > 0:
                flank_pi_dir = sumv / cnt
                coverage["have_flank_pi_dir"][rec_label] += 1

            # inverted
            sumv, cnt = 0.0, 0
            if left_e >= left_s:
                s, n = window_sum_count(left_s, left_e, pi_intervals[chrom]["inverted"])
                sumv += s; cnt += n
            if right_e >= right_s:
                s, n = window_sum_count(right_s, right_e, pi_intervals[chrom]["inverted"])
                sumv += s; cnt += n
            if cnt > 0:
                flank_pi_inv = sumv / cnt
                coverage["have_flank_pi_inv"][rec_label] += 1
        else:
            coverage["miss_chr_pi"][rec_label] += 1

        # --- 10 kb flanking FILTERED-INTENT Hudson FST
        flank_fst = np.nan
        if chrom in fst_intervals:
            sumv, cnt = 0.0, 0
            if left_e >= left_s:
                s, n = window_sum_count(left_s, left_e, fst_intervals[chrom])
                sumv += s; cnt += n
            if right_e >= right_s:
                s, n = window_sum_count(right_s, right_e, fst_intervals[chrom])
                sumv += s; cnt += n
            if cnt > 0:
                flank_fst = sumv / cnt
                coverage["have_flank_fst"][rec_label] += 1
                fst_flank_overlap_counter[rec_label] += cnt
            else:
                coverage["zero_overlap_flank_fst"][rec_label] += 1
        else:
            coverage["miss_chr_fst"][rec_label] += 1

        # --- Assign
        if rec_label == "recurrent":
            cat["direct_recurrent"]["size"].append(size_dir)
            cat["direct_recurrent"]["pi"].append(pi_dir)
            cat["direct_recurrent"]["flank_pi"].append(flank_pi_dir)
            cat["inverted_recurrent"]["size"].append(size_inv)
            cat["inverted_recurrent"]["pi"].append(pi_inv)
            cat["inverted_recurrent"]["flank_pi"].append(flank_pi_inv)
            fst_cat["recurrent"].append(region_fst)
            fst_flank_cat["recurrent"].append(flank_fst)
        else:
            cat["direct_single"]["size"].append(size_dir)
            cat["direct_single"]["pi"].append(pi_dir)
            cat["direct_single"]["flank_pi"].append(flank_pi_dir)
            cat["inverted_single"]["size"].append(size_inv)
            cat["inverted_single"]["pi"].append(pi_inv)
            cat["inverted_single"]["flank_pi"].append(flank_pi_inv)
            fst_cat["single"].append(region_fst)
            fst_flank_cat["single"].append(flank_fst)

        # Collect a few examples for sanity print
        if len(spot_examples) < 10:
            spot_examples.append({
                "region_id": f"{chrom}:{rs}-{re_}",
                "rec": rec_label,
                "pi_dir": pi_dir, "pi_inv": pi_inv,
                "fst_region": region_fst,
                "flank_pi_dir": flank_pi_dir, "flank_pi_inv": flank_pi_inv,
                "flank_fst": flank_fst
            })

    # 5) Coverage diagnostics
    debug("=== COVERAGE DIAGNOSTICS (counts of regions with metric available) ===")
    debug(f"Total matched regions → Recurrent={total_count['recurrent']}, Single-event={total_count['single']}")
    for k, v in coverage.items():
        debug(f"{k}: {dict(v)}")
    if len(fst_intervals) > 0:
        debug("Per-class summed per-site counts contributing to FST means (region windows): "
              f"{dict(fst_overlap_overlap := fst_overlap_counter)}")
        debug("Per-class summed per-site counts contributing to FST means (flank windows): "
              f"{dict(fst_flank_overlap_counter)}")

    debug("Spot-check of first few regions (values shown may include NaN if not available):")
    for ex in spot_examples:
        debug(str(ex))

    # 6) Print summaries
    def print_block(title: str, sizes: List[float], pis: List[float], flank_pis: List[float],
                    total_in_category: int):
        print(title)
        print(f"- Sample size: {describe_vector(sizes, kind='size', n_override=total_in_category)}")
        print(f"- Nucleotide diversity (π): {describe_vector(pis, kind='pi')}")
        print(f"- 10 kb flanking π: {describe_vector(flank_pis, kind='pi')}")
        print()

    n_recur  = total_count["recurrent"]
    n_single = total_count["single"]

    print_block("Direct haplotypes (recurrent region)",
                cat["direct_recurrent"]["size"],
                cat["direct_recurrent"]["pi"],
                cat["direct_recurrent"]["flank_pi"],
                total_in_category=n_recur)

    print_block("Inverted haplotypes (recurrent region)",
                cat["inverted_recurrent"]["size"],
                cat["inverted_recurrent"]["pi"],
                cat["inverted_recurrent"]["flank_pi"],
                total_in_category=n_recur)

    print_block("Direct haplotypes (single-event region)",
                cat["direct_single"]["size"],
                cat["direct_single"]["pi"],
                cat["direct_single"]["flank_pi"],
                total_in_category=n_single)

    print_block("Inverted haplotypes (single-event region)",
                cat["inverted_single"]["size"],
                cat["inverted_single"]["pi"],
                cat["inverted_single"]["flank_pi"],
                total_in_category=n_single)

    print("FST (Hudson, filtered-intent; two categories)")
    print(f"- Recurrent regions: {describe_vector(fst_cat['recurrent'], kind='fst')}")
    print(f"- Single-event regions: {describe_vector(fst_cat['single'], kind='fst')}")
    print()

    print("10 kb flanking FST (Hudson, filtered-intent; two categories)")
    print(f"- Recurrent regions: {describe_vector(fst_flank_cat['recurrent'], kind='fst')}")
    print(f"- Single-event regions: {describe_vector(fst_flank_cat['single'], kind='fst')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err(str(e))
        sys.exit(1)
