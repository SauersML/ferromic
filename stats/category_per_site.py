import os, sys, re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------ Filenames -------------------

INVINFO_TSV  = "inv_info.tsv"
OUTPUT_CSV   = "output.csv"
PI_FALSTA    = "per_site_diversity_output.falsta"
FST_FALSTA   = "per_site_fst_output.falsta"
MAP_TSV      = "map.tsv"  # optional

# ------------------ map.tsv required columns ----

MAP_FILE_COLUMNS = ['Original_Chr', 'Original_Start', 'Original_End',
                    'New_Chr', 'New_Start', 'New_End']

# ------------------ Parameters ------------------

FLANK_BP = 10_000                   # flanks = first/last 10kb within region
MIN_PSITE_BASES_ACCEPT = 200        # min finite sites to accept a per-site mean
MIN_REGION_FST_COVER_FRAC = 0.05    # ≥5% per-site coverage → allow using per-site FST for region
USE_PERSITE_FST_WHEN_COVERED = True # set False to always use CSV region FST (still cross-check)

# ------------------ Regex for FALSTA ------------

RE_PI = re.compile(
    r"^>.*?filtered_pi.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+).*?_group_([01])\b",
    re.IGNORECASE,
)
RE_HUDSON_ANY = re.compile(
    r"^>.*?hudson.*?pairwise.*?fst.*?_chr_?([\w.\-]+)_start_(\d+)_end_(\d+)",
    re.IGNORECASE,
)

# ------------------ Logging helpers -------------

def debug(msg: str):
    print(f"[DEBUG] {msg}", file=sys.stderr, flush=True)

def warn(msg: str):
    print(f"[WARN]  {msg}", file=sys.stderr, flush=True)

def err(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)

# ------------------ Utilities -------------------

def norm_chr(x) -> str:
    s = str(x).strip().lower()
    if s.startswith("chr_"): s = s[4:]
    elif s.startswith("chr"): s = s[3:]
    if not s.startswith("chr"):
        s = f"chr{s}"
    return s

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def sample_sd(arr: np.ndarray) -> float:
    n = arr.size
    if n <= 1: return np.nan
    return float(np.std(arr, ddof=1))

def fmt_stat(median, mean, sd, n, kind: str) -> str:
    if n == 0 or (median is None and mean is None):
        return "NA, NA (NA), N=0"
    if kind == "size":
        med = "NA" if median is None or np.isnan(median) else f"{int(round(median))}"
        mea = "NA" if mean   is None or np.isnan(mean)   else f"{mean:.1f}"
        sdd = "NA" if sd     is None or np.isnan(sd)     else f"{sd:.1f}"
    else:
        med = "NA" if median is None or np.isnan(median) else f"{median:.6f}"
        mea = "NA" if mean   is None or np.isnan(mean)   else f"{mean:.6f}"
        sdd = "NA" if sd     is None or np.isnan(sd)     else f"{sd:.6f}"
    return f"{med}, {mea} ({sdd}), N={n}"

def describe(values: List[float], kind: str, n_override: Optional[int]=None) -> str:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return f"NA, NA (NA), N={0 if n_override is None else int(n_override)}"
    med  = float(np.median(arr))
    mean = float(np.mean(arr))
    sd   = sample_sd(arr)
    n    = int(arr.size if n_override is None else n_override)
    return fmt_stat(med, mean, sd, n, kind)

# ------------------ Optional map.tsv ------------

def maybe_load_map() -> Optional[pd.DataFrame]:
    if not os.path.exists(MAP_TSV):
        debug("No map.tsv found; will use inversion coordinates as-is.")
        return None
    df = pd.read_csv(MAP_TSV, sep="\t", dtype=str)
    if not all(col in df.columns for col in MAP_FILE_COLUMNS):
        warn("map.tsv present but missing required columns; ignoring mapping.")
        return None
    df = df[MAP_FILE_COLUMNS].copy()
    df["Original_Chr"] = df["Original_Chr"].map(norm_chr)
    df["New_Chr"]      = df["New_Chr"].map(norm_chr)
    for c in ["Original_Start","Original_End","New_Start","New_End"]:
        df[c] = to_num(df[c]).astype("Int64")
    df = df.dropna().copy()
    for c in ["Original_Start","Original_End","New_Start","New_End"]:
        df[c] = df[c].astype(int)
    debug(f"Loaded coordinate mapping with {len(df)} rows.")
    return df

def build_map_lookup(map_df: Optional[pd.DataFrame]) -> Dict[Tuple[str,int,int], Tuple[str,int,int]]:
    if map_df is None: return {}
    lut = {}
    for _, r in map_df.iterrows():
        oc, os_, oe = r["Original_Chr"], int(r["Original_Start"]), int(r["Original_End"])
        nc, ns, ne  = r["New_Chr"], int(r["New_Start"]), int(r["New_End"])
        lut[(oc, os_, oe)] = (nc, ns, ne)
    return lut

# ------------------ Load inversion info ---------

def load_invinfo(lut: Dict[Tuple[str,int,int], Tuple[str,int,int]]) -> pd.DataFrame:
    debug(f"Loading inversion mapping from {INVINFO_TSV} ...")
    inv = pd.read_csv(INVINFO_TSV, sep="\t", engine="python", dtype=str)
    inv.columns = [c.strip() for c in inv.columns]
    debug(f"{INVINFO_TSV} columns: {list(inv.columns)}")

    needed = {"Chromosome","Start","End"}
    if not needed.issubset(inv.columns):
        raise RuntimeError(f"{INVINFO_TSV} must contain {sorted(needed)}")

    recur_col = None
    for cand in ["0_single_1_recur_consensus", "0_single_1_recur"]:
        if cand in inv.columns:
            recur_col = cand
            break
    if recur_col is None:
        raise RuntimeError("Missing recurrence column (need '0_single_1_recur_consensus' or '0_single_1_recur').")

    df = pd.DataFrame({
        "chr0":   inv["Chromosome"].map(norm_chr),
        "start0": to_num(inv["Start"]),
        "end0":   to_num(inv["End"]),
        "flag":   to_num(inv[recur_col])
    }).dropna()
    df["start0"] = df["start0"].astype(int)
    df["end0"]   = df["end0"].astype(int)
    df["Recurrence"] = df["flag"].map({0:"Single-event", 1:"Recurrent"})
    df = df.dropna(subset=["Recurrence"])

    rows = []
    for _, r in df.iterrows():
        oc, os, oe = r["chr0"], int(r["start0"]), int(r["end0"])
        key = (oc, os, oe)
        if key in lut:
            nc, ns, ne = lut[key]
        else:
            nc, ns, ne = oc, os, oe
        if ns > ne: ns, ne = ne, ns
        rows.append((nc, ns, ne, r["Recurrence"]))
    out = pd.DataFrame(rows, columns=["chr","start","end","Recurrence"])
    out = out.drop_duplicates(subset=["chr","start","end"]).reset_index(drop=True)
    counts = Counter(out["Recurrence"])
    debug(f"Inversions loaded (valid rows): {len(out)}; Recurrent={counts.get('Recurrent',0)}, Single-event={counts.get('Single-event',0)}")
    return out

# ------------------ Load output.csv --------------

def load_output() -> pd.DataFrame:
    debug(f"Loading per-region summary from {OUTPUT_CSV} ...")
    df = pd.read_csv(OUTPUT_CSV, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    debug(f"{OUTPUT_CSV} columns: {list(df.columns)}")

    need = {"chr","region_start","region_end"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{OUTPUT_CSV} is missing required columns {sorted(need)}")

    keep = {
        "chr","region_start","region_end",
        "0_pi_filtered","1_pi_filtered","0_pi","1_pi",
        "0_num_hap_filter","1_num_hap_filter",
        "0_num_hap_no_filter","1_num_hap_no_filter",
        "hudson_fst_hap_group_0v1"
    }
    present = [c for c in df.columns if c in keep]
    missing = sorted(list(keep - set(present)))
    if missing:
        debug(f"Note: output.csv missing optional columns (OK): {missing}")

    df = df[present].copy()
    df["chr"] = df["chr"].map(norm_chr)
    df["region_start"] = to_num(df["region_start"]).astype("Int64")
    df["region_end"]   = to_num(df["region_end"]).astype("Int64")
    for c in present:
        if c not in {"chr","region_start","region_end"}:
            df[c] = to_num(df[c])
    before = len(df)
    df = df.dropna(subset=["chr","region_start","region_end"]).copy()
    df["region_start"] = df["region_start"].astype(int)
    df["region_end"]   = df["region_end"].astype(int)
    debug(f"{OUTPUT_CSV} rows retained: {len(df)} (dropped {before-len(df)} with missing keys)")
    debug("First 3 normalized rows from output.csv:")
    debug(df[["chr","region_start","region_end"]].head(3).to_string(index=False))
    return df.reset_index(drop=True)

# ------------------ Matching (±1 bp) ------------

def match_regions(out_df: pd.DataFrame, inv_df: pd.DataFrame) -> pd.DataFrame:
    debug("Building ±1 bp candidate keys and performing strict match ...")
    base = out_df[["chr","region_start","region_end"]].copy()
    cands = []
    for ds in (-1,0,1):
        for de in (-1,0,1):
            t = base.copy()
            t["start"] = t["region_start"] + ds
            t["end"]   = t["region_end"]   + de
            t["mp"]    = abs(ds) + abs(de)
            cands.append(t)
    cand = pd.concat(cands, ignore_index=True)
    debug(f"Candidate rows created: {len(cand)} for {len(base)} regions")

    m = cand.merge(inv_df, on=["chr","start","end"], how="inner")
    debug(f"Candidate matches against inv_info: {len(m)}")
    if m.empty:
        raise RuntimeError("No regions matched inv_info under ±1 bp tolerance.")

    key = ["chr","region_start","region_end"]
    mm = m.merge(
        m.groupby(key)["mp"].min().reset_index().rename(columns={"mp":"min_mp"}),
        on=key, how="inner"
    )
    mm = mm[mm["mp"] == mm["min_mp"]].copy()

    counts = mm.groupby(key).size().reset_index(name="n")
    n_ambig = int((counts["n"] != 1).sum())
    if n_ambig:
        warn(f"Ambiguous best matches: {n_ambig} → dropping ambiguous keys")
    mm = mm.merge(counts, on=key, how="left")
    mm = mm[mm["n"] == 1].drop(columns=["n","min_mp"]).copy()

    out = mm.merge(out_df, on=["chr","region_start","region_end"], how="left")
    out["region_id"] = out["chr"].astype(str) + ":" + out["region_start"].astype(str) + "-" + out["region_end"].astype(str)

    n_rec = int((out["Recurrence"] == "Recurrent").sum())
    n_sgl = int((out["Recurrence"] == "Single-event").sum())
    debug(f"Matched unique regions: {len(out)}")
    debug(f"Matched recurrence counts → Recurrent={n_rec}, Single-event={n_sgl}")
    debug("First 5 matched region_ids:")
    debug(out[["region_id","Recurrence"]].head(5).to_string(index=False))
    return out.reset_index(drop=True)

# ------------------ Per-site parsers ------------

class Interval:
    __slots__ = ("start","end","data","header")
    def __init__(self, start: int, end: int, data: np.ndarray, header: str):
        self.start = int(start); self.end = int(end)
        self.data = data; self.header = header

def parse_pi_falsta() -> Dict[str, Dict[str, List[Interval]]]:
    debug(f"Parsing filtered per-site π with explicit group from {PI_FALSTA} ...")
    store: Dict[str, Dict[str, List[Interval]]] = {}
    n_head = n_match = n_bad = 0
    headers_sample = []
    by_orient = Counter()

    with open(PI_FALSTA, "r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line: continue
            if line[0] == ">":
                header = line; n_head += 1
                if len(headers_sample) < 8:
                    headers_sample.append(header.strip())
                continue
            if header is None: continue
            m = RE_PI.search(header)
            if not m:
                header = None; continue
            chrom = norm_chr(m.group(1)); s = int(m.group(2)); e = int(m.group(3)); gid = m.group(4)
            orient = "direct" if gid == "0" else "inverted"
            arr = np.fromstring(line.strip().replace("NA","nan"), sep=",", dtype=np.float64)
            exp = e - s + 1
            if arr.size != exp:
                n_bad += 1; header = None; continue
            store.setdefault(chrom, {"direct":[], "inverted":[]})
            store[chrom][orient].append(Interval(s, e, arr, header))
            n_match += 1; by_orient[orient] += 1
            header = None

    for c in store:
        for o in ("direct","inverted"):
            store[c][o].sort(key=lambda r: r.start)

    debug(f"π headers seen: {n_head}, matched(filtered+grouped): {n_match}, bad-length: {n_bad}")
    debug(f"π intervals by orientation: {dict(by_orient)}")
    if headers_sample:
        debug("Sample π headers:")
        for h in headers_sample:
            debug(f"  {h}")
    debug(f"π chromosomes loaded: {len(store)}; example: {list(store.keys())[:5]}")
    return store

def parse_fst_falsta() -> Dict[str, List[Interval]]:
    debug(f"Scanning Hudson per-site FST from {FST_FALSTA} ...")
    store: Dict[str, List[Interval]] = defaultdict(list)
    n_head = n_hud = n_bad = 0
    token_tally = Counter()
    sample_headers = []

    with open(FST_FALSTA, "r", encoding="utf-8", errors="ignore") as fh:
        header = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line: continue
            if line[0] == ">":
                header = line; n_head += 1; continue
            if header is None: continue

            m = RE_HUDSON_ANY.search(header)
            if not m:
                header = None; continue

            hlow = header.lower()
            for tok, present in [
                ("hudson", True),
                ("pairwise", "pairwise" in hlow),
                ("filtered", "filtered" in hlow or "mask_filtered" in hlow or "filtered_mask" in hlow),
                ("wc", ("weir" in hlow and "cockerham" in hlow) or ("wc" in hlow)),
                ("hap_group", "hap_group" in hlow),
            ]:
                if present:
                    token_tally[tok] += 1

            chrom = norm_chr(m.group(1)); s = int(m.group(2)); e = int(m.group(3))
            arr = np.fromstring(line.strip().replace("NA","nan"), sep=",", dtype=np.float64)
            exp = e - s + 1
            if arr.size != exp:
                n_bad += 1; header = None; continue

            store[chrom].append(Interval(s, e, arr, header))
            n_hud += 1
            if len(sample_headers) < 12:
                sample_headers.append(header.lower())
            header = None

    for c in store:
        store[c].sort(key=lambda r: r.start)

    debug(f"Total headers seen in FST file: {n_head}")
    debug(f"Hudson-like headers captured: {n_hud}, bad-length: {n_bad}")
    debug(f"Token tallies among Hudson headers: {dict(token_tally)}")
    if sample_headers:
        debug("Sample Hudson headers (lowercased):")
        for h in sample_headers[:10]:
            debug(f"  {h}")
    debug(f"FST chromosomes loaded: {len(store)}; example: {list(store.keys())[:5]}")
    return store

# ------------------ Windowing helpers -----------

def window_sum_count(wstart: int, wend: int, intervals: List[Interval]) -> Tuple[float,int]:
    if wend < wstart: return (0.0, 0)
    total = 0.0; n = 0
    for rec in intervals:
        if rec.end < wstart: continue
        if rec.start > wend: break
        s = max(wstart, rec.start); e = min(wend, rec.end)
        if s > e: continue
        a = rec.data[(s - rec.start):(e - rec.start + 1)]
        if a.size == 0: continue
        mask = np.isfinite(a)
        if not np.any(mask): continue
        vals = a[mask]
        total += float(np.sum(vals)); n += int(vals.size)
    return (total, n)

def window_mean(wstart: int, wend: int, intervals: List[Interval]) -> Tuple[float,int]:
    s, n = window_sum_count(wstart, wend, intervals)
    return (np.nan if n == 0 else (s / n), n)

def flank_union_windows(rs: int, re_: int, flank_bp: int) -> Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int], int]:
    """
    Returns (left, right, union, overlap_bp) for flanks INSIDE [rs,re_].
    left  = [rs, min(rs+flank_bp-1, re_)]
    right = [max(rs, re_-flank_bp+1), re_]
    union = left ∪ right (single interval if they overlap, otherwise we return the bounding interval)
    overlap_bp = size of overlap (>=0)
    NOTE: when there is overlap, union == [rs, re_] if region_len <= 2*flank_bp.
    """
    Ls, Le = rs, min(rs + flank_bp - 1, re_)
    Rs, Re = max(rs, re_ - flank_bp + 1), re_
    # normalize invalid
    if Le < Ls: Le = Ls - 1
    if Re < Rs: Rs = Re + 1
    # overlap
    ovl = max(0, min(Le, Re) - max(Ls, Rs) + 1)
    U_s = min(Ls, Rs)
    U_e = max(Le, Re)
    return ( (Ls,Le), (Rs,Re), (U_s,U_e), ovl )

def flank_union_sum_count(rs: int, re_: int, flank_bp: int, intervals: List[Interval]) -> Tuple[float,int,Tuple[int,int],Tuple[int,int],Tuple[int,int],int]:
    (Ls,Le), (Rs,Re), (Us,Ue), ovl = flank_union_windows(rs, re_, flank_bp)
    total = 0.0; count = 0
    # disjoint: sum left + right
    if ovl == 0:
        s1, n1 = window_sum_count(Ls, Le, intervals)
        s2, n2 = window_sum_count(Rs, Re, intervals)
        total, count = (s1 + s2, n1 + n2)
    else:
        # overlapping: use union once (avoid double-count)
        total, count = window_sum_count(Us, Ue, intervals)
    return (total, count, (Ls,Le), (Rs,Re), (Us,Ue), ovl)

# ------------------ Main ------------------------

def main():
    # map
    map_df = maybe_load_map()
    lut = build_map_lookup(map_df)

    # loads
    inv = load_invinfo(lut)
    out = load_output()
    matched = match_regions(out, inv)

    # per-site stores
    pi_store  = parse_pi_falsta()
    fst_store = parse_fst_falsta()

    # sample sizes: use max(filtered, no_filter) per orientation (more realistic)
    def pick_size(r, col_f, col_nf):
        v_f  = r.get(col_f, np.nan)
        v_nf = r.get(col_nf, np.nan)
        if pd.notna(v_f) and pd.notna(v_nf):
            return float(max(v_f, v_nf))
        if pd.notna(v_f):  return float(v_f)
        if pd.notna(v_nf): return float(v_nf)
        return np.nan

    matched["size_dir"] = matched.apply(lambda r: pick_size(r, "0_num_hap_filter", "0_num_hap_no_filter"), axis=1)
    matched["size_inv"] = matched.apply(lambda r: pick_size(r, "1_num_hap_filter", "1_num_hap_no_filter"), axis=1)

    # prefer filtered π from output.csv
    matched["pi_dir"] = matched["0_pi_filtered"].where(matched["0_pi_filtered"].notna(), matched.get("0_pi"))
    matched["pi_inv"] = matched["1_pi_filtered"].where(matched["1_pi_filtered"].notna(), matched.get("1_pi"))

    matched["region_len"] = (matched["region_end"] - matched["region_start"] + 1).astype(int)

    # DEBUG: sample-size distributions
    for k, col in [("direct","size_dir"), ("inverted","size_inv")]:
        arr = pd.to_numeric(matched[col], errors="coerce")
        ok = arr.dropna()
        debug(f"[N-size] {k:7s}: N={ok.size}, median={ok.median() if ok.size else 'NA'}, "
              f"mean={ok.mean() if ok.size else 'NA'}, min={ok.min() if ok.size else 'NA'}, max={ok.max() if ok.size else 'NA'}; "
              f"missing={arr.isna().sum()}")

    # containers
    region_fst_psite = []
    region_fst_psite_n = []
    region_fst_csv = []

    flank_pi_dir_vals = []
    flank_pi_inv_vals = []
    flank_pi_dir_n    = []
    flank_pi_inv_n    = []

    flank_fst_vals = []
    flank_fst_n    = []

    cov = {
        "region_fst_psite_have": Counter(),
        "region_fst_csv_have": Counter(),
        "flank_pi_dir_have": Counter(),
        "flank_pi_inv_have": Counter(),
        "flank_fst_have": Counter(),
        "miss_chr_fst": Counter(),
        "miss_chr_pi": Counter(),
    }

    # region spot-check capture
    xcheck_rows = []

    # iterate regions
    for i, r in matched.iterrows():
        chrom = r["chr"]; rs = int(r["region_start"]); re_ = int(r["region_end"])
        rec_label = "recurrent" if r["Recurrence"] == "Recurrent" else "single"
        L = int(r["region_len"])

        # region per-site FST
        fst_psite_val = np.nan; fst_psite_n = 0
        if chrom in fst_store:
            fst_psite_val, fst_psite_n = window_mean(rs, re_, fst_store[chrom])
            if fst_psite_n >= MIN_PSITE_BASES_ACCEPT:
                cov["region_fst_psite_have"][rec_label] += 1
        else:
            cov["miss_chr_fst"][rec_label] += 1

        region_fst_psite.append(fst_psite_val)
        region_fst_psite_n.append(fst_psite_n)

        # CSV FST (example source)
        csv_fst = r.get("hudson_fst_hap_group_0v1", np.nan)
        if pd.notna(csv_fst):
            cov["region_fst_csv_have"][rec_label] += 1
        region_fst_csv.append(csv_fst)

        # flank windows (INSIDE region)
        (Ls,Le), (Rs,Re), (Us,Ue), ovl = flank_union_windows(rs, re_, FLANK_BP)
        # π direct flank
        v = np.nan; n_used = 0
        if chrom in pi_store:
            s, n, _, _, _, _ = flank_union_sum_count(rs, re_, FLANK_BP, pi_store[chrom]["direct"])
            if n >= MIN_PSITE_BASES_ACCEPT:
                v = s / n; n_used = n; cov["flank_pi_dir_have"][rec_label] += 1
        else:
            cov["miss_chr_pi"][rec_label] += 1
        flank_pi_dir_vals.append(v); flank_pi_dir_n.append(n_used)

        # π inverted flank
        v = np.nan; n_used = 0
        if chrom in pi_store:
            s, n, _, _, _, _ = flank_union_sum_count(rs, re_, FLANK_BP, pi_store[chrom]["inverted"])
            if n >= MIN_PSITE_BASES_ACCEPT:
                v = s / n; n_used = n; cov["flank_pi_inv_have"][rec_label] += 1
        flank_pi_inv_vals.append(v); flank_pi_inv_n.append(n_used)

        # FST flank (per-site, inside region)
        v = np.nan; n_used = 0
        if chrom in fst_store:
            s, n, LL, RR, UU, ov = flank_union_sum_count(rs, re_, FLANK_BP, fst_store[chrom])
            if n >= MIN_PSITE_BASES_ACCEPT:
                v = s / n; n_used = n; cov["flank_fst_have"][rec_label] += 1
        flank_fst_vals.append(v); flank_fst_n.append(n_used)

        # capture spot-check (first 12)
        if len(xcheck_rows) < 12:
            xcheck_rows.append({
                "region_id": r["region_id"],
                "rec": rec_label,
                "region_len": L,
                "flank_left": f"{Ls}-{Le}",
                "flank_right": f"{Rs}-{Re}",
                "flank_union": f"{Us}-{Ue}",
                "flank_overlap_bp": ovl,
                "csv_fst": csv_fst,
                "psite_fst": fst_psite_val,
                "psite_cov_frac": (fst_psite_n / L) if L > 0 else np.nan,
                "flank_fst_n": n_used
            })

    # attach
    matched["fst_psite"] = region_fst_psite
    matched["fst_psite_n"] = region_fst_psite_n
    matched["fst_csv"] = region_fst_csv
    matched["flank_pi_dir"] = flank_pi_dir_vals
    matched["flank_pi_inv"] = flank_pi_inv_vals
    matched["flank_pi_dir_n"] = flank_pi_dir_n
    matched["flank_pi_inv_n"] = flank_pi_inv_n
    matched["flank_fst"] = flank_fst_vals
    matched["flank_fst_n"] = flank_fst_n

    # ---------------- COVERAGE DIAGNOSTICS -------------------------
    n_rec  = int((matched["Recurrence"] == "Recurrent").sum())
    n_sing = int((matched["Recurrence"] == "Single-event").sum())
    debug("=== COVERAGE DIAGNOSTICS (INSIDE-REGION FLANKS) ===")
    debug(f"Total matched regions → Recurrent={n_rec}, Single-event={n_sing}")
    for k, v in cov.items():
        debug(f"{k}: {dict(v)}")

    cov_frac = matched["fst_psite_n"] / matched["region_len"]
    debug(f"[per-site FST] region coverage fraction: "
          f"N={cov_frac.notna().sum()}, "
          f"median={cov_frac.median(skipna=True):.4f}, "
          f"mean={cov_frac.mean(skipna=True):.4f}, "
          f"q10={cov_frac.quantile(0.10):.4f}, q90={cov_frac.quantile(0.90):.4f}")

    debug("Spot-check (~12) of regions (flank windows are INSIDE region):")
    for row in xcheck_rows:
        debug(f"  {row['region_id']:>28s} rec={row['rec']:<9s} L={row['region_len']:<7d} "
              f"Lflank={row['flank_left']:<18s} Rflank={row['flank_right']:<18s} "
              f"U={row['flank_union']:<18s} ovlp={row['flank_overlap_bp']:<6d} "
              f"CSV={row['csv_fst']!s:<10s} psite={row['psite_fst']!s:<12s} "
              f"cov={row['psite_cov_frac']:.3f} flankFSTn={row['flank_fst_n']}")

    # ---------------- Aggregation helpers --------------------------

    def agg_by(rec_key: str, col: str) -> List[float]:
        return pd.to_numeric(
            matched.loc[matched["Recurrence"].eq(rec_key), col], errors="coerce"
        ).dropna().tolist()

    def choose_region_fst_series(rec_key: str) -> List[float]:
        sub = matched.loc[matched["Recurrence"].eq(rec_key)].copy()
        chosen = []
        used_psite = used_csv = 0
        for _, r in sub.iterrows():
            v = np.nan
            if USE_PERSITE_FST_WHEN_COVERED and pd.notna(r["fst_psite"]):
                frac = (r["fst_psite_n"] / r["region_len"]) if r["region_len"] > 0 else 0.0
                if frac >= MIN_REGION_FST_COVER_FRAC and r["fst_psite_n"] >= MIN_PSITE_BASES_ACCEPT:
                    v = float(r["fst_psite"]); used_psite += 1
                elif pd.notna(r["fst_csv"]):
                    v = float(r["fst_csv"]); used_csv += 1
            elif pd.notna(r["fst_csv"]):
                v = float(r["fst_csv"]); used_csv += 1
            chosen.append(v)
        debug(f"[FST choose] {rec_key:11s} → used_per-site={used_psite}, used_csv={used_csv}, "
              f"total_non-NA={np.isfinite(np.array(chosen)).sum()}")
        return [x for x in chosen if np.isfinite(x)]

    # ---------------- PRINT SUMMARIES -------------------------------

    # direct/inverted π (region) + flanking π
    def print_block(title: str, sizes: List[float], pis: List[float], flank_pis: List[float], total_in_cat: int):
        print(title)
        print(f"- Sample size: {describe(sizes, kind='size', n_override=total_in_cat)}")
        print(f"- Nucleotide diversity (π): {describe(pis, kind='pi')}")
        print(f"- 10 kb flanking π (inside region): {describe(flank_pis, kind='pi')}")
        print()

    print_block("Direct haplotypes (recurrent region)",
                agg_by("Recurrent","size_dir"),
                agg_by("Recurrent","pi_dir"),
                agg_by("Recurrent","flank_pi_dir"),
                total_in_cat=n_rec)

    print_block("Inverted haplotypes (recurrent region)",
                agg_by("Recurrent","size_inv"),
                agg_by("Recurrent","pi_inv"),
                agg_by("Recurrent","flank_pi_inv"),
                total_in_cat=n_rec)

    print_block("Direct haplotypes (single-event region)",
                agg_by("Single-event","size_dir"),
                agg_by("Single-event","pi_dir"),
                agg_by("Single-event","flank_pi_dir"),
                total_in_cat=n_sing)

    print_block("Inverted haplotypes (single-event region)",
                agg_by("Single-event","size_inv"),
                agg_by("Single-event","pi_inv"),
                agg_by("Single-event","flank_pi_inv"),
                total_in_cat=n_sing)

    # FST region (CSV-backed; optional per-site override with coverage)
    fst_recur = choose_region_fst_series("Recurrent")
    fst_single= choose_region_fst_series("Single-event")
    print("FST (Hudson; region means; CSV-backed like example)")
    print(f"- Recurrent regions: {describe(fst_recur, kind='fst')}")
    print(f"- Single-event regions: {describe(fst_single, kind='fst')}")
    print()

    # FST flanks (per-site INSIDE region)
    flank_recur = agg_by("Recurrent","flank_fst")
    flank_single= agg_by("Single-event","flank_fst")
    print("10 kb flanking FST (Hudson; per-site inside region)")
    print(f"- Recurrent regions: {describe(flank_recur, kind='fst')}")
    print(f"- Single-event regions: {describe(flank_single, kind='fst')}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err(str(e))
        sys.exit(1)
