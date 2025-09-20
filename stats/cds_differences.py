from __future__ import annotations
import os
import re
import sys
import csv
import math
import shutil
import tempfile
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
from tqdm import tqdm
HAVE_TQDM = True
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===========================
# Utility / Normalization
# ===========================

def normalize_chr(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    s = s.strip()
    if s.upper() in {"X", "Y", "MT", "M"}:
        return "MT" if s.upper() in {"MT", "M"} else s.upper()
    return s

def safe_int(x: str) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None

def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)

def consensus_label(c: int) -> str:
    return "Recurrent" if c == 1 else "Single-event"

def group_label(g: int) -> str:
    return "Direct" if g == 0 else "Inverted"

# ===========================
# Load inv_info.tsv
# ===========================

def load_inv_info(path: str):
    if not os.path.exists(path):
        sys.exit(f"ERROR: inv_info.tsv not found at {path}")

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []

        def find_col(substr: str):
            for h in header:
                if substr == h or substr in h:
                    return h
            return None

        col_chr = find_col("Chromosome")
        col_start = find_col("Start")
        col_end = find_col("End")
        col_cons = find_col("0_single_1_recur_consensus")

        if not all([col_chr, col_start, col_end, col_cons]):
            sys.exit("ERROR: inv_info.tsv missing required columns. Found: " + ", ".join(header))

        rows_all = []
        rows_by_cons = {0: [], 1: []}
        exact_triplets = set()

        for row in reader:
            chr_raw = row.get(col_chr, "")
            start_raw = row.get(col_start, "")
            end_raw = row.get(col_end, "")
            cons_raw = row.get(col_cons, "")

            chr_norm = normalize_chr(chr_raw)
            start = safe_int(start_raw)
            end = safe_int(end_raw)

            try:
                cons = int(str(cons_raw).strip())
            except Exception:
                cons = None

            if cons not in (0, 1):
                continue
            if chr_norm == "" or start is None or end is None:
                continue

            d = dict(row)
            d["_chr_norm"] = chr_norm
            d["_Start_int"] = start
            d["_End_int"] = end
            d["_consensus_int"] = cons

            rows_all.append(d)
            rows_by_cons[cons].append(d)
            exact_triplets.add((chr_norm, start, end))

    return rows_all, rows_by_cons, exact_triplets

# ===========================
# Filename parsing (STRICT)
# ===========================

CDS_RE = re.compile(
    r'^group(?P<phy_group>[01])_'
    r'(?P<gene_name>[A-Za-z0-9]+)_'
    r'(?P<gene_id>[^_]+)_'
    r'(?P<transcript_id>[^_]+)_'
    r'chr(?P<chr>[^_]+)_'
    r'cds_start(?P<cds_start>\d+)_cds_end(?P<cds_end>\d+)_'
    r'inv_start(?P<inv_start>\d+)_inv_end(?P<inv_end>\d+)\.phy$'
)

REGION_RE = re.compile(
    r'^inversion_group(?P<phy_group>[01])_'
    r'(?P<chr>[^_]+)_start(?P<start>\d+)_end(?P<end>\d+)\.phy$'
)

def parse_cds_filename(fn: str) -> Optional[Dict[str, str]]:
    m = CDS_RE.match(fn)
    if not m:
        return None
    g = m.groupdict()
    return {
        "filename": fn,
        "phy_group": int(g["phy_group"]),
        "gene_name": g["gene_name"],
        "gene_id": g["gene_id"],
        "transcript_id": g["transcript_id"],
        "chr": normalize_chr(g["chr"]),
        "cds_start": int(g["cds_start"]),
        "cds_end": int(g["cds_end"]),
        "inv_start": int(g["inv_start"]),
        "inv_end": int(g["inv_end"]),
    }

def parse_region_filename(fn: str) -> Optional[Dict[str, str]]:
    m = REGION_RE.match(fn)
    if not m:
        return None
    g = m.groupdict()
    return {
        "filename": fn,
        "phy_group": int(g["phy_group"]),
        "chr": normalize_chr(g["chr"]),
        "start": int(g["start"]),
        "end": int(g["end"]),
    }

# ===========================
# PHYLIP parsing (STRICT)
# ===========================

class PhylipParseError(Exception):
    pass

# Non-standard: `<name ending _L/_R><sequence>` (no space)
NONSTD_LINE_RE = re.compile(r'^(?P<name>.*?_[LR])(?P<seq>[ACGTRYKMSWBDHVN\-\.\?]+)$', re.IGNORECASE)

def read_nonstandard_phylip(path: str, n: int, m: int) -> List[Tuple[str,str]]:
    out: List[Tuple[str,str]] = []
    with open(path, "r") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]

    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Empty file")
    idx += 1  # past header

    while len(out) < n and idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if line == "":
            continue
        mobj = NONSTD_LINE_RE.match(line)
        if not mobj:
            raise PhylipParseError(f"Non-standard PHYLIP: bad sequence line: '{line[:60]}...'")
        name = mobj.group("name")
        seq = mobj.group("seq").upper()
        if len(seq) != m:
            raise PhylipParseError(f"Sequence length mismatch: expected {m}, got {len(seq)} for {name}")
        out.append((name, seq))

    if len(out) != n:
        raise PhylipParseError(f"Expected {n} sequences; got {len(out)}")
    return out

def read_standard_phylip(path: str, n: int, m: int) -> List[Tuple[str,str]]:
    with open(path, "r") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]

    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Empty file")
    header = lines[idx].strip()
    idx += 1

    names: List[str] = []
    seqs: List[str] = [""] * n

    for i in range(n):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        if idx >= len(lines):
            raise PhylipParseError(f"Truncated first block at sequence {i+1}/{n}")
        line = lines[idx]
        idx += 1
        if len(line) < 10:
            raise PhylipParseError("Standard PHYLIP requires >=10 chars for name field")
        name = line[:10].strip()
        seq_chunk = "".join(line[10:].split()).upper()
        if name == "":
            raise PhylipParseError("Empty name in standard PHYLIP")
        names.append(name)
        seqs[i] += seq_chunk

    while any(len(s) < m for s in seqs):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        for i in range(n):
            if idx >= len(lines):
                raise PhylipParseError("Unexpected EOF in interleaved blocks")
            line = lines[idx].strip()
            idx += 1
            if line == "":
                raise PhylipParseError("Empty line inside interleaved block")
            seq_chunk = "".join(line.split()).upper()
            seqs[i] += seq_chunk

    for i in range(n):
        if len(seqs[i]) != m:
            raise PhylipParseError(f"Length mismatch for '{names[i]}': expected {m}, got {len(seqs[i])}")
    return list(zip(names, seqs))

def read_phylip_sequences_strict(path: str) -> List[Tuple[str,str]]:
    if not os.path.exists(path):
        raise PhylipParseError(f"File not found: {path}")
    with open(path, "r") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]

    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Empty file (no header)")
    header = lines[idx].strip()
    idx += 1
    parts = header.split()
    if len(parts) != 2:
        raise PhylipParseError(f"Malformed header: '{header}'")
    try:
        n = int(parts[0]); m = int(parts[1])
    except Exception:
        raise PhylipParseError(f"Non-integer header values: '{header}'")
    if n < 1 or m < 1:
        raise PhylipParseError(f"Non-positive n or m in header: '{header}'")

    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines):
        raise PhylipParseError("Header present but no sequences")
    first_line = lines[idx].strip()

    if NONSTD_LINE_RE.match(first_line):
        return read_nonstandard_phylip(path, n, m)
    else:
        return read_standard_phylip(path, n, m)

# ===========================
# Fast pairwise differences
# ===========================

def encode_ascii_matrix(seqs: List[str]) -> np.ndarray:
    """Return (n, m) uint8 matrix of ASCII codes."""
    n = len(seqs)
    m = len(seqs[0])
    X = np.empty((n, m), dtype=np.uint8)
    for i, s in enumerate(seqs):
        X[i, :] = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
    return X

def hamming_matrix_chunked(X: np.ndarray, chunk_sites: int) -> np.ndarray:
    """
    Compute full (n,n) Hamming distance matrix with column chunking.
    X: (n, m) uint8
    """
    n, m = X.shape
    D = np.zeros((n, n), dtype=np.int32)
    for start in range(0, m, chunk_sites):
        end = min(start + chunk_sites, m)
        blk = X[:, start:end]               # (n, k)
        # broadcast compare: (n,1,k) != (1,n,k) -> (n,n,k), then sum over k
        diffs = (blk[:, None, :] != blk[None, :, :])
        D += diffs.sum(axis=2, dtype=np.int32)
    return D

def compute_file_metrics_and_pairs(dataset: str,
                                   filename: str,
                                   out_tmp_dir: str,
                                   chunk_sites: int) -> Dict:
    """
    Worker function:
    - Parses PHYLIP
    - Builds (n,m) matrix
    - Computes (n,n) Hamming matrix with chunked broadcasting
    - Writes pairwise TSV to RAM (out_tmp_dir)
    - Returns summary metrics
    """
    try:
        pairs = read_phylip_sequences_strict(filename)
        names = [n for n, _ in pairs]
        seqs = [s for _, s in pairs]
        nseq = len(seqs)
        if nseq < 2:
            return {
                "filename": filename,
                "error": "N_LT_2",
                "error_detail": "Fewer than 2 sequences",
            }
        mlen = len(seqs[0])
        if any(len(s) != mlen for s in seqs):
            return {
                "filename": filename,
                "error": "UNEQUAL_LENGTHS",
                "error_detail": "Sequences have unequal lengths",
            }
        X = encode_ascii_matrix(seqs)

        # Column chunking; allow adaptive backoff on MemoryError
        cur_chunk = max(1024, int(chunk_sites))
        while True:
            try:
                D = hamming_matrix_chunked(X, cur_chunk)
                break
            except MemoryError:
                if cur_chunk <= 1024:
                    raise
                cur_chunk //= 2

        # Summary stats
        iu = np.triu_indices(nseq, 1)
        pair_ndiff = D[iu]
        n_pairs = pair_ndiff.size
        n_ident = int(np.count_nonzero(pair_ndiff == 0))
        prop_ident = (n_ident / n_pairs) if n_pairs > 0 else float('nan')

        # Write pairwise TSV to RAM
        out_name = f"pairs_{dataset}__{filename}.tsv"
        tmp_path = os.path.join(out_tmp_dir, out_name)
        with open(tmp_path, "w", newline="") as pf:
            w = csv.writer(pf, delimiter="\t")
            w.writerow(["sample1","sample2","n_sites","n_diff_sites","prop_sites_different"])
            m = float(mlen)
            # iterate upper triangle pairs once
            idx = 0
            for i in range(nseq):
                for j in range(i+1, nseq):
                    ndiff = int(pair_ndiff[idx])
                    idx += 1
                    w.writerow([names[i], names[j], mlen, ndiff, f"{(ndiff/m):.6f}"])

        return {
            "filename": filename,
            "n_sequences": nseq,
            "n_pairs": int(n_pairs),
            "n_identical_pairs": int(n_ident),
            "prop_identical_pairs": float(prop_ident),
            "tmp_pairs_path": tmp_path,
            "used_chunk": cur_chunk,
        }

    except PhylipParseError as e:
        return {
            "filename": filename,
            "error": "PARSE_ERROR",
            "error_detail": str(e),
        }
    except Exception as e:
        return {
            "filename": filename,
            "error": "UNEXPECTED_ERROR",
            "error_detail": repr(e),
        }

# ===========================
# Main
# ===========================

def main():
    # Config
    chunk_sites = int(os.environ.get("PAIR_CHUNK_SITES", "16384"))
    max_workers_env = os.environ.get("MAX_WORKERS", "").strip()
    max_workers = int(max_workers_env) if max_workers_env.isdigit() else (os.cpu_count() or 1)

    # RAM temp root
    ram_root = os.environ.get("TMPDIR") or "/dev/shm"
    if not (os.path.exists(ram_root) and os.access(ram_root, os.W_OK)):
        print(f"WARNING: RAM temp dir '{ram_root}' not available; using current directory.", file=sys.stderr)
        ram_root = "."

    session_tmp_dir = os.path.join(ram_root, f"phy_pairs_tmp_{os.getpid()}")
    os.makedirs(session_tmp_dir, exist_ok=True)

    print(">>> Loading inv_info.tsv ...")
    rows_all, rows_by_cons, exact_triplets = load_inv_info("inv_info.tsv")
    print(f"    Total inv_info rows (consensus in {{0,1}}): {len(rows_all)}")
    print(f"    Recurrent rows: {len(rows_by_cons[1])}")
    print(f"    Single-event rows: {len(rows_by_cons[0])}")

    # Discover .phy files
    phy_files = sorted([f for f in os.listdir(".") if f.endswith(".phy") and os.path.isfile(f)])
    print(f">>> Found {len(phy_files)} .phy files.")

    # Parse filenames
    cds_files: List[Dict] = []
    region_files: List[Dict] = []
    iterable = tqdm(phy_files, desc="Parsing filenames") if HAVE_TQDM else phy_files
    for fn in iterable:
        p = parse_cds_filename(fn)
        if p:
            cds_files.append(p); continue
        r = parse_region_filename(fn)
        if r:
            region_files.append(r); continue

    print(f"    CDS files parsed: {len(cds_files)}")
    print(f"    Region files parsed: {len(region_files)}")

    cds_by_filename = {d["filename"]: d for d in cds_files}
    region_by_filename = {d["filename"]: d for d in region_files}

    # Categories
    categories = defaultdict(set)  # (dataset, consensus, group) -> set(filenames)
    cds_sanity_exact_match: Dict[str, int] = {}
    cds_dedup_removed: List[Tuple[str, str, int, int]] = []  # (key_str, removed_fn, cons, grp)

    # Skips (everything)
    skipped_records: List[Dict] = []

    def record_skip(dataset: str, cons: Optional[int], grp: Optional[int], fn: str, reason: str, detail: str):
        skipped_records.append({
            "dataset": dataset,
            "consensus": cons if cons in (0,1) else "NA",
            "consensus_label": consensus_label(cons) if cons in (0,1) else "NA",
            "phy_group": grp if grp in (0,1) else "NA",
            "group_label": group_label(grp) if grp in (0,1) else "NA",
            "filename": fn,
            "reason": reason,
            "detail": detail,
        })

    # Map CDS by CDS overlap
    print(">>> Mapping CDS files to consensus categories (by CDS overlap) ...")
    for rec in (tqdm(cds_files, desc="CDS mapping") if HAVE_TQDM else cds_files):
        fn = rec["filename"]
        chr_ = rec["chr"]
        cds_s = rec["cds_start"]; cds_e = rec["cds_end"]
        inv_s = rec["inv_start"]; inv_e = rec["inv_end"]
        grp = rec["phy_group"]

        overl_1 = any((chr_ == row["_chr_norm"]) and intervals_overlap(cds_s, cds_e, row["_Start_int"], row["_End_int"])
                      for row in rows_by_cons[1])
        overl_0 = any((chr_ == row["_chr_norm"]) and intervals_overlap(cds_s, cds_e, row["_Start_int"], row["_End_int"])
                      for row in rows_by_cons[0])

        exact_ok = 1 if (chr_, inv_s, inv_e) in exact_triplets else 0
        cds_sanity_exact_match[fn] = exact_ok

        if not overl_1 and not overl_0:
            record_skip("CDS", None, None, fn, "CDS_NO_OVERLAP",
                        f"CDS [{cds_s},{cds_e}] on chr {chr_} overlaps no inv_info row")
            continue

        if overl_1:
            categories[("CDS", 1, grp)].add(fn)
        if overl_0:
            categories[("CDS", 0, grp)].add(fn)

    # Deduplicate CDS per category
    print(">>> Deduplicating CDS entries per category by (phy_group, chr, cds_start, cds_end, consensus) ...")
    for cons in (1, 0):
        for grp in (0, 1):
            fns = sorted(categories.get(("CDS", cons, grp), []))
            key_to_fn = {}
            for fn in fns:
                r = cds_by_filename[fn]
                key = (grp, r["chr"], r["cds_start"], r["cds_end"], cons)
                if key in key_to_fn:
                    categories[("CDS", cons, grp)].remove(fn)
                    cds_dedup_removed.append(("|".join(map(str, key)), fn, cons, grp))
                    record_skip("CDS", cons, grp, fn, "CDS_DEDUP_REMOVED",
                                f"Duplicate CDS key {key}")
                else:
                    key_to_fn[key] = fn

    # Regions: exact match only
    print(">>> Mapping Region files to consensus categories (by EXACT match) ...")
    for rec in (tqdm(region_files, desc="Region mapping") if HAVE_TQDM else region_files):
        fn = rec["filename"]; chr_ = rec["chr"]; s = rec["start"]; e = rec["end"]; grp = rec["phy_group"]
        if (chr_, s, e) not in exact_triplets:
            record_skip("REGION", None, grp, fn, "REGION_NO_EXACT_MATCH",
                        f"No inv_info exact match for ({chr_},{s},{e})")
            continue
        has_1 = any((chr_ == row["_chr_norm"] and s == row["_Start_int"] and e == row["_End_int"]) for row in rows_by_cons[1])
        has_0 = any((chr_ == row["_chr_norm"] and s == row["_Start_int"] and e == row["_End_int"]) for row in rows_by_cons[0])
        if has_1: categories[("REGION", 1, grp)].add(fn)
        if has_0: categories[("REGION", 0, grp)].add(fn)

    # Unique files per dataset
    cds_fns_all = sorted(set().union(*[categories.get(("CDS", c, g), set()) for c in (1,0) for g in (0,1)]))
    region_fns_all = sorted(set().union(*[categories.get(("REGION", c, g), set()) for c in (1,0) for g in (0,1)]))

    print(f">>> Will compute pairwise for {len(cds_fns_all)} CDS files and {len(region_fns_all)} REGION files.")
    print(f"    Using chunk size: {chunk_sites} sites; max_workers: {max_workers}")
    print(f"    RAM temp dir: {session_tmp_dir} (pairwise TSVs + main TSVs)")

    # ===========================
    # Multiprocess per-file compute (CDS and REGION)
    # ===========================

    results_by_file: Dict[str, Dict] = {}

    def run_pool(dataset: str, files: List[str]):
        if not files:
            return
        desc = f"Computing pairwise ({dataset})"
        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for fn in files:
                tasks.append(ex.submit(compute_file_metrics_and_pairs,
                                       dataset, fn, session_tmp_dir, chunk_sites))
            it = as_completed(tasks)
            it = tqdm(it, total=len(tasks), desc=desc) if HAVE_TQDM else it
            for fut in it:
                res = fut.result()
                fn = res["filename"]
                if "error" in res:
                    # record skip for all categories this file belongs to (for this dataset)
                    for (ds, cons, grp), fset in categories.items():
                        if ds == dataset and fn in fset:
                            record_skip(ds, cons, grp, fn, res["error"], res.get("error_detail",""))
                else:
                    results_by_file[fn] = res

    run_pool("CDS", cds_fns_all)
    run_pool("REGION", region_fns_all)

    # ===========================
    # Write main TSVs to RAM, then move to disk at end
    # ===========================

    cds_out = "cds_identical_proportions.tsv"
    region_out = "region_identical_proportions.tsv"
    skipped_out = "skipped_details.tsv"

    cds_tmp = os.path.join(session_tmp_dir, cds_out)
    region_tmp = os.path.join(session_tmp_dir, region_out)
    skipped_tmp = os.path.join(session_tmp_dir, skipped_out)

    print(f">>> Writing {cds_out} (to RAM) ...")
    with open(cds_tmp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "dataset","consensus","consensus_label","phy_group","group_label","filename",
            "gene_name","gene_id","transcript_id","chr",
            "cds_start","cds_end","inv_start","inv_end",
            "n_sequences","n_pairs","n_identical_pairs","prop_identical_pairs",
            "inv_exact_match"
        ])
        for cons in (1, 0):
            for grp in (0, 1):
                for fn in sorted(categories.get(("CDS", cons, grp), [])):
                    rec = cds_by_filename[fn]
                    met = results_by_file.get(fn, {})
                    prop = met.get("prop_identical_pairs", float('nan'))
                    prop_str = f"{prop:.6f}" if isinstance(prop, float) and not math.isnan(prop) else "NA"
                    w.writerow([
                        "CDS", cons, consensus_label(cons), grp, group_label(grp), fn,
                        rec["gene_name"], rec["gene_id"], rec["transcript_id"], rec["chr"],
                        rec["cds_start"], rec["cds_end"], rec["inv_start"], rec["inv_end"],
                        met.get("n_sequences",""),
                        met.get("n_pairs",""),
                        met.get("n_identical_pairs",""),
                        prop_str,
                        cds_sanity_exact_match.get(fn, 0),
                    ])

    print(f">>> Writing {region_out} (to RAM) ...")
    with open(region_tmp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "dataset","consensus","consensus_label","phy_group","group_label","filename",
            "chr","start","end",
            "n_sequences","n_pairs","n_identical_pairs","prop_identical_pairs"
        ])
        for cons in (1, 0):
            for grp in (0, 1):
                for fn in sorted(categories.get(("REGION", cons, grp), [])):
                    rec = region_by_filename[fn]
                    met = results_by_file.get(fn, {})
                    prop = met.get("prop_identical_pairs", float('nan'))
                    prop_str = f"{prop:.6f}" if isinstance(prop, float) and not math.isnan(prop) else "NA"
                    w.writerow([
                        "REGION", cons, consensus_label(cons), grp, group_label(grp), fn,
                        rec["chr"], rec["start"], rec["end"],
                        met.get("n_sequences",""),
                        met.get("n_pairs",""),
                        met.get("n_identical_pairs",""),
                        prop_str,
                    ])

    print(f">>> Writing {skipped_out} (to RAM) ...")
    with open(skipped_tmp, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["dataset","consensus","consensus_label","phy_group","group_label","filename","reason","detail"])
        for row in skipped_records:
            w.writerow([
                row["dataset"], row["consensus"], row["consensus_label"],
                row["phy_group"], row["group_label"], row["filename"],
                row["reason"], row["detail"]
            ])

    # ===========================
    # Move outputs from RAM to working directory
    # ===========================

    print(">>> Moving outputs from RAM to working directory ...")
    for tmp, final in [(cds_tmp, cds_out), (region_tmp, region_out), (skipped_tmp, skipped_out)]:
        shutil.move(tmp, final)

    # Move pairwise files
    print(">>> Moving per-file pairwise TSVs from RAM ...")
    moved = 0
    for name in os.listdir(session_tmp_dir):
        if name.startswith("pairs_CDS__") or name.startswith("pairs_REGION__"):
            shutil.move(os.path.join(session_tmp_dir, name), name)
            moved += 1
    print(f"    Moved {moved} pairwise TSVs.")

    # ===========================
    # Summaries (means/medians of identity fraction; unweighted)
    # ===========================

    def summarize(dataset: str):
        print(f"\n=== {dataset} SUMMARY (fraction of IDENTICAL comparisons) ===")
        for cons in (1, 0):
            for grp in (0, 1):
                fns = sorted(categories.get((dataset, cons, grp), []))
                vals = []
                used = 0
                skipped = 0
                for fn in fns:
                    met = results_by_file.get(fn)
                    if met and "prop_identical_pairs" in met and isinstance(met["prop_identical_pairs"], float) and not math.isnan(met["prop_identical_pairs"]):
                        vals.append(met["prop_identical_pairs"]); used += 1
                    else:
                        skipped += 1
                mean_str = f"{statistics.mean(vals):.6f}" if vals else "NA"
                median_str = f"{statistics.median(vals):.6f}" if vals else "NA"
                print(f"{consensus_label(cons)}/{group_label(grp)}: n_files={len(fns)}, used={used}, skipped={skipped}, mean={mean_str}, median={median_str}")

    summarize("CDS")
    summarize("REGION")

    # ===========================
    # Gene lists (CDS)
    # ===========================

    def cds_gene_lists_by_consensus(cons_value: int) -> Dict[str, List[str]]:
        gene_to_tx = defaultdict(set)
        for grp in (0, 1):
            for fn in categories.get(("CDS", cons_value, grp), []):
                rec = cds_by_filename[fn]
                gene_to_tx[rec["gene_name"]].add(rec["transcript_id"])
        return {g: sorted(list(txs)) for g, txs in gene_to_tx.items()}

    print("\n=== Gene list (CDS overlapping Single-event intervals) ===")
    g0 = cds_gene_lists_by_consensus(0)
    for gene in sorted(g0.keys()):
        print(f"{gene}\t{';'.join(g0[gene])}")
    print(f"Total unique genes (Single-event): {len(g0)}")

    print("\n=== Gene list (CDS overlapping Recurrent intervals) ===")
    g1 = cds_gene_lists_by_consensus(1)
    for gene in sorted(g1.keys()):
        print(f"{gene}\t{';'.join(g1[gene])}")
    print(f"Total unique genes (Recurrent): {len(g1)}")

    # Sanity check report
    n_cds_exact_ok = sum(cds_sanity_exact_match.get(fn, 0) for fn in cds_fns_all)
    n_cds_total = len(cds_fns_all)
    print(f"\nCDS inv_start/inv_end exact-match sanity: {n_cds_exact_ok}/{n_cds_total} files matched an inv_info interval exactly.")

    # Cleanup RAM dir
    try:
        os.rmdir(session_tmp_dir)
    except OSError:
        pass

    print("\nAll done.")
    print(f"Wrote: {cds_out}")
    print(f"Wrote: {region_out}")
    print(f"Wrote: {skipped_out}")
    print("Pairwise TSVs written for each included file (in current directory).")

if __name__ == "__main__":
    main()
