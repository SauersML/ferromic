import os, sys, math, re, subprocess, errno
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional, DefaultDict
from collections import defaultdict
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================================================================
# HARD-CODED GCS PATHS (NO WILDCARDS, NO DISCOVERY)
# ======================================================================

# Per-chromosome PLINK shards live here (files are chr{1..22,X,Y}.bim/bed/fam)
GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"

# Allow-list of desired variants (POS + target allele). You can keep using the URL:
# format: "<chr>:<pos> <allele>", one per line. Only SNPs with A/C/G/T alleles are honored.
ALLOW_LIST_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/vcf_list.txt"

# Output files (written locally)
OUT_BIM = "subset.bim"
OUT_BED = "subset.bed"
OUT_FAM = "subset.fam"
OUT_PASSED = "passed_snvs.txt"

# I/O concurrency for a *small set* of SNP blocks. We won’t coalesce; we fetch each block exactly.
MAX_FETCH_THREADS = 8

# ======================================================================
# UTILITIES
# ======================================================================

def require_project() -> str:
    pid = os.getenv("GOOGLE_PROJECT")
    if not pid:
        raise RuntimeError("Set GOOGLE_PROJECT for requester-pays.")
    return pid

def run(cmd: List[str], capture: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=capture, text=text)
    except subprocess.CalledProcessError as e:
        # Hard crash, no graceful fallback
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {stderr or e}")

def gsutil_stat_size(gs_uri: str) -> int:
    out = run(["gsutil", "-u", require_project(), "stat", gs_uri]).stdout
    m = re.search(r"Content-Length:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Unable to parse size for {gs_uri}")
    return int(m.group(1))

def gsutil_cat_lines(gs_uri: str) -> Iterable[str]:
    # stream text lines
    proc = subprocess.Popen(["gsutil", "-u", require_project(), "cat", gs_uri],
                            stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    if proc.stdout is None:
        raise RuntimeError("Failed to open gsutil pipe")
    try:
        for line in proc.stdout:
            yield line
    finally:
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"gsutil cat failed for {gs_uri} (exit {ret})")

def gsutil_cat_range(gs_uri: str, start: int, end_inclusive: int) -> bytes:
    # precise ranged read — no fallback
    return subprocess.check_output(["gsutil", "-u", require_project(), "cat", "-r", f"{start}-{end_inclusive}", gs_uri])

def norm_chr(s: str) -> str:
    s = s.strip()
    return s[3:] if s.lower().startswith("chr") else s

# ======================================================================
# BED DECODE (2 bits per genotype; PLINK SNP-major)
# ======================================================================

def build_luts():
    miss4 = np.zeros(256, dtype=np.uint16)
    d1_4  = np.zeros(256, dtype=np.uint16)
    d2_4  = np.zeros(256, dtype=np.uint16)
    miss_k = {1: np.zeros(256, dtype=np.uint8),
              2: np.zeros(256, dtype=np.uint8),
              3: np.zeros(256, dtype=np.uint8)}
    d1_k  = {1: np.zeros(256, dtype=np.uint8),
             2: np.zeros(256, dtype=np.uint8),
             3: np.zeros(256, dtype=np.uint8)}
    d2_k  = {1: np.zeros(256, dtype=np.uint8),
             2: np.zeros(256, dtype=np.uint8),
             3: np.zeros(256, dtype=np.uint8)}
    for b in range(256):
        pairs = [(b >> (2*i)) & 0b11 for i in range(4)]
        def accum(k):
            m = d1 = d2 = 0
            for c in pairs[:k]:
                if c == 0b01: m += 1
                elif c == 0b00: d1 += 2
                elif c == 0b10: d1 += 1; d2 += 1
                elif c == 0b11: d2 += 2
            return m, d1, d2
        m4, d14, d24 = accum(4)
        miss4[b] = m4; d1_4[b] = d14; d2_4[b] = d24
        for k in (1,2,3):
            mk, d1k, d2k = accum(k)
            miss_k[k][b] = mk; d1_k[k][b] = d1k; d2_k[k][b] = d2k
    return miss4, d1_4, d2_4, miss_k, d1_k, d2_k

MISS4, D1_4, D2_4, MISS_K, D1_K, D2_K = build_luts()

def decode_bed_block(block: bytes, n_samples: int) -> Tuple[int,int,int]:
    """
    Decode a *single SNP block* (exactly ceil(n_samples/4) bytes) to:
      (missing_count, doseA1, doseA2)   where doseA1/2 are total allele counts across samples.
    """
    bpf = len(block)
    full = n_samples // 4
    rem  = n_samples % 4
    arr = np.frombuffer(block, dtype=np.uint8)

    miss = D1 = D2 = 0
    if full:
        core = arr[:full]
        miss += int(MISS4[core].sum(dtype=np.int64))
        D1   += int(D1_4[core].sum(dtype=np.int64))
        D2   += int(D2_4[core].sum(dtype=np.int64))
    if rem:
        last = arr[full]
        miss += int(MISS_K[rem][last])
        D1   += int(D1_K[rem][last])
        D2   += int(D2_K[rem][last])
    return miss, D1, D2

# ======================================================================
# DATA CLASSES
# ======================================================================

@dataclass
class Shard:
    chrom: str     # normalized like "17" or "X"
    bim_uri: str
    bed_uri: str
    fam_uri: str
    bim_size: int
    bed_size: int

@dataclass
class Candidate:
    chrom: str
    bp: int
    allele: str
    snp_index: int
    snp_id: str
    a1: str
    a2: str
    bim_line: str

# ======================================================================
# CORE PIPELINE (small targeted set; no coalescing)
# ======================================================================

def load_allow_list(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    allow: DefaultDict[Tuple[str,int], Set[str]] = defaultdict(set)
    order: List[Tuple[str,int,str]] = []
    for raw in r.iter_lines(decode_unicode=True):
        s = (raw or "").strip()
        if not s: continue
        parts = s.split()
        if len(parts) < 2: continue
        loc, al = parts[0], parts[1].upper()
        if al not in {"A","C","G","T"}:  # strict SNP-only
            continue
        if ":" not in loc: continue
        cs, ps = loc.split(":", 1)
        try:
            bp = int(float(ps))
        except:
            continue
        c = norm_chr(cs)
        allow[(c, bp)].add(al)
        order.append((c, bp, al))
    if not allow:
        raise RuntimeError("Allow-list is empty after filtering.")
    return allow, order

def build_shard_for_chr(chr_norm: str) -> Shard:
    # Construct exact filenames; no wildcards
    pref = "chr" + chr_norm
    bim = GCS_DIR + f"{pref}.bim"
    bed = GCS_DIR + f"{pref}.bed"
    fam = GCS_DIR + f"{pref}.fam"
    return Shard(chr_norm, bim, bed, fam, gsutil_stat_size(bim), gsutil_stat_size(bed))

def count_fam_lines(fam_uri: str) -> int:
    n = 0
    for _ in gsutil_cat_lines(fam_uri):
        n += 1
    if n <= 0:
        raise RuntimeError(f"Empty FAM: {fam_uri}")
    return n

def validate_bed_header(bed_uri: str):
    hdr = gsutil_cat_range(bed_uri, 0, 2)
    if hdr != b"\x6c\x1b\x01":
        raise RuntimeError(f"Not SNP-major BED (header {hdr.hex()}): {bed_uri}")

def find_targets_in_bim(bim_uri: str,
                        required: Dict[int, Set[str]],
                        chrom_norm: str) -> List[Candidate]:
    """
    Stream the .bim until we pass max(required positions). Stop early.
    Only return entries where requested allele ∈ {A1,A2}.
    """
    targets = []
    if not required:
        return targets
    max_bp = max(required.keys())
    idx = 0
    found_pos: Set[int] = set()
    for line in gsutil_cat_lines(bim_uri):
        parts = line.strip().split()
        if len(parts) < 6:
            idx += 1; continue
        # BIM: chrom, snp_id, cm, bp, a1, a2
        chrom_raw, snp_id, cm, bp_raw, a1, a2 = parts[:6]
        try:
            bp = int(float(bp_raw))
        except:
            idx += 1; continue
        if bp in required:
            a1u, a2u = a1.upper(), a2.upper()
            for al in list(required[bp]):
                if al == a1u or al == a2u:
                    targets.append(Candidate(chrom_norm, bp, al, idx, snp_id, a1u, a2u, line))
                    found_pos.add(bp)
        # Early exit: if BIM sorted by pos, once we pass max pos & found all, stop
        if bp > max_bp and len(found_pos) == len(required):
            break
        idx += 1
    return targets

def compute_stats_for_candidates(bed_uri: str,
                                 bpf: int,
                                 n_samples: int,
                                 cands: List[Candidate]) -> Dict[int, Tuple[int,int,int]]:
    """
    Fetch *each* SNP block exactly once (best for tiny target sets),
    decode to (missing, D1, D2). Returns map snp_index -> tuple.
    """
    if not cands:
        return {}
    def fetch_one(c: Candidate):
        start = 3 + c.snp_index * bpf
        end   = start + bpf - 1
        blob = gsutil_cat_range(bed_uri, start, end)
        if len(blob) != bpf:
            raise RuntimeError(f"Range size mismatch for {bed_uri} at index {c.snp_index}")
        return c.snp_index, decode_bed_block(blob, n_samples)

    stats: Dict[int, Tuple[int,int,int]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_FETCH_THREADS, len(cands))) as ex:
        futs = [ex.submit(fetch_one, c) for c in cands]
        for f in as_completed(futs):
            snp_idx, tup = f.result()
            stats[snp_idx] = tup
    return stats

def write_subset_fam(from_fam_uri: str):
    with open(OUT_FAM, "w") as fout:
        for ln in gsutil_cat_lines(from_fam_uri):
            fout.write(ln)

def write_subset_bim(cands_in_order: List[Candidate]):
    with open(OUT_BIM, "w") as f:
        for c in cands_in_order:
            f.write(c.bim_line)

def assemble_subset_bed(bed_uri: str, bpf: int, ordered_indices: List[int]):
    with open(OUT_BED, "wb") as f:
        f.write(b"\x6c\x1b\x01")
        for idx in ordered_indices:
            start = 3 + idx * bpf
            end   = start + bpf - 1
            block = gsutil_cat_range(bed_uri, start, end)
            if len(block) != bpf:
                raise RuntimeError(f"Unexpected block size for SNP index {idx}")
            f.write(block)
    # Quick integrity
    expected = 3 + len(ordered_indices)*bpf
    actual = os.path.getsize(OUT_BED)
    if actual != expected:
        raise RuntimeError(f"subset.bed size {actual} != expected {expected}")

# ======================================================================
# DRIVER
# ======================================================================

def main():
    print("== Minimal PLINK subsetting (GCS, requester-pays; small target sets) ==")
    project = require_project()  # fail fast

    # 1) Load allow-list (you can keep it to 3 nearby SNPs for speed)
    allow_map, allow_order = load_allow_list(ALLOW_LIST_URL)
    # Group by chromosome
    by_chr: DefaultDict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for c, bp, al in allow_order:
        by_chr[norm_chr(c)][bp].add(al)

    # 2) For each involved chromosome, build shard (no discovery, hard-coded names)
    all_candidates: List[Candidate] = []
    shards: Dict[str, Shard] = {}
    for c in by_chr.keys():
        shard = build_shard_for_chr(c)
        validate_bed_header(shard.bed_uri)
        shards[c] = shard

    # 3) Compute sample count once (any FAM is fine; all shards share the same N in this dataset)
    #    and get bytes-per-SNP (bpf = ceil(N/4))
    #    Pick the first chromosome present in the allow-list:
    first_chr = next(iter(by_chr.keys()))
    fam_uri = shards[first_chr].fam_uri
    n_samples = count_fam_lines(fam_uri)
    bpf = math.ceil(n_samples/4)
    # quick divisibility sanity on each shard:
    for c, sh in shards.items():
        rem = (sh.bed_size - 3) % bpf
        if rem != 0:
            raise RuntimeError(f"{sh.bed_uri} not divisible by bpf={bpf} (N={n_samples}).")

    print(f"N={n_samples:,} samples | bpf={bpf} bytes/SNP")

    # 4) Map requested (bp, allele) to BIM indices; stop early per shard
    for c, posmap in by_chr.items():
        sh = shards[c]
        hits = find_targets_in_bim(sh.bim_uri, posmap, c)
        if not hits:
            # If you put only 3 nearby SNPs but none show up, crash (by request)
            raise RuntimeError(f"No requested SNPs found in BIM for chr{c}.")
        all_candidates.extend(hits)

    if not all_candidates:
        raise RuntimeError("No candidates after BIM scan.")

    # 5) For tiny sets, fetch each SNP block exactly once (fastest) and compute stats
    per_chr: DefaultDict[str, List[Candidate]] = defaultdict(list)
    for c in all_candidates:
        per_chr[c.chrom].append(c)

    kept: List[Candidate] = []
    for cstr, lst in per_chr.items():
        sh = shards[cstr]
        stats = compute_stats_for_candidates(sh.bed_uri, bpf, n_samples, lst)
        # Filter: call rate ≥95%, and keep one record per (chr,bp,allele).
        chosen: Dict[Tuple[str,int,str], Tuple[Candidate,float,float]] = {}
        for cand in lst:
            st = stats.get(cand.snp_index)
            if st is None:
                continue
            missing, d1, d2 = st
            called = n_samples - missing
            if called <= 0:  # all missing
                continue
            call_rate = called / n_samples
            if call_rate < 0.95:
                continue
            dose = d2 if cand.allele == cand.a2 else d1
            maf = dose / (2*called) if called else 0.0
            key = (cand.chrom, cand.bp, cand.allele)
            prev = chosen.get(key)
            if prev is None or (maf, call_rate) > (prev[1], prev[2]):
                chosen[key] = (cand, maf, call_rate)
        kept.extend([v[0] for v in chosen.values()])

    if not kept:
        raise RuntimeError("All requested SNPs failed call-rate≥95% filter.")

    # 6) Write outputs:
    #    BIM/FAM: deterministic order by (chrom, snp_index) so BED aligns
    kept.sort(key=lambda x: (x.chrom, x.snp_index))
    write_subset_bim(kept)
    write_subset_fam(shards[first_chr].fam_uri)

    # 7) Assemble BED by fetching exactly those blocks (no coalescing)
    #    Group by shard to avoid interleaving different .bed files
    ordered_indices_by_chr: DefaultDict[str, List[int]] = defaultdict(list)
    for c in kept:
        ordered_indices_by_chr[c.chrom].append(c.snp_index)
    # Write bed in the same order as kept[]
    with open(OUT_BED, "wb") as f:
        f.write(b"\x6c\x1b\x01")
    for chrom in [k for k, _ in sorted({(c.chrom,0) for c in kept})]:
        sh = shards[chrom]
        assemble_subset_bed(sh.bed_uri, bpf, ordered_indices_by_chr[chrom])

    # 8) Write a plain text list of passed variants
    with open(OUT_PASSED, "w") as f:
        for c in kept:
            f.write(f"{c.chrom}:{c.bp} {c.allele}\n")

    print(f"== COMPLETE: wrote {OUT_BIM}, {OUT_BED}, {OUT_FAM}, {OUT_PASSED} ==")

if __name__ == "__main__":
    main()
