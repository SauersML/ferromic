import os, sys, math, re, subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional, DefaultDict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from tqdm import tqdm

# ======================================================================
# HARD-CODED GCS PATHS (NO WILDCARDS)
# ======================================================================

GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
ALLOW_LIST_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/vcf_list.txt"

OUT_BIM = "subset.bim"
OUT_BED = "subset.bed"
OUT_FAM = "subset.fam"
OUT_PASSED = "passed_snvs.txt"

# Small-target tuning (fast when you only need a few SNPs)
IO_THREADS_METRICS = 8         # concurrent range fetches for metrics
IO_THREADS_ASSEMBLY = 4        # concurrent runs for final bed assembly (kept modest for stability)
COALESCE_MAX_GAP   = 256*1024  # merge indices if byte gap ≤ 256 KiB
COALESCE_MAX_RUN   = 8*1024*1024  # cap each ranged request to ~8 MiB

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
        err = (e.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{err or e}")

def gsutil_stat_size(gs_uri: str) -> int:
    out = run(["gsutil", "-u", require_project(), "stat", gs_uri]).stdout
    m = re.search(r"Content-Length:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Unable to parse size for {gs_uri}")
    return int(m.group(1))

def gsutil_cat_lines(gs_uri: str) -> Iterable[str]:
    # Streaming text lines with a subprocess pipe
    proc = subprocess.Popen(["gsutil", "-u", require_project(), "cat", gs_uri],
                            stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    if proc.stdout is None:
        raise RuntimeError("Failed to open gsutil pipe")
    for line in proc.stdout:
        yield line
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"gsutil cat failed for {gs_uri} (exit {ret})")

def norm_chr(s: str) -> str:
    s = s.strip()
    return s[3:] if s.lower().startswith("chr") else s

# ======================================================================
# FAST RANGE FETCHER (GCS client; NO FALLBACKS)
# ======================================================================

class RangeFetcher:
    """Persistent GCS ranged fetcher with Requester Pays. No fallbacks."""
    def __init__(self):
        try:
            from google.cloud import storage  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"google-cloud-storage not available: {e}")
        self.project = require_project()
        from google.cloud import storage
        self.client = storage.Client(project=self.project)

    def _blob(self, gs_uri: str):
        if not gs_uri.startswith("gs://"):
            raise RuntimeError(f"Not a gs:// URI: {gs_uri}")
        _, _, rest = gs_uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise RuntimeError(f"Malformed GCS URI: {gs_uri}")
        bucket = self.client.bucket(bucket_name, user_project=self.project)
        return bucket.blob(blob_name)

    def fetch(self, gs_uri: str, start: int, end_inclusive: int) -> bytes:
        # google-cloud-storage uses inclusive end for download_as_bytes(start, end)
        return self._blob(gs_uri).download_as_bytes(start=start, end=end_inclusive)

# ======================================================================
# BED DECODING (vectorized)
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

def decode_run(blob: bytes, bpf: int, n_samples: int):
    """Decode a run (multiple SNPs) of length = len(blob)//bpf."""
    run_len = len(blob) // bpf
    arr = np.frombuffer(blob, dtype=np.uint8)
    blocks = arr.reshape(run_len, bpf)
    full = n_samples // 4
    rem  = n_samples % 4

    if full:
        core = blocks[:, :full]
        miss = MISS4[core].sum(axis=1, dtype=np.int64)
        d1   =  D1_4[core].sum(axis=1, dtype=np.int64)
        d2   =  D2_4[core].sum(axis=1, dtype=np.int64)
    else:
        miss = np.zeros(run_len, dtype=np.int64)
        d1   = np.zeros(run_len, dtype=np.int64)
        d2   = np.zeros(run_len, dtype=np.int64)
    if rem:
        last = blocks[:, full]
        miss += MISS_K[rem][last]
        d1   +=  D1_K[rem][last]
        d2   +=  D2_K[rem][last]
    return miss, d1, d2

# ======================================================================
# DATA CLASSES
# ======================================================================

@dataclass
class Shard:
    chrom: str
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
# HELPERS
# ======================================================================

def load_allow_list(url: str):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    allow: DefaultDict[Tuple[str,int], Set[str]] = defaultdict(set)
    ordered: List[Tuple[str,int,str]] = []
    with tqdm(unit="line", desc="Allow-list", leave=False) as bar:
        for raw in r.iter_lines(decode_unicode=True):
            bar.update(1)
            s = (raw or "").strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 2: continue
            loc, al = parts[0], parts[1].upper()
            if al not in {"A","C","G","T"}:
                continue
            if ":" not in loc: continue
            cs, ps = loc.split(":", 1)
            try:
                bp = int(float(ps))
            except:
                continue
            c = norm_chr(cs)
            allow[(c, bp)].add(al)
            ordered.append((c, bp, al))
    if not allow:
        raise RuntimeError("Allow-list is empty after filtering to SNPs.")
    return allow, ordered

def build_shard_for_chr(c: str) -> Shard:
    pref = f"chr{c}"
    bim = GCS_DIR + f"{pref}.bim"
    bed = GCS_DIR + f"{pref}.bed"
    fam = GCS_DIR + f"{pref}.fam"
    return Shard(c, bim, bed, fam, gsutil_stat_size(bim), gsutil_stat_size(bed))

def validate_bed_header(bed_uri: str, rf: RangeFetcher):
    hdr = rf.fetch(bed_uri, 0, 2)
    if hdr != b"\x6c\x1b\x01":
        raise RuntimeError(f"Not SNP-major BED (header {hdr.hex()}): {bed_uri}")

def count_fam_lines(fam_uri: str, expected_total: Optional[int] = None) -> int:
    n = 0
    with tqdm(desc="FAM lines", unit="line", total=expected_total, leave=False) as bar:
        for _ in gsutil_cat_lines(fam_uri):
            n += 1
            bar.update(1)
    if n <= 0:
        raise RuntimeError(f"Empty FAM: {fam_uri}")
    return n

def coalesce_indices_to_runs(indices: List[int], bpf: int,
                             max_gap: int = COALESCE_MAX_GAP,
                             max_run: int = COALESCE_MAX_RUN) -> List[Tuple[int,int]]:
    if not indices: return []
    idxs = sorted(set(indices))
    runs = []
    i0 = prev = idxs[0]
    run_bytes = bpf
    for x in idxs[1:]:
        byte_gap = (x - prev) * bpf
        if byte_gap <= max_gap and (run_bytes + byte_gap + bpf) <= max_run:
            run_bytes += byte_gap + bpf
            prev = x
        else:
            runs.append((i0, prev))
            i0 = prev = x
            run_bytes = bpf
    runs.append((i0, prev))
    return runs

def find_targets_in_bim(bim_uri: str,
                        required: Dict[int, Set[str]],
                        chrom_norm: str,
                        bim_size: int) -> List[Candidate]:
    targets: List[Candidate] = []
    if not required:
        return targets
    max_bp = max(required.keys())
    idx = 0
    seen = set()
    progressed = 0
    with tqdm(desc=f"BIM chr{chrom_norm}", unit="B", total=bim_size, unit_scale=True, leave=False) as bar:
        for line in gsutil_cat_lines(bim_uri):
            # progress by bytes seen (approx)
            progressed += len(line.encode("utf-8", "ignore"))
            if progressed >= (1<<20):
                bar.update(progressed); progressed = 0

            parts = line.strip().split()
            if len(parts) < 6:
                idx += 1; continue
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
                        seen.add(bp)
            # early stop once we’ve passed max requested position and found all present
            if bp > max_bp and len(seen) == len(required):
                break
            idx += 1
        if progressed:
            bar.update(progressed)
    return targets

# ======================================================================
# METRICS (call-rate/AF) WITH COALESCED RUNS + PROGRESS
# ======================================================================

def metrics_for_candidates(shard: Shard,
                           bpf: int,
                           n_samples: int,
                           cands: List[Candidate],
                           rf: RangeFetcher) -> Dict[int, Tuple[int,int,int]]:
    if not cands:
        return {}
    # coalesce indices to minimize requests (great for “3 close SNPs”)
    idxs = [c.snp_index for c in cands]
    runs = coalesce_indices_to_runs(idxs, bpf)
    total_blocks = sum((i1 - i0 + 1) for i0, i1 in runs)
    total_bytes  = sum((i1 - i0 + 1) * bpf for i0, i1 in runs)

    def worker(i0: int, i1: int):
        start = 3 + i0 * bpf
        end   = 3 + (i1 + 1) * bpf - 1
        blob = rf.fetch(shard.bed_uri, start, end)
        miss, d1, d2 = decode_run(blob, bpf, n_samples)
        # return per-snp tuple list in order
        return i0, i1, miss, d1, d2, len(blob)

    per_snp: Dict[int, Tuple[int,int,int]] = {}
    with ThreadPoolExecutor(max_workers=min(IO_THREADS_METRICS, len(runs))) as ex,\
         tqdm(total=total_blocks, desc=f"Metrics SNPs chr{shard.chrom}", unit="snp", leave=False) as pbar_snp,\
         tqdm(total=total_bytes,  desc=f"Metrics bytes chr{shard.chrom}", unit="B", unit_scale=True, leave=False) as pbar_bytes:
        futs = [ex.submit(worker, i0, i1) for i0, i1 in runs]
        for f in as_completed(futs):
            i0, i1, miss, d1, d2, nbytes = f.result()
            for k, snp_idx in enumerate(range(i0, i1+1)):
                per_snp[snp_idx] = (int(miss[k]), int(d1[k]), int(d2[k]))
            pbar_snp.update(i1 - i0 + 1)
            pbar_bytes.update(nbytes)

    return per_snp

# ======================================================================
# OUTPUT WRITERS WITH PROGRESS
# ======================================================================

def write_subset_bim(cands_in_order: List[Candidate]):
    with open(OUT_BIM, "w") as f, tqdm(total=len(cands_in_order), desc="Write BIM", unit="snp", leave=False) as bar:
        for c in cands_in_order:
            f.write(c.bim_line)
            bar.update(1)

def write_subset_fam(from_fam_uri: str, n_samples: int):
    with open(OUT_FAM, "w") as fout, tqdm(total=n_samples, desc="Write FAM", unit="line", leave=False) as bar:
        for ln in gsutil_cat_lines(from_fam_uri):
            fout.write(ln)
            bar.update(1)

def assemble_subset_bed(shard: Shard, bpf: int, indices: List[int], rf: RangeFetcher):
    runs = coalesce_indices_to_runs(indices, bpf)
    total_snps  = sum((i1 - i0 + 1) for i0, i1 in runs)
    total_bytes = sum((i1 - i0 + 1) * bpf for i0, i1 in runs)

    # header (once)
    mode = "ab" if os.path.exists(OUT_BED) and os.path.getsize(OUT_BED) >= 3 else "wb"
    with open(OUT_BED, mode) as fbed:
        if mode == "wb":
            fbed.write(b"\x6c\x1b\x01")

        def worker(i0: int, i1: int):
            start = 3 + i0 * bpf
            end   = 3 + (i1 + 1) * bpf - 1
            blob = rf.fetch(shard.bed_uri, start, end)
            return i0, i1, blob

        with ThreadPoolExecutor(max_workers=min(IO_THREADS_ASSEMBLY, len(runs))) as ex,\
             tqdm(total=total_snps,  desc=f"BED SNPs chr{shard.chrom}", unit="snp", leave=False) as pbar_snp,\
             tqdm(total=total_bytes, desc=f"BED bytes chr{shard.chrom}", unit="B", unit_scale=True, leave=False) as pbar_bytes:

            futs = [ex.submit(worker, i0, i1) for i0, i1 in runs]
            # preserve order across runs by sorting on i0 when writing
            results = [f.result() for f in as_completed(futs)]
            results.sort(key=lambda x: x[0])

            import bisect
            # carve only requested indices from each run
            indices_sorted = sorted(indices)
            for i0, i1, blob in results:
                # all SNP indices inside this run in order
                left = bisect.bisect_left(indices_sorted, i0)
                right = bisect.bisect_right(indices_sorted, i1)
                for snp_idx in indices_sorted[left:right]:
                    off = (snp_idx - i0) * bpf
                    fbed.write(blob[off:off+bpf])
                pbar_snp.update(i1 - i0 + 1)
                pbar_bytes.update(len(blob))

# ======================================================================
# DRIVER
# ======================================================================

def main():
    print("== Minimal & FAST PLINK subsetting (Requester Pays) ==")

    rf = RangeFetcher()  # hard crash if client missing
    _ = require_project()  # ensure env set

    # 1) Load allow-list (keep it small for speed)
    allow_map, allow_order = load_allow_list(ALLOW_LIST_URL)
    by_chr: DefaultDict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for c, bp, al in allow_order:
        by_chr[norm_chr(c)][bp].add(al)

    # 2) Build shards for involved chromosomes; validate headers
    shards: Dict[str, Shard] = {}
    for c in by_chr.keys():
        sh = build_shard_for_chr(c)
        validate_bed_header(sh.bed_uri, rf)
        shards[c] = sh

    # 3) Determine N and bpf from any involved FAM (same cohort)
    first_chr = next(iter(by_chr.keys()))
    fam_uri   = shards[first_chr].fam_uri
    # If you already know N, you can hard-code it and skip counting for even more speed.
    n_samples = count_fam_lines(fam_uri)
    bpf = math.ceil(n_samples/4)

    # sanity on each shard
    for c, sh in shards.items():
        if ((sh.bed_size - 3) % bpf) != 0:
            raise RuntimeError(f"{sh.bed_uri} not divisible by bpf={bpf} (N={n_samples})")

    print(f"N={n_samples:,} | bpf={bpf} bytes/SNP | chroms={', '.join(sorted(shards.keys(), key=lambda x: (x=='X', x=='Y', x.isdigit() and int(x) or 99, x)))}")

    # 4) Map requested (pos, allele) → BIM indices per chromosome (early stop)
    all_candidates: List[Candidate] = []
    for c, posmap in by_chr.items():
        sh = shards[c]
        hits = find_targets_in_bim(sh.bim_uri, posmap, c, sh.bim_size)
        if not hits:
            raise RuntimeError(f"No requested SNPs found in BIM for chr{c}.")
        all_candidates.extend(hits)

    if not all_candidates:
        raise RuntimeError("No candidates after BIM scan.")

    # 5) Metrics: fetch *coalesced runs* and decode (far fewer requests)
    kept: List[Candidate] = []
    for cstr in sorted(set(c.chrom for c in all_candidates),
                       key=lambda x: (x=='X', x=='Y', x.isdigit() and int(x) or 99, x)):
        sh = shards[cstr]
        cands = [c for c in all_candidates if c.chrom == cstr]
        stats = metrics_for_candidates(sh, bpf, n_samples, cands, rf)

        # Filter: call-rate ≥95%; dedup (chr,bp,allele) preferring higher AF then call-rate
        chosen: Dict[Tuple[str,int,str], Tuple[Candidate,float,float]] = {}
        for cand in cands:
            st = stats.get(cand.snp_index)
            if st is None:
                continue
            missing, d1, d2 = st
            called = n_samples - missing
            if called <= 0: continue
            cr = called / n_samples
            if cr < 0.95: continue
            dose = d2 if cand.allele == cand.a2 else d1
            af = dose / (2*called) if called else 0.0
            key = (cand.chrom, cand.bp, cand.allele)
            prev = chosen.get(key)
            if prev is None or (af, cr) > (prev[1], prev[2]):
                chosen[key] = (cand, af, cr)
        kept.extend([v[0] for v in chosen.values()])

    if not kept:
        raise RuntimeError("All requested SNPs failed call-rate≥95%.")

    # 6) Write outputs in deterministic order (chrom, snp_index)
    kept.sort(key=lambda x: (x.chrom, x.snp_index))
    write_subset_bim(kept)
    write_subset_fam(fam_uri, n_samples)

    # 7) Assemble BED: coalesce + ranged fetch + carve (with progress)
    # Group indices per chromosome and append to OUT_BED in chrom order
    if os.path.exists(OUT_BED):
        os.remove(OUT_BED)
    chrom_order = sorted(set(c.chrom for c in kept),
                         key=lambda x: (x=='X', x=='Y', x.isdigit() and int(x) or 99, x))
    for cstr in chrom_order:
        sh = shards[cstr]
        idxs = [c.snp_index for c in kept if c.chrom == cstr]
        assemble_subset_bed(sh, bpf, idxs, rf)

    # 8) List of passed SNPs (simple)
    with open(OUT_PASSED, "w") as f, tqdm(total=len(kept), desc="Write passed_snvs", unit="snp", leave=False) as bar:
        for c in kept:
            f.write(f"{c.chrom}:{c.bp} {c.allele}\n")
            bar.update(1)

    # integrity check
    expected_size = 3 + len(kept)*bpf
    actual_size = os.path.getsize(OUT_BED)
    if actual_size != expected_size:
        raise RuntimeError(f"subset.bed size {actual_size} != expected {expected_size}")

    print(f"== COMPLETE: {len(kept)} SNPs → {OUT_BIM}, {OUT_BED}, {OUT_FAM}, {OUT_PASSED} ==")

if __name__ == "__main__":
    # Hard crash on any unhandled issue
    main()
