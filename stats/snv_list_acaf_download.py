import os, sys, re, math, subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional, DefaultDict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
from tqdm import tqdm

# ------------------------- HARD-CODED PATHS ----------------------------------

GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
ALLOW_LIST_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/vcf_list.txt"

OUT_BIM    = "subset.bim"
OUT_BED    = "subset.bed"
OUT_FAM    = "subset.fam"
OUT_PASSED = "passed_snvs.txt"

# ------------------------- PERFORMANCE TUNING --------------------------------

# Coalescing by BYTES for ranged reads during metrics and assembly:
MAX_RUN_BYTES = 64 * 1024 * 1024     # ~64 MiB per ranged request (metrics phase)
MAX_BYTE_GAP  =  2 * 1024 * 1024     # merge neighbors if byte gap ≤ 2 MiB (metrics phase)

# For final BED assembly we prioritize stability & resumability (sequential, low RAM):
ASSEMBLY_MAX_RUN_BYTES = 16 * 1024 * 1024  # smaller chunks to keep memory low
ASSEMBLY_MAX_BYTE_GAP  = 256 * 1024

# I/O concurrency (network-bound). Increase if your network can handle it.
IO_THREADS = max(64, (os.cpu_count() or 8) * 8)

# ------------------------------ UTILITIES ------------------------------------

def require_project() -> str:
    pid = os.getenv("GOOGLE_PROJECT")
    if not pid:
        print("FATAL: Set GOOGLE_PROJECT in your environment.", file=sys.stderr)
        sys.exit(1)
    return pid

def run_gsutil(args: List[str], capture: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    cmd = ["gsutil", "-u", require_project()] + args
    return subprocess.run(cmd, check=True, capture_output=capture, text=text)

def gsutil_ls(pattern: str) -> List[str]:
    out = run_gsutil(["ls", pattern]).stdout.strip()
    return sorted([ln for ln in out.splitlines() if ln.strip()]) if out else []

def gsutil_stat_size(gs_uri: str) -> int:
    out = run_gsutil(["stat", gs_uri]).stdout
    m = re.search(r"Content-Length:\s*(\d+)", out)
    if not m:
        print(f"FATAL: Unable to parse size for {gs_uri}", file=sys.stderr)
        sys.exit(1)
    return int(m.group(1))

def gsutil_cat_lines(gs_uri: str) -> Iterable[str]:
    """Stream lines from gs:// URI."""
    proc = subprocess.Popen(["gsutil", "-u", require_project(), "cat", gs_uri],
                            stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    ret = proc.wait()
    if ret != 0:
        print(f"FATAL: gsutil cat failed for {gs_uri} (exit {ret})", file=sys.stderr)
        sys.exit(1)

def norm_chr(s: str) -> str:
    s = s.strip()
    return s[3:] if s.lower().startswith("chr") else s

def looks_like_chr(path: str, chr_norm: str) -> bool:
    p = path.lower()
    if f"chr{chr_norm}" in p:
        return True
    return bool(re.search(rf'(^|[/_\-.]){re.escape(chr_norm)}([/_\-.]|$)', p))

# ----------------------- RANGE FETCHER (PERSISTENT) --------------------------

class RangeFetcher:
    """
    Persistent HTTP range fetcher using google-cloud-storage with Requester Pays.
    Falls back to 'gsutil cat -r start-end' if client import/auth fails.
    'end' is inclusive.
    """
    def __init__(self):
        self.mode = "gsutil"
        self.client = None
        self.project = require_project()
        try:
            from google.cloud import storage  # lazy import
            self.client = storage.Client(project=self.project)
            self.mode = "gcs"
        except Exception as e:
            tqdm.write(f"[RangeFetcher] Using gsutil fallback (client unavailable: {e})")
            self.client = None
            self.mode = "gsutil"

    def _blob(self, gs_uri: str):
        from google.cloud import storage  # type: ignore
        if not gs_uri.startswith("gs://"):
            raise ValueError(f"Not a gs:// URI: {gs_uri}")
        _, _, rest = gs_uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Malformed GCS URI: {gs_uri}")
        bucket = self.client.bucket(bucket_name, user_project=self.project)  # Requester Pays
        return bucket.blob(blob_name)

    def fetch(self, gs_uri: str, start: int, end: int) -> bytes:
        if self.mode == "gcs":
            try:
                return self._blob(gs_uri).download_as_bytes(start=start, end=end)  # inclusive end
            except Exception as e:
                tqdm.write(f"[RangeFetcher] Client range failed ({e}); gsutil fallback: {gs_uri} {start}-{end}")
        # fallback
        return subprocess.check_output(["gsutil", "-u", self.project, "cat", "-r", f"{start}-{end}", gs_uri])

# ------------------------ BED DECODING (VECTORIZED) --------------------------

# PLINK SNP-major:
# 00=A1/A1, 10=A1/A2, 11=A2/A2, 01=missing (2 bits per sample, LSB-first)
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
    """Return (missing, doseA1, doseA2) arrays (length = run_len)."""
    run_len = len(blob) // bpf
    arr = np.frombuffer(blob, dtype=np.uint8)
    blocks = arr.reshape(run_len, bpf)
    full_bytes = n_samples // 4
    last_pairs = n_samples % 4

    if full_bytes > 0:
        blk = blocks[:, :full_bytes]
        miss = MISS4[blk].sum(axis=1, dtype=np.int64)
        d1   =  D1_4[blk].sum(axis=1, dtype=np.int64)
        d2   =  D2_4[blk].sum(axis=1, dtype=np.int64)
    else:
        miss = np.zeros(run_len, dtype=np.int64)
        d1   = np.zeros(run_len, dtype=np.int64)
        d2   = np.zeros(run_len, dtype=np.int64)

    if last_pairs:
        last_col = blocks[:, full_bytes]
        miss += MISS_K[last_pairs][last_col]
        d1   +=  D1_K[last_pairs][last_col]
        d2   +=  D2_K[last_pairs][last_col]

    return miss, d1, d2

# ------------------------------ DATA CLASSES ---------------------------------

@dataclass
class Shard:
    chrom: str
    bim_uri: str
    bed_uri: str
    fam_uri: str
    bim_size: int
    bed_size: int
    variant_count: int = 0
    bpf: Optional[int] = None

@dataclass
class Candidate:
    chrom: str
    bp: int
    allele: str            # allow-list allele (A/C/G/T)
    shard_idx: int
    snp_index: int         # 0-based index within shard
    snp_id: str
    a1: str
    a2: str
    bim_line: str          # write-through if selected

# ------------------------------- PIPELINE ------------------------------------

def load_allow_list(url: str):
    print("START: Loading allow-list …")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    allow_map: DefaultDict[Tuple[str,int], Set[str]] = defaultdict(set)
    allow_raw: DefaultDict[Tuple[str,int], List[str]] = defaultdict(list)
    chr_set: Set[str] = set()
    total = non_acgt = 0

    with tqdm(total=None, unit="line", desc="Allow-list") as bar:
        for raw in r.iter_lines(decode_unicode=True):
            total += 1
            s = (raw or "").strip()
            if not s:
                bar.update(1); continue
            parts = s.split()
            if len(parts) < 2:
                bar.update(1); continue
            loc, al = parts[0], parts[1].upper()
            if al not in {"A","C","G","T"}:
                non_acgt += 1; bar.update(1); continue
            if ":" not in loc:
                bar.update(1); continue
            cs, ps = loc.split(":", 1)
            try: bp = int(float(ps))
            except: bar.update(1); continue
            c = norm_chr(cs)
            allow_map[(c, bp)].add(al)
            allow_raw[(c, bp)].append(s)
            chr_set.add(c)
            bar.update(1)

    print(f"DONE: Allow-list total={total:,}, non-ACGT dropped={non_acgt:,}, unique positions={len(allow_map):,}, chromosomes={len(chr_set)}\n")
    return allow_map, allow_raw, chr_set

def list_relevant_shards(chr_set: Set[str]) -> List[Shard]:
    print("START: Discovering shards on GCS …")
    bim_paths = gsutil_ls(os.path.join(GCS_DIR, "*.bim"))
    if not bim_paths:
        print("FATAL: No .bim files found.", file=sys.stderr); sys.exit(1)
    selected = []
    for p in bim_paths:
        chosen_chr = None
        for c in chr_set:
            if looks_like_chr(p, c):
                chosen_chr = c; break
        if chosen_chr is None:
            continue
        bed = p[:-4] + ".bed"
        fam = p[:-4] + ".fam"
        selected.append(Shard(chrom=chosen_chr,
                              bim_uri=p,
                              bed_uri=bed,
                              fam_uri=fam,
                              bim_size=gsutil_stat_size(p),
                              bed_size=gsutil_stat_size(bed)))
    if not selected:
        print("FATAL: No shards match allow-list chromosomes.", file=sys.stderr); sys.exit(1)
    print(f"DONE: Selected {len(selected)} shards.\n")
    return selected

def list_shards_for_bim_chroms() -> List[Shard]:
    """Resume helper: build shards using chromosomes observed in subset.bim (no allow-list needed)."""
    print("START: Discovering shards for chromosomes in subset.bim …")
    # Gather chroms from local subset.bim
    chroms: List[str] = []
    seen: Set[str] = set()
    with open(OUT_BIM, "r") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 1: continue
            c = norm_chr(p[0])
            if c not in seen:
                seen.add(c); chroms.append(c)
    if not chroms:
        print("FATAL: subset.bim exists but appears empty.", file=sys.stderr); sys.exit(1)

    bim_paths = gsutil_ls(os.path.join(GCS_DIR, "*.bim"))
    if not bim_paths:
        print("FATAL: No .bim files found.", file=sys.stderr); sys.exit(1)

    # Keep only shards whose name matches a chrom in subset.bim; preserve subset.bim chrom order
    shards: List[Shard] = []
    for c in chroms:
        match = None
        for p in bim_paths:
            if looks_like_chr(p, c):
                match = p; break
        if not match:
            print(f"FATAL: Could not locate shard for chromosome {c} referenced by subset.bim.", file=sys.stderr)
            sys.exit(1)
        bed = match[:-4] + ".bed"
        fam = match[:-4] + ".fam"
        shards.append(Shard(chrom=c,
                            bim_uri=match,
                            bed_uri=bed,
                            fam_uri=fam,
                            bim_size=gsutil_stat_size(match),
                            bed_size=gsutil_stat_size(bed)))
    print(f"DONE: Selected {len(shards)} shards (from subset.bim chroms).\n")
    return shards

def scan_bims_collect(shards: List[Shard],
                      allow_map: Dict[Tuple[str,int], Set[str]],
                      allow_raw: Dict[Tuple[str,int], List[str]]) -> List[Candidate]:
    print("START: Streaming BIMs (SNP-only, allele-present) …")
    candidates: List[Candidate] = []
    total_bytes = sum(s.bim_size for s in shards)
    global_scanned = global_kept = 0
    global_nonacgt = global_notallowed = global_allele_absent = 0

    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="BIM bytes") as pbar:
        for sid, sh in enumerate(shards):
            idx = 0
            scanned = kept = nonacgt = notallowed = allele_absent = 0
            pending_bytes = 0
            for line in gsutil_cat_lines(sh.bim_uri):
                pending_bytes += len(line.encode("utf-8", "ignore"))
                parts = line.strip().split()
                if len(parts) < 6:
                    idx += 1; continue
                chr_raw, snp_id, cm, bp_raw, a1, a2 = parts[:6]
                c = norm_chr(chr_raw)
                try:
                    bp = int(float(bp_raw))
                except:
                    idx += 1; continue
                a1u, a2u = a1.upper(), a2.upper()
                # STRICT SNP-only: EXCLUDE INDELS
                if a1u not in {"A","C","G","T"} or a2u not in {"A","C","G","T"}:
                    nonacgt += 1; global_nonacgt += 1; idx += 1; continue
                allow = allow_map.get((c, bp))
                if not allow:
                    notallowed += 1; global_notallowed += 1; idx += 1; continue
                present = [al for al in allow if (al == a1u or al == a2u)]
                if not present:
                    # Print full context: raw allow-list lines and raw BIM line
                    print("\n[ALLELE-ABSENT] Allow-list allele(s) not found in BIM A1/A2")
                    print(f"  Position: {c}:{bp}")
                    print("  Allow-list raw line(s):")
                    for raw in allow_raw.get((c, bp), []):
                        print(f"    {raw}")
                    print(f"  Allow-list parsed alleles: {sorted(list(allow))}")
                    print(f"  BIM raw line: {line.strip()}\n")
                    allele_absent += 1; global_allele_absent += 1
                    idx += 1; continue
                for al in present:
                    candidates.append(Candidate(c, bp, al, sid, idx, snp_id, a1u, a2u, line))
                    kept += 1; global_kept += 1
                scanned += 1; global_scanned += 1
                idx += 1
                if pending_bytes >= (1 << 20):
                    pbar.update(pending_bytes); pending_bytes = 0
            if pending_bytes:
                pbar.update(pending_bytes)
            shards[sid].variant_count = idx
            print(f"[{os.path.basename(sh.bim_uri)}] scanned={idx:,}, kept={kept:,}, non-ACGT={nonacgt:,}, not-allowed={notallowed:,}, allele-absent={allele_absent:,}")

    print(f"DONE: BIM scan — variants scanned={global_scanned:,}, candidates kept={global_kept:,}, allele-absent events={global_allele_absent:,}\n")
    return candidates

def validate_bed_and_choose_fam(shards: List[Shard]) -> int:
    print("START: Validating BED headers + computing bytes-per-SNP (bpf) …")
    bpf_ref = None
    for sh in shards:
        hdr = subprocess.check_output(["gsutil", "-u", require_project(), "cat", "-r", "0-2", sh.bed_uri])
        if hdr != b"\x6c\x1b\x01":
            print(f"FATAL: Not SNP-major BED: {sh.bed_uri} (header={hdr.hex()})", file=sys.stderr); sys.exit(1)
        # We don't need variant_count for bpf; use divisibility by any consistent bpf later.
        # But if we have variant_count (after fresh BIM scan), compute precise bpf:
        if sh.variant_count > 0:
            rem = (sh.bed_size - 3) % sh.variant_count
            if rem != 0:
                print(f"FATAL: BED size not divisible by variant count: {sh.bed_uri}", file=sys.stderr); sys.exit(1)
            sh.bpf = (sh.bed_size - 3) // sh.variant_count
            if bpf_ref is None: bpf_ref = sh.bpf
            elif sh.bpf != bpf_ref:
                print(f"FATAL: Mixed bytes-per-SNP across shards ({sh.bpf} vs {bpf_ref}).", file=sys.stderr); sys.exit(1)

    # If variant_count was unknown (resume path), derive a consistent bpf via FAM selection:
    if bpf_ref is None:
        # Try to infer from existing subset.fam (resume) or select a FAM and write it
        if os.path.exists(OUT_FAM) and os.path.getsize(OUT_FAM) > 0:
            N = sum(1 for _ in open(OUT_FAM, "r"))
            bpf_ref = math.ceil(N/4)
            print(f"INFO: Inferred bpf={bpf_ref} from existing subset.fam (N={N:,}).")
        else:
            # No subset.fam yet — we will select FAM below and set bpf from it then verify
            pass

    # Choose any FAM with ceil(N/4) == bpf (if we still need to write OUT_FAM)
    if not (os.path.exists(OUT_FAM) and os.path.getsize(OUT_FAM) > 0):
        print("START: Selecting compatible FAM (progress = sample lines read) …")
        fams = gsutil_ls(os.path.join(GCS_DIR, "*.fam"))
        if not fams:
            print("FATAL: No .fam files found.", file=sys.stderr); sys.exit(1)

        chosen = None; N = None
        for fam in fams:
            n = 0
            for _ in gsutil_cat_lines(fam):
                n += 1
            if bpf_ref is None:
                # If we had no bpf yet, set bpf_ref from the first FAM — but verify against shards by divisibility
                candidate_bpf = math.ceil(n/4)
                # Quick header/divisibility check across shards
                ok = True
                for sh in shards:
                    rem = (sh.bed_size - 3) % candidate_bpf
                    if rem != 0:
                        ok = False; break
                if not ok:
                    continue
                bpf_ref = candidate_bpf

            if math.ceil(n/4) == bpf_ref:
                chosen = fam; N = n
                print(f"DONE: Selected {os.path.basename(fam)} (N={n:,}, ceil(N/4)={bpf_ref})\n")
                break
        if chosen is None:
            print(f"FATAL: No FAM matches bpf={bpf_ref}.", file=sys.stderr); sys.exit(1)

        print("START: Writing subset.fam …")
        with open(OUT_FAM, "w") as fout:
            for line in tqdm(gsutil_cat_lines(chosen), desc="subset.fam", unit="lines"):
                fout.write(line)
        print(f"DONE: Wrote {OUT_FAM} (N={N:,})\n")
    else:
        print("SKIP: Found existing subset.fam — keeping it.\n")

    # Propagate bpf to all shards
    if bpf_ref is None:
        # Should not happen
        print("FATAL: Could not determine bytes-per-SNP (bpf).", file=sys.stderr); sys.exit(1)
    for sh in shards:
        sh.bpf = int(bpf_ref)

    print(f"DONE: bytes-per-SNP (bpf) = {int(bpf_ref)}\n")
    return sum(1 for _ in open(OUT_FAM, "r"))

def coalesce_by_bytes(indices: List[int], bpf: int, *, max_gap: int, max_run: int) -> List[Tuple[int,int]]:
    """Merge sorted SNP indices into byte-aware runs (gap ≤ max_gap, run ≤ max_run)."""
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

def evaluate_candidates_fast(shards: List[Shard],
                             candidates: List[Candidate],
                             n_samples: int) -> Dict[Tuple[int,int], Tuple[int,int,int]]:
    """
    Compute per-SNP (missing, doseA1, doseA2) for all unique SNP indices by coalesced ranged reads.
    Returns dict[(sid, snp_idx)] = (missing, doseA1, doseA2).
    """
    print("START: Computing call rate & allele frequency (fast mode) …")
    # Unique SNP indices per shard
    snp_by_shard: DefaultDict[int, List[int]] = defaultdict(list)
    for c in candidates:
        snp_by_shard[c.shard_idx].append(c.snp_index)

    runs: List[Tuple[int,int,int,int]] = []  # (sid, i0, i1, total_bytes)
    for sid, idxs in snp_by_shard.items():
        sh = shards[sid]
        bpf = int(sh.bpf)  # type: ignore
        spans = coalesce_by_bytes(idxs, bpf, max_gap=MAX_BYTE_GAP, max_run=MAX_RUN_BYTES)
        for i0, i1 in spans:
            runs.append((sid, i0, i1, (i1 - i0 + 1) * bpf))
    # Large runs first to keep the pipe full
    runs.sort(key=lambda x: x[3], reverse=True)

    total_blocks = sum((i1 - i0 + 1) for _, i0, i1, _ in runs)
    total_bytes  = sum(sz for *_, sz in runs)
    print(f"INFO: {len(runs)} ranged requests | ~{total_blocks:,} SNP blocks | ~{total_bytes/1024/1024:.1f} MiB")

    fetcher = RangeFetcher()
    per_snp_stats: Dict[Tuple[int,int], Tuple[int,int,int]] = {}

    def worker(sid: int, i0: int, i1: int):
        sh = shards[sid]
        bpf = int(sh.bpf)  # type: ignore
        start = 3 + i0 * bpf
        end   = 3 + (i1 + 1) * bpf - 1  # inclusive
        blob = fetcher.fetch(sh.bed_uri, start, end)
        miss, d1, d2 = decode_run(blob, bpf, n_samples)
        return sid, i0, i1, miss, d1, d2, len(blob)

    blocks_done = 0
    bytes_done  = 0
    with ThreadPoolExecutor(max_workers=IO_THREADS) as ex, \
         tqdm(total=total_blocks, desc="Metrics SNPs", unit="snp") as pbar_snp, \
         tqdm(total=total_bytes,  desc="Metrics bytes", unit="B", unit_scale=True, unit_divisor=1024, leave=False) as pbar_bytes:

        futs = [ex.submit(worker, sid, i0, i1) for sid, i0, i1, _ in runs]
        for fut in as_completed(futs):
            sid, i0, i1, miss, d1, d2, nbytes = fut.result()
            # store per-SNP results
            for off, snp_idx in enumerate(range(i0, i1+1)):
                per_snp_stats[(sid, snp_idx)] = (int(miss[off]), int(d1[off]), int(d2[off]))
            nblocks = i1 - i0 + 1
            blocks_done += nblocks
            bytes_done  += nbytes
            pbar_snp.update(nblocks)
            pbar_bytes.update(nbytes)

    print(f"DONE: Metrics for {blocks_done:,} SNPs (~{bytes_done/1024/1024:.1f} MiB)\n")
    return per_snp_stats

def select_winners(candidates: List[Candidate],
                   per_snp_stats: Dict[Tuple[int,int], Tuple[int,int,int]],
                   n_samples: int) -> List[int]:
    """
    Keep variants with call rate ≥95% and deduplicate by (chr,bp,allele),
    preferring higher target-allele frequency (then higher call rate).
    """
    print("START: Filtering (call-rate≥95%) and deduplicating …")
    kept: Dict[Tuple[str,int,str], Tuple[int,float,float]] = {}
    dropped_cr = 0; considered = 0

    for i, c in enumerate(candidates):
        st = per_snp_stats.get((c.shard_idx, c.snp_index))
        if st is None:
            continue
        missing, d1, d2 = st
        called = n_samples - missing
        call_rate = (called / n_samples) if n_samples else 0.0
        if call_rate < 0.95:
            dropped_cr += 1; continue
        dose = d2 if c.allele == c.a2 else d1
        freq = (dose / (2*called)) if called > 0 else 0.0
        considered += 1
        key = (c.chrom, c.bp, c.allele)
        prev = kept.get(key)
        if prev is None or (freq, call_rate) > (prev[1], prev[2]):
            kept[key] = (i, freq, call_rate)

    winners = [i for (i, _, _) in kept.values()]
    print(f"DONE: Considered={considered:,}, dropped(call-rate<95%)={dropped_cr:,}, unique kept={len(winners):,}\n")
    return winners

def write_winners_outputs(shards: List[Shard], candidates: List[Candidate], winners: List[int]):
    """Write subset.bim and passed_snvs.txt in BIM order across shards."""
    # Winners grouped by shard in BIM order
    by_shard: DefaultDict[int, List[int]] = defaultdict(list)
    for i in winners:
        by_shard[candidates[i].shard_idx].append(i)
    for sid in list(by_shard.keys()):
        by_shard[sid].sort(key=lambda i: candidates[i].snp_index)

    print(f"START: Writing {OUT_BIM} and {OUT_PASSED} …")
    total_selected = sum(len(v) for v in by_shard.values())
    with open(OUT_BIM, "w") as fbim, open(OUT_PASSED, "w") as ftxt:
        for sid in range(len(shards)):
            for i in by_shard.get(sid, []):
                c = candidates[i]
                fbim.write(c.bim_line)
                ftxt.write(f"{c.chrom}:{c.bp} {c.allele}\n")
    print(f"DONE: Wrote {OUT_BIM} (variants={total_selected:,}), {OUT_PASSED}\n")

def load_winners_from_subset_bim() -> Tuple[List[str], List[Tuple[str,str]]]:
    """
    Return:
      - chrom_order: list of chromosomes in the order they appear in subset.bim
      - winners_records: list of (chrom, snp_id) in BIM order (one per line)
    """
    chrom_order: List[str] = []
    seen: Set[str] = set()
    winners_records: List[Tuple[str,str]] = []
    with open(OUT_BIM, "r") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 2: continue
            chrom = norm_chr(p[0]); snp_id = p[1]
            winners_records.append((chrom, snp_id))
            if chrom not in seen:
                seen.add(chrom); chrom_order.append(chrom)
    if not winners_records:
        print("FATAL: subset.bim exists but contains no variants.", file=sys.stderr); sys.exit(1)
    return chrom_order, winners_records

def map_snpids_to_indices(shards: List[Shard], winners_records: List[Tuple[str,str]]) -> Dict[int, List[int]]:
    """
    Build per-shard ordered list of SNP indices for the winners in subset.bim.
    Streaming scan of each shard's BIM, but only look up needed SNP IDs.
    """
    # Build per-chrom required snp_id sets in BIM order (to preserve order later)
    required_by_chrom: DefaultDict[str, List[str]] = defaultdict(list)
    needed_sets: DefaultDict[str, Set[str]] = defaultdict(set)
    for chrom, snp_id in winners_records:
        if snp_id not in needed_sets[chrom]:
            needed_sets[chrom].add(snp_id)
            required_by_chrom[chrom].append(snp_id)

    idx_map_by_chrom: Dict[str, Dict[str, int]] = {sh.chrom: {} for sh in shards}

    print("START: Mapping winner SNP IDs to shard indices …")
    for sh in shards:
        needed = needed_sets.get(sh.chrom, set())
        if not needed:
            continue
        found = 0
        idx_map = idx_map_by_chrom[sh.chrom]
        idx = 0
        for line in tqdm(gsutil_cat_lines(sh.bim_uri), desc=f"Index {os.path.basename(sh.bim_uri)}", unit="lines", leave=False):
            p = line.split()
            if len(p) < 2:
                idx += 1; continue
            snp_id = p[1]
            if snp_id in needed:
                idx_map[snp_id] = idx
                found += 1
                if found == len(needed):
                    break
            idx += 1
        if found < len(needed):
            missing = len(needed) - found
            print(f"WARNING: {missing} SNP IDs from subset.bim not located in {os.path.basename(sh.bim_uri)}; continuing…")

    # Now order per shard according to subset.bim order
    sel_by_shard: DefaultDict[int, List[int]] = defaultdict(list)
    chrom_to_sid = {sh.chrom: sid for sid, sh in enumerate(shards)}
    for chrom, snp_id in winners_records:
        sid = chrom_to_sid.get(chrom)
        if sid is None:
            print(f"FATAL: Chromosome {chrom} from subset.bim has no matching shard.", file=sys.stderr)
            sys.exit(1)
        idx = idx_map_by_chrom.get(chrom, {}).get(snp_id)
        if idx is None:
            print(f"FATAL: SNP ID {snp_id} from subset.bim not found in shard {chrom}.", file=sys.stderr)
            sys.exit(1)
        sel_by_shard[sid].append(idx)

    print("DONE: Mapped winner SNP IDs to shard indices.\n")
    return sel_by_shard

def compute_written_blocks(bpf: int) -> int:
    """How many SNP blocks are already present in OUT_BED (resumable)."""
    if not os.path.exists(OUT_BED):
        return 0
    sz = os.path.getsize(OUT_BED)
    if sz < 3:
        return 0
    body = sz - 3
    if body < 0 or (body % bpf) != 0:
        print("WARNING: Existing subset.bed has unexpected size; rewriting from scratch.")
        # Rewrite header so we can append correctly
        with open(OUT_BED, "wb") as f:
            f.write(b"\x6c\x1b\x01")
        return 0
    return body // bpf

def assemble_bed_resume(shards: List[Shard],
                        sel_by_shard: Dict[int, List[int]]):
    """
    Assemble subset.bed in strict BIM order, RESUMABLE by counting blocks.
    - Writes header once (3 bytes) if file is new.
    - Computes already-written blocks and skips them deterministically.
    - Sequential per-shard, per-run fetching with conservative coalescing to keep RAM low.
    """
    # Determine bpf and total winners
    bpf_any = next(int(sh.bpf) for sh in shards if sh.bpf is not None)
    total_selected = sum(len(v) for v in sel_by_shard.values())
    already = compute_written_blocks(bpf_any)
    if already > total_selected:
        print("WARNING: subset.bed appears larger than expected; rewriting from scratch.")
        with open(OUT_BED, "wb") as f:
            f.write(b"\x6c\x1b\x01")
        already = 0

    remaining = max(0, total_selected - already)
    if remaining == 0:
        print(f"SKIP: subset.bed already complete (variants={total_selected:,}).\n")
        return

    print(f"START: Assembling {OUT_BED} via ranged reads (resuming) …")
    print(f"INFO: total winners={total_selected:,}, already written={already:,}, remaining={remaining:,}")

    # Open for append (header must already exist)
    if not os.path.exists(OUT_BED) or os.path.getsize(OUT_BED) < 3:
        with open(OUT_BED, "wb") as f:
            f.write(b"\x6c\x1b\x01")

    fetcher = RangeFetcher()

    # Skip across shards according to BIM order (sid ascending)
    # Build a pointer of where to start in each shard
    to_write_by_shard: Dict[int, List[int]] = {}
    skip_left = already
    for sid in range(len(shards)):
        lst = sel_by_shard.get(sid, [])
        if not lst:
            continue
        if skip_left >= len(lst):
            skip_left -= len(lst)
            continue
        # keep only the part after skipping
        to_write_by_shard[sid] = lst[skip_left:]
        skip_left = 0
    # Sanity
    left = sum(len(v) for v in to_write_by_shard.values())
    if left != remaining:
        print(f"WARNING: resume accounting mismatch (computed {left}, expected {remaining}); proceeding with {left}.")

    # Plan sequential runs (per shard, increasing index order)
    total_bytes_planned = 0
    for sid, idxs in to_write_by_shard.items():
        sh = shards[sid]
        spans = coalesce_by_bytes(idxs, int(sh.bpf), max_gap=ASSEMBLY_MAX_BYTE_GAP, max_run=ASSEMBLY_MAX_RUN_BYTES)
        total_bytes_planned += sum((i1 - i0 + 1) * int(sh.bpf) for i0, i1 in spans)

    import bisect
    wrote = 0
    with open(OUT_BED, "ab") as fbed, \
         tqdm(total=remaining, desc="BED SNPs (resume)", unit="snp") as pbar_snp, \
         tqdm(total=total_bytes_planned, desc="BED bytes (planned)", unit="B", unit_scale=True, unit_divisor=1024, leave=False) as pbar_bytes:

        for sid in range(len(shards)):
            idxs = to_write_by_shard.get(sid, [])
            if not idxs:
                continue
            sh = shards[sid]
            bpf = int(sh.bpf)  # type: ignore
            spans = coalesce_by_bytes(idxs, bpf, max_gap=ASSEMBLY_MAX_BYTE_GAP, max_run=ASSEMBLY_MAX_RUN_BYTES)
            # Sequentially fetch each span and write only selected indices (in order)
            for i0, i1 in spans:
                start = 3 + i0 * bpf
                end   = 3 + (i1 + 1) * bpf - 1
                blob = fetcher.fetch(sh.bed_uri, start, end)
                # carve
                start_ptr = bisect.bisect_left(idxs, i0)
                ptr = start_ptr
                while ptr < len(idxs) and idxs[ptr] <= i1:
                    snp_idx = idxs[ptr]
                    off = (snp_idx - i0) * bpf
                    fbed.write(blob[off:off+bpf])
                    wrote += 1
                    pbar_snp.update(1)
                    ptr += 1
                pbar_bytes.update(len(blob))

    # Final integrity
    expected_size = 3 + total_selected * bpf_any
    actual_size = os.path.getsize(OUT_BED)
    if actual_size != expected_size:
        print(f"FATAL: BED size {actual_size} != expected {expected_size}", file=sys.stderr)
        sys.exit(1)
    print(f"DONE: Wrote {OUT_BED} (variants={total_selected:,}).\n")

# ------------------------------- DRIVER --------------------------------------

def main():
    print("=== STREAMED PLINK SUBSETTER (FAST, SNP-only; RESUMABLE) ===\n")

    # FAST-PATH: If subset.bim + subset.fam already exist, skip heavy stages and only assemble BED (resumable).
    winners_ready = os.path.exists(OUT_BIM) and os.path.getsize(OUT_BIM) > 0
    fam_ready     = os.path.exists(OUT_FAM) and os.path.getsize(OUT_FAM) > 0

    if winners_ready and fam_ready:
        print("RESUME MODE: Found existing subset.bim and subset.fam. Skipping candidate/metrics/selection.\n")
        # Build shards from subset.bim chroms (preserve order as written previously)
        shards = list_shards_for_bim_chroms()

        # Determine bpf from subset.fam (and propagate to shards); verify BED divisibility
        N = sum(1 for _ in open(OUT_FAM, "r"))
        bpf = math.ceil(N/4)
        for sh in shards:
            sh.bpf = bpf
            # quick divisibility sanity
            if (sh.bed_size - 3) % bpf != 0:
                print(f"FATAL: {os.path.basename(sh.bed_uri)} not compatible with derived bpf={bpf}.", file=sys.stderr)
                sys.exit(1)

        # Map winners (subset.bim) to per-shard SNP indices
        chrom_order, winners_records = load_winners_from_subset_bim()
        sel_by_shard = map_snpids_to_indices(shards, winners_records)

        # If subset.bed already complete, exit; else resume assembly
        total_selected = len(winners_records)
        expected_size = 3 + total_selected * bpf
        if os.path.exists(OUT_BED) and os.path.getsize(OUT_BED) == expected_size:
            print(f"ALL DONE: subset.bed already complete (variants={total_selected:,}).")
            return

        assemble_bed_resume(shards, sel_by_shard)
        print("=== COMPLETE (RESUME) ===")
        return

    # FULL PIPELINE PATH
    # 1) Allow-list
    allow_map, allow_raw, chr_set = load_allow_list(ALLOW_LIST_URL)

    # 2) Shards
    shards = list_relevant_shards(chr_set)

    # 3) BIM scan -> candidates (strict SNP-only; verbose allele-absent reporting)
    candidates = scan_bims_collect(shards, allow_map, allow_raw)
    if not candidates:
        print("No candidates after BIM scan. Writing empty outputs.")
        open(OUT_BIM, "w").close()
        with open(OUT_BED, "wb") as f: f.write(b"\x6c\x1b\x01")
        open(OUT_FAM, "w").close()
        open(OUT_PASSED, "w").close()
        return

    # 4) Validate BED geometry, choose & write FAM (no sample filtering)
    n_samples = validate_bed_and_choose_fam(shards)

    # 5) Evaluate (persistent ranges + vectorized decode + aggressive coalescing)
    per_snp_stats = evaluate_candidates_fast(shards, candidates, n_samples)

    # 6) Filter call-rate≥95% and deduplicate per (chr,bp,allele)
    winners = select_winners(candidates, per_snp_stats, n_samples)
    if not winners:
        print("All candidates failed call-rate≥95%. Writing empty subset.")
        open(OUT_BIM, "w").close()
        with open(OUT_BED, "wb") as f: f.write(b"\x6c\x1b\x01")
        open(OUT_PASSED, "w").close()
        return

    # 7) Write subset.bim + passed_snvs.txt
    write_winners_outputs(shards, candidates, winners)

    # 8) Assemble subset.bed (fresh run; not resume)
    # Recreate ordered winners per shard
    chrom_order, winners_records = load_winners_from_subset_bim()
    # Reorder shards to match subset.bim chrom order
    shards = list_shards_for_bim_chroms()
    # Re-populate BPF into the newly created shards list before assembly.
    # The BPF value is derived from the final subset.fam file, which is the
    # source of truth for the number of samples (N).
    N = sum(1 for _ in open(OUT_FAM, "r"))
    bpf = math.ceil(N/4)
    for sh in shards:
        sh.bpf = bpf
    sel_by_shard = map_snpids_to_indices(shards, winners_records)
    assemble_bed_resume(shards, sel_by_shard)

    print("=== COMPLETE ===")

if __name__ == "__main__":
    main()
