import os, sys, re, math, subprocess, shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable, Optional, DefaultDict
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import requests
import tempfile

# ------------------------- HARD-CODED PATHS ----------------------------------

GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
ALLOW_LIST_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/vcf_list.txt"

OUT_BIM = "subset.bim"
OUT_BED = "subset.bed"
OUT_FAM = "subset.fam"
OUT_PASSED = "passed_snvs.txt"

# ------------------------- PERFORMANCE TUNING --------------------------------

# Range coalescing by bytes:
MAX_RUN_BYTES = 8 * 1024 * 1024     # cap a single ranged request to ~8 MiB (tunable)
MAX_BYTE_GAP  = 256 * 1024          # merge neighboring SNPs if byte gap ≤ 256 KiB (tunable)

# I/O concurrency (network-bound; increase if network allows):
IO_THREADS = max(32, (os.cpu_count() or 8) * 4)

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
    proc = subprocess.Popen(["gsutil", "-u", require_project(), "cat", gs_uri],
                            stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line
    ret = proc.wait()
    if ret != 0:
        print(f"FATAL: gsutil cat failed for {gs_uri} (exit {ret})", file=sys.stderr)
        sys.exit(1)

def gsutil_cat_range(gs_uri: str, start: int, end: int) -> bytes:
    return subprocess.check_output(["gsutil", "-u", require_project(), "cat", "-r", f"{start}-{end}", gs_uri])

def norm_chr(s: str) -> str:
    s = s.strip()
    return s[3:] if s.lower().startswith("chr") else s

def looks_like_chr(path: str, chr_norm: str) -> bool:
    p = path.lower()
    if f"chr{chr_norm}" in p:
        return True
    return bool(re.search(rf'(^|[/_\-.]){re.escape(chr_norm)}([/_\-.]|$)', p))

# ------------------------ BED DECODING (VECTOR) ------------------------------

# PLINK SNP-major encoding:
# 00=A1/A1, 10=A1/A2, 11=A2/A2, 01=missing  (LSB-first per 2-bit genotype)
# We compute per-SNP counts: missing, doseA1, doseA2 using LUTs and numpy take/sum.

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
        # Full 4 pairs
        m = d1 = d2 = 0
        for c in pairs:
            if c == 0b01: m += 1
            elif c == 0b00: d1 += 2
            elif c == 0b10: d1 += 1; d2 += 1
            elif c == 0b11: d2 += 2
        miss4[b] = m; d1_4[b] = d1; d2_4[b] = d2
        # Partial k=1..3
        for k in (1,2,3):
            m = d1 = d2 = 0
            for c in pairs[:k]:
                if c == 0b01: m += 1
                elif c == 0b00: d1 += 2
                elif c == 0b10: d1 += 1; d2 += 1
                elif c == 0b11: d2 += 2
            miss_k[k][b] = m; d1_k[k][b] = d1; d2_k[k][b] = d2
    return miss4, d1_4, d2_4, miss_k, d1_k, d2_k

MISS4, D1_4, D2_4, MISS_K, D1_K, D2_K = build_luts()

def decode_run_stats(blob: bytes, bpf: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a contiguous run blob for snps [i0..i1], return arrays of length run_len:
      missing_count, doseA1, doseA2
    """
    run_len = len(blob) // bpf
    arr = np.frombuffer(blob, dtype=np.uint8)
    blocks = arr.reshape(run_len, bpf)

    full_bytes = n_samples // 4
    last_pairs = n_samples % 4

    # Full bytes contributions
    if full_bytes > 0:
        blk_full = blocks[:, :full_bytes]  # (run_len, full_bytes)
        miss = MISS4[blk_full].sum(axis=1).astype(np.int32)
        d1   = D1_4[blk_full].sum(axis=1).astype(np.int32)
        d2   = D2_4[blk_full].sum(axis=1).astype(np.int32)
    else:
        miss = np.zeros(run_len, dtype=np.int32)
        d1   = np.zeros(run_len, dtype=np.int32)
        d2   = np.zeros(run_len, dtype=np.int32)

    # Trailing byte (if any subjects spill)
    if last_pairs:
        last_col = blocks[:, full_bytes]  # (run_len,)
        miss += MISS_K[last_pairs][last_col].astype(np.int32)
        d1   += D1_K[last_pairs][last_col].astype(np.int32)
        d2   += D2_K[last_pairs][last_col].astype(np.int32)

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
    bim_line: str          # original BIM line (write-through if selected)

# ------------------------------- PIPELINE ------------------------------------

def load_allow_list(url: str):
    print("START: Loading allow-list …")
    r = requests.get(url, stream=True)
    r.raise_for_status()

    allow_map: DefaultDict[Tuple[str,int], Set[str]] = defaultdict(set)
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
            chr_set.add(c)
            bar.update(1)

    print(f"DONE: Allow-list total={total:,}, non-ACGT dropped={non_acgt:,}, unique positions={len(allow_map):,}, chromosomes={len(chr_set)}\n")
    return allow_map, chr_set

def list_relevant_shards(chr_set: Set[str]) -> List[Shard]:
    print("START: Discovering shards on GCS …")
    bim_paths = gsutil_ls(os.path.join(GCS_DIR, "*.bim"))
    if not bim_paths:
        print("FATAL: No .bim files found.", file=sys.stderr); sys.exit(1)
    # Select only chromosomes present in allow-list by filename
    selected = []
    print("INFO: Selecting shards whose names match allowed chromosomes …")
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

def scan_bims_collect(shards: List[Shard], allow_map: Dict[Tuple[str,int], Set[str]]) -> List[Candidate]:
    print("START: Streaming BIMs (SNP-only, allele-present) …")
    candidates: List[Candidate] = []
    total_bytes = sum(s.bim_size for s in shards)
    # Counters
    global_scanned = global_kept = 0
    global_nonacgt = global_notallowed = global_allele_absent = 0

    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="BIM bytes") as pbar:
        for sid, sh in enumerate(shards):
            idx = 0
            scanned = kept = nonacgt = notallowed = allele_absent = 0
            pending_bytes = 0
            bim_name = os.path.basename(sh.bim_uri)
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
                # SNP-only: EXCLUDE INDELS (strict)
                if a1u not in {"A","C","G","T"} or a2u not in {"A","C","G","T"}:
                    nonacgt += 1; global_nonacgt += 1; idx += 1; continue

                allow = allow_map.get((c, bp))
                if not allow:
                    notallowed += 1; global_notallowed += 1; idx += 1; continue

                present = [al for al in allow if (al == a1u or al == a2u)]
                if not present:
                    # VERBOSE: print raw allow-list line(s) and raw BIM line
                    # Find all raw allowed alleles for this (c,bp) and show both parsed and raw BIM
                    print("\n[ALLELE-ABSENT] Allow-list allele not in BIM A1/A2")
                    print(f"  Position: {c}:{bp}")
                    print(f"  BIM line: {line.strip()}")
                    print(f"  Allowed alleles at this position: {sorted(list(allow))}")
                    allele_absent += 1; global_allele_absent += 1
                    idx += 1; continue

                for al in present:
                    candidates.append(Candidate(chrom=c, bp=bp, allele=al,
                                                shard_idx=sid, snp_index=idx,
                                                snp_id=snp_id, a1=a1u, a2=a2u,
                                                bim_line=line))
                    kept += 1; global_kept += 1

                scanned += 1; global_scanned += 1
                idx += 1

                if pending_bytes >= (1 << 20):
                    pbar.update(pending_bytes); pending_bytes = 0
            if pending_bytes:
                pbar.update(pending_bytes)

            shards[sid].variant_count = idx
            print(f"[{bim_name}] scanned={idx:,}, kept={kept:,}, non-ACGT={nonacgt:,}, not-allowed={notallowed:,}, allele-absent={allele_absent:,}")

    print(f"DONE: BIM scan — variants scanned={global_scanned:,}, candidates kept={global_kept:,}, allele-absent events={global_allele_absent:,}\n")
    return candidates

def validate_bed_and_choose_fam(shards: List[Shard]) -> int:
    print("START: Validating BED headers + bytes-per-SNP; selecting compatible FAM …")
    bpf_ref = None
    for sh in shards:
        # confirm header and compute bpf (3-byte header)
        hdr = gsutil_cat_range(sh.bed_uri, 0, 2)
        if hdr != b"\x6c\x1b\x01":
            print(f"FATAL: Not SNP-major BED: {sh.bed_uri} (header={hdr.hex()})", file=sys.stderr); sys.exit(1)
        if sh.variant_count == 0:
            continue
        rem = (sh.bed_size - 3) % sh.variant_count
        if rem != 0:
            print(f"FATAL: BED size not divisible by variant count: {sh.bed_uri}", file=sys.stderr); sys.exit(1)
        sh.bpf = (sh.bed_size - 3) // sh.variant_count
        if bpf_ref is None: bpf_ref = sh.bpf
        elif sh.bpf != bpf_ref:
            print(f"FATAL: Mixed bytes-per-SNP across shards ({sh.bpf} vs {bpf_ref}).", file=sys.stderr); sys.exit(1)
    if bpf_ref is None:
        print("FATAL: No usable shards.", file=sys.stderr); sys.exit(1)
    bpf = int(bpf_ref)

    # Choose FAM with ceil(N/4) == bpf
    fams = gsutil_ls(os.path.join(GCS_DIR, "*.fam"))
    if not fams:
        print("FATAL: No .fam files found.", file=sys.stderr); sys.exit(1)

    chosen = None
    chosen_N = None
    for fam in fams:
        n = 0
        for _ in tqdm(gsutil_cat_lines(fam), desc=f"FAM {os.path.basename(fam)}", unit="samples", leave=False):
            n += 1
        if math.ceil(n/4) == bpf:
            chosen = fam; chosen_N = n
            print(f"DONE: Selected {os.path.basename(fam)} (N={n:,}, ceil(N/4)={bpf})")
            break
    if chosen is None:
        print(f"FATAL: No FAM matches bpf={bpf}.", file=sys.stderr); sys.exit(1)

    print(f"START: Writing {OUT_FAM} …")
    with open(OUT_FAM, "w") as fout:
        for line in tqdm(gsutil_cat_lines(chosen), desc="subset.fam", unit="lines"):
            fout.write(line)
    print(f"DONE: Wrote {OUT_FAM} (N={chosen_N:,})\n")
    return chosen_N  # N samples

def coalesce_by_bytes(indices: List[int], bpf: int) -> List[Tuple[int,int]]:
    """
    Merge sorted SNP indices into runs while:
      - byte gap between consecutive indices ≤ MAX_BYTE_GAP, and
      - total run size ≤ MAX_RUN_BYTES.
    Returns list of (i0, i1) inclusive SNP index spans.
    """
    if not indices:
        return []
    idxs = sorted(set(indices))
    runs = []
    i0 = prev = idxs[0]
    run_bytes = bpf  # first block
    for x in idxs[1:]:
        byte_gap = (x - prev) * bpf
        if byte_gap <= MAX_BYTE_GAP and (run_bytes + byte_gap + bpf) <= MAX_RUN_BYTES:
            run_bytes += byte_gap + bpf
            prev = x
        else:
            runs.append((i0, prev))
            i0 = prev = x
            run_bytes = bpf
    runs.append((i0, prev))
    return runs

def evaluate_candidates_fast(shards: List[Shard], candidates: List[Candidate], n_samples: int):
    """
    Compute call rate and target-allele frequency for all candidate SNP indices,
    using coalesced ranged reads and vectorized decode.

    Returns:
      per_snp_stats: dict[(shard_idx, snp_index)] -> (missing:int, doseA1:int, doseA2:int)
    """
    print("START: Computing call rate & allele frequency (fast mode) …")
    # Unique SNP indices per shard
    snp_by_shard: DefaultDict[int, List[int]] = defaultdict(list)
    for c in candidates:
        snp_by_shard[c.shard_idx].append(c.snp_index)

    # Build runs
    runs: List[Tuple[int, int, int, int]] = []  # (sid, i0, i1, bytes)
    for sid, idxs in snp_by_shard.items():
        sh = shards[sid]
        bpf = int(sh.bpf)  # type: ignore
        spans = coalesce_by_bytes(sorted(set(idxs)), bpf)
        for i0, i1 in spans:
            nblocks = i1 - i0 + 1
            runs.append((sid, i0, i1, nblocks * bpf))
    # Prioritize larger spans first
    runs.sort(key=lambda x: x[3], reverse=True)

    total_blocks = sum((i1 - i0 + 1) for _, i0, i1, _ in runs)
    total_bytes = sum(sz for *_, sz in runs)
    per_snp_stats: Dict[Tuple[int,int], Tuple[int,int,int]] = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(sid: int, i0: int, i1: int):
        sh = shards[sid]
        bpf = int(sh.bpf)  # type: ignore
        start = 3 + i0 * bpf
        end   = 3 + (i1 + 1) * bpf - 1
        blob = gsutil_cat_range(sh.bed_uri, start, end)
        miss, d1, d2 = decode_run_stats(blob, bpf, n_samples)
        return (sid, i0, i1, miss, d1, d2, len(blob))

    print(f"INFO: {len(runs)} ranged requests planned | ~{total_blocks:,} SNP blocks | ~{total_bytes/1024/1024:.1f} MiB")
    blocks_done = 0
    bytes_done = 0

    with ThreadPoolExecutor(max_workers=IO_THREADS) as ex, \
         tqdm(total=total_blocks, desc="Metrics SNPs", unit="snp") as pbar_blocks, \
         tqdm(total=total_bytes, desc="Metrics bytes", unit="B", unit_scale=True, unit_divisor=1024, leave=False) as pbar_bytes:

        futs = [ex.submit(worker, sid, i0, i1) for (sid, i0, i1, _) in runs]
        for fut in as_completed(futs):
            sid, i0, i1, miss, d1, d2, blob_bytes = fut.result()
            # Store per-SNP stats
            for off, snp_idx in enumerate(range(i0, i1+1)):
                per_snp_stats[(sid, snp_idx)] = (int(miss[off]), int(d1[off]), int(d2[off]))
            # Progress
            nblocks = i1 - i0 + 1
            blocks_done += nblocks
            bytes_done += blob_bytes
            pbar_blocks.update(nblocks)
            pbar_bytes.update(blob_bytes)

    print(f"DONE: Metrics computed for {blocks_done:,} SNPs (~{bytes_done/1024/1024:.1f} MiB).\n")
    return per_snp_stats

def select_winners(candidates: List[Candidate],
                   per_snp_stats: Dict[Tuple[int,int], Tuple[int,int,int]],
                   n_samples: int) -> List[int]:
    """
    Apply call rate ≥95% and deduplicate by (chr,bp,allele).
    Choose the candidate with higher target-allele frequency (then higher call-rate).
    """
    print("START: Filtering (call-rate≥95%) and deduplicating …")
    kept: Dict[Tuple[str,int,str], Tuple[int, float, float]] = {}  # key -> (cand_idx, freq, call_rate)
    dropped_callrate = 0
    considered = 0

    for i, c in enumerate(candidates):
        st = per_snp_stats.get((c.shard_idx, c.snp_index))
        if st is None:
            continue
        missing, d1, d2 = st
        called = n_samples - missing
        call_rate = (called / n_samples) if n_samples else 0.0
        if call_rate < 0.95:
            dropped_callrate += 1
            continue
        # freq for the allow-list allele at this SNP
        dose = d2 if c.allele == c.a2 else d1
        freq = (dose / (2*called)) if called > 0 else 0.0
        considered += 1
        key = (c.chrom, c.bp, c.allele)
        prev = kept.get(key)
        if prev is None or (freq, call_rate) > (prev[1], prev[2]):
            kept[key] = (i, freq, call_rate)

    winners = [v[0] for v in kept.values()]
    print(f"DONE: Considered={considered:,}, dropped (call-rate<95%)={dropped_callrate:,}, unique kept={len(winners):,}\n")
    return winners

def write_outputs(shards: List[Shard], candidates: List[Candidate], winners: List[int]):
    """
    Write:
      - subset.bim (selected BIM lines, BIM order across shards)
      - passed_snvs.txt ("chr:pos allele" in same order)
      - subset.bed (via coalesced ranged reads in BIM order)
    """
    # Group winners by shard in BIM index order
    by_shard: DefaultDict[int, List[int]] = defaultdict(list)
    for i in winners:
        by_shard[candidates[i].shard_idx].append(i)
    for sid in list(by_shard.keys()):
        by_shard[sid].sort(key=lambda i: candidates[i].snp_index)

    # subset.bim + passed_snvs.txt
    print(f"START: Writing {OUT_BIM} and {OUT_PASSED} …")
    total_selected = sum(len(v) for v in by_shard.values())
    with open(OUT_BIM, "w") as fbim, open(OUT_PASSED, "w") as ftxt:
        wrote = 0
        for sid in range(len(shards)):
            lst = by_shard.get(sid, [])
            for i in lst:
                c = candidates[i]
                fbim.write(c.bim_line)
                ftxt.write(f"{c.chrom}:{c.bp} {c.allele}\n")
                wrote += 1
    print(f"DONE: Wrote {OUT_BIM} (variants={total_selected:,}), {OUT_PASSED}.\n")

    # subset.bed via ranged reads (in BIM order)
    print(f"START: Assembling {OUT_BED} via ranged reads …")
    total_blocks = total_selected
    total_bytes_planned = 0
    run_specs: List[Tuple[int,int,int,int]] = []  # (sid, i0, i1, bytes)

    for sid in range(len(shards)):
        lst = by_shard.get(sid, [])
        if not lst: continue
        sh = shards[sid]
        bpf = int(sh.bpf)  # type: ignore
        idxs = [candidates[i].snp_index for i in lst]
        spans = coalesce_by_bytes(idxs, bpf)
        for i0, i1 in spans:
            run_specs.append((sid, i0, i1, (i1 - i0 + 1) * bpf))
            total_bytes_planned += (i1 - i0 + 1) * bpf

    # Prioritize large spans
    run_specs.sort(key=lambda x: x[3], reverse=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_run(sid: int, i0: int, i1: int):
        sh = shards[sid]
        bpf = int(sh.bpf)  # type: ignore
        start = 3 + i0 * bpf
        end   = 3 + (i1 + 1) * bpf - 1
        blob = gsutil_cat_range(sh.bed_uri, start, end)
        return sid, i0, i1, bpf, blob

    with open(OUT_BED, "wb") as fbed, \
         tqdm(total=total_blocks, desc="BED SNPs", unit="snp") as pbar_blocks, \
         tqdm(total=total_bytes_planned, desc="BED bytes", unit="B", unit_scale=True, unit_divisor=1024, leave=False) as pbar_bytes, \
         ThreadPoolExecutor(max_workers=IO_THREADS) as ex:

        # Write header
        fbed.write(b"\x6c\x1b\x01")

        # Pre-build per-shard selected index lists (for carving)
        sel_by_shard: Dict[int, List[int]] = {}
        for sid in range(len(shards)):
            lst = by_shard.get(sid, [])
            if lst:
                sel_by_shard[sid] = [candidates[i].snp_index for i in lst]

        futs = [ex.submit(fetch_run, *spec[:3]) for spec in run_specs]
        for fut, spec in zip(as_completed(futs), run_specs):
            sid, i0, i1, bpf, blob = fut.result()
            # Carve only selected indices in [i0..i1]
            sel = sel_by_shard[sid]
            # Move a pointer across sel for this window
            # Find first index >= i0 (binary search)
            import bisect
            start_ptr = bisect.bisect_left(sel, i0)
            ptr = start_ptr
            while ptr < len(sel) and sel[ptr] <= i1:
                snp_idx = sel[ptr]
                off = (snp_idx - i0) * bpf
                fbed.write(blob[off:off+bpf])
                pbar_blocks.update(1)
                ptr += 1
            pbar_bytes.update(len(blob))

    # Final integrity
    bpf_any = None
    for sh in shards:
        if sh.bpf is not None:
            bpf_any = int(sh.bpf); break
    n_variants = sum(1 for _ in open(OUT_BIM, "r"))
    expected_size = 3 + n_variants * int(bpf_any)  # type: ignore
    actual_size = os.path.getsize(OUT_BED)
    if actual_size != expected_size:
        print(f"FATAL: BED size {actual_size} != expected {expected_size}", file=sys.stderr)
        sys.exit(1)
    print(f"DONE: Wrote {OUT_BED} (variants={n_variants:,}).\n")

# ------------------------------- DRIVER --------------------------------------

def main():
    print("=== STREAMED PLINK SUBSETTER (FAST, SNP-only) ===\n")

    # Step 1: Allow-list
    allow_map, chr_set = load_allow_list(ALLOW_LIST_URL)

    # Step 2: Shards
    shards = list_relevant_shards(chr_set)

    # Step 3: BIM scan -> candidates (strict SNP-only; print allele-absent events verbosely)
    candidates = scan_bims_collect(shards, allow_map)
    if not candidates:
        print("No candidates after BIM scan. Writing empty outputs.")
        open(OUT_BIM, "w").close()
        with open(OUT_BED, "wb") as f: f.write(b"\x6c\x1b\x01")
        open(OUT_FAM, "w").close()
        open(OUT_PASSED, "w").close()
        return

    # Step 4: Validate BED geometry, choose & write FAM (no sample filtering)
    n_samples = validate_bed_and_choose_fam(shards)

    # Step 5: Evaluate candidates (vectorized + coalesced + parallel)
    per_snp_stats = evaluate_candidates_fast(shards, candidates, n_samples)

    # Step 6: Filter call-rate≥95% and deduplicate per (chr,bp,allele)
    winners = select_winners(candidates, per_snp_stats, n_samples)
    if not winners:
        print("All candidates failed call-rate≥95%. Writing empty subset.")
        open(OUT_BIM, "w").close()
        with open(OUT_BED, "wb") as f: f.write(b"\x6c\x1b\x01")
        open(OUT_PASSED, "w").close()
        return

    # Step 7: Write outputs (subset.bim, passed_snvs.txt, subset.bed)
    write_outputs(shards, candidates, winners)

    print("=== COMPLETE ===")

if __name__ == "__main__":
    main()
