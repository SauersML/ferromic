import os, sys, math, re, subprocess
from typing import List, Tuple, Dict, Iterable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from google.cloud import storage  # hard crash if missing

# ------------------------------ HARD-CODED PATHS ------------------------------

# Requester-pays GCS PLINK shards (hard-coded)
GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
CHR     = "17"

BIM_URI = GCS_DIR + f"chr{CHR}.bim"
BED_URI = GCS_DIR + f"chr{CHR}.bed"
FAM_URI = GCS_DIR + f"chr{CHR}.fam"

# Tag SNP targets for the chr17q21 inversion (bp, inversion_allele)
# G = inversion allele; A = direct allele (after orientation we count G copies 0/1/2).
TARGETS: List[Tuple[int, str]] = [
    (46003698, "G"),
    (45996523, "G"),
    (45974480, "G"),
]

# Single-output hard-call matrix
OUT_TSV = "imputed_inversion_hardcalls.tsv"

# ------------------------------ ENV & SHELL UTILS -----------------------------

def require_project() -> str:
    pid = os.getenv("GOOGLE_PROJECT")
    if not pid:
        raise RuntimeError("Set GOOGLE_PROJECT for requester-pays access.")
    return pid

def run(cmd: List[str]) -> str:
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return cp.stdout

def gsutil_stat_size(gs_uri: str) -> int:
    out = run(["gsutil", "-u", require_project(), "stat", gs_uri])
    m = re.search(r"Content-Length:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Unable to parse size for {gs_uri}")
    return int(m.group(1))

def gsutil_cat_lines(gs_uri: str) -> Iterable[str]:
    # Stream text lines from gs:// (Requester Pays)
    proc = subprocess.Popen(
        ["gsutil", "-u", require_project(), "cat", gs_uri],
        stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True
    )
    if proc.stdout is None:
        raise RuntimeError("Failed to open gsutil pipe")
    for line in proc.stdout:
        yield line
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"gsutil cat failed for {gs_uri} (exit {ret})")

# ------------------------------ GCS RANGE FETCHER -----------------------------

class RangeFetcher:
    """Persistent HTTP range fetcher using google-cloud-storage (Requester Pays)."""
    def __init__(self):
        self.project = require_project()
        self.client = storage.Client(project=self.project)

    def _blob(self, gs_uri: str):
        if not gs_uri.startswith("gs://"):
            raise RuntimeError(f"Not a gs:// URI: {gs_uri}")
        _, _, rest = gs_uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if (not bucket_name) or (not blob_name):
            raise RuntimeError(f"Malformed GCS URI: {gs_uri}")
        bucket = self.client.bucket(bucket_name, user_project=self.project)
        return bucket.blob(blob_name)

    def fetch(self, gs_uri: str, start: int, end_inclusive: int) -> bytes:
        # download_as_bytes uses inclusive end
        return self._blob(gs_uri).download_as_bytes(start=start, end=end_inclusive)

# ------------------------------ PLINK BED DECODING ----------------------------

# 2-bit encoding (SNP-major):
# 00=A1/A1, 10=A1/A2, 11=A2/A2, 01=missing
def build_lut_4perbyte() -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):
        for i in range(4):
            code = (b >> (2*i)) & 0b11
            if   code == 0b00: lut[b, i] = 0
            elif code == 0b10: lut[b, i] = 1
            elif code == 0b11: lut[b, i] = 2
            else:              lut[b, i] = -1  # missing
    return lut

LUT4 = build_lut_4perbyte()

def decode_a2count_per_snp(block: bytes, n_samples: int) -> np.ndarray:
    """Decode one SNP block into per-sample A2 counts {0,1,2,-1}."""
    arr = np.frombuffer(block, dtype=np.uint8)
    expanded = LUT4[arr]           # shape (bpf, 4)
    flat = expanded.reshape(-1)    # length bpf*4
    return flat[:n_samples].copy()

# ------------------------------ BIM SCAN (CHR17 ONLY) -------------------------

@dataclass
class Hit:
    bp: int
    snp_index: int
    a1: str
    a2: str
    snp_id: str

_bp_num_re = re.compile(r"^\d+(?:\.\d+)?$")

def _parse_bp_int(s: str) -> int:
    # Accept integer or float string that is integral (e.g., "12345" or "12345.0")
    if not _bp_num_re.match(s):
        return -1
    f = float(s)
    i = int(f)
    return i if f == float(i) else -1

def find_chr17_targets_best_effort(
    bim_uri: str,
    bim_size: int,
    wanted: Dict[int, str]
) -> Tuple[List[Hit], List[int]]:
    """
    Stream chr17.bim and collect rows for the requested BPs that actually contain
    the requested allele. If a bp lacks that allele, it is omitted and recorded as missing.
    """
    if not wanted:
        raise RuntimeError("No targets provided.")
    wanted = {bp: al.upper() for bp, al in wanted.items()}
    target_bps = sorted(wanted.keys())

    print("[DEBUG] Target requests:",
          ", ".join(f"{bp}:{wanted[bp]}" for bp in target_bps))

    seen_rows: Dict[int, List[Hit]] = {bp: [] for bp in target_bps}
    chosen: Dict[int, Hit] = {}

    max_bp = max(target_bps)
    idx = 0
    progressed = 0

    with tqdm(total=bim_size, unit="B", unit_scale=True, desc="Scan BIM chr17") as bar:
        for ln in gsutil_cat_lines(bim_uri):
            progressed += len(ln.encode("utf-8", "ignore"))
            if progressed >= (1 << 20):  # ~1 MiB progress updates
                bar.update(progressed)
                progressed = 0

            parts = ln.strip().split()
            if len(parts) < 6:
                idx += 1
                continue

            # columns: chrom, snp_id, cm, bp, a1, a2
            snp_id = parts[1]
            bp_val = _parse_bp_int(parts[3])
            if bp_val < 0:
                idx += 1
                continue
            a1 = parts[4].upper()
            a2 = parts[5].upper()

            if bp_val in seen_rows:
                hit = Hit(bp=bp_val, snp_index=idx, a1=a1, a2=a2, snp_id=snp_id)
                seen_rows[bp_val].append(hit)
                contains = (a1 == wanted[bp_val]) or (a2 == wanted[bp_val])
                print(f"[HIT] bp={bp_val} idx={idx} snp_id={snp_id} alleles={a1}/{a2} "
                      f"contains_target={contains}")
                if (bp_val not in chosen) and contains:
                    chosen[bp_val] = hit
                    print(f"[SELECT] bp={bp_val} -> idx={idx} ({a1}/{a2}) contains '{wanted[bp_val]}'")

            if bp_val > max_bp and all(len(seen_rows[b]) > 0 for b in target_bps):
                print("[DEBUG] Passed max target bp and visited all targets; stopping scan.")
                break

            idx += 1

        if progressed:
            bar.update(progressed)

    missing: List[int] = []
    for bp in target_bps:
        if bp in chosen:
            k = chosen[bp]
            print(f"[KEEP] bp={bp} -> idx={k.snp_index} snp_id={k.snp_id} alleles={k.a1}/{k.a2}")
        else:
            rows = seen_rows.get(bp, [])
            if rows:
                print(f"[WARN] No row at {bp} contained requested allele '{wanted[bp]}'.")
            else:
                print(f"[WARN] Target position {bp} never observed in BIM.")
            missing.append(bp)

    hits = list(chosen.values())
    return hits, missing

# ------------------------------ FAM (IDs and N) -------------------------------

def read_fam_ids(fam_uri: str) -> Tuple[List[str], List[str]]:
    """Return (FID_list, IID_list) with a progress bar."""
    fids: List[str] = []
    iids: List[str] = []

    # First pass: count lines for a nice progress bar
    n = 0
    for _ in gsutil_cat_lines(fam_uri):
        n += 1
    if n <= 0:
        raise RuntimeError(f"Empty FAM: {fam_uri}")

    # Second pass: actually read IDs
    with tqdm(total=n, desc="Read FAM", unit="line") as bar:
        i = 0
        for ln in gsutil_cat_lines(fam_uri):
            p = ln.rstrip("\n").split()
            if len(p) < 2:
                raise RuntimeError(f"Malformed FAM line {i}: {ln!r}")
            fids.append(p[0]); iids.append(p[1])
            i += 1
            bar.update(1)
    return fids, iids

# ------------------------------ HARD-CALL LOGIC -------------------------------

def hard_calls_from_tags(per_bp_dosage: Dict[int, np.ndarray],
                         selected_bps: List[int],
                         n_samples: int) -> np.ndarray:
    """
    Strict unanimity-only per-sample calls across selected tag SNPs:
      - any missing at any tag -> NaN (no-call)
      - all values 2 -> 2  (G/G everywhere)
      - all values 1 -> 1  (G/A everywhere)
      - all values 0 -> 0  (A/A everywhere)
      - any mixture -> NaN (no-call)
    Returns float array with NaN for no-call; integers for calls.
    """
    if not selected_bps:
        return np.full(n_samples, np.nan, dtype=np.float32)

    M = np.vstack([per_bp_dosage[bp] for bp in selected_bps])  # shape (k_tags, N_samples)
    out = np.full(M.shape[1], np.nan, dtype=np.float32)

    for i in range(M.shape[1]):
        col = M[:, i]
        if np.any(col < 0):
            # any missing tag genotype -> no-call
            continue
        v0 = col[0]
        if np.all(col == v0):
            out[i] = float(v0)  # 0, 1, or 2
        # else: mixed -> leave as NaN (no-call)
    return out

def write_single_inversion_hardcalls_tsv(
    iids: List[str],
    calls: np.ndarray,
    selected_bps: List[int],
    chr_label: str,
    out_path: str = OUT_TSV,
) -> None:
    """
    Write a TSV with one hard-call column:
        SampleID <TAB> chr17-<start>-INV-<length>-HARD
    Values: 0, 1, 2, or blank for no-call.
    """
    if not iids:
        raise RuntimeError("No samples found (empty IID list).")

    if not selected_bps:
        print("[WARN] No usable tag SNPs found; writing header and blank entries.")
        start_bp = min(bp for bp, _ in TARGETS)
        end_bp   = max(bp for bp, _ in TARGETS)
    else:
        start_bp = min(selected_bps)
        end_bp   = max(selected_bps)

    inv_id = f"{chr_label}-{start_bp}-INV-{end_bp - start_bp}-HARD"
    print(f"[WRITE] Building '{out_path}' (N={len(iids)}), column='{inv_id}' "
          f"from {len(selected_bps)} tag(s): {selected_bps}")

    with open(out_path, "w") as fo, tqdm(total=len(iids), desc="Write TSV", unit="sample") as bar:
        fo.write(f"SampleID\t{inv_id}\n")
        for iid, v in zip(iids, calls):
            fo.write(f"{iid}\t{'' if np.isnan(v) else str(int(v))}\n")
            bar.update(1)

    print(f"[DONE] Wrote {out_path} with unanimity-only hard calls '{inv_id}'.")

# ------------------------------ MAIN ------------------------------------------

def main():
    print("== chr17 inversion hard-calls (strict unanimity) via ranged PLINK fetch ==")

    # Ensure Requester-Pays project; initialize range fetcher
    _ = require_project()
    rf = RangeFetcher()

    # Stat sizes (for progress bars and sanity)
    bim_size = gsutil_stat_size(BIM_URI)
    bed_size = gsutil_stat_size(BED_URI)
    if bed_size < 3:
        raise RuntimeError(f"BED too small: {BED_URI}")

    # Read FAM IDs (gets N)
    fids, iids = read_fam_ids(FAM_URI)
    n_samples = len(iids)
    bpf = math.ceil(n_samples / 4)  # bytes per SNP in SNP-major PLINK
    if ((bed_size - 3) % bpf) != 0:
        raise RuntimeError(f"{BED_URI} not divisible by bpf={bpf} (N={n_samples})")

    # Build requested map: bp -> inversion allele (G for all here)
    wanted: Dict[int, str] = {bp: al.upper() for bp, al in TARGETS}

    # Find only the needed SNP indices in chr17.bim (best effort; diagnostics)
    hits, missing_bps = find_chr17_targets_best_effort(BIM_URI, bim_size, wanted)

    if not hits:
        print("[FATAL] No usable tag SNP rows found that contain requested allele(s).")
        raise RuntimeError("No tag SNPs available to compute hard-calls.")

    # Sort hits by their SNP index (BED order) to fetch one tight contiguous range
    hits.sort(key=lambda h: h.snp_index)

    # Orientation: inversion allele location (A1 vs A2)
    orient: Dict[int, str] = {}
    for h in hits:
        inv = wanted[h.bp]
        if inv == h.a1:
            orient[h.bp] = "A1"
        elif inv == h.a2:
            orient[h.bp] = "A2"
        else:
            raise RuntimeError(f"Selected row at {h.bp} lacks allele {inv}: {h.a1}/{h.a2}")
        print(f"[ORIENT] bp={h.bp} snp_id={h.snp_id} alleles={h.a1}/{h.a2} -> {orient[h.bp]}")

    # Ranged fetch: grab one contiguous run from min..max SNP index
    i0 = hits[0].snp_index
    i1 = hits[-1].snp_index
    total_snps = i1 - i0 + 1
    start = 3 + i0 * bpf
    end   = 3 + (i1 + 1) * bpf - 1
    total_bytes = end - start + 1

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Fetch BED bytes") as bar:
        blob = rf.fetch(BED_URI, start, end)  # one HTTP GET
        bar.update(len(blob))

    if len(blob) != total_bytes:
        raise RuntimeError(f"Fetched {len(blob)} bytes but expected {total_bytes}")

    # Decode only the selected tag SNP blocks; map to counts of the inversion allele (G)
    per_bp_dosage: Dict[int, np.ndarray] = {}
    for h in hits:
        off = (h.snp_index - i0) * bpf
        block = blob[off:off + bpf]
        a2count = decode_a2count_per_snp(block, n_samples)  # {0,1,2,-1}
        if orient[h.bp] == "A2":
            dos = a2count
        else:
            dos = np.where(a2count >= 0, 2 - a2count, -1).astype(np.int8)
        per_bp_dosage[h.bp] = dos
        print(f"[DECODE] bp={h.bp} decoded G-copies (sample0..4) = {dos[:5].tolist()}")

    selected_bps = [h.bp for h in hits]
    if missing_bps:
        print(f"[NOTE] Tag SNPs without matching allele (absent): {missing_bps}")

    # Strict unanimity-only hard calls
    calls = hard_calls_from_tags(per_bp_dosage, selected_bps, n_samples)

    # Write single-column hard-call matrix
    write_single_inversion_hardcalls_tsv(
        iids=iids,
        calls=calls,
        selected_bps=selected_bps,
        chr_label=f"chr{CHR}",
        out_path=OUT_TSV,
    )

    n_called = int(np.sum(~np.isnan(calls)))
    print(f"== DONE: wrote {OUT_TSV} for N={n_samples:,} samples "
          f"(called {n_called:,}; no-call {n_samples - n_called:,}; "
          f"bpf={bpf}, fetched {total_snps} SNP blocks, {total_bytes/1024:.1f} KiB) ==")

if __name__ == "__main__":
    main()
