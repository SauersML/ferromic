import os, sys, math, re, subprocess
from typing import List, Tuple, Dict, Iterable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from google.cloud import storage  # hard crash if missing

# ------------------------------ HARD-CODED PATHS ------------------------------

# Requester-pays GCS PLINK shards (hard-coded)
GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
CHR     = "22"

BIM_URI = GCS_DIR + f"chr{CHR}.bim"
BED_URI = GCS_DIR + f"chr{CHR}.bed"
FAM_URI = GCS_DIR + f"chr{CHR}.fam"

# ------------------------------ TARGET (SINGLE SNP) ---------------------------

# hg38 / GRCh38
# rs12628403 at chr22:38962032 (+ strand)
# Deletion-tagging allele = C ; Non-deletion (normal) = A
TARGET_BP = 38962032
DELETION_ALLELE = "C"   # count copies of C (0/1/2); blank = missing
NORMAL_ALLELE   = "A"

# Single-output hard-call matrix
OUT_TSV = "apobec3_deletion_hardcalls.tsv"

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

# ------------------------------ BIM SCAN (CHR22 SINGLE) -----------------------

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

def find_chr22_target_single(bim_uri: str, bim_size: int, target_bp: int) -> Hit:
    print(f"[DEBUG] Looking for chr22 target bp={target_bp} expecting alleles {NORMAL_ALLELE}/{DELETION_ALLELE}")
    idx = 0
    progressed = 0
    chosen: Hit = None  # type: ignore

    with tqdm(total=bim_size, unit="B", unit_scale=True, desc="Scan BIM chr22") as bar:
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

            if bp_val == target_bp:
                allele_set = {a1, a2}
                is_AC = (allele_set == {NORMAL_ALLELE, DELETION_ALLELE})
                print(f"[HIT] bp={bp_val} idx={idx} snp_id={snp_id} alleles={a1}/{a2} is_AC_pair={is_AC}")
                if is_AC:
                    chosen = Hit(bp=bp_val, snp_index=idx, a1=a1, a2=a2, snp_id=snp_id)
                    print(f"[SELECT] bp={bp_val} -> idx={idx} ({a1}/{a2}) A/C row chosen")
                    break  # single SNP: done
            idx += 1

        if progressed:
            bar.update(progressed)

    if chosen is None:
        raise RuntimeError(f"Target position {target_bp} not found with A/C alleles in BIM.")
    print(f"[KEEP] bp={chosen.bp} -> idx={chosen.snp_index} snp_id={chosen.snp_id} alleles={chosen.a1}/{chosen.a2}")
    return chosen

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

def write_single_marker_hardcalls_tsv(
    iids: List[str],
    dosages_c: np.ndarray,
    hit: Hit,
    chr_label: str,
    out_path: str = OUT_TSV,
) -> None:
    """
    Write a TSV with one hard-call column for counts of the deletion-tagging allele (C):
        SampleID <TAB> chr22-38962032-rs12628403-C-HARD
    Values: 0, 1, 2, or blank for no-call.
    """
    if not iids:
        raise RuntimeError("No samples found (empty IID list).")

    marker_id = f"{chr_label}-{hit.bp}-rs12628403-C-HARD"
    print(f"[WRITE] Building '{out_path}' (N={len(iids)}), column='{marker_id}'")

    with open(out_path, "w") as fo, tqdm(total=len(iids), desc="Write TSV", unit="sample") as bar:
        fo.write(f"SampleID\t{marker_id}\n")
        for iid, v in zip(iids, dosages_c):
            fo.write(f"{iid}\t{'' if v < 0 else str(int(v))}\n")
            bar.update(1)

    print(f"[DONE] Wrote {out_path} with allele-C hard calls '{marker_id}'.")

# ------------------------------ MAIN ------------------------------------------

def main():
    print("== chr22 rs12628403 (APOBEC3A/B deletion tag) hard-calls via ranged PLINK fetch ==")
    print(f"[CONFIG] Counting copies of deletion allele '{DELETION_ALLELE}' (normal '{NORMAL_ALLELE}')")

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

    # Locate the single SNP row in chr22.bim (must be A/C)
    hit = find_chr22_target_single(BIM_URI, bim_size, TARGET_BP)

    # Orientation: is deletion allele the A1 or A2 column?
    if DELETION_ALLELE == hit.a1:
        orient = "A1"
    elif DELETION_ALLELE == hit.a2:
        orient = "A2"
    else:
        raise RuntimeError(f"Selected row at {hit.bp} lacks allele {DELETION_ALLELE}: {hit.a1}/{hit.a2}")
    print(f"[ORIENT] bp={hit.bp} snp_id={hit.snp_id} alleles={hit.a1}/{hit.a2} -> {orient} carries '{DELETION_ALLELE}'")

    # Ranged fetch: bytes for this single SNP index
    i0 = hit.snp_index
    start = 3 + i0 * bpf
    end   = 3 + (i0 + 1) * bpf - 1
    total_bytes = end - start + 1

    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Fetch BED bytes") as bar:
        blob = rf.fetch(BED_URI, start, end)  # one HTTP GET
        bar.update(len(blob))

    if len(blob) != total_bytes:
        raise RuntimeError(f"Fetched {len(blob)} bytes but expected {total_bytes}")

    # Decode this SNP to counts of the deletion allele (C): {0,1,2,-1}
    a2count = decode_a2count_per_snp(blob, n_samples)  # A2 copies
    if orient == "A2":
        dos_c = a2count
    else:
        dos_c = np.where(a2count >= 0, 2 - a2count, -1).astype(np.int8)

    print(f"[DECODE] bp={hit.bp} decoded C-copies (sample0..4) = {dos_c[:5].tolist()}")

    # Write single-column hard-call matrix
    write_single_marker_hardcalls_tsv(
        iids=iids,
        dosages_c=dos_c,
        hit=hit,
        chr_label=f"chr{CHR}",
        out_path=OUT_TSV,
    )

    n_called = int(np.sum(dos_c >= 0))
    print(f"== DONE: wrote {OUT_TSV} for N={n_samples:,} samples "
          f"(called {n_called:,}; no-call {n_samples - n_called:,}; "
          f"bpf={bpf}, fetched 1 SNP block, {(total_bytes/1024):.1f} KiB) ==")

if __name__ == "__main__":
    main()
