import os, sys, math, re, subprocess
from typing import List, Tuple, Dict, Iterable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

# ------------------------------ HARD-CODED PATHS ------------------------------

GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
CHR     = "17"

BIM_URI = GCS_DIR + f"chr{CHR}.bim"
BED_URI = GCS_DIR + f"chr{CHR}.bed"
FAM_URI = GCS_DIR + f"chr{CHR}.fam"

# Targets: (bp, inversion_allele)
TARGETS: List[Tuple[int, str]] = [
    (46003698, "G"),
    (45996523, "G"),
    (45974480, "G"),
]

OUT_TSV = "imputed_inversion_dosages.tsv"

# ------------------------------ ENV & SHELL UTILS -----------------------------

def require_project() -> str:
    pid = os.getenv("GOOGLE_PROJECT")
    if not pid:
        raise RuntimeError("Set GOOGLE_PROJECT for requester-pays access.")
    return pid

def run(cmd: List[str]) -> str:
    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return cp.stdout
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{msg or e}")

def gsutil_stat_size(gs_uri: str) -> int:
    out = run(["gsutil", "-u", require_project(), "stat", gs_uri])
    m = re.search(r"Content-Length:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Unable to parse size for {gs_uri}")
    return int(m.group(1))

def gsutil_cat_lines(gs_uri: str) -> Iterable[str]:
    # stream text lines from gs:// (Requester Pays)
    proc = subprocess.Popen(["gsutil", "-u", require_project(), "cat", gs_uri],
                            stdout=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    if proc.stdout is None:
        raise RuntimeError("Failed to open gsutil pipe")
    for line in proc.stdout:
        yield line
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"gsutil cat failed for {gs_uri} (exit {ret})")

# ------------------------------ GCS RANGE FETCHER -----------------------------

class RangeFetcher:
    """Persistent HTTP range fetcher using google-cloud-storage (no fallbacks)."""
    def __init__(self):
        try:
            from google.cloud import storage  # noqa
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
        # download_as_bytes uses inclusive end
        return self._blob(gs_uri).download_as_bytes(start=start, end=end_inclusive)

# ------------------------------ PLINK BED DECODING ----------------------------

# LUT: for each byte (2 bits per sample, 4 samples), return A2 allele count per sample (or -1 for missing)
# 00=A1/A1 -> A2 count 0
# 10=A1/A2 -> A2 count 1
# 11=A2/A2 -> A2 count 2
# 01=missing -> -1
def build_lut_4perbyte() -> np.ndarray:
    lut = np.zeros((256, 4), dtype=np.int8)
    for b in range(256):
        for i in range(4):
            code = (b >> (2*i)) & 0b11
            if code == 0b00: lut[b, i] = 0
            elif code == 0b10: lut[b, i] = 1
            elif code == 0b11: lut[b, i] = 2
            else: lut[b, i] = -1  # missing
    return lut

LUT4 = build_lut_4perbyte()

def decode_a2count_per_snp(block: bytes, n_samples: int) -> np.ndarray:
    """
    Decode one SNP block (length = bpf bytes) into per-sample A2 allele counts.
    Returns int8 array of length n_samples with values in {0,1,2,-1}.
    """
    arr = np.frombuffer(block, dtype=np.uint8)
    expanded = LUT4[arr]                 # shape (bpf, 4)
    flat = expanded.reshape(-1)          # length bpf*4
    return flat[:n_samples].copy()

# ------------------------------ BIM SCAN (CHR17 ONLY) -------------------------

@dataclass
class Hit:
    bp: int
    snp_index: int
    a1: str
    a2: str
    snp_id: str

def find_chr17_targets(bim_uri: str,
                       bim_size: int,
                       wanted: Dict[int, str]) -> List[Hit]:
    """
    Stream the chr17.bim and stop as soon as we've passed the max requested bp
    AND found all present targets.
    """
    max_bp = max(wanted.keys())
    found: Dict[int, Hit] = {}
    idx = 0
    progressed = 0
    with tqdm(total=bim_size, unit="B", unit_scale=True, desc="Scan BIM chr17") as bar:
        for ln in gsutil_cat_lines(bim_uri):
            progressed += len(ln.encode("utf-8", "ignore"))
            if progressed >= (1 << 20):   # update roughly per MiB
                bar.update(progressed)
                progressed = 0

            p = ln.strip().split()
            if len(p) < 6:
                idx += 1
                continue
            # chrom_raw = p[0]
            snp_id = p[1]
            bp_s   = p[3]
            a1u    = p[4].upper()
            a2u    = p[5].upper()
            try:
                bp = int(float(bp_s))
            except:
                idx += 1
                continue
            # keep if it's one of the requested positions
            if bp in wanted and bp not in found:
                found[bp] = Hit(bp=bp, snp_index=idx, a1=a1u, a2=a2u, snp_id=snp_id)
            # quit early when we passed the biggest target and found all that exist
            if bp > max_bp and len(found) == len(wanted):
                break
            idx += 1

        if progressed:
            bar.update(progressed)

    # Ensure all targets exist in BIM
    missing = [bp for bp in wanted.keys() if bp not in found]
    if missing:
        raise RuntimeError(f"Requested bp not found in BIM: {missing}")
    # Validate alleles are SNPs
    for h in found.values():
        if h.a1 not in {"A","C","G","T"} or h.a2 not in {"A","C","G","T"}:
            raise RuntimeError(f"Non-SNP allele at {h.bp}: {h.a1}/{h.a2}")
    # return in genomic order of input TARGETS (or any order you prefer)
    return [found[bp] for bp in sorted(found.keys())]

# ------------------------------ FAM (IDs and N) --------------------------------

def read_fam_ids(fam_uri: str) -> Tuple[List[str], List[str]]:
    """Return (FID_list, IID_list) and show progress."""
    fids: List[str] = []
    iids: List[str] = []
    # First, count lines to size the progress bar quickly
    # (This is fast: gsutil cat | wc -l would cost another process; we just count once.)
    n = 0
    for _ in gsutil_cat_lines(fam_uri):
        n += 1
    if n <= 0:
        raise RuntimeError(f"Empty FAM: {fam_uri}")

    # Read again to capture IDs with a progress bar
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

# ------------------------------ MAIN ------------------------------------------

def main():
    print("== chr17 inversion dosage (3 SNPs) via ranged PLINK fetch ==")

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
    n_samples = len(fids)
    bpf = math.ceil(n_samples / 4)
    if ((bed_size - 3) % bpf) != 0:
        raise RuntimeError(f"{BED_URI} not divisible by bpf={bpf} (N={n_samples})")

    # Build requested map: bp -> inversion allele (always 'G' here, but keep general)
    wanted: Dict[int, str] = {bp: al.upper() for bp, al in TARGETS}

    # Find only the needed SNP indices in chr17.bim (early-stop scan)
    hits = find_chr17_targets(BIM_URI, bim_size, wanted)  # returns sorted by bp
    # sort the three in BED (BIM) order for a single tight BED range
    hits.sort(key=lambda h: h.snp_index)

    # Validate that G exists among A1/A2 for each target; remember orientation
    orient: Dict[int, str] = {}  # bp -> 'A1' or 'A2'
    for h in hits:
        inv = wanted[h.bp]
        if inv == h.a1:
            orient[h.bp] = "A1"
        elif inv == h.a2:
            orient[h.bp] = "A2"
        else:
            raise RuntimeError(
                f"Inversion allele {inv} not present at {h.bp} (BIM has {h.a1}/{h.a2})"
            )

    # Ranged fetch: grab one contiguous run from min to max SNP index
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

    # Carve out exactly the 3 SNP blocks and decode per-sample dosages
    # Pre-allocate result arrays for each target (int8)
    per_bp_dosage: Dict[int, np.ndarray] = {}
    for h in hits:
        off = (h.snp_index - i0) * bpf
        block = blob[off:off + bpf]
        a2count = decode_a2count_per_snp(block, n_samples)  # {0,1,2,-1}
        if orient[h.bp] == "A2":
            # dosage for inversion allele G is A2 count directly
            dos = a2count
        else:
            # inversion allele is A1 -> dosage = 2 - A2count (except missing)
            dos = np.where(a2count >= 0, 2 - a2count, -1).astype(np.int8)
        per_bp_dosage[h.bp] = dos

    # Write TSV with progress: FID IID and the 3 columns (bp_G)
    # Column order: follow TARGETS order provided in the specification
    header_cols = ["FID", "IID"] + [f"{bp}_G" for bp, _ in TARGETS]
    with open(OUT_TSV, "w") as fo, tqdm(total=n_samples, desc="Write TSV", unit="sample") as bar:
        fo.write("\t".join(header_cols) + "\n")
        for i in range(n_samples):
            row = [fids[i], iids[i]]
            for bp, _ in TARGETS:
                v = per_bp_dosage[bp][i]
                row.append("." if v < 0 else str(int(v)))
            fo.write("\t".join(row) + "\n")
            bar.update(1)

    print(f"== DONE: wrote {OUT_TSV} for N={n_samples:,} samples "
          f"(bpf={bpf}, fetched {total_snps} SNP blocks, {total_bytes/1024:.1f} KiB) ==")

if __name__ == "__main__":
    # hard crash on any fatal error
    main()
