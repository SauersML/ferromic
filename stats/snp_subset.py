import os
import sys
import math
import re
import shutil
import tempfile
import subprocess
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests

# --- CONFIGURATION ------------------------------------------------------------

# Input: list of SNPs to find. Format: "CHR:BP" (e.g., "1:10583")
TARGET_SNPS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/stats/all_unique_snps_sorted.txt"

# Input: Directory containing sharded PLINK files (BED/BIM/FAM)
ACAF_PLINK_GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"

# Output filenames
OUTPUT_BIM_FILENAME = "subset.bim"
OUTPUT_FAM_FILENAME = "subset.fam"
OUTPUT_BED_FILENAME = "subset.bed"

# Performance knobs
MAX_GAP_SNPS = 64           # coalesce indices separated by <= this many SNPs into one range
MAX_WORKERS = min(16, (os.cpu_count() or 8) * 2)  # parallel ranged reads

# -----------------------------------------------------------------------------

def _require_project() -> str:
    pid = os.getenv("GOOGLE_PROJECT")
    if not pid:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    return pid

# ----------------------- gsutil helpers --------------------------------------

def gsutil_run(args: List[str], *, capture: bool = True, text: bool = False) -> subprocess.CompletedProcess:
    project_id = _require_project()
    cmd = ["gsutil", "-u", project_id] + args
    return subprocess.run(cmd, check=True, capture_output=capture, text=text)

def gsutil_ls(pattern: str) -> List[str]:
    out = gsutil_run(["ls", pattern], capture=True, text=True).stdout.strip()
    if not out:
        return []
    return sorted([line for line in out.splitlines() if line.strip()])

def gsutil_cat_stream(gs_uri: str):
    project_id = _require_project()
    cmd = ["gsutil", "-u", project_id, "cat", gs_uri]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, errors="replace")
    try:
        for line in proc.stdout:
            yield line
    finally:
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"gsutil cat failed for {gs_uri} with code {ret}")

def gsutil_cat_range(gs_uri: str, start: int, end: int) -> bytes:
    project_id = _require_project()
    cmd = ["gsutil", "-u", project_id, "cat", "-r", f"{start}-{end}", gs_uri]
    return subprocess.check_output(cmd)

def gsutil_size(gs_uri: str) -> int:
    out = gsutil_run(["stat", gs_uri], capture=True, text=True).stdout
    m = re.search(r"Content-Length:\s*(\d+)", out)
    if not m:
        raise RuntimeError(f"Unable to read size for {gs_uri}")
    return int(m.group(1))

# ----------------------- Requester-Pays aware fetcher ------------------------

class RangeFetcher:
    """
    Fetch byte ranges from GCS using the Python client with Requester Pays,
    falling back to gsutil if the client is unavailable or unauthed.
    """
    def __init__(self):
        self.mode = "gsutil"
        self.client = None
        self.project = _require_project()
        try:
            from google.cloud import storage  # lazy import
            self.client = storage.Client(project=self.project)
            # Quick smoke test: ensure we can build a blob with user_project
            self.mode = "gcs"
        except Exception as e:
            tqdm.write(f"Note: google-cloud-storage unavailable or not authed ({e}); using gsutil fallback.")
            self.client = None
            self.mode = "gsutil"

    def _blob_with_user_project(self, gs_uri: str):
        from google.cloud import storage  # type: ignore
        # Parse "gs://bucket/path"
        if not gs_uri.startswith("gs://"):
            raise ValueError(f"Not a gs:// URI: {gs_uri}")
        _, _, rest = gs_uri.partition("gs://")
        bucket_name, _, blob_name = rest.partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Malformed GCS URI: {gs_uri}")
        # IMPORTANT: user_project passed here enables Requester Pays billing
        bucket = self.client.bucket(bucket_name, user_project=self.project)
        return bucket.blob(blob_name)

    def fetch(self, gs_uri: str, start: int, end: int) -> bytes:
        if self.mode == "gcs":
            try:
                blob = self._blob_with_user_project(gs_uri)
                return blob.download_as_bytes(start=start, end=end)  # inclusive
            except Exception as e:
                # Fall back to gsutil transparently on any client error
                tqdm.write(f"Note: GCS client range fetch failed ({e}); falling back to gsutil for {gs_uri}.")
                return gsutil_cat_range(gs_uri, start, end)
        else:
            return gsutil_cat_range(gs_uri, start, end)

# ----------------------- Core logic ------------------------------------------

def fetch_target_snps(url: str) -> set:
    print("--- STEP 1: Fetching Target SNPs ---")
    try:
        r = requests.get(url)
        r.raise_for_status()
        snps = {line.strip() for line in r.text.splitlines() if line.strip()}
        print(f"Loaded {len(snps):,} target SNPs.\n")
        return snps
    except requests.RequestException as e:
        print(f"FATAL: Could not fetch SNP list from {url}. Error: {e}", file=sys.stderr)
        sys.exit(1)

def identify_matches_and_geometry(gcs_dir_path: str, target_snps_set: set) -> Tuple[List[dict], int]:
    """
    Stream .bim shards, write subset .bim, and collect per-shard metadata:
      {
        "order": shard_order_index,
        "bim": <gcs_uri>,
        "bed": <gcs_uri>,
        "match_indices": [i, j, ...],  # ascending within this shard
        "variant_count": V_shard,
        "bytes_per_snp": bpf_remote
      }
    Enforces SNP-major and identical bytes_per_snp across shards containing matches.
    """
    print("--- STEP 2: Scanning .bim files and validating shard geometry ---")
    bim_files = gsutil_ls(os.path.join(gcs_dir_path, "*.bim"))
    if not bim_files:
        print(f"FATAL: No .bim files found under {gcs_dir_path}", file=sys.stderr)
        sys.exit(1)

    matched_shards: List[dict] = []
    total_matches = 0
    common_bpf = None

    with open(OUTPUT_BIM_FILENAME, "w") as fout:
        for order, bim_path in enumerate(tqdm(bim_files, desc="Scanning .bim shards")):
            bed_path = bim_path.replace(".bim", ".bed")
            idx = 0
            V_shard = 0
            match_indices: List[int] = []

            for line in gsutil_cat_stream(bim_path):
                parts = line.strip().split()  # whitespace-safe
                if len(parts) >= 4:
                    key = f"{parts[0]}:{parts[3]}"  # chr:bp
                    if key in target_snps_set:
                        fout.write(line)  # keep line as-is
                        match_indices.append(idx)
                idx += 1
            V_shard = idx

            if V_shard == 0:
                tqdm.write(f"WARNING: {bim_path} appears empty; skipping.")
                continue
            if not match_indices:
                continue

            # Validate SNP-major header and compute bytes/SNP from actual BED size
            hdr = gsutil_cat_range(bed_path, 0, 2)
            if hdr != b"\x6c\x1b\x01":
                raise RuntimeError(f"{bed_path} is not SNP-major .bed (header={hdr.hex()})")

            B = gsutil_size(bed_path)
            if B < 3:
                raise RuntimeError(f"{bed_path} too small (size={B})")

            rem = (B - 3) % V_shard
            if rem != 0:
                raise RuntimeError(f"{bed_path} has non-integer bytes/SNP: (B-3)={B-3}, V={V_shard}")
            bpf = (B - 3) // V_shard

            if common_bpf is None:
                common_bpf = bpf
            elif bpf != common_bpf:
                raise RuntimeError(
                    f"Inconsistent bytes/SNP across shards: {bed_path} has {bpf}, expected {common_bpf}"
                )

            matched_shards.append({
                "order": order,
                "bim": bim_path,
                "bed": bed_path,
                "match_indices": match_indices,
                "variant_count": V_shard,
                "bytes_per_snp": bpf
            })
            total_matches += len(match_indices)

    print(f"Found {total_matches:,} matching SNPs across {len(matched_shards)} shards.")
    print(f"Wrote local BIM: ./{OUTPUT_BIM_FILENAME}\n")
    return matched_shards, (common_bpf or 0)

def select_and_copy_fam(gcs_dir_path: str, required_bpf: int) -> int:
    """
    Copy a .fam whose ceil(N/4) equals required_bpf. Return N (sample count).
    """
    print("--- STEP 3: Selecting compatible .fam and copying locally ---")
    fam_files = gsutil_ls(os.path.join(gcs_dir_path, "*.fam"))
    if not fam_files:
        print(f"FATAL: No .fam files found under {gcs_dir_path}", file=sys.stderr)
        sys.exit(1)

    compatible: List[Tuple[str, int]] = []
    for fam in fam_files:
        n = 0
        for _ in gsutil_cat_stream(fam):
            n += 1
        if math.ceil(n / 4) == required_bpf:
            compatible.append((fam, n))

    if not compatible:
        raise RuntimeError(
            f"No .fam in directory has ceil(N/4) equal to shards' bytes-per-SNP ({required_bpf})."
        )

    fam_src, N = compatible[0]
    tqdm.write(f"Selected {os.path.basename(fam_src)} (N={N:,}, ceil(N/4)={required_bpf}).")
    gsutil_run(["cp", fam_src, OUTPUT_FAM_FILENAME], capture=True, text=True)
    print(f"Copied local FAM: ./{OUTPUT_FAM_FILENAME}\n")
    return N

def coalesce_indices(indices: List[int], max_gap: int) -> List[List[int]]:
    """Group sorted indices into runs where consecutive elements differ by <= max_gap."""
    if not indices:
        return []
    runs: List[List[int]] = []
    current = [indices[0]]
    for i in indices[1:]:
        if i - current[-1] <= max_gap:
            current.append(i)
        else:
            runs.append(current)
            current = [i]
    runs.append(current)
    return runs

def process_shard_to_temp(shard: dict, fetcher: RangeFetcher, tmpdir: str, pbar: tqdm) -> Tuple[int, str, int]:
    """
    For one shard:
      - Build runs over match_indices.
      - For each run, fetch a contiguous byte range and carve out only required blocks.
      - Append to a shard temp file (headerless bed-part).
    Returns (order, temp_path, n_variants_written).
    """
    bed = shard["bed"]
    bpf = shard["bytes_per_snp"]
    indices = shard["match_indices"]  # ascending
    runs = coalesce_indices(indices, MAX_GAP_SNPS)

    temp_path = os.path.join(tmpdir, f"{os.path.basename(bed)}.subset.part")
    n_written = 0

    with open(temp_path, "wb") as fout:
        for run in runs:
            start_idx = run[0]
            end_idx = run[-1]
            start_off = 3 + start_idx * bpf
            end_off = 3 + (end_idx + 1) * bpf - 1  # inclusive
            blob_bytes = fetcher.fetch(bed, start_off, end_off)

            # Carve only the wanted indices
            for idx in run:
                offset_in_blob = (idx - start_idx) * bpf
                chunk = blob_bytes[offset_in_blob:offset_in_blob + bpf]
                if len(chunk) != bpf:
                    raise IOError(f"Short slice in {bed} at idx {idx}: {len(chunk)} vs {bpf}")
                fout.write(chunk)
                n_written += 1
                pbar.update(1)

    return shard["order"], temp_path, n_written

def assemble_bed_parallel(matched_shards: List[dict]) -> int:
    """
    Parallel assembly:
      - Each shard writes a headerless temp .bed part in parallel (coalesced ranged reads).
      - Then concatenate parts in BIM-order with a single 3-byte header.
    Returns number of variants written.
    """
    print("--- STEP 4: Assembling subset .bed with coalesced, parallel ranged reads ---")

    total_snps = sum(len(s["match_indices"]) for s in matched_shards)
    if total_snps == 0:
        print("No matching SNPs to extract; skipping .bed creation.\n")
        return 0

    fetcher = RangeFetcher()
    tmpdir = tempfile.mkdtemp(prefix="subset_bed_parts_")
    parts_in_order: List[Tuple[int, str, int]] = []
    n_written_total = 0

    try:
        with tqdm(total=total_snps, desc="Extracting genotype blocks") as pbar:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [
                    ex.submit(process_shard_to_temp, shard, fetcher, tmpdir, pbar)
                    for shard in matched_shards
                ]
                for fut in as_completed(futures):
                    order, part_path, n_written = fut.result()
                    parts_in_order.append((order, part_path, n_written))
                    n_written_total += n_written

        # Order parts by shard order (same as BIM writing order)
        parts_in_order.sort(key=lambda x: x[0])

        # Concatenate into final BED (add header once)
        with open(OUTPUT_BED_FILENAME, "wb") as fout:
            fout.write(b"\x6c\x1b\x01")
            for _, part_path, _ in parts_in_order:
                with open(part_path, "rb") as fpart:
                    shutil.copyfileobj(fpart, fout)
    finally:
        # Cleanup no matter what
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Wrote local BED: ./{OUTPUT_BED_FILENAME}\n")
    return n_written_total

def final_integrity_checks(n_variants_written: int, matched_shards: List[dict], fam_N: int):
    """
    Validate what PLINK will compute: BIM count, BED size, bytes/SNP consistency.
    """
    print("--- STEP 5: Final integrity checks ---")
    with open(OUTPUT_BIM_FILENAME, "r") as f:
        bim_lines = sum(1 for _ in f)
    if bim_lines != n_variants_written:
        raise AssertionError(
            f"BIM line count ({bim_lines}) != variants written to BED ({n_variants_written})"
        )

    bpf = matched_shards[0]["bytes_per_snp"] if matched_shards else math.ceil(fam_N / 4)
    actual = os.path.getsize(OUTPUT_BED_FILENAME) if n_variants_written > 0 else 0
    expected = 3 + n_variants_written * bpf if n_variants_written > 0 else 0
    if actual != expected:
        raise AssertionError(
            f"BED size mismatch: got {actual}, expected {expected} "
            f"(variants={n_variants_written}, bpf={bpf})"
        )

    fam_bpf = math.ceil(fam_N / 4)
    if n_variants_written > 0 and fam_bpf != bpf:
        raise AssertionError(
            f".fam implies bytes/SNP={fam_bpf} but shards use {bpf}. Wrong .fam?"
        )

    print("Integrity checks passed.\n")

def main():
    print("Starting PLINK subset creation process...\n")

    targets = fetch_target_snps(TARGET_SNPS_URL)
    matched_shards, common_bpf = identify_matches_and_geometry(ACAF_PLINK_GCS_DIR, targets)

    if not matched_shards:
        print("--- FINAL RESULT ---")
        print("No matching SNPs were found. No .bed will be created.")
        if os.path.exists(OUTPUT_BIM_FILENAME) and os.path.getsize(OUTPUT_BIM_FILENAME) == 0:
            os.remove(OUTPUT_BIM_FILENAME)
        sys.exit(0)

    fam_N = select_and_copy_fam(ACAF_PLINK_GCS_DIR, common_bpf)

    n_written = assemble_bed_parallel(matched_shards)

    final_integrity_checks(n_written, matched_shards, fam_N)

    print("--- FINAL RESULT ---")
    print("Successfully created a PLINK 1.9-readable subset fileset:")
    print(f"  --> {OUTPUT_FAM_FILENAME} (N={fam_N:,})")
    print(f"  --> {OUTPUT_BIM_FILENAME} (variants={n_written:,})")
    print(f"  --> {OUTPUT_BED_FILENAME}")
    print("\nProcess complete.")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        msg = e.stderr if isinstance(e.stderr, str) and e.stderr else str(e)
        print(f"FATAL: A gsutil command failed. {msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
