import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOI = "doi:10.7910/DVN/7RVV9N"
BASE_URL = "https://dataverse.harvard.edu"

OUTPUT_DIR = Path("stats")
INV_PROPERTIES_PATH = Path("data/inv_properties.tsv")

SELECTION_GZ_NAME = "Selection_Summary_Statistics_01OCT2025.tsv.gz"
SELECTION_TSV_NAME = "Selection_Summary_Statistics_01OCT2025.tsv"
SELECTION_TSV_PATH = OUTPUT_DIR / SELECTION_TSV_NAME

PHY_ZIP_URL = "https://sharedspace.s3.msi.umn.edu/public_internet/all_phy/phy_files.zip"
PHY_ZIP_LOCAL = Path("phy_files.zip")


def get_dataset_metadata():
    """Fetch dataset metadata from the Dataverse API."""
    api_url = f"{BASE_URL}/api/datasets/:persistentId/?persistentId={DOI}"
    print("Fetching dataset metadata...")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))

    status = data.get("status")
    if status != "OK":
        raise ValueError(f"API returned non-OK status: {status}")

    return data["data"]


def calculate_md5(filepath: Path) -> str:
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def format_size(size_bytes: int) -> str:
    """Format file size in a human-readable form."""
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def unzip_file(gz_path: Path) -> Path:
    """Unzip a .gz file with progress reporting."""
    if gz_path.suffix == ".gz":
        output_path = gz_path.with_suffix("")
    else:
        output_path = Path(str(gz_path).replace(".gz", ""))

    if output_path.exists():
        print(f"Unzipped file {output_path.name} already exists, skipping extraction")
        return output_path

    print(f"Unzipping {gz_path.name}...")

    total_size = gz_path.stat().st_size
    bytes_read = 0
    chunk_size = 8192 * 1024  # 8 MB

    with gzip.open(gz_path, "rb") as f_in, output_path.open("wb") as f_out:
        while True:
            chunk = f_in.read(chunk_size)
            if not chunk:
                break
            f_out.write(chunk)
            bytes_read += len(chunk)
            if total_size > 0:
                progress = bytes_read / total_size * 100
                print(
                    f"\rUnzipped: {format_size(bytes_read)} "
                    f"({progress:.1f}% of {format_size(total_size)})",
                    end="",
                )
            else:
                print(f"\rUnzipped: {format_size(bytes_read)}", end="")

    print()
    print(f"✓ Unzipped to {output_path.name} ({format_size(output_path.stat().st_size)})")
    return output_path


def download_file(file_id: int, filename: str, expected_md5: str | None):
    """
    Download a file from Dataverse.

    Returns
    -------
    (path, is_gz)
        path : Path to downloaded file
        is_gz : bool indicating whether the file is a .gz archive
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    is_gz = filename.endswith(".gz")

    if output_path.exists():
        print(f"✓ File {filename} already exists")
        if expected_md5:
            print("  Verifying MD5 checksum...")
            actual_md5 = calculate_md5(output_path)
            if actual_md5 == expected_md5:
                print("  ✓ MD5 verified")
                return output_path, is_gz
            print(
                f"  ✗ MD5 mismatch! Expected {expected_md5}, got {actual_md5}. "
                f"Re-downloading..."
            )
        else:
            return output_path, is_gz

    download_url = f"{BASE_URL}/api/access/datafile/{file_id}"
    print(f"Downloading {filename}...")

    req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content_length = response.headers.get("Content-Length")
        file_size = int(content_length) if content_length is not None else None
        if file_size is not None:
            print(f"  File size: {format_size(file_size)}")

        bytes_read = 0
        chunk_size = 8192 * 1024

        with output_path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_read += len(chunk)
                mb_read = bytes_read / (1024 * 1024)
                if file_size:
                    progress = bytes_read / file_size * 100
                    print(
                        f"\r  Progress: {progress:.1f}% ({mb_read:.1f} MB)",
                        end="",
                    )
                else:
                    print(f"\r  Downloaded: {mb_read:.1f} MB", end="")

    print()
    print("✓ Download complete")

    if expected_md5:
        print("  Verifying MD5 checksum...")
        actual_md5 = calculate_md5(output_path)
        if actual_md5 != expected_md5:
            output_path.unlink(missing_ok=True)
            raise ValueError(
                f"MD5 mismatch after download! Expected {expected_md5}, got {actual_md5}"
            )
        print("  ✓ MD5 verified")

    return output_path, is_gz


def install_liftover():
    """Download and install the UCSC liftOver tool if not already present."""
    which_result = subprocess.run(
        ["which", "liftOver"],
        capture_output=True,
        text=True,
    )
    
    if which_result.returncode == 0:
        print("✓ liftOver tool already installed")
        return
    
    print("liftOver tool not found. Installing...")
    
    # Determine the appropriate binary URL based on system architecture
    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "linux" and machine == "x86_64":
        liftover_url = "http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver"
    elif system == "darwin" and machine == "x86_64":
        liftover_url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.x86_64/liftOver"
    elif system == "darwin" and machine == "arm64":
        liftover_url = "http://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver"
    else:
        raise RuntimeError(
            f"Unsupported platform: {system} {machine}. "
            "Please install liftOver manually from UCSC."
        )
    
    # Create a local bin directory if it doesn't exist
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    
    liftover_path = local_bin / "liftOver"
    
    print(f"  Downloading liftOver from {liftover_url}...")
    req = urllib.request.Request(liftover_url, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as response:
        with liftover_path.open("wb") as f:
            f.write(response.read())
    
    # Make it executable
    liftover_path.chmod(0o755)
    
    print(f"✓ liftOver installed to {liftover_path}")
    
    # Add to PATH for current session
    current_path = os.environ.get("PATH", "")
    if str(local_bin) not in current_path:
        os.environ["PATH"] = f"{local_bin}:{current_path}"
        print(f"  Added {local_bin} to PATH for this session")


def download_and_unzip_phy():
    """Download and unzip the PHY files."""
    import zipfile
    
    if PHY_ZIP_LOCAL.exists():
        print(f"✓ PHY zip file {PHY_ZIP_LOCAL.name} already exists")
    else:
        print(f"Downloading PHY files from {PHY_ZIP_URL}...")
        req = urllib.request.Request(PHY_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req) as response:
            content_length = response.headers.get("Content-Length")
            file_size = int(content_length) if content_length is not None else None
            if file_size is not None:
                print(f"  File size: {format_size(file_size)}")
            
            bytes_read = 0
            chunk_size = 8192 * 1024
            
            with PHY_ZIP_LOCAL.open("wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_read += len(chunk)
                    mb_read = bytes_read / (1024 * 1024)
                    if file_size:
                        progress = bytes_read / file_size * 100
                        print(
                            f"\r  Progress: {progress:.1f}% ({mb_read:.1f} MB)",
                            end="",
                        )
                    else:
                        print(f"\r  Downloaded: {mb_read:.1f} MB", end="")
        
        print()
        print("✓ Download complete")
    
    # Unzip the file
    phy_extract_dir = Path("phy_files")
    if phy_extract_dir.exists() and any(phy_extract_dir.iterdir()):
        print(f"✓ PHY files already extracted to {phy_extract_dir}")
    else:
        print(f"Extracting {PHY_ZIP_LOCAL.name}...")
        phy_extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(PHY_ZIP_LOCAL, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            print(f"  Extracting {total_files} files...")
            
            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, phy_extract_dir)
                if i % 100 == 0 or i == total_files:
                    print(f"\r  Extracted: {i}/{total_files} files", end="")
            
            print()
        
        print(f"✓ Extracted to {phy_extract_dir}")


def main():
    """
    Download the target selection summary statistics file if present in the dataset.
    """
    # Download and unzip PHY files first
    download_and_unzip_phy()
    
    # Install liftOver tool if not present
    install_liftover()
    
    OUTPUT_DIR.mkdir(exist_ok=True)

    metadata = get_dataset_metadata()
    files_metadata = metadata["latestVersion"]["files"]

    target_filename = SELECTION_GZ_NAME
    print(f"Looking for {target_filename} in dataset...")

    for data_file in files_metadata:
        data_info = data_file.get("dataFile", {})
        filename = data_info.get("filename")
        if filename != target_filename:
            continue

        file_id = data_info["id"]
        checksum = data_info.get("checksum", {})
        expected_md5 = (
            checksum.get("value") if checksum.get("type") == "MD5" else None
        )

        downloaded_path, is_gz = download_file(file_id, filename, expected_md5)
        if is_gz and downloaded_path.exists():
            unzip_file(downloaded_path)
        break
    else:
        print(f"✗ Could not find file '{target_filename}' in the dataset.")
        return 1

    return 0


def liftover_coordinates_batch(regions, from_build: str, to_build: str):
    """
    Liftover a batch of genomic regions using the UCSC liftOver binary.

    Parameters
    ----------
    regions : list of dict
        Each dict has keys: 'chrom', 'start', 'end'.
        'chrom' may be with or without 'chr' prefix.
    from_build : str
        Source genome build (e.g. 'hg38').
    to_build : str
        Target genome build (e.g. 'hg19').

    Returns
    -------
    (final_regions, stats_dict)
        final_regions : list of dict
            Successfully lifted regions in the original ordering (failed removed),
            each with 'chrom', 'start', 'end'.
        stats_dict : dict
            Summary with keys:
            'total', 'lifted', 'failed', 'unmapped', 'failed_indices'.

    Raises
    ------
    RuntimeError
        If liftOver is missing or all regions fail to liftover.
    """
    if not regions:
        raise RuntimeError("No regions provided for liftover")

    print(f"  Lifting over {len(regions)} regions from {from_build} to {to_build}...")
    temp_dir = Path(tempfile.gettempdir())
    pid = os.getpid()

    bed_input_path = temp_dir / f"liftover_input_{pid}.bed"
    bed_output_path = temp_dir / f"liftover_output_{pid}.bed"
    unmapped_path = temp_dir / f"liftover_unmapped_{pid}.bed"

    region_prefix = "region_"
    region_map = {}

    with bed_input_path.open("w") as bed_input:
        for i, region in enumerate(regions):
            chrom = str(region["chrom"])
            if not chrom.startswith("chr"):
                chrom = f"chr{chrom}"
            start = int(region["start"])
            end = int(region["end"])
            region_id = f"{region_prefix}{i}"
            bed_input.write(f"{chrom}\t{start}\t{end}\t{region_id}\n")
            region_map[region_id] = region

    OUTPUT_DIR.mkdir(exist_ok=True)
    chain_file = OUTPUT_DIR / f"{from_build}To{to_build.capitalize()}.over.chain.gz"

    if not chain_file.exists():
        chain_url = (
            f"https://hgdownload.soe.ucsc.edu/goldenPath/"
            f"{from_build}/liftOver/{from_build}To{to_build.capitalize()}.over.chain.gz"
        )
        print(f"  Downloading chain file: {chain_file.name}...")
        req = urllib.request.Request(chain_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, chain_file.open("wb") as out:
            out.write(response.read())
        print("  ✓ Downloaded chain file")

    cmd = [
        "liftOver",
        str(bed_input_path),
        str(chain_file),
        str(bed_output_path),
        str(unmapped_path),
    ]
    run_result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if run_result.returncode != 0:
        bed_input_path.unlink(missing_ok=True)
        bed_output_path.unlink(missing_ok=True)
        unmapped_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"liftOver failed with exit code {run_result.returncode}: {run_result.stderr}"
        )

    lifted_map = {}
    lifted_regions = []

    with bed_output_path.open() as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            region_id = fields[3]
            chrom = fields[0]
            if chrom.startswith("chr"):
                chrom = chrom[3:]
            start = int(fields[1])
            end = int(fields[2])
            lifted_region = {"chrom": chrom, "start": start, "end": end}
            lifted_regions.append(lifted_region)
            lifted_map[region_id] = lifted_region

    unmapped_regions = []
    if unmapped_path.exists():
        with unmapped_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) >= 4:
                    region_id = fields[3]
                    original = region_map.get(region_id)
                    if original is not None:
                        unmapped_regions.append(
                            {
                                "region_id": region_id,
                                "chrom": original["chrom"],
                                "start": int(original["start"]),
                                "end": int(original["end"]),
                                "size": int(original["end"]) - int(original["start"]),
                            }
                        )

    total = len(regions)
    lifted_count = len(lifted_regions)
    failed_indices = [
        i for i in range(total) if f"{region_prefix}{i}" not in lifted_map
    ]
    failed = len(failed_indices)

    bed_input_path.unlink(missing_ok=True)
    bed_output_path.unlink(missing_ok=True)
    unmapped_path.unlink(missing_ok=True)

    if lifted_count == 0:
        raise RuntimeError("All regions failed to liftover")

    if failed > 0:
        print(f"  ⚠ {failed}/{total} regions failed to liftover")

    final_regions = [
        lifted_map[f"{region_prefix}{i}"]
        for i in range(total)
        if f"{region_prefix}{i}" in lifted_map
    ]

    stats = {
        "total": total,
        "lifted": lifted_count,
        "failed": failed,
        "unmapped": unmapped_regions,
        "failed_indices": failed_indices,
    }

    return final_regions, stats


def compare_selection_coefficients():
    """
    Summarize relationships between inversion properties and mean selection coefficients.

    All liftover failures and overlapping inversions are excluded prior to aggregation.
    No formal hypothesis testing is performed.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS: MEAN SELECTION COEFFICIENT (S) vs INVERSION PROPERTIES")
    print("=" * 80 + "\n")

    # Step 1: Load inversion regions with metadata
    print("Step 1: Loading inversion regions with metadata...")
    inversions = []

    with INV_PROPERTIES_PATH.open() as f:
        header = f.readline().strip().split("\t")
        try:
            num_recurrent_idx = header.index("Number_recurrent_events")
            inverted_af_idx = header.index("Inverted_AF")
        except ValueError as e:
            raise RuntimeError(
                f"Required columns missing in {INV_PROPERTIES_PATH}: {e}"
            )

        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) <= max(num_recurrent_idx, inverted_af_idx):
                continue

            chrom = fields[0].replace("chr", "")
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError:
                continue

            size = end - start

            num_recurrent = None
            value = fields[num_recurrent_idx]
            if value not in ("", "NA", "na"):
                try:
                    num_recurrent = int(value)
                except ValueError:
                    num_recurrent = None

            inverted_af = None
            value = fields[inverted_af_idx]
            if value not in ("", "NA", "na"):
                try:
                    inverted_af = float(value)
                except ValueError:
                    inverted_af = None

            inversions.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "size": size,
                    "num_recurrent": num_recurrent,
                    "inverted_af": inverted_af,
                }
            )

    if not inversions:
        print("✗ No inversion records loaded.")
        return

    print(f"  Total inversions loaded: {len(inversions)}")
    print(
        f"  Total bp in inversions: {sum(inv['size'] for inv in inversions):,}"
    )

    # Step 2: Liftover coordinates
    print("\nStep 2: Coordinate liftover (hg38 → hg19)")
    all_regions = inversions

    try:
        lifted_regions, liftover_stats = liftover_coordinates_batch(
            all_regions, "hg38", "hg19"
        )
    except RuntimeError as e:
        print(f"\n✗ CRITICAL: Liftover failed completely: {e}")
        print("  Cannot proceed with analysis.")
        sys.exit(1)

    if liftover_stats.get("failed", 0) > 0:
        print(
            f"\n  Excluding {liftover_stats['failed']} regions "
            f"that failed liftover from analysis..."
        )
        failed_indices = set(liftover_stats.get("failed_indices", []))
        new_inversions = []
        lifted_idx = 0

        for i in range(len(all_regions)):
            if i in failed_indices:
                continue
            lifted_inv = dict(lifted_regions[lifted_idx])
            lifted_inv["num_recurrent"] = all_regions[i]["num_recurrent"]
            lifted_inv["inverted_af"] = all_regions[i]["inverted_af"]
            new_inversions.append(lifted_inv)
            lifted_idx += 1

        inversions = new_inversions
        print(f"  Remaining after liftover filter: {len(inversions)} inversions")
    else:
        for i in range(len(lifted_regions)):
            lifted_regions[i]["num_recurrent"] = all_regions[i]["num_recurrent"]
            lifted_regions[i]["inverted_af"] = all_regions[i]["inverted_af"]
        inversions = lifted_regions

    if not inversions:
        print("✗ No inversions remain after liftover.")
        return

    # Step 3: De-overlap inversions
    print("\nStep 3: De-overlapping inversions (>10% reciprocal overlap)...")

    by_chrom = defaultdict(list)
    for idx, inv in enumerate(inversions):
        by_chrom[inv["chrom"]].append((idx, inv))

    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x[1]["start"])

    excluded_indices = set()
    overlap_count = 0

    for chrom, invs in by_chrom.items():
        n = len(invs)
        for i in range(n):
            idx_i, inv_i = invs[i]
            if idx_i in excluded_indices:
                continue
            for j in range(i + 1, n):
                idx_j, inv_j = invs[j]
                if idx_j in excluded_indices:
                    continue
                if inv_j["start"] >= inv_i["end"]:
                    break
                overlap_start = max(inv_i["start"], inv_j["start"])
                overlap_end = min(inv_i["end"], inv_j["end"])
                if overlap_start < overlap_end:
                    overlap_bp = overlap_end - overlap_start
                    size_i = inv_i["end"] - inv_i["start"]
                    size_j = inv_j["end"] - inv_j["start"]
                    if (
                        overlap_bp / size_i > 0.10
                        or overlap_bp / size_j > 0.10
                    ):
                        if size_i >= size_j:
                            excluded_indices.add(idx_j)
                        else:
                            excluded_indices.add(idx_i)
                        overlap_count += 1
                        break

    non_overlapping = [
        inv for idx, inv in enumerate(inversions) if idx not in excluded_indices
    ]

    print(f"  Overlapping inversion pairs detected: {overlap_count}")
    print(f"  Excluded inversions: {len(excluded_indices)}")
    print(f"  Retained non-overlapping inversions: {len(non_overlapping)}")

    if not non_overlapping:
        print("✗ No inversions remain after de-overlap filtering.")
        return

    # Step 4: Index inversions and collect S values
    print("\nStep 4: Collecting S values per inversion...")
    inv_by_chrom = defaultdict(list)
    inv_stats = []

    for idx, inv in enumerate(non_overlapping):
        inv_id = f"inv_{idx}"
        inv["id"] = inv_id
        size = inv["end"] - inv["start"]
        inv_stats.append(
            {
                "id": inv_id,
                "chrom": inv["chrom"],
                "size_bp": size,
                "num_recurrent": inv["num_recurrent"],
                "inverted_af": inv["inverted_af"],
                "s_values": [],
                "n_snps": 0,
            }
        )
        inv_by_chrom[inv["chrom"]].append((idx, inv))

    for chrom in inv_by_chrom:
        inv_by_chrom[chrom].sort(key=lambda x: x[1]["start"])

    print(f"  Indexed {len(non_overlapping)} non-overlapping inversions")
    print("  Scanning selection summary statistics...")

    if not SELECTION_TSV_PATH.exists():
        raise RuntimeError(
            f"Selection summary statistics file not found: {SELECTION_TSV_PATH}"
        )

    variants_processed = 0
    variants_matched = 0

    with SELECTION_TSV_PATH.open() as f:
        # Read header (skip metadata lines starting with "##")
        for line in f:
            if line.startswith("##") or not line.strip():
                continue
            header_fields = line.strip().split("\t")
            break
        else:
            raise RuntimeError("Could not locate header line in selection file")

        try:
            chrom_idx = header_fields.index("CHROM")
            pos_idx = header_fields.index("POS")
            s_idx = header_fields.index("S")
            filter_idx = header_fields.index("FILTER")
        except ValueError as e:
            raise RuntimeError(f"Required columns missing in selection file: {e}")

        for line in f:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            fields = line.split("\t")
            if len(fields) <= max(chrom_idx, pos_idx, s_idx, filter_idx):
                continue
            if fields[filter_idx] != "PASS":
                continue

            chrom = fields[chrom_idx]
            if chrom.startswith("chr"):
                chrom = chrom[3:]

            try:
                pos = int(fields[pos_idx])
                s_val = float(fields[s_idx])
            except ValueError:
                continue

            variants_processed += 1

            inv_list = inv_by_chrom.get(chrom)
            if not inv_list:
                continue

            # Inversions are non-overlapping and sorted; linear scan is acceptable
            for inv_idx, inv in inv_list:
                if inv["start"] <= pos <= inv["end"]:
                    inv_stats[inv_idx]["s_values"].append(s_val)
                    inv_stats[inv_idx]["n_snps"] += 1
                    variants_matched += 1
                    break

            if variants_processed % 1_000_000 == 0:
                print(
                    f"    Processed {variants_processed // 1_000_000}M variants, "
                    f"matched {variants_matched}..."
                )

    print(f"  Total variants processed: {variants_processed:,}")
    print(f"  Variants matched to inversions: {variants_matched:,}")

    # Step 5: Build per-inversion summary table (no hypothesis testing)
    print("\nStep 5: Preparing per-inversion summaries...")

    model_data = []
    for inv in inv_stats:
        if inv["n_snps"] > 0 and inv["s_values"]:
            mean_s = float(np.mean(inv["s_values"]))
            model_data.append(
                {
                    "id": inv["id"],
                    "chrom": inv["chrom"],
                    "size_bp": inv["size_bp"],
                    "n_snps": inv["n_snps"],
                    "num_recurrent": inv["num_recurrent"],
                    "inverted_af": inv["inverted_af"],
                    "mean_s": mean_s,
                }
            )

    df = pd.DataFrame(model_data)

    print(f"  Inversions with variant data: {len(df)}")

    if df.empty:
        print("✗ No inversions had mapped selection coefficients (S).")
        return

    print(f"  Mean of mean_S across inversions: {df['mean_s'].mean():.6f}")
    print(f"  Median of mean_S across inversions: {df['mean_s'].median():.6f}")

    # Log-scale covariates for descriptive convenience
    df["log_size"] = np.log(df["size_bp"])
    df["log_n_snps"] = np.log(df["n_snps"])

    # Save descriptive results
    output_file = OUTPUT_DIR / "selection_comparison_results.txt"
    with output_file.open("w") as out:
        out.write(
            "Analysis: Mean Selection Coefficient (S) vs Inversion Properties\n"
        )
        out.write("=" * 80 + "\n\n")
        out.write(
            "This file reports descriptive summaries of per-inversion mean "
            "selection coefficients and covariates.\n"
        )

        out.write("Data Summary:\n")
        out.write(
            f"  Non-overlapping inversions after liftover: "
            f"{len(non_overlapping)}\n"
        )
        out.write(
            f"  Inversions with ≥1 mapped variant (S): {len(df)}\n\n"
        )

        out.write("Mean S across inversions:\n")
        out.write(
            f"  Mean:   {df['mean_s'].mean():.6f}\n"
            f"  Median: {df['mean_s'].median():.6f}\n"
            f"  Min:    {df['mean_s'].min():.6f}\n"
            f"  Max:    {df['mean_s'].max():.6f}\n\n"
        )

        numeric_cols = [
            "mean_s",
            "num_recurrent",
            "inverted_af",
            "size_bp",
            "n_snps",
            "log_size",
            "log_n_snps",
        ]
        cols_present = [c for c in numeric_cols if c in df.columns]

        if len(cols_present) > 1:
            corr = df[cols_present].corr()
            out.write(
                "Pairwise Pearson correlations (descriptive only, no tests):\n"
            )
            out.write(corr.to_string(float_format=lambda v: f"{v: .3f}"))
            out.write("\n")

    print(f"\n  Descriptive results saved to: {output_file}")

    # Step 6: Simple visualization (no inferential overlay)
    print("\nStep 6: Creating visualization...")

    if "num_recurrent" in df.columns and df["num_recurrent"].notna().any():
        valid = df["num_recurrent"].notna()
        fig, ax = plt.subplots(figsize=(10, 7))

        sizes = 20 + 200 * (
            df.loc[valid, "n_snps"] / df.loc[valid, "n_snps"].max()
        )

        ax.scatter(
            df.loc[valid, "num_recurrent"],
            df.loc[valid, "mean_s"],
            s=sizes,
            alpha=0.4,
            edgecolors="black",
            linewidth=0.5,
            label="Inversions (point size ∝ n_snps)",
        )

        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Number of Recurrent Events", fontsize=13)
        ax.set_ylabel("Mean Selection Coefficient (S)", fontsize=13)
        ax.set_title(
            "Mean Selection Coefficient vs. Recurrence Count",
            fontsize=15,
            pad=20,
        )
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = OUTPUT_DIR / "recurrence_selection_scatter.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()

        print(f"  Plot saved to: {plot_file}")
    else:
        print("  Skipping plot: 'num_recurrent' not available or all missing.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if not INV_PROPERTIES_PATH.exists():
        print(f"✗ Error: Required file not found: {INV_PROPERTIES_PATH}")
        sys.exit(1)

    if not SELECTION_TSV_PATH.exists():
        print("Selection data not found locally. Downloading...")
        exit_code = main()
        if exit_code != 0:
            print("✗ Download failed")
            sys.exit(exit_code)
        print()
    
    # Ensure liftOver is installed before running analysis
    install_liftover()

    compare_selection_coefficients()
