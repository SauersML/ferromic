import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import requests

# --- CONFIGURATION ---
TARGET_SNPS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/stats/all_unique_snps_sorted.txt"
ACAF_PLINK_GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
OUTPUT_FILENAME = "found_snps_in_acaf.txt"
MAX_WORKERS = os.cpu_count()

def fetch_target_snps(url):
    """Downloads the list of target SNPs into a set for fast lookups."""
    print("--- STEP 1: Fetching Target SNPs ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        snp_set = {line.strip() for line in response.text.splitlines() if line.strip()}
        print(f"Successfully loaded {len(snp_set):,} unique target SNPs into memory.\n")
        return snp_set
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not fetch SNP list from URL. Error: {e}", file=sys.stderr)
        sys.exit(1)

def get_bim_file_metadata(gcs_dir_path):
    """
    Gets the path and size of all .bim files using a robust two-step process.
    Returns a list of tuples: [(path, size_in_bytes), ...].
    """
    print("--- STEP 2: Locating Remote .bim Files and Their Sizes ---")
    project_id = os.getenv("GOOGLE_PROJECT")
    if not project_id:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        # Step 1: Get all file paths using the reliable 'gsutil ls' command.
        ls_command = ["gsutil", "-u", project_id, "ls", os.path.join(gcs_dir_path, "*.bim")]
        process_ls = subprocess.run(ls_command, capture_output=True, text=True, check=True)
        bim_files = sorted(process_ls.stdout.strip().split("\n"))

        if not bim_files or not bim_files[0]:
            print(f"FATAL: 'gsutil ls' found no .bim files in {gcs_dir_path}", file=sys.stderr)
            sys.exit(1)

        # Step 2: Get sizes for all found files in a single, efficient command.
        du_command = ["gsutil", "-u", project_id, "du", "-s"] + bim_files
        process_du = subprocess.run(du_command, capture_output=True, text=True, check=True)
        
        size_map = {}
        for line in process_du.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                size_bytes = int(parts[0])
                path = parts[-1]
                size_map[path] = size_bytes

        files_with_sizes = [(path, size_map.get(path, 0)) for path in bim_files]
        
        total_size_gb = sum(size for _, size in files_with_sizes) / (1024**3)
        print(f"Found {len(files_with_sizes)} .bim files to process, total size {total_size_gb:.2f} GiB.\n")
        return files_with_sizes

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FATAL: A gsutil command failed. Error: {e}", file=sys.stderr)
        sys.exit(1)

def stream_and_find_matches(bim_gcs_path, target_snps_set):
    """(Worker Function) Streams a single .bim file and returns a list of found SNPs."""
    project_id = os.getenv("GOOGLE_PROJECT")
    cat_command = ["gsutil", "-u", project_id, "cat", bim_gcs_path]
    found_snps_in_file = []
    
    try:
        process = subprocess.Popen(cat_command, stdout=subprocess.PIPE, text=True, errors="replace")
        for line in process.stdout:
            parts = line.strip().split()
            if len(parts) < 4: continue
            
            # .bim format: <chr> <snp_id> <cm> <pos> <a1> <a2>
            current_snp_id = f"{parts[0]}:{parts[3]}"
            if current_snp_id in target_snps_set:
                found_snps_in_file.append(current_snp_id)
        
        process.wait()
        return found_snps_in_file
    except Exception as e:
        # Let the main process handle and report the error
        raise e

def main():
    """Main execution function."""
    target_snps = fetch_target_snps(TARGET_SNPS_URL)
    files_to_process = get_bim_file_metadata(ACAF_PLINK_GCS_DIR)
    
    total_bytes_to_process = sum(size for _, size in files_to_process)
    unique_found_snps = set()

    print(f"--- STEP 3: Scanning All Files in Parallel (using up to {MAX_WORKERS} processes) ---")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_metadata = {
            executor.submit(stream_and_find_matches, path, target_snps): (path, size)
            for path, size in files_to_process
        }

        with tqdm(total=total_bytes_to_process, unit='B', unit_scale=True, desc="Scanning Dataset", unit_divisor=1024) as pbar:
            for future in as_completed(future_to_metadata):
                path, size = future_to_metadata[future]
                try:
                    matches_in_file = future.result()
                    unique_found_snps.update(matches_in_file)
                except Exception as exc:
                    pbar.write(f"ERROR processing {os.path.basename(path)}: {exc}", file=sys.stderr)
                finally:
                    pbar.update(size)

    print("\n--- STEP 4: Final Results Summary ---")
    
    num_targets_total = len(target_snps)
    num_targets_found = len(unique_found_snps)
    
    percentage_found = (num_targets_found / num_targets_total * 100) if num_targets_total > 0 else 0

    print("Scan complete.")
    print(f"Found {num_targets_found:,} of your {num_targets_total:,} target SNPs.")
    print(f"Match Percentage: {percentage_found:.2f}%")

    with open(OUTPUT_FILENAME, "w") as f_out:
        for snp in sorted(list(unique_found_snps)):
            f_out.write(f"{snp}\n")
            
    print(f"\nA list of the {num_targets_found:,} unique matching SNPs has been saved to: ./{OUTPUT_FILENAME}\n")

if __name__ == "__main__":
    main()
