import os
import sys
import re
import requests
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---

# The public URL to list of target SNPs.
TARGET_SNPS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/stats/all_unique_snps_sorted.txt"

# The GCS directory path for the All of Us ACAF Threshold PLINK dataset.
ACAF_PLINK_GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"

# The name of the output file for found SNPs.
OUTPUT_FILENAME = "found_snps_in_acaf.txt"

# Number of parallel processes to run. Defaults to the number of CPU cores.
MAX_WORKERS = os.cpu_count()


def fetch_target_snps(url):
    """
    Downloads the list of target SNPs from a URL and returns them as a Python set
    for extremely fast lookups.
    """
    print("--- STEP 1: Fetching Target SNPs ---")
    print(f"Fetching target SNP list from URL:\n  {url}")
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
    Uses 'gsutil ls -L' to get the path and size of all .bim files. This is
    more efficient than separate 'ls' and 'du' commands.
    Returns a list of tuples: [(path, size_in_bytes), ...].
    """
    print("--- STEP 2: Locating Remote .bim Files and Their Sizes ---")
    print(f"Searching in GCS directory:\n  {gcs_dir_path}")
    project_id = os.getenv("GOOGLE_PROJECT")
    if not project_id:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # 'gsutil ls -L' provides detailed file info, including size.
    list_command = ["gsutil", "-u", project_id, "ls", "-L", os.path.join(gcs_dir_path, "*.bim")]
    
    print(f"Executing gsutil command: {' '.join(list_command)}")

    try:
        process = subprocess.run(list_command, capture_output=True, text=True, check=True)
        files_with_sizes = []
        
        # Regex to robustly parse the 'gsutil ls -L' output for file size and path.
        # It looks for a number (the size) followed later by a gs:// path on the same line.
        for line in process.stdout.strip().split('\n'):
            match = re.match(r'^\s*(\d+)\s+.*?\s+(gs://.*)$', line)
            if match:
                size_bytes = int(match.group(1))
                path = match.group(2).strip()
                files_with_sizes.append((path, size_bytes))

        if not files_with_sizes:
            print(f"FATAL: 'gsutil' found no .bim files in {gcs_dir_path}", file=sys.stderr)
            sys.exit(1)
        
        # Sort files for consistent processing order (e.g., chr1, chr2, ...)
        files_with_sizes.sort(key=lambda x: x[0])
        
        total_size_gb = sum(size for _, size in files_with_sizes) / (1024**3)
        print(f"\nFound {len(files_with_sizes)} .bim files to process, with a total size of {total_size_gb:.2f} GiB.\n")
        return files_with_sizes

    except FileNotFoundError:
        print("FATAL: 'gsutil' command not found. Is it installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("FATAL: 'gsutil ls -L' command failed with error:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def stream_and_find_matches(bim_gcs_path, target_snps_set):
    """
    (This function is run in a separate process)
    Streams a SINGLE .bim file from GCS using 'gsutil cat', finds all matches
    against the target_snps_set, and returns them as a list.
    """
    project_id = os.getenv("GOOGLE_PROJECT") # Must re-get env var in new process
    cat_command = ["gsutil", "-u", project_id, "cat", bim_gcs_path]
    
    found_snps_in_file = []
    
    try:
        process = subprocess.Popen(cat_command, stdout=subprocess.PIPE, text=True, errors="replace")
        for line in process.stdout:
            try:
                # .bim format: <chr> <snp_id> <cm> <pos> <a1> <a2>
                parts = line.strip().split()
                if len(parts) < 4: continue
                
                chromosome = parts[0]
                position = parts[3]
                current_snp_id = f"{chromosome}:{position}"

                if current_snp_id in target_snps_set:
                    found_snps_in_file.append(current_snp_id)
            except IndexError:
                # This can happen on malformed lines. Silently skip.
                continue
        
        process.wait()
        if process.returncode != 0:
            # Errors from gsutil will be printed to stderr of the main process
            pass
            
        return found_snps_in_file
    except Exception as e:
        # Propagate exceptions to be handled by the main process
        raise e


def main():
    """
    Main execution function to orchestrate fetching, parallel scanning, and reporting.
    """
    target_snps = fetch_target_snps(TARGET_SNPS_URL)
    files_to_process = get_bim_file_metadata(ACAF_PLINK_GCS_DIR)
    
    total_bytes_to_process = sum(size for _, size in files_to_process)
    unique_found_snps = set()

    print(f"--- STEP 3: Scanning All Files in Parallel (using up to {MAX_WORKERS} processes) ---")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a dictionary to map a running process (future) to its file info
        future_to_metadata = {
            executor.submit(stream_and_find_matches, path, target_snps): (path, size)
            for path, size in files_to_process
        }

        # Setup the TQDM progress bar
        pbar = tqdm(total=total_bytes_to_process, unit='B', unit_scale=True, desc="Scanning dataset", unit_divisor=1024)
        
        for future in as_completed(future_to_metadata):
            path, size = future_to_metadata[future]
            try:
                # Get the list of matches found in the completed file
                matches_in_file = future.result()
                # Add these matches to our set of unique found SNPs
                unique_found_snps.update(matches_in_file)
            except Exception as exc:
                print(f"\nERROR: File '{os.path.basename(path)}' generated an exception: {exc}", file=sys.stderr)
            finally:
                # Update the progress bar by the size of the file that just finished
                pbar.update(size)
        pbar.close()

    print("\n--- STEP 4: Final Results Summary ---")
    
    num_targets_total = len(target_snps)
    num_targets_found = len(unique_found_snps)
    
    if num_targets_total > 0:
        percentage_found = (num_targets_found / num_targets_total) * 100
    else:
        percentage_found = 0.0

    print(f"Scan complete.")
    print(f"Found {num_targets_found:,} of your {num_targets_total:,} target SNPs.")
    print(f"Match Percentage: {percentage_found:.2f}%")

    # Write the unique, sorted list of found SNPs to the output file.
    with open(OUTPUT_FILENAME, "w") as f_out:
        for snp in sorted(list(unique_found_snps)):
            f_out.write(f"{snp}\n")
            
    print(f"\nA complete list of the {num_targets_found:,} unique matching SNPs has been saved to:\n  ./{OUTPUT_FILENAME}\n")

if __name__ == "__main__":
    main()
