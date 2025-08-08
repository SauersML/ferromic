import os
import sys
import requests
import subprocess

# The public URL to list of target SNPs.
TARGET_SNPS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/stats/all_unique_snps_sorted.txt"

# The GCS directory path for the All of Us ACAF Threshold PLINK dataset.
ACAF_PLINK_GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"

# The name of the output file that will be created in the same directory as the script.
OUTPUT_FILENAME = "found_snps_in_acaf.txt"


def fetch_target_snps(url):
    """
    Downloads the list of target SNPs from a URL and returns them as a Python set
    for fast lookups.
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


def get_bim_paths_with_gsutil(gcs_dir_path):
    """
    Uses 'gsutil' to find ALL .bim files in a GCS directory.
    This is robust for 'requester pays' buckets.
    """
    print("--- STEP 2: Locating All Remote .bim Files ---")
    print(f"Searching in GCS directory:\n  {gcs_dir_path}")
    project_id = os.getenv("GOOGLE_PROJECT")
    if not project_id:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    list_command = [
        "gsutil", "-u", project_id, "ls", os.path.join(gcs_dir_path, "*.bim"),
    ]

    print("\n--- DEBUG: Executing gsutil command to find all .bim files ---")
    print(f"  $ {' '.join(list_command)}")

    try:
        process = subprocess.run(
            list_command, capture_output=True, text=True, check=True
        )
        # Sort the file paths to process them in a consistent order (chr1, chr2, etc.)
        bim_files = sorted(process.stdout.strip().split("\n"))

        if not bim_files or not bim_files[0]:
            print(f"FATAL: 'gsutil' found no .bim files in {gcs_dir_path}", file=sys.stderr)
            sys.exit(1)

        print(f"\nFound {len(bim_files)} .bim files to process.\n")
        return bim_files

    except FileNotFoundError:
        print("FATAL: 'gsutil' command not found. Is it installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"FATAL: 'gsutil ls' command failed with error:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def stream_and_find_matches(bim_gcs_path, target_snps_set):
    """
    Streams a SINGLE .bim file from GCS using 'gsutil cat' and identifies matches.
    """
    project_id = os.getenv("GOOGLE_PROJECT")
    cat_command = ["gsutil", "-u", project_id, "cat", bim_gcs_path]

    found_snps_in_file = []
    lines_processed = 0

    try:
        process = subprocess.Popen(
            cat_command, stdout=subprocess.PIPE, text=True, errors="replace"
        )

        for line in process.stdout:
            lines_processed += 1
            if lines_processed % 1_000_000 == 0:
                print(f"\r  ...lines processed in this file: {lines_processed:,}", end="", flush=True)

            try:
                parts = line.split()
                # The first column ALREADY contains the 'chr' prefix (e.g., 'chr1')
                chromosome = parts[0]
                position = parts[3]
                
                # The first column is the full chromosome name, so we don't add "chr".
                current_snp_id = f"{chromosome}:{position}"

                if current_snp_id in target_snps_set:
                    found_snps_in_file.append(current_snp_id)
            except IndexError:
                print(f"\rWarning: Skipping malformed line #{lines_processed}: {line.strip()}", file=sys.stderr)

        process.wait()
        if process.returncode != 0:
            print(f"\nWarning: 'gsutil cat' for {bim_gcs_path} exited with status {process.returncode}", file=sys.stderr)

        print(f"\rFinished processing {os.path.basename(bim_gcs_path)}. Total lines: {lines_processed:,}. Matches found: {len(found_snps_in_file):,}.")
        return found_snps_in_file

    except FileNotFoundError:
        print("FATAL: 'gsutil' command not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while streaming {bim_gcs_path}: {e}", file=sys.stderr)
        return []


def main():
    """
    Main execution function.
    """
    target_snps = fetch_target_snps(TARGET_SNPS_URL)
    
    # --- FIX: Get the LIST of all .bim files ---
    all_bim_paths = get_bim_paths_with_gsutil(ACAF_PLINK_GCS_DIR)
    
    all_found_matches = []

    print("--- STEP 3: Iterating Through All .bim Files ---")
    
    # --- FIX: Loop through every .bim file found ---
    for i, bim_path in enumerate(all_bim_paths):
        print(f"\n--- Processing file {i+1}/{len(all_bim_paths)}: {bim_path} ---")
        matches_in_this_file = stream_and_find_matches(bim_path, target_snps)
        all_found_matches.extend(matches_in_this_file)

    print("\n--- STEP 4: Final Results Summary ---")
    print(f"Search complete. Found a total of {len(all_found_matches):,} matches for your {len(target_snps):,} target SNPs across all files.")

    with open(OUTPUT_FILENAME, "w") as f_out:
        for snp in sorted(all_found_matches):
            f_out.write(f"{snp}\n")
            
    print(f"\nA complete list of the matching SNPs has been saved to the local file:\n  ./{OUTPUT_FILENAME}\n")


if __name__ == "__main__":
    main()
