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
    for fast lookups, with extensive debugging output.
    """
    print("--- STEP 1: Fetching Target SNPs ---")
    print(f"Fetching target SNP list from URL:\n  {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        lines = response.text.splitlines()
        
        print("\n--- DEBUG: First 5 raw lines from target SNP file ---")
        for i, line in enumerate(lines[:5]):
            print(f"  Line {i+1}: '{line.strip()}'")
            
        # Create a set of non-empty lines for O(1) average time complexity lookups.
        snp_set = {line.strip() for line in lines if line.strip()}
        
        print("\n--- DEBUG: First 5 parsed SNPs added to target set (sorted for display) ---")
        # Sort for consistent debug output
        for snp in sorted(list(snp_set))[:5]:
            print(f"  - '{snp}'")
        
        print(f"\nSuccessfully loaded {len(snp_set):,} unique target SNPs into memory.\n")
        return snp_set
        
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not fetch SNP list from URL. Error: {e}", file=sys.stderr)
        sys.exit(1)


def get_bim_path_with_gsutil(gcs_dir_path):
    """
    Uses the command-line tool 'gsutil' to find the .bim file in a GCS directory.
    This is robust for 'requester pays' buckets.
    """
    print("--- STEP 2: Locating Remote .bim File ---")
    print(f"Searching in GCS directory:\n  {gcs_dir_path}")
    project_id = os.getenv("GOOGLE_PROJECT")
    if not project_id:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Construct the command to list .bim files in the directory
    list_command = [
        "gsutil",
        "-u",
        project_id,
        "ls",
        os.path.join(gcs_dir_path, "*.bim"),
    ]

    print("\n--- DEBUG: Executing gsutil command to find .bim file ---")
    print(f"  $ {' '.join(list_command)}")

    try:
        process = subprocess.run(
            list_command,
            capture_output=True,
            text=True,
            check=True,
        )
        bim_files = process.stdout.strip().split("\n")

        if not bim_files or not bim_files[0]:
            print(f"FATAL: 'gsutil' found no .bim files in {gcs_dir_path}", file=sys.stderr)
            sys.exit(1)

        if len(bim_files) > 1:
            print(f"Warning: Found multiple .bim files. Using the first one: {bim_files[0]}", file=sys.stderr)

        found_path = bim_files[0]
        print(f"\nSuccessfully found .bim file via gsutil:\n  {found_path}\n")
        return found_path

    except FileNotFoundError:
        print("FATAL: 'gsutil' command not found. Is it installed and in your PATH?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"FATAL: 'gsutil ls' command failed with error:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def stream_and_find_matches(bim_gcs_path, target_snps_set):
    """
    Streams the .bim file from GCS using 'gsutil cat' and identifies matches,
    with extensive debugging output.
    """
    print("--- STEP 3: Streaming .bim File and Finding Matches ---")
    project_id = os.getenv("GOOGLE_PROJECT")

    # Command to stream the file content to standard output
    cat_command = ["gsutil", "-u", project_id, "cat", bim_gcs_path]
    
    print("\n--- DEBUG: Executing gsutil command to stream .bim content ---")
    print(f"  $ {' '.join(cat_command)}\n")

    found_snps = []
    lines_processed = 0

    try:
        process = subprocess.Popen(
            cat_command,
            stdout=subprocess.PIPE,
            text=True,
            errors="replace",
        )

        for line in process.stdout:
            lines_processed += 1
            
            # --- Detailed debug block for the first 5 lines ---
            if lines_processed <= 5:
                print(f"--- DEBUG: Processing .bim line {lines_processed} ---")
                print(f"  Raw line:       '{line.strip()}'")
                
            try:
                parts = line.split()
                chromosome = parts[0]
                position = parts[3]
                current_snp_id = f"chr{chromosome}:{position}"

                if lines_processed <= 5:
                    print(f"  Parsed CHR:     '{chromosome}'")
                    print(f"  Parsed POS:     '{position}'")
                    print(f"  Constructed ID: '{current_snp_id}'")

                if current_snp_id in target_snps_set:
                    # Print with newlines to ensure it's not overwritten by the progress counter
                    print(f"\n>>> MATCH FOUND! SNP: {current_snp_id} (on .bim line ~{lines_processed:,}) <<<\n")
                    found_snps.append(current_snp_id)
                
                # After the initial debug, switch to a quieter progress counter
                elif lines_processed > 5 and lines_processed % 500_000 == 0:
                    print(f"\r  Lines processed: {lines_processed:,}", end="", flush=True)

            except IndexError:
                print(f"\rWarning: Skipping malformed line #{lines_processed}: {line.strip()}", file=sys.stderr)

        process.wait()
        if process.returncode != 0:
            print(f"\nWarning: 'gsutil cat' exited with non-zero status {process.returncode}", file=sys.stderr)

        print(f"\rFinished processing. Total lines in .bim file: {lines_processed:,}\n")
        return found_snps

    except FileNotFoundError:
        print("FATAL: 'gsutil' command not found.", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main execution function.
    """
    target_snps = fetch_target_snps(TARGET_SNPS_URL)
    bim_path = get_bim_path_with_gsutil(ACAF_PLINK_GCS_DIR)
    found_matches = stream_and_find_matches(bim_path, target_snps)

    print("--- STEP 4: Final Results Summary ---")
    print(f"Found {len(found_matches)} of your {len(target_snps)} target SNPs in the ACAF dataset.")

    with open(OUTPUT_FILENAME, "w") as f_out:
        # Sort results for consistent output
        for snp in sorted(found_matches):
            f_out.write(f"{snp}\n")
            
    print(f"\nA complete list of the matching SNPs has been saved to the local file:\n  ./{OUTPUT_FILENAME}\n")


if __name__ == "__main__":
    main()
