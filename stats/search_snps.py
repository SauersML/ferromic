import os
import sys
import requests
import subprocess

# --- Configuration ---
TARGET_SNPS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/stats/all_unique_snps_sorted.txt"
ACAF_PLINK_GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"
OUTPUT_FILENAME = "found_snps_in_acaf.txt"


def fetch_target_snps(url):
    """
    Downloads the list of target SNPs from a URL and returns them as a Python set
    for fast lookups.
    """
    print(f"Fetching target SNP list from:\n  {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        snp_set = {line.strip() for line in response.text.splitlines() if line.strip()}
        print(f"Successfully loaded {len(snp_set):,} unique target SNPs into memory.\n")
        return snp_set
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not fetch SNP list from URL. Error: {e}", file=sys.stderr)
        sys.exit(1)


def get_bim_path_with_gsutil(gcs_dir_path):
    """
    Uses the command-line tool 'gsutil' to find the .bim file in a GCS directory.
    This is robust for 'requester pays' buckets.
    """
    print(f"Using 'gsutil' to find .bim file in:\n  {gcs_dir_path}")
    project_id = os.getenv("GOOGLE_PROJECT")
    if not project_id:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # Construct the command to list .bim files in the directory
    # The '*' is a wildcard that gsutil understands.
    list_command = [
        "gsutil",
        "-u",
        project_id,
        "ls",
        os.path.join(gcs_dir_path, "*.bim"),
    ]

    try:
        # Execute the command
        process = subprocess.run(
            list_command,
            capture_output=True,
            text=True,
            check=True,  # Raise an exception if gsutil returns a non-zero exit code
        )
        # The output will be a string of one or more file paths, separated by newlines
        bim_files = process.stdout.strip().split("\n")

        if not bim_files or not bim_files[0]:
            print(f"FATAL: 'gsutil' found no .bim files in {gcs_dir_path}", file=sys.stderr)
            sys.exit(1)

        if len(bim_files) > 1:
            print(f"Warning: Found multiple .bim files. Using the first one: {bim_files[0]}", file=sys.stderr)

        found_path = bim_files[0]
        print(f"Found .bim file via gsutil:\n  {found_path}\n")
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
    Streams the .bim file from GCS using 'gsutil cat' and identifies matches.
    """
    print("Starting to stream the .bim file via 'gsutil cat'...")
    project_id = os.getenv("GOOGLE_PROJECT")

    # Command to stream the file content to standard output
    cat_command = ["gsutil", "-u", project_id, "cat", bim_gcs_path]

    found_snps = []
    lines_processed = 0

    try:
        # Start the 'gsutil cat' process
        process = subprocess.Popen(
            cat_command,
            stdout=subprocess.PIPE,
            text=True,  # Decodes the output stream as text
            errors="replace", # Handle potential decoding errors gracefully
        )

        # Iterate directly over the output stream, line by line
        for line in process.stdout:
            lines_processed += 1
            if lines_processed % 1_000_000 == 0:
                print(f"\r  Lines processed: {lines_processed:,}", end="", flush=True)

            try:
                parts = line.split()
                chromosome = parts[0]
                position = parts[3]
                current_snp_id = f"chr{chromosome}:{position}"

                if current_snp_id in target_snps_set:
                    print(f"\r  MATCH FOUND: {current_snp_id} (line ~{lines_processed:,})")
                    found_snps.append(current_snp_id)
            except IndexError:
                print(f"\rWarning: Skipping malformed line #{lines_processed}: {line.strip()}", file=sys.stderr)

        # Wait for the process to finish and check its return code
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

    print("--- Results Summary ---")
    print(f"Found {len(found_matches)} of your {len(target_snps)} target SNPs in the ACAF dataset.")

    with open(OUTPUT_FILENAME, "w") as f_out:
        # Sort results for consistent output
        for snp in sorted(found_matches):
            f_out.write(f"{snp}\n")
            
    print(f"\nA complete list of the matching SNPs has been saved to:\n  ./{OUTPUT_FILENAME}\n")


if __name__ == "__main__":
    main()
