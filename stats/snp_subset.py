import os
import sys
import subprocess
import math
import requests
from google.cloud import storage
from tqdm import tqdm

# --- CONFIGURATION ---

# Input: list of SNPs to find.
TARGET_SNPS_URL = "https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/stats/all_unique_snps_sorted.txt"

# Input: The GCS directory containing the full PLINK fileset.
# This points to the All of Us Controlled Tier ACAF Threshold dataset.
ACAF_PLINK_GCS_DIR = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/acaf_threshold/plink_bed/"

# Output: The names for new, local, subsetted PLINK files.
OUTPUT_BIM_FILENAME = "subset.bim"
OUTPUT_FAM_FILENAME = "subset.fam"
OUTPUT_BED_FILENAME = "subset.bed"

def fetch_target_snps(url):
    """Downloads the list of target SNPs into a set for fast lookups."""
    print("--- STEP 1: Fetching Target SNPs ---")
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Create a set for O(1) average time complexity lookups.
        snp_set = {line.strip() for line in response.text.splitlines() if line.strip()}
        print(f"Successfully loaded {len(snp_set):,} unique target SNPs into memory.\n")
        return snp_set
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not fetch SNP list from {url}. Error: {e}", file=sys.stderr)
        sys.exit(1)

def process_fam_file(gcs_dir_path, output_filename):
    """
    Copies a single .fam file (they are all identical) and counts the number of samples.
    This count is CRITICAL for calculating byte offsets in the .bed file.
    """
    print(f"--- STEP 2: Processing Sample Information (.fam file) ---")
    project_id = os.getenv("GOOGLE_PROJECT")
    if not project_id:
        print("FATAL: GOOGLE_PROJECT environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        # List all .fam files and grab the first one.
        ls_command = ["gsutil", "-u", project_id, "ls", os.path.join(gcs_dir_path, "*.fam")]
        process_ls = subprocess.run(ls_command, capture_output=True, text=True, check=True)
        all_fam_files = process_ls.stdout.strip().split("\n")
        if not all_fam_files or not all_fam_files[0]:
            print(f"FATAL: No .fam files found in {gcs_dir_path}", file=sys.stderr)
            sys.exit(1)
        source_fam_path = all_fam_files[0]
        
        # Copy the single .fam file to our local directory.
        print(f"Copying sample file {os.path.basename(source_fam_path)} to ./{output_filename}")
        cp_command = ["gsutil", "-u", project_id, "cp", source_fam_path, output_filename]
        subprocess.run(cp_command, check=True, capture_output=True)

        # Count the number of lines (samples) in the new file.
        with open(output_filename, 'r') as f:
            num_samples = sum(1 for _ in f)

        print(f"Found {num_samples:,} samples in the dataset.")
        print(f"Created local file: ./{output_filename}\n")
        return num_samples

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FATAL: A gsutil command for the .fam file failed. Error: {e.stderr}", file=sys.stderr)
        sys.exit(1)

def process_bim_files(gcs_dir_path, target_snps_set):
    """
    Streams all .bim files, creates a local subset .bim file, and returns a
    map of the locations (file path and index) of the matched SNPs.
    """
    print(f"--- STEP 3: Identifying and Mapping Target SNPs (.bim files) ---")
    project_id = os.getenv("GOOGLE_PROJECT")
    snp_locations = {}
    
    # Get a list of all .bim files to process.
    ls_command = ["gsutil", "-u", project_id, "ls", os.path.join(gcs_dir_path, "*.bim")]
    process_ls = subprocess.run(ls_command, capture_output=True, text=True, check=True)
    bim_files = sorted(process_ls.stdout.strip().split("\n"))

    print(f"Streaming {len(bim_files)} .bim files to find matches...")
    with open(OUTPUT_BIM_FILENAME, "w") as f_out:
        for bim_gcs_path in tqdm(bim_files, desc="Scanning .bim files"):
            snp_index = 0
            found_in_file = []
            
            cat_command = ["gsutil", "-u", project_id, "cat", bim_gcs_path]
            process_cat = subprocess.Popen(cat_command, stdout=subprocess.PIPE, text=True, errors="replace")
            
            for line in process_cat.stdout:
                parts = line.strip().split('\t') # .bim is tab-separated
                if len(parts) >= 4:
                    # Construct SNP ID from bim format: <chr> <id> <pos> <bp> <a1> <a2>
                    current_snp_id = f"{parts[0]}:{parts[3]}"
                    if current_snp_id in target_snps_set:
                        f_out.write(line)
                        found_in_file.append(snp_index)
                snp_index += 1
            
            process_cat.wait()
            if found_in_file:
                snp_locations[bim_gcs_path] = found_in_file

    total_snps_found = sum(len(indices) for indices in snp_locations.values())
    print(f"Found a total of {total_snps_found:,} matching SNP records across all files.")
    print(f"Created local file: ./{OUTPUT_BIM_FILENAME}\n")
    return snp_locations

def process_bed_files(snp_locations, num_samples):
    """
    Performs targeted byte-range reads on remote .bed files based on the
    snp_locations map and assembles the local subset.bed file.
    """
    print(f"--- STEP 4: Assembling Subset Genotype Data (.bed file) ---")
    
    # Initialize the GCS client. It will use Application Default Credentials.
    storage_client = storage.Client()
    
    # In a SNP-major .bed file, each SNP's data is a fixed-size block.
    # The size is determined by the number of samples.
    bytes_per_snp = math.ceil(num_samples / 4)
    print(f"Calculated block size per SNP: {bytes_per_snp} bytes.")

    total_snps_to_write = sum(len(indices) for indices in snp_locations.values())

    with open(OUTPUT_BED_FILENAME, "wb") as f_out:
        # Write the mandatory 3-byte PLINK .bed header.
        # Magic numbers: 0x6c (108), 0x1b (27), 0x01 (1 = SNP-major)
        f_out.write(b'\x6c\x1b\x01')

        with tqdm(total=total_snps_to_write, desc="Extracting genotype data") as pbar:
            # Iterate through each file that had matches.
            for bim_gcs_path, indices in snp_locations.items():
                bed_gcs_path = bim_gcs_path.replace('.bim', '.bed')
                
                try:
                    # Get the blob object for the remote .bed file.
                    blob = storage.Blob.from_string(bed_gcs_path, client=storage_client)
                    
                    # For each matched SNP in this file, download its specific data block.
                    for snp_index in indices:
                        # Calculate the exact start position of the data block in the remote file.
                        # The first 3 bytes are the header.
                        offset = 3 + (snp_index * bytes_per_snp)
                        
                        # Download JUST this specific byte range into memory.
                        # The 'end' parameter is inclusive.
                        chunk = blob.download_as_bytes(start=offset, end=offset + bytes_per_snp - 1)
                        
                        # Write the downloaded chunk to our local subset.bed file.
                        f_out.write(chunk)
                        pbar.update(1)
                
                except Exception as e:
                    pbar.write(f"Warning: Could not process {bed_gcs_path}. Error: {e}", file=sys.stderr)

    print(f"Finished writing genotype data.")
    print(f"Created local file: ./{OUTPUT_BED_FILENAME}\n")

def main():
    """Main execution function to orchestrate the subsetting process."""
    print("Starting PLINK subset creation process...\n")
    
    target_snps = fetch_target_snps(TARGET_SNPS_URL)
    
    num_samples = process_fam_file(ACAF_PLINK_GCS_DIR, OUTPUT_FAM_FILENAME)
    
    snp_locations_map = process_bim_files(ACAF_PLINK_GCS_DIR, target_snps)
    
    if not snp_locations_map:
        print("--- FINAL RESULT ---")
        print("No matching SNPs were found in the dataset. No .bed file will be created.")
        # Clean up empty .bim file
        if os.path.exists(OUTPUT_BIM_FILENAME):
             if os.path.getsize(OUTPUT_BIM_FILENAME) == 0:
                  os.remove(OUTPUT_BIM_FILENAME)
        sys.exit(0)
        
    process_bed_files(snp_locations_map, num_samples)
    
    print("--- FINAL RESULT ---")
    print("Successfully created a subsetted PLINK fileset.")
    print(f" --> {OUTPUT_FAM_FILENAME}")
    print(f" --> {OUTPUT_BIM_FILENAME}")
    print(f" --> {OUTPUT_BED_FILENAME}")
    print("\nProcess complete.")

if __name__ == "__main__":
    main()
