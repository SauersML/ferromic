import json
from pathlib import Path

# --- Configuration ---
INPUT_DIR = "final_imputation_models"
OUTPUT_FILE = "all_unique_snps_sorted.txt"
# ---------------------

def natural_sort_key(snp_id: str) -> tuple:
    """A simplified key for sorting SNP IDs like 'chr10:123' correctly."""
    chrom_part, pos_part = snp_id.split(':')
    chrom_val = chrom_part[3:] # Removes 'chr' prefix
    
    if chrom_val == 'X':
        chrom_num = 23
    elif chrom_val == 'Y':
        chrom_num = 24
    else:
        chrom_num = int(chrom_val)
        
    return (chrom_num, int(pos_part))

# 1. Find all files and extract SNP IDs into a set for uniqueness.
all_snp_ids = set()
for file_path in Path(INPUT_DIR).glob("*.snps.json"):
    with file_path.open('r') as f:
        data = json.load(f)
        for record in data:
            all_snp_ids.add(record['id'])

# 2. Sort the unique IDs using the custom sort key.
sorted_snps = sorted(list(all_snp_ids), key=natural_sort_key)

# 3. Write the sorted list to the output file.
with open(OUTPUT_FILE, 'w') as f_out:
    for snp_id in sorted_snps:
        f_out.write(f"{snp_id}\n")

print(f"Process complete. {len(sorted_snps)} unique SNPs saved to {OUTPUT_FILE}.")
