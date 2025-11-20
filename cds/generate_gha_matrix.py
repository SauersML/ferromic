import os
import sys
import glob
import json
import re
import argparse

# Ensure we can import pipeline_lib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import pipeline_lib as lib
except ImportError:
    print("Error: Could not import pipeline_lib. Ensure it is in the same directory.", file=sys.stderr)
    sys.exit(1)

def scan_regions():
    """Scans for combined_inversion_*.phy files and extracts identifiers."""
    files = glob.glob('combined_inversion_*.phy')
    regions = []
    for f in files:
        try:
            info = lib.parse_region_filename(f)
            regions.append(info['label'])
        except Exception as e:
            print(f"Warning: Skipping file {f} due to parse error: {e}", file=sys.stderr)
    return sorted(list(set(regions)))

def scan_genes_and_batch(batch_size=4):
    """Scans for gene files, filters them, and groups them into batches."""
    glob_pattern = 'combined_*.phy'
    print(f"Scanning genes with glob: {glob_pattern}", file=sys.stderr)
    all_combined = glob.glob(glob_pattern)
    print(f"Total combined files found: {len(all_combined)}", file=sys.stderr)

    files = [f for f in all_combined if 'inversion' not in os.path.basename(f)]
    print(f"Gene files (excluding inversion): {len(files)}", file=sys.stderr)

    metadata = lib.load_gene_metadata()

    valid_genes = []
    for f in files:
        try:
            info = lib.parse_gene_filename(f, metadata)
            valid_genes.append(info['label'])
        except Exception as e:
            print(f"Warning: Skipping gene file {f} due to parse/metadata error: {e}", file=sys.stderr)

    valid_genes.sort()

    batches = []
    for i in range(0, len(valid_genes), batch_size):
        batch = valid_genes[i:i + batch_size]
        # Store batch as a comma-separated string for easy passing in GHA matrix
        batches.append(",".join(batch))

    return batches

def main():
    # Just to be safe, allow overriding batch size via env or args, but default is 4
    parser = argparse.ArgumentParser(description="Generate GHA Matrix JSON")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of genes per PAML job")
    args = parser.parse_args()

    print("Scanning regions...", file=sys.stderr)
    regions = scan_regions()
    print(f"Found {len(regions)} regions.", file=sys.stderr)

    print("Scanning genes...", file=sys.stderr)
    gene_batches = scan_genes_and_batch(batch_size=args.batch_size)
    print(f"Found {len(gene_batches) * args.batch_size} genes (approx) in {len(gene_batches)} batches.", file=sys.stderr)

    output = {
        "regions": regions,
        "gene_batches": gene_batches
    }

    # Output raw JSON to stdout so GHA can capture it
    print(json.dumps(output))

if __name__ == "__main__":
    main()
