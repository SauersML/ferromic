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

def normalize_region_override(override_str):
    """
    Parses and normalizes a region override string.
    Input format: chr22:100-200
    Returns: (label, chrom, start, end)
             label format: chr22_100_200
    """
    m = re.match(r"^(chr[0-9a-zA-Z]+):(\d+)-(\d+)$", override_str)
    if not m:
        raise ValueError(f"Invalid region override format: {override_str}. Expected chr:start-end")

    chrom, start, end = m.groups()
    start, end = int(start), int(end)
    if start > end:
        start, end = end, start

    label = f"{chrom}_{start}_{end}"
    return label, chrom, start, end

def scan_regions(region_override=None):
    """
    Scans for combined_inversion_*.phy files.
    If region_override is set, checks if that specific region file exists and returns only it.
    Otherwise, filters them to ensure they match an entry in data/inv_properties.tsv.
    """
    files = glob.glob('combined_inversion_*.phy')

    if region_override:
        target_label, t_chrom, t_start, t_end = normalize_region_override(region_override)
        print(f"Override active. Looking for region: {target_label}", file=sys.stderr)

        # Scan found files to match the requested label
        found = False
        for f in files:
            try:
                info = lib.parse_region_filename(f)
                if info['label'] == target_label:
                    found = True
                    break
            except Exception as e:
                pass

        if found:
            print(f"Found matching region file for {target_label}.", file=sys.stderr)
            return [target_label]
        else:
            print(f"Error: Override region file for {target_label} not found among {len(files)} candidates.", file=sys.stderr)
            # We error out as requested
            sys.exit(1)

    # Normal Logic
    regions = []
    # Convert pipeline_lib allowed list to a set for fast lookup
    # allowed entries are tuples: (chrom, start, end)
    allowed_set = set(lib.ALLOWED_REGIONS)

    print(f"Filtering found files against {len(allowed_set)} allowed regions from inv_properties.tsv", file=sys.stderr)

    for f in files:
        try:
            info = lib.parse_region_filename(f)
            # Create key from filename info to match whitelist format
            file_key = (info['chrom'], info['start'], info['end'])

            if file_key in allowed_set:
                regions.append(info['label'])
            else:
                pass

        except Exception as e:
            print(f"Warning: Skipping file {f} due to parse error: {e}", file=sys.stderr)

    # Sort and dedup
    final_regions = sorted(list(set(regions)))
    print(f"Total regions scheduled for analysis: {len(final_regions)}", file=sys.stderr)

    return final_regions

def scan_genes_and_batch(batch_size=4, region_override=None):
    """Scans for gene files, filters them, and groups them into batches."""
    glob_pattern = 'combined_*.phy'
    print(f"Scanning genes with glob: {glob_pattern}", file=sys.stderr)
    all_combined = glob.glob(glob_pattern)

    files = [f for f in all_combined if 'inversion' not in os.path.basename(f)]
    print(f"Gene files (excluding inversion): {len(files)}", file=sys.stderr)

    metadata = lib.load_gene_metadata()

    target_region_info = None
    if region_override:
        target_label, t_chrom, t_start, t_end = normalize_region_override(region_override)
        target_region_info = (t_chrom, t_start, t_end)
        print(f"Filtering genes to overlap with {target_label} ({t_chrom}:{t_start}-{t_end})", file=sys.stderr)

    valid_genes = []
    for f in files:
        try:
            info = lib.parse_gene_filename(f, metadata)

            if target_region_info:
                # Check overlap
                t_chrom, t_start, t_end = target_region_info
                if info['chrom'] == t_chrom:
                    # Overlap check: not (End < Start OR Start > End)
                    if not (info['end'] < t_start or info['start'] > t_end):
                         valid_genes.append(info['label'])
            else:
                valid_genes.append(info['label'])

        except Exception as e:
            print(f"Warning: Skipping gene file {f} due to parse/metadata error: {e}", file=sys.stderr)

    valid_genes.sort()
    print(f"Valid genes selected: {len(valid_genes)}", file=sys.stderr)

    batches = []
    for i in range(0, len(valid_genes), batch_size):
        batch = valid_genes[i:i + batch_size]
        batches.append(",".join(batch))

    return batches

def main():
    parser = argparse.ArgumentParser(description="Generate GHA Matrix JSON")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of genes per PAML job")
    parser.add_argument("--region-override", type=str, help="Specific region override (e.g., chr22:100-200)")
    args = parser.parse_args()

    print("Scanning regions...", file=sys.stderr)
    regions = scan_regions(region_override=args.region_override)
    print(f"Found {len(regions)} regions.", file=sys.stderr)

    print("Scanning genes...", file=sys.stderr)
    gene_batches = scan_genes_and_batch(batch_size=args.batch_size, region_override=args.region_override)
    print(f"Found {len(gene_batches) * args.batch_size} genes (approx) in {len(gene_batches)} batches.", file=sys.stderr)

    output = {
        "regions": regions,
        "gene_batches": gene_batches
    }

    print(json.dumps(output))

if __name__ == "__main__":
    main()
