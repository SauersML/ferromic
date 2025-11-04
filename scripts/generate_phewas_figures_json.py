#!/usr/bin/env python3
"""
Generate the PheWAS Manhattan plot list for special-figures.json.

Uses the same logic as forest.py: include any inversion that has
at least one phenotype association with Q_GLOBAL <= 0.05.

This script does NOT require pandas - uses pure Python CSV reading.
"""

import os
import sys
import json
import csv

# Paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(REPO_ROOT, "data/phewas_results.tsv")
INV_PROPERTIES_FILE = os.path.join(REPO_ROOT, "data/inv_properties.tsv")
OUTPUT_JSON = os.path.join(REPO_ROOT, "web/figures-site/data/phewas-figures.json")

def load_inv_region_map(inv_properties_path):
    """Load mapping from OrigID to chr:start-end format from inv_properties.tsv."""
    if not os.path.exists(inv_properties_path):
        print(f"WARNING: {inv_properties_path} not found!")
        return {}

    mapping = {}
    with open(inv_properties_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Get OrigID (e.g., "chr10-79542902-INV-674513")
            orig_id = row.get('OrigID', '').strip()
            if not orig_id:
                continue

            # Get Chromosome, Start, End
            chrom = row.get('Chromosome', '').strip()
            if not chrom:
                continue
            # Ensure chr prefix
            if not chrom.lower().startswith('chr'):
                chrom = f'chr{chrom}'

            start_str = row.get('Start', '').strip().replace(',', '')
            end_str = row.get('End', '').strip().replace(',', '')

            try:
                start_int = int(start_str)
                end_int = int(end_str)
                # Format with commas for thousands: chr10:79,542,902-80,217,415
                mapping[orig_id] = f"{chrom}:{start_int:,}-{end_int:,}"
            except (ValueError, TypeError):
                continue

    return mapping

def map_inversion(value, mapping):
    """Map an inversion ID to its genomic region label."""
    if not value:
        return value
    key = str(value).strip()
    return mapping.get(key, key)

def inversion_to_filename(inv_label):
    """
    Convert 'chr6:76,109,081-76,158,474' to 'phewas_plots/phewas_chr6_76_109_081-76_158_474.pdf'
    """
    # Remove 'chr' prefix temporarily
    inv_label = inv_label.replace("chr", "", 1)

    # Split on colon
    parts = inv_label.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid inversion label format: {inv_label}")

    chrom = parts[0]
    range_part = parts[1]  # e.g., "76,109,081-76,158,474"

    # Replace commas with underscores
    range_part = range_part.replace(",", "_")

    return f"phewas_plots/phewas_chr{chrom}_{range_part}.pdf"

def calculate_inversion_size(inv_label):
    """Calculate inversion size in bp from 'chr6:76,109,081-76,158,474' format."""
    try:
        range_part = inv_label.split(":")[1]
        start_str, end_str = range_part.replace(",", "").split("-")
        start = int(start_str)
        end = int(end_str)
        return end - start
    except (IndexError, ValueError):
        return 0

def load_and_filter(path, inv_mapping):
    """
    Load phewas_results.tsv and filter using forest.py logic:
    - Valid OR > 0
    - Valid Q_GLOBAL
    - Q_GLOBAL <= 0.05

    Returns a set of inversions (in chr:start-end format) that have significant associations.
    """
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: '{path}' not found.")

    significant_inversions = set()

    with open(path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')

        # Check required columns
        required = ["Phenotype", "Inversion", "OR", "Q_GLOBAL"]
        if not all(col in reader.fieldnames for col in required):
            missing = [c for c in required if c not in reader.fieldnames]
            raise SystemExit(f"ERROR: missing required column(s): {', '.join(missing)}")

        for row in reader:
            # Skip if missing phenotype
            phenotype = row.get('Phenotype', '').strip()
            if not phenotype:
                continue

            # Parse and validate OR
            try:
                or_val = float(row.get('OR', '').strip())
                if or_val <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            # Parse and validate Q_GLOBAL
            try:
                q_global = float(row.get('Q_GLOBAL', '').strip())
            except (ValueError, TypeError):
                continue

            # Same threshold as forest.py: FDR-significant only
            if q_global <= 0.05:
                # Map inversion to chr:start-end format
                inv_id = row.get('Inversion', '').strip()
                inv_label = map_inversion(inv_id, inv_mapping)
                significant_inversions.add(inv_label)

    if not significant_inversions:
        raise SystemExit("No FDR-significant hits at q <= 0.05 in Q_GLOBAL.")

    return significant_inversions

def generate_figure_entries(inversions):
    """Generate JSON figure entries for each inversion with significant associations."""
    figures = []

    for inv in sorted(inversions):
        # Calculate size
        size_bp = calculate_inversion_size(inv)
        size_kb = size_bp // 1000

        # Generate filename
        filename = inversion_to_filename(inv)

        # Create entry
        entry = {
            "title": f"PheWAS Manhattan — {inv}",
            "filename": filename,
            "description": f"Phenome-wide association plot for the {size_kb} kbp inversion at {inv}."
        }
        figures.append(entry)

    return figures

def main():
    print(f"Loading inversion mapping from {INV_PROPERTIES_FILE}...")
    inv_mapping = load_inv_region_map(INV_PROPERTIES_FILE)
    print(f"  Loaded {len(inv_mapping)} inversion mappings")

    print(f"\nLoading and filtering data from {INPUT_FILE}...")
    significant_inversions = load_and_filter(INPUT_FILE, inv_mapping)

    print(f"\nFound {len(significant_inversions)} inversions with significant associations (q <= 0.05)")
    print("\nInversions to include:")
    for inv in sorted(significant_inversions):
        print(f"  - {inv}")

    print("\nGenerating figure entries...")
    figures = generate_figure_entries(significant_inversions)

    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_JSON)
    os.makedirs(output_dir, exist_ok=True)

    # Write output
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(figures, f, indent=2)

    print(f"\nWrote {len(figures)} figure entries to {OUTPUT_JSON}")
    print("\nThis file will be automatically loaded by the Next.js site during build.")

if __name__ == "__main__":
    main()
