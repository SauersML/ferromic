import os
import re
from collections import defaultdict
import sys

def parse_specific_phy_file(filename, group_type):
    """
    Parses a PHYLIP-like file based on specific rules for different group types.

    Args:
        filename (str): The path to the file to parse.
        group_type (str): The type of file ('group0', 'group1', 'outgroup'),
                          which determines the parsing rules.

    Returns:
        list: A list of (taxon_name, sequence, group_type) tuples, or None if a fatal error occurs.
    """
    sequences = []
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except IOError as e:
        print(f"  [!] FATAL: Could not read file '{filename}': {e}", file=sys.stderr)
        return None

    if not lines:
        print(f"  [!] FAILURE: File '{filename}' is empty or contains only whitespace.", file=sys.stderr)
        return None

    # Determine if the first line is a PHYLIP header (e.g., "2 372") and should be skipped.
    first_line_parts = lines[0].split()
    start_index = 0
    if len(first_line_parts) == 2:
        try:
            int(first_line_parts[0])
            int(first_line_parts[1])
            start_index = 1  # It's a header, so we start processing from the next line.
        except ValueError:
            pass  # Not a header, process from the first line.

    # Apply parsing rules based on the file's group type
    for i, line in enumerate(lines[start_index:], start=start_index + 1):
        taxon_name, seq = None, None

        if group_type in ['group0', 'group1']:
            # Rule: Find the last _L or _R. Taxon is everything before and including it.
            # Sequence is everything after it.
            split_pos_L = line.rfind('_L')
            split_pos_R = line.rfind('_R')
            split_pos = max(split_pos_L, split_pos_R)

            if split_pos != -1:
                # The taxon name includes the _L or _R, so the split is after that.
                taxon_name = line[:split_pos + 2]
                seq = line[split_pos + 2:]
            else:
                print(f"  [!] FAILURE: In '{filename}' (line {i}), could not find '_L' or '_R' delimiter. Line: '{line[:80]}...'", file=sys.stderr)
                return None  # This is a fatal error for the file.

        elif group_type == 'outgroup':
            # Rule: Split on the first whitespace.
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                taxon_name, seq = parts
            else:
                print(f"  [!] FAILURE: In '{filename}' (line {i}), could not split line into taxon and sequence. Line: '{line[:80]}...'", file=sys.stderr)
                return None  # This is a fatal error for the file.

        if taxon_name and seq:
            # Clean any whitespace from within the sequence string itself
            cleaned_seq = ''.join(seq.split())
            # Add the group_type to the returned data structure
            sequences.append((taxon_name.strip(), cleaned_seq, group_type))
        else:
            # This case should not be reached with the logic above, but is a safeguard.
            print(f"  [!] FAILURE: In '{filename}' (line {i}), parsing logic failed unexpectedly.", file=sys.stderr)
            return None

    return sequences

def _validate_and_get_length(sequences, require_divisible_by_three=True, context=""):
    """
    Validate that all sequences are the same length and, optionally, divisible by 3.
    Returns the common length if valid, else None.
    """
    if not sequences:
        print(f"  [!] FAILURE: No sequences found for {context}.", file=sys.stderr)
        return None

    expected_length = None
    for taxon, seq, _ in sequences:
        if expected_length is None:
            if require_divisible_by_three and (len(seq) % 3 != 0):
                print(f"  [!] FAILURE: Taxon '{taxon}' has length {len(seq)}, which is not divisible by 3. Skipping.", file=sys.stderr)
                return None
            expected_length = len(seq)
        else:
            if len(seq) != expected_length:
                print(f"  [!] FAILURE: Taxon '{taxon}' has length {len(seq)}, but expected {expected_length}. Skipping.", file=sys.stderr)
                return None
    return expected_length

def _write_combined_output(output_filename, sequences, alignment_length):
    try:
        with open(output_filename, 'w') as f_out:
            # Write the header with the number of sequences and the CORRECT sequence length.
            f_out.write(f"{len(sequences)} {alignment_length}\n")

            # Iterate through all sequences, now including the group_type
            for taxon, seq, group_type in sequences:
                final_taxon_name = taxon

                # Prepend 0 for direct (group0) and 1 for inverted (group1)
                if group_type == 'group0':
                    final_taxon_name = f"0{taxon}"
                elif group_type == 'group1':
                    final_taxon_name = f"1{taxon}"

                # Write with the modified taxon name and two spaces for separation
                f_out.write(f"{final_taxon_name}  {seq}\n")

        print(f"  -> SUCCESS: Created '{output_filename}'")
        return True
    except IOError as e:
        print(f"  [!] FATAL: Could not write to output file '{output_filename}': {e}", file=sys.stderr)
        return False

def find_and_combine_phy_files():
    """
    Finds and processes .phy files for:
      1) Gene trios: (group0, group1, outgroup) -> combined_<gene>_<ENST>_<coords>.phy
      2) Overall region pairs: (inversion_group0, inversion_group1) -> combined_inversion_<chr>_start<start>_end<end>.phy

    For gene trios, the sequences must all have identical lengths and be divisible by 3.
    For region pairs, the sequences must all have identical lengths (no multiple-of-3 requirement).
    """
    # Regex to extract key parts from filenames, handling the optional ENSG ID. (Gene files)
    pattern_gene = re.compile(
        r"^(group0|group1|outgroup)_([A-Za-z0-9\._-]+?)_(?:ENSG[0-9\.]+_)?(ENST[0-9\.]+)_(chr.+)\.phy$"
    )

    # Regex for overall region (inversion) files: inversion_group{0|1}_{CHR}_start{S}_end{E}.phy
    pattern_region = re.compile(
        r"^inversion_(group0|group1)_([A-Za-z0-9]+)_start(\d+)_end(\d+)\.phy$"
    )

    gene_groups = defaultdict(dict)     # identifier -> {'group0': file, 'group1': file, 'outgroup': file}
    region_groups = defaultdict(dict)   # identifier -> {'group0': file, 'group1': file}

    # Discover files and group by identifier
    for filename in os.listdir('.'):
        m_gene = pattern_gene.match(filename)
        if m_gene:
            group_type, gene_name, enst_id, coords = m_gene.groups()
            gene_name_norm = gene_name.upper()  # normalize case so names match
            identifier = f"{gene_name_norm}_{enst_id}"  # drop coords from the join key
            gene_groups[identifier][group_type] = filename
            continue

        m_region = pattern_region.match(filename)
        if m_region:
            group_type, chrom, start, end = m_region.groups()
            identifier = f"inversion_{chrom}_start{start}_end{end}"
            region_groups[identifier][group_type] = filename
            continue

    if not gene_groups and not region_groups:
        print("No files matching gene (group0_|group1_|outgroup_) or region (inversion_group0_|inversion_group1_) patterns were found.", file=sys.stderr)
        return

    print(f"Found {len(gene_groups)} gene identifiers and {len(region_groups)} region identifiers.")

    # ---------- PRE-RUN SUMMARY (exact-match only) ----------
    gene_pair_ids = [i for i,d in gene_groups.items() if ('group0' in d and 'group1' in d)]
    gene_trio_ids = [i for i in gene_pair_ids if 'outgroup' in gene_groups[i]]

    gene_cached = sum(os.path.exists(f"combined_{i}.phy") for i in gene_trio_ids)
    gene_to_make = len(gene_trio_ids) - gene_cached

    region_pair_ids = [i for i,d in region_groups.items() if ('group0' in d and 'group1' in d)]
    region_cached = sum(os.path.exists(f"combined_{i}.phy") for i in region_pair_ids)
    region_to_make = len(region_pair_ids) - region_cached

    print("=== PRE-RUN SUMMARY ===")
    print(f"Genes: pairs={len(gene_pair_ids)}, trios_with_outgroup={len(gene_trio_ids)}, cached_skip={gene_cached}, will_make_now={gene_to_make}")
    print(f"Regions: pairs={len(region_pair_ids)}, cached_skip={region_cached}, will_make_now={region_to_make}")
    print("Now checking for complete sets (trios for genes, pairs for regions)...")

    trios_processed_count = 0
    pairs_processed_count = 0

    # ---- Process GENE TRIOS ----
    for identifier, files_dict in sorted(gene_groups.items()):
        if not ('group0' in files_dict and 'group1' in files_dict and 'outgroup' in files_dict):
            continue

        output_filename = f"combined_{identifier}.phy"
        if os.path.exists(output_filename):
            print(f"  - SKIP (cached): {output_filename}")
            continue

        print(f"\n--- Checking Gene Trio: {identifier} ---")
        is_valid = True
        collected = []

        for group_type in ['group0', 'group1', 'outgroup']:
            filename = files_dict[group_type]
            print(f"  - Parsing '{filename}' with '{group_type}' rules...")
            seqs = parse_specific_phy_file(filename, group_type)
            if seqs is None:
                is_valid = False
                break
            collected.extend(seqs)

        if not is_valid:
            print(f"   Skipping trio for '{identifier}' due to parsing failure.")
            continue

        print("  - All files parsed. Validating sequence consistency (and /3 for coding)...")
        alignment_length = _validate_and_get_length(collected, require_divisible_by_three=True, context=f"trio '{identifier}'")
        if alignment_length is None:
            continue

        if _write_combined_output(output_filename, collected, alignment_length):
            trios_processed_count += 1

    # ---- Process REGION PAIRS ----
    for identifier, files_dict in sorted(region_groups.items()):
        if not ('group0' in files_dict and 'group1' in files_dict):
            continue

        output_filename = f"combined_{identifier}.phy"
        if os.path.exists(output_filename):
            print(f"  - SKIP (cached): {output_filename}")
            continue

        print(f"\n--- Checking Region Pair: {identifier} ---")
        is_valid = True
        collected = []

        for group_type in ['group0', 'group1']:
            filename = files_dict[group_type]
            print(f"  - Parsing '{filename}' with '{group_type}' rules...")
            seqs = parse_specific_phy_file(filename, group_type)
            if seqs is None:
                is_valid = False
                break
            collected.extend(seqs)

        # Attempt to include a region outgroup when present using the required naming convention:
        # outgroup_inversion_{chrom_label}_start{start}_end{end}.phy
        if is_valid:
            m_id = re.match(r"^inversion_([A-Za-z0-9]+)_start(\d+)_end(\d+)$", identifier)
            if m_id:
                chrom_label, start, end = m_id.groups()
                outgroup_filename = f"outgroup_inversion_{chrom_label}_start{start}_end{end}.phy"
                if os.path.exists(outgroup_filename):
                    print(f"  - Parsing '{outgroup_filename}' with 'outgroup' rules...")
                    out_seqs = parse_specific_phy_file(outgroup_filename, 'outgroup')
                    if out_seqs is None:
                        is_valid = False
                    else:
                        collected.extend(out_seqs)

        if not is_valid:
            print(f"   Skipping region pair for '{identifier}' due to parsing failure.")
            continue

        print("  - All files parsed. Validating sequence consistency...")
        alignment_length = _validate_and_get_length(collected, require_divisible_by_three=False, context=f"region pair '{identifier}'")
        if alignment_length is None:
            continue

        if _write_combined_output(output_filename, collected, alignment_length):
            pairs_processed_count += 1

    total = trios_processed_count + pairs_processed_count
    print("-" * 20)
    print(f"Operation complete. Successfully created {trios_processed_count} combined gene .phy files and {pairs_processed_count} combined region .phy files (total {total}).")

if __name__ == "__main__":
    find_and_combine_phy_files()
