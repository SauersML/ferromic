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
        list: A list of (taxon_name, sequence) tuples, or None if a fatal error occurs.
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
            pass # Not a header, process from the first line.

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
                return None # This is a fatal error for the file.

        elif group_type == 'outgroup':
            # Rule: Split on the first whitespace.
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                taxon_name, seq = parts
            else:
                print(f"  [!] FAILURE: In '{filename}' (line {i}), could not split line into taxon and sequence. Line: '{line[:80]}...'", file=sys.stderr)
                return None # This is a fatal error for the file.
        
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

def find_and_combine_phy_files():
    """
    Finds and processes trios of .phy files, providing detailed logs.
    It identifies file trios (group0, group1, outgroup) by a common identifier,
    parses them, validates sequence length consistency, and combines them into a
    single new .phy file with a correct header.
    """
    # Regex to extract key parts from filenames, handling the optional ENSG ID.
    file_pattern = re.compile(
        r"^(group0|group1|outgroup)_([A-Z0-9\._-]+?)_(?:ENSG[0-9\.]+_)?(ENST[0-9\.]+)_(chr.+)\.phy$"
    )

    file_groups = defaultdict(dict)
    
    # Group files by a common identifier derived from the filename
    for filename in os.listdir('.'):
        match = file_pattern.match(filename)
        if match:
            group_type, gene_name, enst_id, coords = match.groups()
            identifier = f"{gene_name}_{enst_id}_{coords}"
            file_groups[identifier][group_type] = filename
            
    if not file_groups:
        print("No files matching the required naming pattern (e.g., group0_...) were found.", file=sys.stderr)
        return

    print(f"Found {len(file_groups)} unique identifiers. Now checking for complete trios...")
    
    trios_processed_count = 0
    # Sort for deterministic order
    for identifier, files_dict in sorted(file_groups.items()):
        # A complete trio must have all three group types.
        if not ('group0' in files_dict and 'group1' in files_dict and 'outgroup' in files_dict):
            continue

        print(f"\n--- Checking Trio: {identifier} ---")
        is_valid_trio = True
        all_sequences_for_trio = []
        expected_dna_length = None

        # Step 1: Parse all three files and collect sequences.
        for group_type in ['group0', 'group1', 'outgroup']:
            filename = files_dict[group_type]
            print(f"  - Parsing '{filename}' with '{group_type}' rules...")
            sequences_from_file = parse_specific_phy_file(filename, group_type)
            
            if sequences_from_file is None:
                is_valid_trio = False
                break # A fatal error occurred during parsing.
            
            all_sequences_for_trio.extend(sequences_from_file)
        
        if not is_valid_trio:
            print(f"   Skipping trio for '{identifier}' due to parsing failure.")
            continue

        # Step 2: Validate the collected sequences for length and consistency.
        print("  - All files parsed. Validating sequence consistency...")
        if not all_sequences_for_trio:
            print(f"  [!] FAILURE: No sequences found for trio '{identifier}'. Skipping.", file=sys.stderr)
            continue
            
        for taxon, seq, _ in all_sequences_for_trio:
            # The first valid sequence sets the standard length.
            if expected_dna_length is None:
                if len(seq) % 3 != 0:
                    print(f"  [!] FAILURE: Taxon '{taxon}' has length {len(seq)}, which is not divisible by 3. Skipping trio.", file=sys.stderr)
                    is_valid_trio = False
                    break
                expected_dna_length = len(seq)
            # All subsequent sequences must match the standard length.
            elif len(seq) != expected_dna_length:
                print(f"  [!] FAILURE: Taxon '{taxon}' has length {len(seq)}, but expected {expected_dna_length}. Skipping trio.", file=sys.stderr)
                is_valid_trio = False
                break
        
        if not is_valid_trio:
            continue

        # Step 3: If all checks pass, write the combined file with the CORRECT header.
        print("  - Validation successful.")
        num_sequences = len(all_sequences_for_trio)
        
        # The alignment length in the header must match the actual length of the 
        # sequences being written. Since we are writing the full DNA sequences,
        # we must use `expected_dna_length`.
        alignment_length = expected_dna_length
        output_filename = f"combined_{identifier}.phy"
        
        try:
            with open(output_filename, 'w') as f_out:
                # Write the header with the number of sequences and the CORRECT sequence length.
                f_out.write(f"{num_sequences} {alignment_length}\n")
                
                # Iterate through all sequences, now including the group_type
                for taxon, seq, group_type in all_sequences_for_trio:
                    final_taxon_name = taxon
                    
                    # Prepend 0 for direct (group0) and 1 for inverted (group1)
                    if group_type == 'group0':
                        final_taxon_name = f"0{taxon}"
                    elif group_type == 'group1':
                        final_taxon_name = f"1{taxon}"
                    
                    # Write with the modified taxon name and two spaces for separation
                    f_out.write(f"{final_taxon_name}  {seq}\n")
                    
            print(f"  -> SUCCESS: Created '{output_filename}'")
            trios_processed_count += 1
        except IOError as e:
            print(f"  [!] FATAL: Could not write to output file '{output_filename}': {e}", file=sys.stderr)
    
    print("-" * 20)
    print(f"Operation complete. Successfully created {trios_processed_count} combined .phy files.")

if __name__ == "__main__":
    find_and_combine_phy_files()
