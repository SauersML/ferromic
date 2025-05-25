import random
import sys
import collections

# hg38 chromosome lengths (1-based, inclusive)
HG38_CHROM_LENGTHS = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559,
    'chr4': 190214555, 'chr5': 181538259, 'chr6': 170805979,
    'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
    'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189,
    'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285,
    'chr19': 58617616, 'chr20': 64444167, 'chr21': 46709983,
    'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415,
}

def do_regions_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Checks if two 1-based, closed interval regions overlap."""
    return max(start1, start2) <= min(end1, end2)

def parse_and_validate_input(input_tsv_path: str) -> tuple[
    collections.defaultdict[str, list[tuple[int, int]]], # exclusion_map
    list[dict], # regions_to_process
    str # header_line (string, not including newline)
]:
    """
    Reads the input TSV. Validates all regions and builds an exclusion map
    from these original regions. Collects regions for permutation.
    Raises errors on invalid data or file issues.
    Input TSV: first 3 cols chr, start, end (1-based). Others carried over.
    """
    exclusion_map = collections.defaultdict(list)
    regions_to_process = []
    
    # If input_tsv_path doesn't exist, open() will raise FileNotFoundError.
    with open(input_tsv_path, 'r') as infile:
        header_line_content = infile.readline()
        if not header_line_content.strip():
            raise ValueError(f"Error: Input file '{input_tsv_path}' is empty or has no header line.")
        header = header_line_content.strip()

        for line_number, line_content in enumerate(infile, 2):
            line_strip = line_content.strip()
            if not line_strip: # Skip purely empty lines if any
                continue
            
            fields = line_strip.split('\t')

            if len(fields) < 3:
                raise ValueError(f"Error (L{line_number}): Insufficient columns ({len(fields)}). Expected >=3. Line: '{line_strip}'")

            seqnames = fields[0]
            try:
                original_start = int(fields[1])
                original_end = int(fields[2])
            except ValueError as e:
                raise ValueError(f"Error (L{line_number}): Non-integer start/end coordinates. Line: '{line_strip}'. Details: {e}")

            if seqnames not in HG38_CHROM_LENGTHS:
                raise ValueError(f"Error (L{line_number}): Chromosome '{seqnames}' unknown. Line: '{line_strip}'")

            chromosome_length = HG38_CHROM_LENGTHS[seqnames]
            
            if not (1 <= original_start <= original_end <= chromosome_length):
                raise ValueError(
                    f"Error (L{line_number}): Invalid coordinates {seqnames}:{original_start}-{original_end}. "
                    f"Must be 1 <= start <= end <= chromosome_length ({chromosome_length}). Line: '{line_strip}'"
                )
            
            # span_val is (length - 1). For a 1bp region (start=1, end=1), span_val is 0.
            # For a 2bp region (start=1, end=2), span_val is 1.
            span_val = original_end - original_start 

            exclusion_map[seqnames].append((original_start, original_end))
            
            regions_to_process.append({
                'fields': fields,
                'line_num': line_number,
                'seqnames': seqnames,
                'original_start': original_start,
                'original_end': original_end,
                'span_val': span_val, 
                'chromosome_length': chromosome_length
            })

    if not regions_to_process:
        raise ValueError(f"Error: No valid data lines found in '{input_tsv_path}' after the header.")

    for chrom_key in exclusion_map: # Sort for consistent behavior if needed later, minor impact.
        exclusion_map[chrom_key].sort()
        
    print(f"Parsed and validated {len(regions_to_process)} regions from '{input_tsv_path}'. These will also form the exclusion set.")
    return exclusion_map, regions_to_process, header


def permute_coordinates_with_self_exclusion(
    input_tsv_path: str,
    output_tsv_path: str = "permuted.tsv", # Fixed default output path
    max_retries_per_region: int = 1000
):
    """
    Permutes regions from input_tsv_path, ensuring permuted regions do not
    overlap with ANY original region from the same input_tsv_path.
    Carries over all columns. Crashes on failure to find a placement.
    """
    
    exclusion_map, regions_to_process, header = parse_and_validate_input(input_tsv_path)
    
    permuted_count = 0
    
    print(f"Starting permutation for {len(regions_to_process)} regions. Output: '{output_tsv_path}'.")
    
    with open(output_tsv_path, 'w') as outfile:
        outfile.write(header + '\n')

        for region_info in regions_to_process:
            fields = region_info['fields']
            line_num = region_info['line_num']
            seqnames = region_info['seqnames']
            span = region_info['span_val'] 
            chromosome_length = region_info['chromosome_length']

            # new_start is 1-based. max_possible_new_start allows new_end to reach chromosome_length.
            # new_end = new_start + span. So, new_start + span <= chromosome_length.
            # new_start <= chromosome_length - span.
            max_possible_new_start = chromosome_length - span
            
            if max_possible_new_start < 1:
                # This implies region is effectively the entire chromosome or too large to place.
                # Given prior validation (1 <= start <= end <= chrom_length),
                # this means start=1 and end=chromosome_length (span = chromosome_length - 1).
                # Such a region cannot be moved to a new location that doesn't overlap an original region (itself).
                raise RuntimeError(
                    f"Error (L{line_num}): Region {seqnames}:{region_info['original_start']}-{region_info['original_end']} "
                    f"(span {span}) effectively covers the entire usable chromosome length or cannot be placed given "
                    f"max_possible_new_start is {max_possible_new_start}. Non-overlapping permutation is impossible."
                )

            found_placement = False
            for _ in range(max_retries_per_region):
                new_start = random.randint(1, max_possible_new_start) 
                new_end = new_start + span 

                is_overlapping = False
                if seqnames in exclusion_map: # Should always be true if region_info exists
                    for ex_start, ex_end in exclusion_map[seqnames]:
                        if do_regions_overlap(new_start, new_end, ex_start, ex_end):
                            is_overlapping = True
                            break 
                
                if not is_overlapping:
                    permuted_fields = list(fields) 
                    permuted_fields[1] = str(new_start)
                    permuted_fields[2] = str(new_end)
                    outfile.write('\t'.join(permuted_fields) + '\n')
                    permuted_count += 1
                    found_placement = True
                    break 
            
            if not found_placement:
                raise RuntimeError(
                    f"Error (L{line_num}): Max retries ({max_retries_per_region}) exhausted for region "
                    f"{seqnames}:{region_info['original_start']}-{region_info['original_end']}. "
                    f"Could not find a random non-overlapping placement. CRASHING."
                )

    print(f"\nProcessing complete. Successfully permuted {permuted_count} lines to '{output_tsv_path}'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <input_tsv_file>")
        print("Example: python your_script_name.py original_inversions_with_ids.tsv")
        print("       (Output will be 'permuted.tsv' by default in the current directory)")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    
    print(f"Input TSV file: {input_filename}")
    
    permute_coordinates_with_self_exclusion(input_filename) # Output defaults to "permuted.tsv"
