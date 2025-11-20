import random
import sys
import collections
import os
import glob

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
    collections.defaultdict[str, list[tuple[int, int]]],
    list[dict],
    str
]:
    """
    Reads input TSV, validates regions, builds exclusion map from these regions.
    Returns: exclusion_map, list of regions_to_process, header_line string.
    Raises errors on invalid data/file issues.
    Input TSV: first 3 cols chr, start, end (1-based). Others carried over.
    """
    exclusion_map = collections.defaultdict(list)
    regions_to_process = []
    
    with open(input_tsv_path, 'r') as infile:
        header_line_content = infile.readline()
        if not header_line_content.strip():
            raise ValueError(f"Error: Input file '{input_tsv_path}' is empty or lacks a header.")
        header = header_line_content.strip()

        for line_number, line_content in enumerate(infile, 2):
            line_strip = line_content.strip()
            if not line_strip:
                continue 
            
            fields = line_strip.split('\t')

            if len(fields) < 3:
                raise ValueError(f"Error (L{line_number}): Insufficient columns ({len(fields)}). Line: '{line_strip}'")

            seqnames = fields[0]
            try:
                original_start = int(fields[1])
                original_end = int(fields[2])
            except ValueError as e:
                raise ValueError(f"Error (L{line_number}): Non-integer coordinates. Line: '{line_strip}'. {e}")

            if seqnames not in HG38_CHROM_LENGTHS:
                raise ValueError(f"Error (L{line_number}): Chromosome '{seqnames}' unknown. Line: '{line_strip}'")

            chromosome_length = HG38_CHROM_LENGTHS[seqnames]
            
            if not (1 <= original_start <= original_end <= chromosome_length):
                raise ValueError(
                    f"Error (L{line_number}): Invalid coords {seqnames}:{original_start}-{original_end}. "
                    f"Constraint: 1 <= start <= end <= chrom_len ({chromosome_length}). Line: '{line_strip}'"
                )
            
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
        raise ValueError(f"Error: No valid data lines found in '{input_tsv_path}' post-header.")

    for chrom_key in exclusion_map:
        exclusion_map[chrom_key].sort()
        
    print(f"Validated {len(regions_to_process)} regions from '{input_tsv_path}' for permutation and exclusion.")
    return exclusion_map, regions_to_process, header


def permute_coordinates_with_self_exclusion(
    input_tsv_path: str,
    output_tsv_path: str = "permuted.tsv",
    mapping_tsv_path: str = "map.tsv", # New parameter for the mapping file
    max_retries_per_region: int = 1000
):
    """
    Permutes regions from input_tsv_path, excluding overlaps with ANY original region
    from the same input file. Carries over all columns. Crashes on placement failure.
    Outputs a permuted TSV and a mapping TSV.
    """
    
    exclusion_map, regions_to_process, header = parse_and_validate_input(input_tsv_path)
    
    permuted_count = 0
    coordinate_mappings = [] # To store original and new coordinate pairs

    print(f"Starting permutation for {len(regions_to_process)} regions. Output: '{output_tsv_path}'. Mapping: '{mapping_tsv_path}'.")
    
    with open(output_tsv_path, 'w') as outfile:
        outfile.write(header + '\n')

        for region_info in regions_to_process:
            fields = region_info['fields']
            line_num = region_info['line_num']
            seqnames = region_info['seqnames']
            span = region_info['span_val'] 
            chromosome_length = region_info['chromosome_length']
            
            original_start_val = region_info['original_start']
            original_end_val = region_info['original_end']

            max_possible_new_start = chromosome_length - span
            
            if max_possible_new_start < 1:
                raise RuntimeError(
                    f"Error (L{line_num}): Region {seqnames}:{original_start_val}-{original_end_val} "
                    f"(span {span}) cannot be placed. Max new start {max_possible_new_start}. "
                    f"Likely too large or covers entire chromosome, making non-overlapping permutation impossible."
                )

            found_placement = False
            for _ in range(max_retries_per_region):
                new_start = random.randint(1, max_possible_new_start) 
                new_end = new_start + span 

                is_overlapping = False
                if seqnames in exclusion_map:
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
                    
                    # Store mapping information
                    coordinate_mappings.append({
                        'original_chr': seqnames,
                        'original_start': original_start_val,
                        'original_end': original_end_val,
                        'new_chr': seqnames,  # Chromosome remains the same
                        'new_start': new_start,
                        'new_end': new_end
                    })
                    found_placement = True
                    break 
            
            if not found_placement:
                raise RuntimeError(
                    f"Error (L{line_num}): Max retries ({max_retries_per_region}) for region "
                    f"{seqnames}:{original_start_val}-{original_end_val}. "
                    f"Could not find non-overlapping placement. CRASHING."
                )

    print(f"\nProcessing complete. Successfully permuted {permuted_count} lines to '{output_tsv_path}'.")

    # Write the mapping file
    with open(mapping_tsv_path, 'w') as mapfile:
        mapfile.write("Original_Chr\tOriginal_Start\tOriginal_End\tNew_Chr\tNew_Start\tNew_End\n")
        for mapping_entry in coordinate_mappings:
            mapfile.write(
                f"{mapping_entry['original_chr']}\t{mapping_entry['original_start']}\t{mapping_entry['original_end']}\t"
                f"{mapping_entry['new_chr']}\t{mapping_entry['new_start']}\t{mapping_entry['new_end']}\n"
            )
    print(f"Coordinate mapping written to '{mapping_tsv_path}'.")


if __name__ == "__main__":
    input_filename = None
    # os.getcwd() is not strictly needed if glob is called without a path, it defaults to cwd.
    tsv_files_in_cwd = glob.glob("*.tsv") 

    if len(sys.argv) == 2: # User explicitly provided an input file
        input_filename = sys.argv[1]
        print(f"Using explicitly provided input TSV file: {input_filename}")
    elif len(sys.argv) == 1: # No arguments provided, try auto-detection
        if len(tsv_files_in_cwd) == 1:
            input_filename = tsv_files_in_cwd[0]
            print(f"Automatically using the single TSV file found in current directory: {input_filename}")
        elif len(tsv_files_in_cwd) == 0:
            print("Error: No .tsv files found in the current directory and no input file provided.")
            print("Usage: python script_name.py [<input_tsv_file>]") # Assuming script_name.py
            print("       If <input_tsv_file> is omitted, and only one .tsv file exists here, it's used automatically.")
            sys.exit(1)
        else: # More than one .tsv file found
            print("Error: Multiple .tsv files found in the current directory. Please specify which one to use.")
            print("Found: " + ", ".join(tsv_files_in_cwd))
            print("Usage: python script_name.py <input_tsv_file>") # Assuming script_name.py
            sys.exit(1)
    else: # Incorrect number of arguments
        print("Usage: python script_name.py [<input_tsv_file>]") # Assuming script_name.py
        print("       Provide zero arguments to auto-detect a single .tsv file, or one argument for the input file path.")
        sys.exit(1)
    
    output_file = "permuted.tsv" # Default output name for permuted regions
    mapping_file = "map.tsv"     # Default output name for the coordinate map
    
    print(f"Permuted output will be written to '{output_file}' in the current directory.")
    print(f"Coordinate mapping will be written to '{mapping_file}' in the current directory.")
    
    permute_coordinates_with_self_exclusion(
        input_filename, 
        output_tsv_path=output_file,
        mapping_tsv_path=mapping_file # Pass the mapping file name
    )
