import random
import sys

# Hardcoded hg38 chromosome lengths
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

def permute_genomic_coordinates(input_tsv_path, output_tsv_path="permuted.tsv"):
    """
    Reads a TSV file, permutes start and end coordinates, and writes to a new TSV.

    The permutation keeps the chromosome and the distance (end - start) constant.
    New start/end coordinates are chosen randomly within the valid bounds of the
    same chromosome.
    """
    processed_lines = 0
    warning_lines = 0

    try:
        with open(input_tsv_path, 'r') as infile, \
             open(output_tsv_path, 'w') as outfile:

            header_line = infile.readline()
            if not header_line:
                print(f"Error: Input file '{input_tsv_path}' appears to be empty.")
                return
            outfile.write(header_line) # Write header as is (includes newline)

            for line_number, line_content in enumerate(infile, 2): # Starts from file line 2
                original_line_for_output = line_content # Preserve original line ending
                fields = line_content.strip().split('\t')

                if len(fields) < 3:
                    # print(f"Warning (Line {line_number}): Insufficient columns ({len(fields)}). Original: '{line_content.strip()}'. Writing as is.")
                    outfile.write(original_line_for_output)
                    warning_lines += 1
                    continue

                seqnames = fields[0]
                original_start_str = fields[1]
                original_end_str = fields[2]

                try:
                    original_start = int(original_start_str)
                    original_end = int(original_end_str)
                except ValueError:
                    # print(f"Warning (Line {line_number}): Non-integer start/end. Original: '{line_content.strip()}'. Writing as is.")
                    outfile.write(original_line_for_output)
                    warning_lines += 1
                    continue

                if seqnames not in HG38_CHROM_LENGTHS:
                    # print(f"Warning (Line {line_number}): Chromosome '{seqnames}' not in hardcoded lengths. Original: '{line_content.strip()}'. Writing as is.")
                    outfile.write(original_line_for_output)
                    warning_lines += 1
                    continue

                chromosome_length = HG38_CHROM_LENGTHS[seqnames]
                
                # Calculate the span or "distance" to be preserved
                # span = original_end - original_start
                # This interpretation ensures new_end = new_start + span
                span = original_end - original_start

                if span < 0:
                    # print(f"Warning (Line {line_number}): End coordinate ({original_end}) is less than start ({original_start}). Original: '{line_content.strip()}'. Writing as is.")
                    outfile.write(original_line_for_output)
                    warning_lines += 1
                    continue
                
                # Determine the valid range for the new start position
                # new_start must be >= 1
                # new_start + span must be <= chromosome_length
                # So, new_start <= chromosome_length - span
                max_possible_new_start = chromosome_length - span

                if max_possible_new_start < 1:
                    # This occurs if span >= chromosome_length
                    if span == chromosome_length: # Segment is exactly chromosome length
                        new_start = 1
                    else: # span > chromosome_length, segment is too large
                        # print(f"Warning (Line {line_number}): Segment span ({span}) for {seqnames} ({original_start}-{original_end}) "
                        #       f"is greater than chromosome length ({chromosome_length}). Original: '{line_content.strip()}'. Writing as is.")
                        outfile.write(original_line_for_output)
                        warning_lines += 1
                        continue
                else:
                    new_start = random.randint(1, max_possible_new_start)
                
                new_end = new_start + span

                fields[1] = str(new_start)
                fields[2] = str(new_end)
                
                outfile.write('\t'.join(fields) + '\n')
                processed_lines += 1

        print(f"\nProcessing complete.")
        print(f"Successfully processed and permuted {processed_lines} data lines.")
        if warning_lines > 0:
            print(f"Encountered {warning_lines} lines with warnings (written as is to output). Check console for details if warnings were un-commented.")
        print(f"Output written to '{output_tsv_path}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_tsv_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <input_tsv_file>")
        print("Example: python permute_coords.py variants_input.tsv")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    output_filename = "permuted.tsv" 
    
    print(f"Input file: {input_filename}")
    print(f"Output file: {output_filename}")
    
    permute_genomic_coordinates(input_filename, output_filename)
