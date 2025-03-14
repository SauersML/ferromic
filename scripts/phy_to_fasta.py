"""
Convert PHY files to FASTA format.

This script processes all PHY files in the current directory (or a specified directory)
and converts them to FASTA format with the same base name but .fa extension.
Optionally translates DNA sequences to protein sequences.
"""

import os
import sys
import re
import argparse
from Bio.Seq import Seq
from Bio.SeqUtils import six_frame_translations

def translate_dna_to_protein(dna_sequence):
    """
    Translate a DNA sequence to a protein sequence using the standard genetic code
    
    Parameters:
    dna_sequence (str): DNA sequence to translate
    
    Returns:
    str: Protein sequence
    """
    try:
        # Create a Seq object and translate it
        seq_obj = Seq(dna_sequence)
        # Translate using standard genetic code, stop at first stop codon
        protein_seq = str(seq_obj.translate())
        return protein_seq
    except Exception as e:
        print(f"Error translating DNA sequence: {e}")
        return dna_sequence

def convert_phy_to_fasta(phy_file_path, pattern=None, wrap_length=60, translate_to_protein=False):
    """
    Convert a PHY file to FASTA format
    
    Parameters:
    phy_file_path (str): Path to the PHY file
    pattern (str, optional): Regular expression pattern to identify the ID/sequence boundary
    wrap_length (int, optional): Number of characters per line for the sequence
    translate_to_protein (bool, optional): Whether to translate DNA sequences to protein
    
    Returns:
    bool: True if conversion was successful, False otherwise
    """
    # Get the output file path
    if translate_to_protein:
        fa_file_path = os.path.splitext(phy_file_path)[0] + '_protein.fa'
    else:
        fa_file_path = os.path.splitext(phy_file_path)[0] + '.fa'
    
    # Use default pattern if not provided
    if pattern is None:
        pattern = r'(_[LR])([ACGTN])'
    
    try:
        # Open the input file for reading
        with open(phy_file_path, 'r') as phy_file:
            # Read the first line (metadata)
            metadata_line = next(phy_file, "").strip()
            if not metadata_line:
                print(f"Error: Empty file: {phy_file_path}")
                return False
            
            # Parse the metadata
            metadata = metadata_line.split()
            if len(metadata) >= 2:
                try:
                    num_sequences = int(metadata[0])
                    seq_length = int(metadata[1])
                    print(f"File contains {num_sequences} sequences of length {seq_length}")
                except ValueError:
                    print(f"Warning: Invalid metadata format: {metadata_line}")
                    num_sequences = None
                    seq_length = None
            else:
                print(f"Warning: Invalid metadata format: {metadata_line}")
                num_sequences = None
                seq_length = None
            
            # Open the output file for writing
            with open(fa_file_path, 'w') as fa_file:
                # Process each sequence line
                sequence_count = 0
                for line_num, line in enumerate(phy_file, 2):  # Start line counting from 2
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # Find the pattern to split ID and sequence
                    match = re.search(pattern, line)
                    if match:
                        # Split at the position between the ID and the sequence
                        split_position = match.start() + len(match.group(1))
                        seq_id = line[:split_position]
                        seq_data = line[split_position:]
                        
                        # Check if the sequence length matches the expected length
                        if seq_length is not None and len(seq_data) != seq_length:
                            print(f"Warning: Line {line_num} has sequence length {len(seq_data)}, expected {seq_length}")
                        
                        # Translate sequence if required
                        if translate_to_protein:
                            seq_data = translate_dna_to_protein(seq_data)
                        
                        # Write in FASTA format
                        fa_file.write(f'>{seq_id}\n')
                        
                        # Wrap sequence data to the specified length
                        for i in range(0, len(seq_data), wrap_length):
                            fa_file.write(seq_data[i:i+wrap_length] + '\n')
                        
                        sequence_count += 1
                    else:
                        # Try fallback to length-based approach if pattern not found
                        if seq_length is not None and len(line) > seq_length:
                            split_position = len(line) - seq_length
                            seq_id = line[:split_position]
                            seq_data = line[split_position:]
                            
                            # Translate sequence if required
                            if translate_to_protein:
                                seq_data = translate_dna_to_protein(seq_data)
                                
                            fa_file.write(f'>{seq_id}\n')
                            for i in range(0, len(seq_data), wrap_length):
                                fa_file.write(seq_data[i:i+wrap_length] + '\n')
                            
                            sequence_count += 1
                            print(f"Warning: Pattern not found for line {line_num}, using length-based splitting")
                        else:
                            print(f"Warning: Could not parse line {line_num}: {line}")
                
                # Check if the number of sequences matches the metadata
                if num_sequences is not None and sequence_count != num_sequences:
                    print(f"Warning: Expected {num_sequences} sequences, but found {sequence_count}")
        
        return True
    except FileNotFoundError:
        print(f"Error: File not found: {phy_file_path}")
    except Exception as e:
        print(f"Error processing file {phy_file_path}: {e}")
    
    return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Convert PHY files to FASTA format')
    parser.add_argument('directory', nargs='?', default='.',
                        help='Directory containing PHY files (default: current directory)')
    parser.add_argument('--pattern', default=None,
                        help='Regular expression pattern to identify the ID/sequence boundary. '
                             'Default is "(_[LR])([ACGTN])" to match the end of ID followed by sequence start.')
    parser.add_argument('--wrap', type=int, default=60,
                        help='Number of characters per line for the sequence (default: 60)')
    parser.add_argument('--protein', action='store_true',
                        help='Translate DNA sequences to protein sequences')
    return parser.parse_args()

def main():
    """Main function to process all PHY files in a directory"""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Process all PHY files in the directory
        phy_files = [f for f in os.listdir(args.directory) if f.endswith('.phy')]
        if not phy_files:
            print(f"No PHY files found in directory: {args.directory}")
            return
        
        success_count = 0
        for filename in phy_files:
            phy_file_path = os.path.join(args.directory, filename)
            output_type = "protein FASTA" if args.protein else "FASTA"
            print(f"Converting {phy_file_path} to {output_type} format...")
            if convert_phy_to_fasta(phy_file_path, pattern=args.pattern, wrap_length=args.wrap, translate_to_protein=args.protein):
                output_file = os.path.splitext(phy_file_path)[0] + ('_protein.fa' if args.protein else '.fa')
                print(f"Conversion complete: {output_file}")
                success_count += 1
        
        print(f"Conversion completed for {success_count} out of {len(phy_files)} PHY files.")
    except FileNotFoundError:
        print(f"Error: Directory not found: {args.directory}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
