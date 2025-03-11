import os
import re
import csv
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_fasta(fasta_file):
    """Parse a FASTA file and return a dictionary of sequences."""
    sequences = {}
    current_header = None
    current_sequence = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save the previous sequence if it exists
                if current_header is not None:
                    sequences[current_header] = current_sequence
                
                current_header = line[1:]  # Remove the '>' character
                current_sequence = ""
            else:
                current_sequence += line.upper()  # Convert to uppercase for consistency
        
        # Save the last sequence
        if current_header is not None:
            sequences[current_header] = current_sequence
    
    return sequences

def extract_gene_info(filename):
    """Extract gene ID and genomic coordinates from filename.
    
    Expected format: group[0/1]_GENE_ID_...chr[X]_start[N]_end[M].fa
    """
    # Extract gene ID
    gene_match = re.search(r'group\d+_([^_]+)_', filename)
    gene_id = gene_match.group(1) if gene_match else ""
    
    # Extract full gene identifier (everything between groupX_ and .fa)
    full_id_match = re.search(r'group\d+_(.+?)\.fa', filename)
    full_id = full_id_match.group(1) if full_id_match else ""
    
    # Extract chromosome information - properly separate chr from start
    chr_match = re.search(r'_(chr\w+)_', filename)
    chr_info = chr_match.group(1) if chr_match else ""
    
    # Extract start position
    start_match = re.search(r'_start(\d+)_', filename)
    start_pos = int(start_match.group(1)) if start_match else 0
    
    # Extract end position
    end_match = re.search(r'_end(\d+)\.fa', filename)
    end_pos = int(end_match.group(1)) if end_match else 0
    
    return gene_id, full_id, chr_info, start_pos, end_pos

def check_valid_dna(sequences):
    """Check if all sequences contain only valid DNA letters (A, C, G, T).
    Return a list of any invalid characters found."""
    invalid_chars = set()
    
    for header, sequence in sequences.items():
        for char in set(sequence):  # Use set for efficiency - only check each unique character once
            if char not in 'ACGT':
                invalid_chars.add(char)
    
    return list(invalid_chars)

def find_differences(group0_sequences, group1_sequences):
    """Find both fixed and all differences between two groups efficiently.
    
    Returns:
    - fixed_differences: list of tuples (position, group0_nuc, group1_nuc)
    - all_differences: dict of all positions with differences
    """
    # Get sequence length (assuming all sequences are the same length)
    seq_length = len(next(iter(group0_sequences.values())))
    
    # Initialize data structures
    fixed_differences = []
    all_differences = {}
    
    # Process sequence positions in chunks for better efficiency
    chunk_size = 1000
    
    for chunk_start in range(0, seq_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_length)
        
        # Process each position in the chunk
        for pos in range(chunk_start, chunk_end):
            # Collect nucleotides at this position for each group
            g0_nucs = set(seq[pos] for seq in group0_sequences.values() if pos < len(seq))
            g1_nucs = set(seq[pos] for seq in group1_sequences.values() if pos < len(seq))
            
            # Skip if both groups have identical nucleotide sets
            if g0_nucs == g1_nucs:
                continue
                
            # Only include valid nucleotides
            g0_valid = {n for n in g0_nucs if n in 'ACGT'}
            g1_valid = {n for n in g1_nucs if n in 'ACGT'}
            
            # Skip if either group has no valid nucleotides
            if not g0_valid or not g1_valid:
                continue
                
            # Record as a difference
            all_differences[pos] = {'g0': g0_valid, 'g1': g1_valid}
            
            # Check if it's a fixed difference (each group has exactly one nucleotide)
            if len(g0_valid) == 1 and len(g1_valid) == 1:
                g0_nuc = next(iter(g0_valid))
                g1_nuc = next(iter(g1_valid))
                
                if g0_nuc != g1_nuc:
                    fixed_differences.append((pos, g0_nuc, g1_nuc))
    
    return fixed_differences, all_differences

def plot_differences(gene_id, chr_info, start_pos, group0_sequences, group1_sequences, all_differences, fixed_differences):
    """Create a visualization of the differences between the two groups."""
    # Create a nucleotide color map
    nuc_colors = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', '-': 'white', 'N': 'purple'}
    
    # Convert fixed differences to a set of positions for quick lookup
    fixed_pos = {pos for pos, _, _ in fixed_differences}
    
    # Get all sample IDs
    g0_samples = list(group0_sequences.keys())
    g1_samples = list(group1_sequences.keys())
    all_samples = g0_samples + g1_samples
    
    # Count number of samples
    n_samples = len(all_samples)
    
    # Prepare the figure
    fig_height = max(10, n_samples * 0.3)  # Adjust height based on sample count
    fig_width = 15
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create an empty matrix for visualization
    # Only include positions with differences to save space
    diff_positions = sorted(all_differences.keys())
    
    # Create visualization matrix
    n_diffs = len(diff_positions)
    viz_matrix = np.ones((n_samples, n_diffs)) * -1  # -1 represents missing/invalid data
    
    # Fill in the matrix with nucleotide indices
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': 5}
    
    for i, sample in enumerate(all_samples):
        is_g0 = i < len(g0_samples)
        sequence = group0_sequences[sample] if is_g0 else group1_sequences[sample]
        
        for j, pos in enumerate(diff_positions):
            if pos < len(sequence):
                nuc = sequence[pos]
                if nuc in nuc_to_idx:
                    viz_matrix[i, j] = nuc_to_idx[nuc]
    
    # Create a custom colormap
    cmap = ListedColormap(['green', 'blue', 'orange', 'red', 'purple', 'lightgrey'])
    
    # Create heatmap
    plt.imshow(viz_matrix, aspect='auto', cmap=cmap, interpolation='none')
    
    # Highlight fixed difference columns
    fixed_indices = [diff_positions.index(pos) for pos in fixed_pos if pos in diff_positions]
    for idx in fixed_indices:
        plt.axvspan(idx-0.5, idx+0.5, color='yellow', alpha=0.3)
    
    # Add position labels on x-axis (genomic coordinates)
    plt.xticks(range(len(diff_positions)), 
               [start_pos + pos for pos in diff_positions], 
               rotation=90, fontsize=8)
    
    # Add group and sample labels on y-axis
    plt.yticks(range(n_samples), 
               [f"G0: {s}" if i < len(g0_samples) else f"G1: {s}" 
                for i, s in enumerate(all_samples)],
               fontsize=8)
    
    # Add a legend for nucleotides
    legend_elements = [mpatches.Patch(color=c, label=n) for n, c in 
                      [('A', 'green'), ('C', 'blue'), ('G', 'orange'), 
                       ('T', 'red'), ('N', 'purple'), ('-', 'lightgrey')]]
    
    # Add fixed difference to legend
    legend_elements.append(mpatches.Patch(color='yellow', alpha=0.3, label='Fixed Difference'))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add title and labels
    plt.title(f"{gene_id} ({chr_info}:{start_pos}+) - {len(fixed_differences)} Fixed Differences")
    plt.xlabel("Genomic Position")
    plt.ylabel("Sample")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = f"{gene_id}_{chr_info}_differences.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file

def find_file_pairs():
    """Find all group0/group1 file pairs in the current directory."""
    group0_files = glob.glob("group0_*.fa")
    file_pairs = []
    
    for group0_file in group0_files:
        # Extract the gene info part
        gene_part = group0_file.replace("group0_", "")
        # Construct the expected group1 filename
        group1_file = "group1_" + gene_part
        
        if os.path.exists(group1_file):
            file_pairs.append((group0_file, group1_file))
    
    return file_pairs

def process_file_pair(file_pair):
    """Process a single file pair and return results."""
    group0_file, group1_file = file_pair
    
    # Extract gene info from the filename
    gene_id, full_id, chr_info, start_pos, end_pos = extract_gene_info(group0_file)
    
    print(f"Processing {gene_id} ({chr_info}:{start_pos}-{end_pos})...")
    
    try:
        # Parse the FASTA files
        group0_sequences = parse_fasta(group0_file)
        group1_sequences = parse_fasta(group1_file)
        
        # Verify that sequences exist in each group
        if not group0_sequences:
            raise ValueError(f"No valid sequences found in {group0_file}")
        if not group1_sequences:
            raise ValueError(f"No valid sequences found in {group1_file}")
        
        # Check for invalid DNA characters
        g0_invalid = check_valid_dna(group0_sequences)
        g1_invalid = check_valid_dna(group1_sequences)
        
        if g0_invalid:
            raise ValueError(f"Invalid characters found in {group0_file}: {', '.join(g0_invalid)}")
        if g1_invalid:
            raise ValueError(f"Invalid characters found in {group1_file}: {', '.join(g1_invalid)}")
        
        # Get sequence length (assuming all sequences are the same length)
        group0_seq_length = len(next(iter(group0_sequences.values())))
        group1_seq_length = len(next(iter(group1_sequences.values())))
        
        if group0_seq_length != group1_seq_length:
            raise ValueError(f"Sequence lengths differ between groups: "
                            f"Group 0 = {group0_seq_length}, Group 1 = {group1_seq_length}")
        
        # Find differences efficiently
        fixed_differences, all_differences = find_differences(group0_sequences, group1_sequences)
        
        # Only create visualization if there are fixed differences
        viz_file = ""
        if fixed_differences:
            viz_file = plot_differences(gene_id, chr_info, start_pos, group0_sequences, group1_sequences, 
                                       all_differences, fixed_differences)
            print(f"  Created visualization: {viz_file}")
        
        # Prepare results
        results = {
            'gene_id': gene_id,
            'full_id': full_id,
            'chr_info': chr_info,
            'start_pos': start_pos,
            'fixed_differences': fixed_differences,
            'viz_file': viz_file,
            'status': 'success'
        }
        
    except Exception as e:
        results = {
            'gene_id': gene_id,
            'full_id': full_id,
            'chr_info': chr_info,
            'start_pos': start_pos,
            'error': str(e),
            'status': 'error'
        }
        print(f"  Error processing {gene_id}: {str(e)}")
    
    return results

def main():
    output_file = "fixed_differences.csv"
    start_time = time.time()
    
    try:
        file_pairs = find_file_pairs()
        
        if not file_pairs:
            print("No matching group0/group1 file pairs found in the current directory.")
            return 1
        
        print(f"Found {len(file_pairs)} group0/group1 file pairs to process.")
        
        # Process all file pairs
        results = []
        
        # Determine if we should use multiprocessing based on the number of file pairs
        use_multiprocessing = len(file_pairs) > 4
        
        if use_multiprocessing:
            # Use multiprocessing for efficiency
            with ProcessPoolExecutor() as executor:
                # Submit all tasks
                future_to_pair = {executor.submit(process_file_pair, pair): pair for pair in file_pairs}
                
                # Collect results as they complete
                for future in as_completed(future_to_pair):
                    results.append(future.result())
        else:
            # Process sequentially for small numbers of files
            for pair in file_pairs:
                results.append(process_file_pair(pair))
        
        # Write results to CSV
        with open(output_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write header
            csv_writer.writerow(['Gene', 'FullGeneID', 'Chromosome', 'Position', 'Group0_Nucleotide', 'Group1_Nucleotide', 'Visualization'])
            
            # Track statistics
            total_differences = 0
            genes_with_differences = 0
            genes_without_differences = 0
            error_count = 0
            
            # Write each result
            for result in sorted(results, key=lambda x: x['gene_id']):
                if result['status'] == 'error':
                    # Write error entry
                    csv_writer.writerow([
                        result['gene_id'],
                        result['full_id'],
                        result['chr_info'],
                        f"Error: {result['error']}",
                        "",
                        "",
                        ""
                    ])
                    error_count += 1
                elif not result.get('fixed_differences', []):
                    # No fixed differences
                    csv_writer.writerow([
                        result['gene_id'],
                        result['full_id'],
                        result['chr_info'],
                        "No fixed differences",
                        "",
                        "",
                        ""
                    ])
                    genes_without_differences += 1
                else:
                    # Write each fixed difference
                    for i, (pos, g0_nuc, g1_nuc) in enumerate(result['fixed_differences']):
                        # Calculate genomic coordinate (1-based, per user requirement)
                        genomic_pos = result['start_pos'] + pos
                        
                        # Only include viz file on first row
                        viz = result['viz_file'] if i == 0 else ""
                        
                        csv_writer.writerow([
                            result['gene_id'],
                            result['full_id'],
                            result['chr_info'],
                            genomic_pos,
                            g0_nuc,
                            g1_nuc,
                            viz
                        ])
                    
                    total_differences += len(result['fixed_differences'])
                    genes_with_differences += 1
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print(f"Analysis complete in {elapsed_time:.2f} seconds.")
        print(f"  Total genes processed: {len(file_pairs)}")
        print(f"  Genes with fixed differences: {genes_with_differences}")
        print(f"  Genes without fixed differences: {genes_without_differences}")
        print(f"  Genes with errors: {error_count}")
        print(f"  Total fixed differences found: {total_differences}")
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
