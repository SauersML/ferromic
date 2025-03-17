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
    """
    Parse a FASTA file and return a dictionary of sequences.
    Keys: sequence headers (minus '>'), Values: full uppercase sequence strings.
    """
    sequences = {}
    current_header = None
    current_sequence = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences[current_header] = current_sequence
                current_header = line[1:]  # remove '>'
                current_sequence = ""
            else:
                current_sequence += line.upper()
        if current_header is not None:
            sequences[current_header] = current_sequence
    
    return sequences

def extract_gene_info(filename):
    """
    Extract gene ID, plus additional info from a filename of format:
      group[0/1]_GENEID_chrX_startNNN_endMMM.fa
    Also extracts transcript ID if present, e.g. _ENST#########_ in the filename.
    """
    gene_match = re.search(r'group\d+_(.+?)_chr', filename)
    gene_id = gene_match.group(1) if gene_match else ""
    
    full_id_match = re.search(r'group\d+_(.+?)\.fa', filename)
    full_id = full_id_match.group(1) if full_id_match else ""
    
    chr_match = re.search(r'_(chr\w+)_', filename)
    chr_info = chr_match.group(1) if chr_match else ""
    
    start_match = re.search(r'_start(\d+)_', filename)
    start_pos = int(start_match.group(1)) if start_match else 0
    
    end_match = re.search(r'_end(\d+)\.fa', filename)
    end_pos = int(end_match.group(1)) if end_match else 0
    
    transcript_match = re.search(r'_(ENST\d+\.\d+)_', filename)
    transcript_id = transcript_match.group(1) if transcript_match else ""
    
    return gene_id, full_id, chr_info, start_pos, end_pos, transcript_id

def check_valid_dna(sequences):
    """
    Check that each sequence in 'sequences' contains only A, C, G, T.
    Returns a list of invalid characters found if any exist.
    """
    invalid_chars = set()
    for header, seq in sequences.items():
        for char in set(seq):
            if char not in "ACGT":
                invalid_chars.add(char)
    return list(invalid_chars)

def find_differences(group0_sequences, group1_sequences):
    """
    Compare two sets of sequences (group0, group1) that should be the same length.
    Return:
      fixed_differences: list of (pos, group0_nuc, group1_nuc)
      all_differences: dict with pos -> { 'g0': set_of_nucs, 'g1': set_of_nucs }
    """
    seq_length = len(next(iter(group0_sequences.values())))
    fixed_differences = []
    all_differences = {}
    
    chunk_size = 1000
    for chunk_start in range(0, seq_length, chunk_size):
        chunk_end = min(seq_length, chunk_start + chunk_size)
        for pos in range(chunk_start, chunk_end):
            g0_nucs = set(seq[pos] for seq in group0_sequences.values() if pos < len(seq))
            g1_nucs = set(seq[pos] for seq in group1_sequences.values() if pos < len(seq))
            if g0_nucs == g1_nucs:
                continue
            g0_valid = {n for n in g0_nucs if n in "ACGT"}
            g1_valid = {n for n in g1_nucs if n in "ACGT"}
            if not g0_valid or not g1_valid:
                continue
            all_differences[pos] = {"g0": g0_valid, "g1": g1_valid}
            if len(g0_valid) == 1 and len(g1_valid) == 1:
                g0_nuc = next(iter(g0_valid))
                g1_nuc = next(iter(g1_valid))
                if g0_nuc != g1_nuc:
                    fixed_differences.append((pos, g0_nuc, g1_nuc))
    
    return fixed_differences, all_differences

def parse_gtf_for_cds_regions(gtf_file, transcript_id):
    """
    Parse the GTF file for a specific transcript ID and gather all its CDS regions.
    Return a sorted list of (start, end) for these CDS segments, plus the strand.
    """
    cds_regions = []
    strand = None
    try:
        with open(gtf_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9:
                    continue
                if fields[2] != "CDS":
                    continue
                attributes = fields[8]
                transcript_base = transcript_id.split('.')[0]
                if not (f'transcript_id "{transcript_id}"' in attributes or 
                        f'transcript_id "{transcript_base}"' in attributes):
                    continue
                start = int(fields[3])
                end = int(fields[4])
                strand = fields[6]
                cds_regions.append((start, end))
    except Exception as e:
        print(f"Error parsing GTF file for {transcript_id}: {str(e)}")
        return [], None
    
    cds_regions.sort()
    return cds_regions, strand

def build_spliced_to_genomic_map(cds_regions, strand):
    """
    Given a list of (start, end) CDS coordinates and the strand,
    create a dictionary mapping from 0-based spliced positions to 1-based genomic positions.
    """
    spliced_to_genomic = {}
    if not cds_regions:
        return spliced_to_genomic
    regions = list(cds_regions)
    if strand == "-":
        regions.reverse()
    spliced_pos = 0
    for start, end in regions:
        length = end - start + 1
        if strand == "+":
            for i in range(length):
                spliced_to_genomic[spliced_pos + i] = start + i
        else:
            # negative strand
            for i in range(length):
                spliced_to_genomic[spliced_pos + i] = end - i
        spliced_pos += length
    return spliced_to_genomic

def plot_differences(gene_id, chr_info, start_pos, group0_sequences, group1_sequences,
                     all_differences, fixed_differences, genomic_positions):
    """
    Create a heatmap-like visualization of differences.
    X-axis: spliced positions (labeled with genomic_positions).
    Y-axis: all samples from both groups.
    """
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "-": 5}
    g0_samples = list(group0_sequences.keys())
    g1_samples = list(group1_sequences.keys())
    all_samples = g0_samples + g1_samples
    
    diff_positions = sorted(all_differences.keys())
    fixed_pos = {p for p, _, _ in fixed_differences}
    
    n_samples = len(all_samples)
    n_diffs = len(diff_positions)
    
    fig_height = max(10, n_samples * 0.3)
    fig_width = 15
    plt.figure(figsize=(fig_width, fig_height))
    
    viz_matrix = np.ones((n_samples, n_diffs)) * -1
    
    for i, sample in enumerate(all_samples):
        sequence = group0_sequences[sample] if i < len(g0_samples) else group1_sequences[sample]
        for j, pos in enumerate(diff_positions):
            if pos < len(sequence):
                nuc = sequence[pos]
                if nuc in nuc_to_idx:
                    viz_matrix[i, j] = nuc_to_idx[nuc]
    
    cmap = ListedColormap(["green", "blue", "orange", "red", "purple", "lightgrey"])
    plt.imshow(viz_matrix, aspect="auto", cmap=cmap, interpolation="none")
    
    # highlight columns with fixed differences
    for fx in fixed_pos:
        if fx in diff_positions:
            idx = diff_positions.index(fx)
            plt.axvspan(idx - 0.5, idx + 0.5, color="yellow", alpha=0.3)
    
    plt.xticks(range(n_diffs), [genomic_positions[p] for p in diff_positions], rotation=90, fontsize=8)
    plt.yticks(range(n_samples), 
               [f"G0: {sample}" if i < len(g0_samples) else f"G1: {sample}" 
                for i, sample in enumerate(all_samples)],
               fontsize=8)
    
    legend_elements = [
        mpatches.Patch(color="green", label="A"),
        mpatches.Patch(color="blue", label="C"),
        mpatches.Patch(color="orange", label="G"),
        mpatches.Patch(color="red", label="T"),
        mpatches.Patch(color="purple", label="N"),
        mpatches.Patch(color="lightgrey", label="-")
    ]
    legend_elements.append(mpatches.Patch(color="yellow", alpha=0.3, label="Fixed Difference"))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.title(f"{gene_id} ({chr_info}) - {len(fixed_differences)} Fixed Differences")
    plt.xlabel("Genomic Position (spliced)")
    plt.ylabel("Sample")
    plt.tight_layout()
    
    output_file = f"{gene_id}_{chr_info}_differences.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    return output_file

def find_file_pairs():
    """
    Look for files named group0_*.fa and group1_*.fa in the directory
    and pair them based on replacing the 'group0_' prefix with 'group1_'.
    """
    group0_files = glob.glob("group0_*.fa")
    file_pairs = []
    for g0 in group0_files:
        gene_part = g0.replace("group0_", "")
        g1 = "group1_" + gene_part
        if os.path.exists(g1):
            file_pairs.append((g0, g1))
    return file_pairs

def process_file_pair(file_pair, gtf_file=None):
    """
    Process one pair of group0 / group1 FASTA files:
      1) Parse them
      2) Ensure valid DNA
      3) Ensure lengths match
      4) Identify fixed differences
      5) (If transcript_id is present and GTF is given) parse CDS
         and verify spliced lengths match the .fa length
      6) Build spliced->genomic mapping
      7) Convert differences
      8) Return results or error
    """
    group0_file, group1_file = file_pair
    gene_id, full_id, chr_info, start_pos, end_pos, transcript_id = extract_gene_info(group0_file)
    
    print(f"Processing {gene_id} ({chr_info}:{start_pos}-{end_pos})...")
    try:
        # parse FASTA
        g0_seqs = parse_fasta(group0_file)
        g1_seqs = parse_fasta(group1_file)
        if not g0_seqs:
            raise ValueError(f"No valid sequences found in {group0_file}")
        if not g1_seqs:
            raise ValueError(f"No valid sequences found in {group1_file}")
        
        # check DNA validity
        g0_invalid = check_valid_dna(g0_seqs)
        g1_invalid = check_valid_dna(g1_seqs)
        if g0_invalid:
            raise ValueError(f"Invalid characters found in {group0_file}: {', '.join(g0_invalid)}")
        if g1_invalid:
            raise ValueError(f"Invalid characters found in {group1_file}: {', '.join(g1_invalid)}")
        
        # check lengths
        g0_length = len(next(iter(g0_seqs.values())))
        g1_length = len(next(iter(g1_seqs.values())))
        if g0_length != g1_length:
            raise ValueError(f"Sequence lengths differ: Group0={g0_length} vs Group1={g1_length}")
        
        # find differences
        fixed_diff, all_diff = find_differences(g0_seqs, g1_seqs)
        
        # parse GTF if transcript is known
        spliced_map = {}
        if gtf_file and transcript_id:
            cds_regions, strand = parse_gtf_for_cds_regions(gtf_file, transcript_id)
            if cds_regions:
                # ensure the combined length of these CDS segments matches the FASTA length
                total_cds_len = sum((end - start + 1) for start, end in cds_regions)
                if total_cds_len != g0_length:
                    raise ValueError(f"CDS length mismatch: sum(CDS)={total_cds_len} vs FASTA={g0_length}")
                spliced_map = build_spliced_to_genomic_map(cds_regions, strand)
                print(f"  Built mapping for {len(spliced_map)} spliced positions")
                if strand == "-":
                    print(f"  Note: {gene_id} is on the negative strand")

        # Check sequence similarity between representatives
        g0_seq = next(iter(g0_seqs.values()))
        g1_seq = next(iter(g1_seqs.values()))
        similarity = calculate_similarity(g0_seq, g1_seq)
        if similarity < 0.9:
            raise ValueError(f"Sequences are only {similarity:.2%} similar, which is below the 90% threshold")
    
        # if mapping is empty but we do have a transcript, it might be missing in GTF
        if transcript_id and gtf_file and not spliced_map and fixed_diff:
            raise ValueError("Could not map CDS for this transcript; GTF data may be missing or incomplete.")
        
        # convert fixed differences to genomic coords
        mapped_fixed = []
        for pos, g0_nuc, g1_nuc in fixed_diff:
            # Always record the difference with the position in the original sequence
            mapped_fixed.append((pos, pos, g0_nuc, g1_nuc))
            # Debug message if needed
            if not (transcript_id and spliced_map):
                print("NO TRANSCRIPT")
        
        # create visualization only if there are fixed differences
        # build a visualization position map for all differences
        vis_positions = {}
        for p in all_diff.keys():
            vis_positions[p] = p  # Just use positions in sequence
        
        viz_file = ""
        if fixed_diff:
            try:
                viz_file = plot_differences(gene_id, chr_info, start_pos,
                                           g0_seqs, g1_seqs, all_diff, fixed_diff,
                                           vis_positions)
                print(f"  Created visualization: {viz_file}")
            except Exception as e:
                print(f"  Warning: Couldn't create visualization: {str(e)}")
        
        results = {
            "gene_id": gene_id,
            "full_id": full_id,
            "chr_info": chr_info,
            "start_pos": start_pos,
            "fixed_differences": mapped_fixed,
            "viz_file": viz_file,
            "status": "success"
        }
        return results
    
    except Exception as e:
        error_context = ""
        if 'pos' in locals():
            error_context = f" (at position {pos})"
        print(f"  Error processing {gene_id}{error_context}: {str(e)}")
        return {
            "gene_id": gene_id,
            "full_id": full_id,
            "chr_info": chr_info,
            "start_pos": start_pos,
            "error": str(e),
            "status": "error"
        }

def calculate_similarity(seq1, seq2):
    """Calculate the percentage of matching positions between two sequences."""
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)

def main():
    """
    Main entry point. Searches for group0_*.fa and group1_*.fa pairs, optionally
    accepts a GTF file as sys.argv[1]. Processes each pair to find differences,
    map them to CDS coordinates if possible, and writes fixed_differences.csv.
    """
    output_file = "fixed_differences.csv"
    start_time = time.time()
    gtf_file = None
    
    if len(sys.argv) > 1:
        gtf_file = sys.argv[1]
        if not os.path.exists(gtf_file):
            print(f"Warning: GTF file {gtf_file} not found. Will use naive coordinate mapping.")
            gtf_file = None
        else:
            print(f"Using GTF file {gtf_file} for splicing-aware coordinate mapping")
    else:
        print("No GTF file provided. Using naive coordinate mapping (may be inaccurate for spliced genes).")
        print("Usage: python script.py [gtf_file]")
    
    try:
        pairs = find_file_pairs()
        if not pairs:
            print("No matching group0_*.fa / group1_*.fa pairs found.")
            return 1
        
        print(f"Found {len(pairs)} group0/group1 file pairs to process.")
        results = []
        
        use_multiprocessing = (len(pairs) > 4 and gtf_file is None)
        if use_multiprocessing:
            with ProcessPoolExecutor() as executor:
                future_map = {executor.submit(process_file_pair, pair, gtf_file): pair for pair in pairs}
                for future in as_completed(future_map):
                    results.append(future.result())
        else:
            for pair in pairs:
                results.append(process_file_pair(pair, gtf_file))
        
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Gene", "FullGeneID", "Chromosome", "Position",
                             "Group0_Nucleotide", "Group1_Nucleotide", "Visualization"])
            
            total_diff = 0
            genes_with_diff = 0
            genes_no_diff = 0
            errors = 0
            
            # sort by gene_id
            for res in sorted(results, key=lambda x: x["gene_id"]):
                if res["status"] == "error":
                    writer.writerow([res["gene_id"], res["full_id"], res["chr_info"],
                                     f"Error: {res['error']}", "", "", ""])
                    errors += 1
                else:
                    fd = res.get("fixed_differences", [])
                    if not fd:
                        writer.writerow([res["gene_id"], res["full_id"], res["chr_info"],
                                         "No fixed differences", "", "", ""])
                        genes_no_diff += 1
                    else:
                        genes_with_diff += 1
                        total_diff += len(fd)
                        for i, (spliced_pos, genomic_pos, g0n, g1n) in enumerate(fd):
                            viz = res["viz_file"] if i == 0 else ""
                            writer.writerow([res["gene_id"], res["full_id"], res["chr_info"],
                                             genomic_pos, g0n, g1n, viz])
        
        elapsed = time.time() - start_time
        print(f"Analysis complete in {elapsed:.2f} seconds.")
        print(f"  Total genes processed: {len(pairs)}")
        print(f"  Genes with fixed differences: {genes_with_diff}")
        print(f"  Genes without fixed differences: {genes_no_diff}")
        print(f"  Genes with errors: {errors}")
        print(f"  Total fixed differences found: {total_diff}")
        print(f"Results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
