import os
import gzip
import requests
import shutil
import glob
import time
import re
import multiprocessing
from collections import defaultdict

# --- Configuration ---
METADATA_FILE = 'phy_metadata.tsv'
AXT_URL = 'http://hgdownload.soe.ucsc.edu/goldenpath/hg38/vsPanTro5/hg38.panTro5.net.axt.gz'
AXT_GZ_FILENAME = 'hg38.panTro5.net.axt.gz'
AXT_FILENAME = 'hg38.panTro5.net.axt'

# --- For diagnostics: Set to an ENST ID string to get verbose logs for one transcript ---
DEBUG_TRANSCRIPT = None 

# --- Utility Functions ---

class Logger:
    """A simple class to manage and summarize warnings."""
    def __init__(self, max_prints=10):
        self.warnings = defaultdict(list)
        self.max_prints = max_prints

    def add(self, category, message):
        self.warnings[category].append(message)

    def report(self):
        print("\n--- Validation & Processing Summary ---")
        if not self.warnings:
            print("All checks passed without warnings.")
            return
        for category, messages in self.warnings.items():
            print(f"\nCategory '{category}': {len(messages)} total warnings.")
            for msg in messages[:self.max_prints]:
                print(f"  - {msg}")
            if len(messages) > self.max_prints:
                print(f"  ... and {len(messages) - self.max_prints} more.")
        print("-" * 35)

logger = Logger()

def download_axt_file():
    if os.path.exists(AXT_FILENAME) or os.path.exists(AXT_GZ_FILENAME): return
    print(f"Downloading '{AXT_GZ_FILENAME}' from UCSC...")
    try:
        with requests.get(AXT_URL, stream=True) as r:
            r.raise_for_status()
            with open(AXT_GZ_FILENAME, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192 * 4): f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Error downloading file: {e}"); exit(1)

def ungzip_file():
    if not os.path.exists(AXT_GZ_FILENAME):
        if not os.path.exists(AXT_FILENAME): print(f"FATAL: AXT file not found."); exit(1)
        return
    if os.path.exists(AXT_FILENAME): return
    print(f"Decompressing '{AXT_GZ_FILENAME}'...")
    try:
        with gzip.open(AXT_GZ_FILENAME, 'rb') as f_in, open(AXT_FILENAME, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
    except Exception as e:
        print(f"FATAL: Error decompressing file: {e}"); exit(1)

def read_phy_sequences(filename):
    """
    Robustly reads ALL sequences from a phylip file, handling cases
    where there is no space between the sample name and the sequence data.
    Returns a list of sequence strings.
    """
    sequences = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2: return []
            # Skip the header line (lines[0])
            for line in lines[1:]:
                line = line.strip()
                if not line: continue
                match = re.search(r'[ACGTN-]+$', line, re.IGNORECASE)
                if match:
                    sequences.append(match.group(0).upper())
    except Exception:
        pass
    return sequences

# --- Core Logic ---

def validate_inputs_and_parse_metadata():
    """
    Parses metadata, calculating expected length from coordinate chunks and
    validating that ALL sequences in the input .phy files match this length.
    """
    if not os.path.exists(METADATA_FILE):
        print(f"FATAL: Metadata file '{METADATA_FILE}' not found."); exit(1)
    
    print("Validating input files against metadata...")
    validated_transcripts = []
    seen_cds_keys = set()

    with open(METADATA_FILE, 'r') as f:
        next(f, None)
        for line_num, line in enumerate(f, 2):
            parts = [p.strip() for p in line.strip().split('\t')]
            if len(parts) < 9: continue
            
            phy_fname, t_id, gene, chrom, _, start, end, _, coords_str = parts[:9]
            cds_key = (t_id, coords_str)
            if cds_key in seen_cds_keys: continue
            seen_cds_keys.add(cds_key)

            try:
                segments = [(int(s), int(e)) for s, e in (p.split('-') for p in coords_str.split(';'))]
                expected_len = sum(e - s + 1 for s, e in segments)
                if expected_len == 0: continue
            except (ValueError, IndexError):
                logger.add("Metadata Parsing Error", f"L{line_num}: Could not parse coordinate chunks for {t_id}.")
                continue
            
            g0_fname = phy_fname.replace("group1_", "group0_") if "group1_" in phy_fname else phy_fname
            g1_fname = phy_fname.replace("group0_", "group1_") if "group0_" in phy_fname else phy_fname
            
            g0_seqs = read_phy_sequences(g0_fname)
            g1_seqs = read_phy_sequences(g1_fname)
            
            if not g0_seqs or not g1_seqs:
                logger.add("Missing Input File", f"{t_id}: group0 or group1 .phy file not found or is empty.")
                continue

            # Validate ALL sequences in each file
            valid_g0 = all(len(s) == expected_len for s in g0_seqs)
            valid_g1 = all(len(s) == expected_len for s in g1_seqs)

            if not valid_g0 or not valid_g1:
                g0_lengths = set(len(s) for s in g0_seqs)
                g1_lengths = set(len(s) for s in g1_seqs)
                logger.add("Input Length Mismatch", f"{t_id}: Phy lengths (g0:{g0_lengths}, g1:{g1_lengths}) != calculated length from coords ({expected_len}).")
                continue

            cds_info = {'gene_name': gene, 'transcript_id': t_id, 'chromosome': 'chr' + chrom,
                        'expected_len': expected_len, 'start': start, 'end': end}
            validated_transcripts.append({'info': cds_info, 'segments': segments})
    
    return validated_transcripts

def process_axt_chunk(chunk_start, chunk_end, coord_map):
    """Worker process with robust AXT block parsing."""
    results = defaultdict(dict)
    with open(AXT_FILENAME, 'r') as f:
        f.seek(chunk_start)
        if chunk_start != 0: f.readline()

        while f.tell() < chunk_end:
            line = f.readline()
            if not line: break
            if not line.strip() or line.startswith('#'): continue

            header = line.strip().split()
            if len(header) != 9: continue
            
            human_seq_line = f.readline()
            chimp_seq_line = f.readline()
            if not human_seq_line or not chimp_seq_line: break
            
            human_seq = human_seq_line.strip().upper()
            chimp_seq = chimp_seq_line.strip().upper()

            axt_chr, human_pos = header[1], int(header[2])
            chr_coord_map = coord_map.get(axt_chr)
            if not chr_coord_map: continue

            for h_char, c_char in zip(human_seq, chimp_seq):
                if h_char != '-':
                    target_list = chr_coord_map.get(human_pos)
                    if target_list:
                        for t_id, target_idx in target_list:
                            if target_idx not in results[t_id]:
                                 results[t_id][target_idx] = c_char
                    human_pos += 1
    
    return dict(results)

def build_chimp_sequences(validated_transcripts):
    if not validated_transcripts: return

    print("Pre-computing overlap-aware coordinate map...")
    coord_map, scaffold_map = {}, {}
    
    for t in validated_transcripts:
        info, segments = t['info'], t['segments']
        t_id, chrom = info['transcript_id'], info['chromosome']
        scaffold_map[t_id] = ['-'] * info['expected_len']
        
        if chrom not in coord_map: coord_map[chrom] = {}
        
        target_idx = 0
        for seg_start, seg_end in segments:
            for coord in range(seg_start, seg_end + 1):
                if coord not in coord_map[chrom]: coord_map[chrom][coord] = []
                coord_map[chrom][coord].append((t_id, target_idx))
                target_idx += 1

    print(f"Processing '{AXT_FILENAME}' in parallel...")
    file_size = os.path.getsize(AXT_FILENAME)
    try: num_procs = len(os.sched_getaffinity(0))
    except AttributeError: num_procs = multiprocessing.cpu_count()
    print(f"Using {num_procs} available CPU cores.")
    
    chunk_size = file_size // num_procs
    chunks = [(i * chunk_size, (i + 1) * chunk_size, coord_map) for i in range(num_procs)]
    chunks[-1] = (chunks[-1][0], file_size, coord_map) # Ensure last chunk goes to the end

    start_time = time.time()
    with multiprocessing.Pool(processes=num_procs) as pool:
        list_of_results_dicts = pool.starmap(process_axt_chunk, chunks)
    print(f"Finished parallel AXT processing in {time.time() - start_time:.2f} seconds.")

    print("Merging results and writing outgroup files...")
    for results_dict in list_of_results_dicts:
        for t_id, positions in results_dict.items():
            if t_id in scaffold_map:
                for target_idx, base in positions.items():
                    if scaffold_map[t_id][target_idx] == '-':
                        scaffold_map[t_id][target_idx] = base

    files_written = 0
    for t in validated_transcripts:
        info = t['info']
        t_id, gene, chrom, start, end = info['transcript_id'], info['gene_name'], info['chromosome'], info['start'], info['end']
        final_sequence = "".join(scaffold_map[t_id])

        if not final_sequence or final_sequence.count('-') == len(final_sequence):
            logger.add("No Alignment Found", f"No chimp alignment found for {t_id}.")
            continue
            
        if DEBUG_TRANSCRIPT == t_id:
            print(f"\n--- DEBUG: {t_id} ---\nFinal sequence (len={len(final_sequence)}): {final_sequence[:100]}...\n")

        fname = f"outgroup_{gene}_{t_id}_{chrom}_start{start}_end{end}.phy"
        with open(fname, 'w') as f_out:
            f_out.write(f" 1 {len(final_sequence)}\n")
            f_out.write(f"{'panTro5':<10}{final_sequence}\n")
        files_written += 1
    
    print(f"Wrote {files_written} outgroup phylip files.")

def calculate_and_print_differences():
    print("\n--- Final Difference Calculation & Statistics ---")
    key_regex = re.compile(r"(ENST[0-9]+\.[0-9]+)_(chr.+?)_start([0-9]+)_end([0-9]+)")
    cds_groups = defaultdict(dict)
    for f in glob.glob('*.phy'):
        match = key_regex.search(os.path.basename(f))
        if match: cds_groups[match.groups()][os.path.basename(f).split('_')[0]] = f

    # Data structures for new stats
    per_transcript_divergence = {}
    total_fixed_diffs = 0
    g0_matches_chimp_at_fixed = 0
    g1_matches_chimp_at_fixed = 0
    per_transcript_g0_match_scores = {}
    per_transcript_g1_match_scores = {}
    
    substitution_diffs = defaultdict(list)
    comparable_sets = 0
    
    print("Analyzing each comparable transcript set...")
    for identifier, group_files in cds_groups.items():
        if 'group0' in group_files and 'group1' in group_files and 'outgroup' in group_files:
            g0_seqs = read_phy_sequences(group_files['group0'])
            g1_seqs = read_phy_sequences(group_files['group1'])
            out_seq = read_phy_sequences(group_files['outgroup'])[0] # Chimp is single sequence
            
            # --- Verification Step ---
            g0_len_set = set(len(s) for s in g0_seqs)
            g1_len_set = set(len(s) for s in g1_seqs)
            if not (len(g0_len_set) == 1 and len(g1_len_set) == 1):
                logger.add("Intra-file Length Mismatch", f"Not all sequences in a .phy file have the same length for {identifier[0]}.")
                continue
            
            expected_len = g0_len_set.pop()
            if expected_len != g1_len_set.pop() or expected_len != len(out_seq):
                logger.add("Final Comparison Error", f"Length mismatch between groups for {identifier[0]}.")
                continue
            
            comparable_sets += 1
            n = expected_len
            t_id = identifier[0]
            gene_name = group_files['group0'].split('_')[1]
            
            # --- Stat 1: Substitution Divergence (Ignoring Dashes) ---
            diff_count = sum(1 for a, b in zip(g0_seqs[0], out_seq) if a != b and '-' not in (a, b))
            sub_div = (diff_count / n) * 100
            per_transcript_divergence[f"{gene_name} ({t_id})"] = sub_div
            substitution_diffs['g0_out'].append(sub_div)

            # --- Stat 2: Fixed Difference Analysis ---
            local_fixed_diffs = 0
            local_g0_matches = 0
            local_g1_matches = 0

            for i in range(n):
                g0_alleles = {s[i] for s in g0_seqs if s[i] != '-'}
                g1_alleles = {s[i] for s in g1_seqs if s[i] != '-'}

                if len(g0_alleles) == 1 and len(g1_alleles) == 1 and g0_alleles != g1_alleles:
                    # This is a fixed difference
                    local_fixed_diffs += 1
                    total_fixed_diffs += 1
                    g0_allele = g0_alleles.pop()
                    g1_allele = g1_alleles.pop()
                    chimp_allele = out_seq[i]

                    if chimp_allele == g0_allele:
                        g0_matches_chimp_at_fixed += 1
                        local_g0_matches += 1
                    elif chimp_allele == g1_allele:
                        g1_matches_chimp_at_fixed += 1
                        local_g1_matches += 1
            
            if local_fixed_diffs > 0:
                per_transcript_g0_match_scores[f"{gene_name} ({t_id})"] = (local_g0_matches / local_fixed_diffs) * 100
                per_transcript_g1_match_scores[f"{gene_name} ({t_id})"] = (local_g1_matches / local_fixed_diffs) * 100

    if comparable_sets == 0:
        print("CRITICAL: No complete sets found to compare after processing."); return
    print(f"Successfully analyzed {comparable_sets} complete CDS sets.")

    # --- Print All New Statistics ---
    print("\n" + "="*50)
    print(" REPORT")
    print("="*50)
    
    # Report Substitution Divergence
    sorted_divergence = sorted(per_transcript_divergence.items(), key=lambda item: item[1])
    print("\n--- Human-Chimp Substitution Divergence (Ignoring Indels) ---")
    print("Top 5 LOWEST Divergence (Most Conserved):")
    for (gene_id, div) in sorted_divergence[:5]:
        print(f"  - {gene_id:<40}: {div:.4f}%")
    print("\nTop 5 HIGHEST Divergence (Most Divergent):")
    for (gene_id, div) in sorted_divergence[-5:]:
        print(f"  - {gene_id:<40}: {div:.4f}%")
    avg_div = sum(substitution_diffs['g0_out']) / len(substitution_diffs['g0_out'])
    print(f"\nOverall Average Substitution Divergence: {avg_div:.4f}%")

    # Report Fixed Difference Statistics
    print("\n--- Fixed Difference Ancestry Analysis ---")
    if total_fixed_diffs > 0:
        g0_match_perc = (g0_matches_chimp_at_fixed / total_fixed_diffs) * 100
        g1_match_perc = (g1_matches_chimp_at_fixed / total_fixed_diffs) * 100
        print(f"Total fixed differences found between group0 and group1: {total_fixed_diffs}")
        print(f"  - Group 0 allele matched Chimp: {g0_match_perc:.2f}% of the time.")
        print(f"  - Group 1 allele matched Chimp: {g1_match_perc:.2f}% of the time.")

        sorted_g0_scores = sorted(per_transcript_g0_match_scores.items(), key=lambda item: item[1])
        print("\nTop 5 CDS where Group 0 allele LEAST resembles Chimp (at fixed diffs):")
        for (gene_id, score) in sorted_g0_scores[:5]:
            print(f"  - {gene_id:<40}: {score:.2f}% match")

        sorted_g1_scores = sorted(per_transcript_g1_match_scores.items(), key=lambda item: item[1])
        print("\nTop 5 CDS where Group 1 allele LEAST resembles Chimp (at fixed diffs):")
        for (gene_id, score) in sorted_g1_scores[:5]:
            print(f"  - {gene_id:<40}: {score:.2f}% match")
    else:
        print("No fixed differences were found between group0 and group1.")
    print("="*50 + "\n")


def main():
    print("--- Starting Corrected Chimp CDS Phylip Generation with Stats ---")
    download_axt_file()
    ungzip_file()
    validated_transcripts = validate_inputs_and_parse_metadata()
    if not validated_transcripts:
        print("No valid transcripts found after initial validation.")
    else:
        build_chimp_sequences(validated_transcripts)
        calculate_and_print_differences()
    logger.report()
    print("--- Script finished. ---")

if __name__ == '__main__':
    main()
