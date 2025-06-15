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

def read_phy_sequence(filename):
    """Robustly reads a sequence from a phylip file using regex."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2: return ""
            line = lines[1].strip()
            match = re.search(r'[ACGTN-]+$', line, re.IGNORECASE)
            if match: return match.group(0)
    except Exception:
        pass
    return ""

# --- Core Logic ---

def validate_inputs_and_parse_metadata():
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
            g0_seq = read_phy_sequence(g0_fname)
            g1_seq = read_phy_sequence(g1_fname)
            
            if not g0_seq or not g1_seq:
                logger.add("Missing Input File", f"{t_id}: group0 or group1 .phy file not found or is empty.")
                continue

            if len(g0_seq) != expected_len or len(g1_seq) != expected_len:
                logger.add("Input Length Mismatch", f"{t_id}: Phy length (g0:{len(g0_seq)}, g1:{len(g1_seq)}) != calculated length from coords ({expected_len}).")
                continue

            cds_info = {'gene_name': gene, 'transcript_id': t_id, 'chromosome': 'chr' + chrom,
                        'expected_len': expected_len, 'start': start, 'end': end}
            validated_transcripts.append({'info': cds_info, 'segments': segments})
    
    return validated_transcripts

def process_axt_chunk(chunk_start, chunk_end, coord_map):
    """
    Worker process with robust AXT block parsing to prevent IndexErrors.
    """
    results = defaultdict(dict)
    with open(AXT_FILENAME, 'r') as f:
        f.seek(chunk_start)
        # Ensure we start on a full line, not mid-line
        if chunk_start != 0: f.readline()

        while f.tell() < chunk_end:
            line = f.readline()
            if not line: break

            if not line.strip() or line.startswith('#'): continue

            # --- ROBUSTNESS FIX ---
            # A valid header line MUST have 9 columns. If not, skip it entirely
            # and do NOT attempt to read the next two lines as sequences.
            header = line.strip().split()
            if len(header) != 9:
                # This is a malformed line, not a real header.
                # Simply continue to the next line to resynchronize.
                continue
            
            # If we are here, we have a valid 9-column header.
            # Now it is safe to read the next two lines.
            human_seq = f.readline()
            chimp_seq = f.readline()
            
            # Check that we didn't hit the end of the file/chunk prematurely
            if not human_seq or not chimp_seq:
                break
            
            human_seq = human_seq.strip().upper()
            chimp_seq = chimp_seq.strip().upper()
            # --- END OF FIX ---

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
    coord_map = {}
    scaffold_map = {}
    
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
    chunk_boundaries = [0] + [i * chunk_size for i in range(1, num_procs)] + [file_size]
    chunks = [(chunk_boundaries[i], chunk_boundaries[i+1], coord_map) for i in range(num_procs)]

    start_time = time.time()
    with multiprocessing.Pool(processes=num_procs) as pool:
        list_of_results_dicts = pool.starmap(process_axt_chunk, chunks)
    print(f"Finished parallel AXT processing in {time.time() - start_time:.2f} seconds.")

    print("Merging results and writing outgroup files...")
    for results_dict in list_of_results_dicts:
        for t_id, positions in results_dict.items():
            if t_id not in scaffold_map: continue
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
    print("\n--- Final Difference Calculation ---")
    key_regex = re.compile(r"(ENST[0-9]+\.[0-9]+)_(chr.+?)_start([0-9]+)_end([0-9]+)")
    cds_groups = defaultdict(dict)
    for f in glob.glob('*.phy'):
        match = key_regex.search(os.path.basename(f))
        if match: cds_groups[match.groups()][os.path.basename(f).split('_')[0]] = f

    diffs = defaultdict(list)
    comparable_sets = 0
    for identifier, group_files in cds_groups.items():
        if 'group0' in group_files and 'group1' in group_files and 'outgroup' in group_files:
            seq0 = read_phy_sequence(group_files['group0'])
            seq1 = read_phy_sequence(group_files['group1'])
            out_seq = read_phy_sequence(group_files['outgroup'])
            
            if not (seq0 and seq1 and out_seq and len(seq0) == len(seq1) == len(out_seq)):
                logger.add("Final Comparison Error", f"Length mismatch for {identifier[0]}. This should not happen now.")
                continue

            comparable_sets += 1
            n = len(seq0)
            diffs['g0_g1'].append(sum(1 for a, b in zip(seq0, seq1) if a.upper() != b.upper()) / n * 100)
            diffs['g0_out'].append(sum(1 for a, b in zip(seq0, out_seq) if a.upper() != b.upper()) / n * 100)
            diffs['g1_out'].append(sum(1 for a, b in zip(seq1, out_seq) if a.upper() != b.upper()) / n * 100)

    if comparable_sets == 0:
        print("CRITICAL: No complete sets found to compare after processing."); return

    print(f"Successfully compared {comparable_sets} complete CDS sets.")
    
    avg_g0_g1 = sum(diffs['g0_g1']) / len(diffs['g0_g1']) if diffs['g0_g1'] else 0
    avg_g0_out = sum(diffs['g0_out']) / len(diffs['g0_out']) if diffs['g0_out'] else 0
    avg_g1_out = sum(diffs['g1_out']) / len(diffs['g1_out']) if diffs['g1_out'] else 0

    print("\nAverage Pairwise Sequence Difference (%)\n")
    print(f"{'':<10} {'group0':<10} {'group1':<10} {'outgroup':<10}")
    print("-" * 45)
    print(f"{'group0':<10} {'-':<10} {avg_g0_g1:<10.4f} {avg_g0_out:<10.4f}")
    print(f"{'group1':<10} {'-':<10} {'-':<10} {avg_g1_out:<10.4f}")
    print(f"{'outgroup':<10} {'-':<10} {'-':<10} {'-':<10}\n")

def main():
    print("--- Starting Corrected Chimp CDS Phylip Generation ---")
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
