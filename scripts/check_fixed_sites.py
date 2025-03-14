import pandas as pd
import subprocess
import os
import re
import sys
from collections import defaultdict
import time
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored output
init()

def extract_chromosome(chrom_field):
    """
    Extract just the chromosome name from a string like 'chr15_start24675868'.
    We match a pattern like 'chr' followed by digits or X/Y/M.
    """
    match = re.match(r'(chr(?:\d+|X|Y|M))', str(chrom_field))
    if match:
        return match.group(1)
    return chrom_field

def check_vcf_indexed(vcf_path):
    """
    Check if a VCF file is indexed (with .tbi or .csi).
    """
    tbi_path = vcf_path + ".tbi"
    csi_path = vcf_path + ".csi"
    return os.path.exists(tbi_path) or os.path.exists(csi_path)

def find_position_in_vcf(chr_name, position, vcf_path):
    """
    Look up a genomic position in a VCF file (which may or may not be indexed).
    Return:
      (found_in_file, exact_match, nearest_position, distance, vcf_line)
    where:
      found_in_file: bool
      exact_match: bool
      nearest_position: int or None
      distance: int or None
      vcf_line: str or a message
    """
    position = int(position)
    is_indexed = check_vcf_indexed(vcf_path)
    
    try:
        if is_indexed:
            # First, exact position
            cmd = f"tabix {vcf_path} {chr_name}:{position}-{position}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            lines = result.stdout.strip()
            if lines:
                return True, True, position, 0, lines
            
            # If not found, search for nearest in a 1000bp window
            window = 1000
            cmd_window = f"tabix {vcf_path} {chr_name}:{max(1, position-window)}-{position+window}"
            w_res = subprocess.run(cmd_window, shell=True, capture_output=True, text=True)
            w_lines = w_res.stdout.strip().split('\n') if w_res.stdout.strip() else []
            
            if w_lines:
                nearest = ""
                nearest_pos = 0
                min_d = float('inf')
                for line in w_lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pos = int(parts[1])
                        dist = abs(pos - position)
                        if dist < min_d:
                            min_d = dist
                            nearest_pos = pos
                            nearest = line
                return True, False, nearest_pos, min_d, nearest
            
            # If no result, search a 5000bp window
            wide = 5000
            cmd_wide = f"tabix {vcf_path} {chr_name}:{max(1, position-wide)}-{position+wide}"
            wide_res = subprocess.run(cmd_wide, shell=True, capture_output=True, text=True)
            wide_lines = wide_res.stdout.strip().split('\n') if wide_res.stdout.strip() else []
            
            if wide_lines:
                nearest = ""
                nearest_pos = 0
                min_d = float('inf')
                for line in wide_lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pos = int(parts[1])
                        dist = abs(pos - position)
                        if dist < min_d:
                            min_d = dist
                            nearest_pos = pos
                            nearest = line
                return True, False, nearest_pos, min_d, nearest
            
            return False, False, None, None, "No positions found in the VCF for this region"
        
        else:
            # Non-indexed VCF handling
            # 1) exact
            exact_cmd = f"zgrep -m 1 '^{chr_name}[[:space:]]+{position}[[:space:]]' {vcf_path}"
            exact_res = subprocess.run(exact_cmd, shell=True, capture_output=True, text=True)
            if exact_res.stdout.strip():
                return True, True, position, 0, exact_res.stdout.strip()
            
            # 2) nearest
            find_nearest_cmd = f"""
            zcat {vcf_path} | awk -v target={position} '
            BEGIN {{min_diff = 1000000000; closest = ""}}
            !/^#/ && $1=="{chr_name}" {{
                diff = $2 - target;
                if (diff < 0) diff = -diff;
                if (diff < min_diff) {{
                    min_diff = diff;
                    closest = $0;
                    closest_pos = $2;
                }}
            }}
            END {{
                if (closest != "") {{
                    print closest;
                    print "NEAREST_POS=" closest_pos;
                    print "DISTANCE=" min_diff;
                }} else {{
                    print "No positions found for chromosome {chr_name}";
                }}
            }}
            """
            near_res = subprocess.run(find_nearest_cmd, shell=True, capture_output=True, text=True)
            output = near_res.stdout.strip()
            
            if "No positions found" in output:
                return False, False, None, None, "No positions found in the VCF for this chromosome"
            
            pos_m = re.search(r'NEAREST_POS=(\d+)', output)
            dist_m = re.search(r'DISTANCE=(\d+)', output)
            if pos_m and dist_m:
                n_pos = int(pos_m.group(1))
                dist = int(dist_m.group(1))
                v_line = output.split("NEAREST_POS=")[0].strip()
                return True, False, n_pos, dist, v_line
            
            return False, False, None, None, "Failed to parse nearest position information"
    
    except Exception as e:
        return False, False, None, None, f"Error searching VCF: {str(e)}"

def format_vcf_line(vcf_line, highlight_pos=True):
    """
    For a single VCF line of text, colorize the chromosome, position,
    reference, and alternate allele columns for better CLI readability.
    """
    if not vcf_line or (isinstance(vcf_line, str) and "Error" in vcf_line):
        return vcf_line
    parts = vcf_line.split('\t')
    if len(parts) < 3:
        return vcf_line
    formatted = []
    for i, part in enumerate(parts):
        if i == 0:  # chrom
            formatted.append(f"{Fore.CYAN}{part}{Style.RESET_ALL}")
        elif i == 1 and highlight_pos:  # position
            formatted.append(f"{Fore.YELLOW}{part}{Style.RESET_ALL}")
        elif i == 3:  # ref
            formatted.append(f"{Fore.GREEN}{part}{Style.RESET_ALL}")
        elif i == 4:  # alt
            formatted.append(f"{Fore.RED}{part}{Style.RESET_ALL}")
        else:
            formatted.append(part)
    return '\t'.join(formatted)

def find_vcf_file(chrom_name, vcf_dir):
    """
    Attempt to locate a VCF file for the given chromosome by building a standard
    filename pattern, or by scanning the directory for something that matches
    the chromosome prefix.
    """
    clean_chrom = extract_chromosome(chrom_name)
    vcf_pattern = f"{clean_chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
    candidate = os.path.join(vcf_dir, vcf_pattern)
    if os.path.exists(candidate):
        return candidate, clean_chrom
    try:
        listing = os.listdir(vcf_dir)
        matches = [f for f in listing if f.startswith(clean_chrom) and f.endswith(".vcf.gz")]
        if matches:
            return os.path.join(vcf_dir, matches[0]), clean_chrom
    except:
        pass
    return None, clean_chrom

def validate_positions(csv_file="fixed_differences.csv", vcf_dir="../vcfs"):
    """
    Validate the genomic positions in the CSV file (which now contains raw genomic
    coordinates) against the appropriate VCF. We will search each position, attempt
    to find an exact or nearest match in the VCF, and summarize the results.
    """
    print(f"{Fore.BLUE}=== VCF Position Validator (Raw Genomic Coordinates) ==={Style.RESET_ALL}")
    print(f"CSV file: {csv_file}")
    print(f"VCF directory: {vcf_dir}\n")
    
    if not os.path.exists(csv_file):
        print(f"{Fore.RED}Error: CSV file '{csv_file}' not found.{Style.RESET_ALL}")
        return
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"{Fore.RED}Error reading CSV: {str(e)}{Style.RESET_ALL}")
        return
    
    # Peek at the data
    print("Sample rows from CSV:")
    print(df.head(3).to_string())
    print()
    
    # Filter out rows that have "No fixed differences" or "Error" in the Position field
    valid_rows = df[~df['Position'].astype(str).str.contains('No fixed differences|Error', case=False)]
    if valid_rows.empty:
        print(f"{Fore.YELLOW}No valid positions found in {csv_file}{Style.RESET_ALL}")
        return
    
    print(f"Found {len(valid_rows)} positions to check.\n")
    
    # We'll track stats
    total_positions = 0
    found_exact = 0
    found_nearby = 0
    not_found = 0
    
    # Remember which chromosome's VCF we've already looked up
    processed_chroms = {}
    
    for idx, row in valid_rows.iterrows():
        gene = row.get('Gene', 'N/A')
        chrom = row.get('Chromosome', '')
        try:
            pos = int(row.get('Position', ''))
        except:
            continue  # skip if not int
        g0_nuc = row.get('Group0_Nucleotide', 'N/A')
        g1_nuc = row.get('Group1_Nucleotide', 'N/A')
        
        if not isinstance(chrom, str) or not chrom:
            continue
        
        # Look for the VCF
        cleaned_chrom = extract_chromosome(chrom)
        if cleaned_chrom in processed_chroms:
            vcf_file, disp_chrom = processed_chroms[cleaned_chrom]
        else:
            vcf_file, disp_chrom = find_vcf_file(cleaned_chrom, vcf_dir)
            processed_chroms[cleaned_chrom] = (vcf_file, disp_chrom)
        
        if not vcf_file:
            print(f"{Fore.RED}No VCF file found for chromosome {cleaned_chrom}{Style.RESET_ALL}")
            continue
        
        total_positions += 1
        position_desc = f"{disp_chrom}:{pos}"
        print(f"{Fore.BLUE}Checking position {position_desc} ({gene}){Style.RESET_ALL}")
        print(f"  Fixed difference: {Fore.GREEN}{g0_nuc}{Style.RESET_ALL} vs {Fore.RED}{g1_nuc}{Style.RESET_ALL}")
        
        found, exact, nearest_pos, distance, vline = find_position_in_vcf(disp_chrom, pos, vcf_file)
        if found and exact:
            found_exact += 1
            print(f"  {Fore.GREEN}Exact position found{Style.RESET_ALL}")
            print(format_vcf_line(vline))
        elif found:
            found_nearby += 1
            print(f"  {Fore.YELLOW}Position not found, nearest is {distance} bp away at {nearest_pos}{Style.RESET_ALL}")
            print(format_vcf_line(vline))
        else:
            not_found += 1
            print(f"  {Fore.RED}Position not found in VCF{Style.RESET_ALL}")
            print(vline)
        
        print()
    
    # Summaries
    print(f"{Fore.BLUE}=== Summary ==={Style.RESET_ALL}")
    print(f"Total positions checked: {total_positions}")
    print(f"Exact matches: {found_exact}")
    print(f"Nearest found: {found_nearby}")
    print(f"Not found: {not_found}")
    if total_positions:
        ex_pct = found_exact / total_positions * 100
        nr_pct = found_nearby / total_positions * 100
        nf_pct = not_found / total_positions * 100
        print(f"Exact: {ex_pct:.1f}%  Near: {nr_pct:.1f}%  Not found: {nf_pct:.1f}%")

def main():
    csv_file = "fixed_differences.csv"
    vcf_dir = "../vcfs"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        vcf_dir = sys.argv[2]
    
    validate_positions(csv_file, vcf_dir)

if __name__ == "__main__":
    main()
