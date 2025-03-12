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
    """Extract just the chromosome name from a string like 'chr15_start24675868'."""
    # Match specifically chr followed by digits or X/Y/M
    match = re.match(r'(chr(?:\d+|X|Y|M))', chrom_field)
    if match:
        return match.group(1)
    return chrom_field  # Return original if no match

def extract_position_from_field(pos_field):
    """
    Extract position from a field like 'chr15_start24675868'.
    Returns None if it's not a position field.
    """
    # Check if the field is already just a position (integer or string of digits)
    if isinstance(pos_field, (int, float)) or (isinstance(pos_field, str) and pos_field.isdigit()):
        return pos_field
        
    # Try to extract position from start coordinate in the chromosome field
    match = re.search(r'_start(\d+)', str(pos_field))
    if match:
        return int(match.group(1))
        
    # If we can't parse it as a position, return None
    if isinstance(pos_field, str) and ("No fixed" in pos_field or "Error" in pos_field):
        return None
        
    return pos_field  # Return original if no match but not an error message

def check_vcf_indexed(vcf_path):
    """Check if a VCF file is indexed (with .tbi or .csi)."""
    tbi_path = vcf_path + ".tbi"
    csi_path = vcf_path + ".csi"
    return os.path.exists(tbi_path) or os.path.exists(csi_path)

def find_position_in_vcf(chr_name, position, vcf_path):
    """
    Check if a position exists in a VCF file.
    
    Returns:
        tuple: (exists, exact_match, nearest_position, distance, vcf_line)
    """
    position = int(position)
    is_indexed = check_vcf_indexed(vcf_path)
    
    # First try to find the exact position
    try:
        if is_indexed:
            # Use tabix for indexed files (faster)
            cmd = f"tabix {vcf_path} {chr_name}:{position}-{position}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout.strip():
                return True, True, position, 0, result.stdout.strip()
                
            # If not found, search for nearest positions
            # Try a window of 1000bp
            window = 1000
            window_cmd = f"tabix {vcf_path} {chr_name}:{max(1, position-window)}-{position+window}"
            window_result = subprocess.run(window_cmd, shell=True, capture_output=True, text=True)
            
            if window_result.stdout.strip():
                lines = window_result.stdout.strip().split('\n')
                nearest_line = ""
                nearest_pos = 0
                min_distance = float('inf')
                
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pos = int(parts[1])
                        distance = abs(pos - position)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_pos = pos
                            nearest_line = line
                
                return True, False, nearest_pos, min_distance, nearest_line
            
            # Try an even wider window if no results (5000bp)
            wider_window = 5000
            wider_cmd = f"tabix {vcf_path} {chr_name}:{max(1, position-wider_window)}-{position+wider_window}"
            wider_result = subprocess.run(wider_cmd, shell=True, capture_output=True, text=True)
            
            if wider_result.stdout.strip():
                lines = wider_result.stdout.strip().split('\n')
                nearest_line = ""
                nearest_pos = 0
                min_distance = float('inf')
                
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pos = int(parts[1])
                        distance = abs(pos - position)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_pos = pos
                            nearest_line = line
                
                return True, False, nearest_pos, min_distance, nearest_line
            
            # No nearby positions found
            return False, False, None, None, "No positions found in the VCF for this region"
            
        else:
            # For non-indexed files, first check if the exact position exists
            exact_cmd = f"zgrep -m 1 '^{chr_name}\\s\\+{position}\\s' {vcf_path}"
            exact_result = subprocess.run(exact_cmd, shell=True, capture_output=True, text=True)
            
            if exact_result.stdout.strip():
                return True, True, position, 0, exact_result.stdout.strip()
            
            # If not, find the nearest position
            # This might be slow for large VCFs
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
            }}'
            """
            
            nearest_result = subprocess.run(find_nearest_cmd, shell=True, capture_output=True, text=True)
            output = nearest_result.stdout.strip()
            
            if "No positions found" in output:
                return False, False, None, None, "No positions found in the VCF for this chromosome"
            
            # Extract nearest position and distance
            pos_match = re.search(r'NEAREST_POS=(\d+)', output)
            dist_match = re.search(r'DISTANCE=(\d+)', output)
            
            if pos_match and dist_match:
                nearest_pos = int(pos_match.group(1))
                distance = int(dist_match.group(1))
                vcf_line = output.split("NEAREST_POS=")[0].strip()
                return True, False, nearest_pos, distance, vcf_line
            
            return False, False, None, None, "Failed to parse nearest position information"
            
    except Exception as e:
        return False, False, None, None, f"Error searching VCF: {str(e)}"

def format_vcf_line(vcf_line, highlight_pos=True):
    """Format a VCF line for display, optionally highlighting the position column."""
    if not vcf_line or isinstance(vcf_line, str) and "Error" in vcf_line:
        return vcf_line
        
    parts = vcf_line.split('\t')
    if len(parts) < 3:
        return vcf_line
        
    formatted_parts = []
    for i, part in enumerate(parts):
        if i == 0:  # Chromosome
            formatted_parts.append(f"{Fore.CYAN}{part}{Style.RESET_ALL}")
        elif i == 1 and highlight_pos:  # Position
            formatted_parts.append(f"{Fore.YELLOW}{part}{Style.RESET_ALL}")
        elif i == 3:  # Reference allele
            formatted_parts.append(f"{Fore.GREEN}{part}{Style.RESET_ALL}")
        elif i == 4:  # Alternative allele
            formatted_parts.append(f"{Fore.RED}{part}{Style.RESET_ALL}")
        else:
            formatted_parts.append(part)
            
    return '\t'.join(formatted_parts)

def find_vcf_file(chrom_name, vcf_dir):
    """Find the matching VCF file for a chromosome."""
    # Extract just the chromosome part (e.g., "chr15" from "chr15_start24675868")
    clean_chrom = extract_chromosome(chrom_name)
    
    if clean_chrom == chrom_name:  # No change was made
        print(f"{Fore.YELLOW}Warning: Could not extract clean chromosome from {chrom_name}{Style.RESET_ALL}")
    
    # Build the expected VCF filename pattern
    vcf_pattern = f"{clean_chrom}.fixedPH.simpleINV.mod.all.wAA.myHardMask98pc.vcf.gz"
    vcf_path = os.path.join(vcf_dir, vcf_pattern)
    
    # Check if the file exists
    if os.path.exists(vcf_path):
        return vcf_path, clean_chrom
    
    # If not found, try to find any VCF file matching the chromosome
    try:
        all_vcfs = [f for f in os.listdir(vcf_dir) if f.startswith(clean_chrom) and f.endswith('.vcf.gz')]
        
        if all_vcfs:
            return os.path.join(vcf_dir, all_vcfs[0]), clean_chrom
    except:
        pass
    
    return None, clean_chrom

def validate_positions(csv_file="fixed_differences.csv", vcf_dir="../vcfs"):
    """Validate positions from fixed_differences.csv against VCF files."""
    print(f"{Fore.BLUE}=== VCF Position Validator ==={Style.RESET_ALL}")
    print(f"Checking positions from {csv_file} against VCF files in {vcf_dir}\n")
    
    # Make sure the CSV file exists
    if not os.path.exists(csv_file):
        print(f"{Fore.RED}Error: CSV file {csv_file} not found{Style.RESET_ALL}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        
        # Print the first few rows to debug
        print("Sample data from CSV:")
        print(df.head(2).to_string())
        print()
        
        # Filter out rows with errors or no fixed differences
        valid_rows = df[~df['Position'].astype(str).str.contains('No fixed differences|Error', case=False)]
        
        if valid_rows.empty:
            print(f"{Fore.YELLOW}No valid positions found in {csv_file}{Style.RESET_ALL}")
            return
            
        print(f"Found {len(valid_rows)} positions to check")
        
        # Process each row individually for more control
        total_positions = 0
        found_exact = 0
        found_nearby = 0
        not_found = 0
        
        # Track chromosomes we've already processed
        processed_chroms = {}
        
        # Process each row
        for _, row in valid_rows.iterrows():
            chromosome = row['Chromosome']
            position = row['Position']
            gene = row['Gene']
            g0_nuc = row['Group0_Nucleotide']
            g1_nuc = row['Group1_Nucleotide']
            
            # Skip invalid rows
            if not isinstance(chromosome, str) or chromosome.lower() == 'nan':
                continue
                
            # Clean up chromosome and position if needed
            clean_chrom = extract_chromosome(chromosome)
            
            # For positions that might be encoded in the chromosome field
            if isinstance(position, str) and ("No fixed" in position or "Error" in position):
                continue
                
            # Certain rows may have position embedded in the Chromosome field
            if not isinstance(position, (int, float)) and not (isinstance(position, str) and position.isdigit()):
                pos_from_chrom = extract_position_from_field(chromosome)
                if pos_from_chrom:
                    position = pos_from_chrom
                    print(f"{Fore.YELLOW}Extracted position {position} from chromosome field{Style.RESET_ALL}")
            
            # Try to convert position to int
            try:
                position = int(position)
            except (ValueError, TypeError):
                print(f"{Fore.RED}Invalid position {position} for {gene} on {clean_chrom}{Style.RESET_ALL}")
                continue
            
            # Find the VCF file for this chromosome (only look it up once per chromosome)
            if clean_chrom in processed_chroms:
                vcf_file, display_chrom = processed_chroms[clean_chrom]
            else:
                vcf_file, display_chrom = find_vcf_file(clean_chrom, vcf_dir)
                processed_chroms[clean_chrom] = (vcf_file, display_chrom)
            
            if not vcf_file:
                print(f"{Fore.RED}VCF file not found for {chromosome} (cleaned to {clean_chrom}){Style.RESET_ALL}")
                continue
                
            # Only show the chromosome header once
            if clean_chrom not in processed_chroms:
                print(f"\n{Fore.BLUE}=== Checking positions on {display_chrom} ==={Style.RESET_ALL}")
                print(f"Using VCF file: {os.path.basename(vcf_file)}")
                
                # Check if VCF is indexed
                indexed = check_vcf_indexed(vcf_file)
                if not indexed:
                    print(f"{Fore.YELLOW}Warning: VCF file is not indexed. Searches may be slow.{Style.RESET_ALL}")
            
            # Process this position
            total_positions += 1
            
            print(f"\n{Fore.CYAN}► Position {display_chrom}:{position} ({gene}){Style.RESET_ALL}")
            print(f"  Fixed difference: Group0={Fore.GREEN}{g0_nuc}{Style.RESET_ALL}, Group1={Fore.RED}{g1_nuc}{Style.RESET_ALL}")
            
            # Check if position exists in VCF
            found, exact, nearest_pos, distance, vcf_line = find_position_in_vcf(display_chrom, position, vcf_file)
            
            if found and exact:
                print(f"  {Fore.GREEN}✓ Exact position found in VCF{Style.RESET_ALL}")
                print(f"  {format_vcf_line(vcf_line)}")
                found_exact += 1
            elif found:
                print(f"  {Fore.YELLOW}⚠ Position not found, nearest is {distance} bp away at {nearest_pos}{Style.RESET_ALL}")
                print(f"  {format_vcf_line(vcf_line)}")
                found_nearby += 1
            else:
                print(f"  {Fore.RED}✗ Position not found in VCF{Style.RESET_ALL}")
                print(f"  {vcf_line}")
                not_found += 1
        
        # Print summary
        print(f"\n{Fore.BLUE}=== Summary ==={Style.RESET_ALL}")
        print(f"Total positions checked: {total_positions}")
        if total_positions > 0:
            print(f"Exact matches: {found_exact} ({found_exact/total_positions*100:.1f}%)")
            print(f"Nearest position found: {found_nearby} ({found_nearby/total_positions*100:.1f}%)")
            print(f"No matches found: {not_found} ({not_found/total_positions*100:.1f}%)")
        else:
            print("No positions were checked.")
        
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Get command line arguments if provided
    csv_file = "fixed_differences.csv"
    vcf_dir = "../vcfs"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        vcf_dir = sys.argv[2]
        
    validate_positions(csv_file, vcf_dir)
