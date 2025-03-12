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
    match = re.match(r'(chr(?:\d+|X|Y|M))', str(chrom_field))
    if match:
        return match.group(1)
    return chrom_field  # Return original if no match

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
            exact_cmd = f"zgrep -m 1 '^{chr_name}[[:space:]]+{position}[[:space:]]' {vcf_path}"
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
    clean_chrom = extract_chromosome(chrom_name)
    
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

def extract_transcript_info_from_fullgeneid(field):
    """
    Extract transcript ID, gene ID, and position info from FullGeneID field.
    Format example: ABCC1_ENSG00000103222.20_ENST00000399410.8_chr16_start15949752_end16141277
    """
    if not isinstance(field, str):
        return None, None, None, None, None
        
    parts = field.split('_')
    
    # Initialize variables
    gene_name = None
    gene_id = None
    transcript_id = None
    start_pos = None
    end_pos = None
    
    for part in parts:
        if part.startswith("ENSG"):
            gene_id = part.split('.')[0]  # Remove version number
        elif part.startswith("ENST"):
            transcript_id = part.split('.')[0]  # Remove version number
        elif part.startswith("start"):
            start_pos = int(part[5:])  # Extract numeric part after "start"
        elif part.startswith("end"):
            end_pos = int(part[3:])  # Extract numeric part after "end"
    
    # First part is typically the gene name
    if len(parts) > 0:
        gene_name = parts[0]
            
    return gene_name, gene_id, transcript_id, start_pos, end_pos

def parse_gtf_for_cds_regions(gtf_file):
    """
    Parse the GTF file to build a map of CDS regions for each transcript.
    
    Returns:
        tuple: (cds_regions, transcript_info, gene_to_transcripts)
               cds_regions: transcript_id -> list of (start, end) tuples for CDS regions
               transcript_info: transcript_id -> dict with gene info, strand
               gene_to_transcripts: gene_name -> list of transcript_ids
    """
    print(f"{Fore.BLUE}Parsing GTF file for CDS regions: {gtf_file}{Style.RESET_ALL}")
    
    cds_regions = defaultdict(list)
    transcript_info = {}  # Store additional transcript info
    gene_to_transcripts = defaultdict(list)
    
    try:
        with open(gtf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 9 or fields[2] != 'CDS':
                    continue
                    
                # Extract basic information
                chromosome = fields[0]
                start = int(fields[3])  # 1-based inclusive
                end = int(fields[4])    # 1-based inclusive
                strand = fields[6]
                attributes = fields[8]
                
                # Parse attributes
                transcript_id_match = re.search(r'transcript_id "([^"]+)"', attributes)
                gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
                gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
                
                if not transcript_id_match:
                    continue
                    
                # Extract IDs without version numbers
                transcript_id = transcript_id_match.group(1).split('.')[0]
                gene_id = gene_id_match.group(1).split('.')[0] if gene_id_match else None
                gene_name = gene_name_match.group(1) if gene_name_match else None
                
                # Store the CDS region
                cds_regions[transcript_id].append((start, end))
                
                # Store transcript info if not already stored
                if transcript_id not in transcript_info:
                    transcript_info[transcript_id] = {
                        'chromosome': chromosome,
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'strand': strand
                    }
                    
                # Map gene name to transcript
                if gene_name and transcript_id not in gene_to_transcripts[gene_name]:
                    gene_to_transcripts[gene_name].append(transcript_id)
        
        # Sort CDS regions by start position for each transcript
        for transcript_id, regions in cds_regions.items():
            cds_regions[transcript_id] = sorted(regions)
        
        print(f"  Parsed CDS regions for {len(cds_regions)} transcripts across {len(gene_to_transcripts)} genes")
        return cds_regions, transcript_info, gene_to_transcripts
        
    except Exception as e:
        print(f"{Fore.RED}Error parsing GTF file: {str(e)}{Style.RESET_ALL}")
        return {}, {}, {}

def build_genomic_to_spliced_map(cds_regions, transcript_info):
    """
    Build a mapping from genomic positions to positions in the spliced CDS.
    
    Args:
        cds_regions: dict mapping transcript_id -> list of (start, end) tuples
        transcript_info: dict mapping transcript_id -> transcript metadata
        
    Returns:
        dict mapping transcript_id -> dict of genomic_position -> spliced_position
    """
    genomic_to_spliced = {}
    
    for transcript_id, regions in cds_regions.items():
        if transcript_id not in transcript_info:
            continue
            
        strand = transcript_info[transcript_id]['strand']
        transcript_map = {}
        
        # Sort regions by start position
        sorted_regions = sorted(regions)
        
        # For negative strand, we need to reverse the order of regions
        if strand == '-':
            sorted_regions.reverse()
        
        # Build the mapping
        spliced_pos = 1  # 1-based
        
        for start, end in sorted_regions:
            segment_length = end - start + 1
            
            if strand == '+':
                # For positive strand, map in ascending order
                for i in range(segment_length):
                    genomic_pos = start + i
                    transcript_map[genomic_pos] = spliced_pos + i
            else:
                # For negative strand, map in descending order
                for i in range(segment_length):
                    genomic_pos = end - i
                    transcript_map[genomic_pos] = spliced_pos + i
                    
            spliced_pos += segment_length
            
        genomic_to_spliced[transcript_id] = transcript_map
        
    return genomic_to_spliced

def genomic_to_spliced_position(genomic_pos, transcript_id, genomic_to_spliced):
    """
    Convert a genomic position to its corresponding position in the spliced CDS.
    
    Args:
        genomic_pos: Genomic position (1-based)
        transcript_id: Transcript ID
        genomic_to_spliced: Mapping from genomic to spliced positions
        
    Returns:
        int or None: Position in spliced CDS, or None if position is not in CDS
    """
    if transcript_id not in genomic_to_spliced:
        return None
        
    transcript_map = genomic_to_spliced[transcript_id]
    return transcript_map.get(genomic_pos)

def validate_positions(csv_file="fixed_differences.csv", vcf_dir="../vcfs", gtf_file="gencode.v47.basic.annotation.gtf"):
    """Validate positions from fixed_differences.csv against VCF files using GTF for spliced coordinate mapping."""
    print(f"{Fore.BLUE}=== VCF Position Validator (With Spliced CDS Coordinate Mapping) ==={Style.RESET_ALL}")
    print(f"Checking positions from {csv_file} against VCF files in {vcf_dir}\n")
    print(f"Using GTF file {gtf_file} for spliced CDS coordinate mapping\n")
    
    # Make sure files exist
    if not os.path.exists(csv_file):
        print(f"{Fore.RED}Error: CSV file {csv_file} not found{Style.RESET_ALL}")
        return
        
    if not os.path.exists(gtf_file):
        print(f"{Fore.RED}Error: GTF file {gtf_file} not found{Style.RESET_ALL}")
        return
    
    # Parse GTF file to get CDS regions
    cds_regions, transcript_info, gene_to_transcripts = parse_gtf_for_cds_regions(gtf_file)
    
    if not cds_regions:
        print(f"{Fore.RED}Error: Failed to parse CDS regions from GTF{Style.RESET_ALL}")
        return
        
    # Build genomic to spliced CDS position mapping
    print(f"{Fore.BLUE}Building genomic to spliced CDS position mapping...{Style.RESET_ALL}")
    genomic_to_spliced = build_genomic_to_spliced_map(cds_regions, transcript_info)
    print(f"  Built mapping for {len(genomic_to_spliced)} transcripts")
    
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
        positions_in_cds = 0
        found_exact = 0
        found_nearby = 0
        not_found = 0
        
        # Track chromosomes we've already processed
        processed_chroms = {}
        
        # Process each row
        for _, row in valid_rows.iterrows():
            gene_name = row['Gene']
            full_gene_id = row['FullGeneID'] if 'FullGeneID' in row else None
            chromosome = row['Chromosome']
            position = row['Position']
            g0_nuc = row['Group0_Nucleotide']
            g1_nuc = row['Group1_Nucleotide']
            
            # Skip invalid rows
            if not isinstance(chromosome, str) or chromosome.lower() == 'nan':
                continue
                
            # For positions that might be encoded in the chromosome field
            if isinstance(position, str) and ("No fixed" in position or "Error" in position):
                continue
                
            print(f"\n{Fore.BLUE}Processing entry for gene: {gene_name}{Style.RESET_ALL}")
            
            # Extract transcript information from FullGeneID
            extracted_gene_name = None
            gene_id = None
            transcript_id = None
            cds_start = None
            cds_end = None
            
            if full_gene_id and isinstance(full_gene_id, str):
                extracted_gene_name, gene_id, transcript_id, cds_start, cds_end = extract_transcript_info_from_fullgeneid(full_gene_id)
                print(f"  FullGeneID info: Gene={extracted_gene_name}, TranscriptID={transcript_id}, CDSRange={cds_start}-{cds_end}")
            
            # Get transcript IDs for this gene
            transcript_ids = []
            
            # First try the specific transcript from FullGeneID
            if transcript_id:
                transcript_ids.append(transcript_id)
                
            # Then add any additional transcripts for this gene
            if gene_name in gene_to_transcripts:
                for t_id in gene_to_transcripts[gene_name]:
                    if t_id not in transcript_ids:
                        transcript_ids.append(t_id)
            
            if not transcript_ids:
                print(f"  {Fore.YELLOW}No transcripts found for gene {gene_name}{Style.RESET_ALL}")
                
            # Clean up chromosome and position
            clean_chrom = extract_chromosome(chromosome)
            
            try:
                position = int(position)
            except (ValueError, TypeError):
                print(f"  {Fore.RED}Invalid position format: {position}{Style.RESET_ALL}")
                continue
            
            # Map genomic position to spliced CDS position
            best_spliced_position = None
            best_transcript = None
            
            for t_id in transcript_ids:
                spliced_pos = genomic_to_spliced_position(position, t_id, genomic_to_spliced)
                if spliced_pos:
                    best_spliced_position = spliced_pos
                    best_transcript = t_id
                    break
                    
            # If position maps to a spliced CDS position, report it
            if best_spliced_position:
                print(f"  {Fore.GREEN}✓ Genomic position {position} maps to spliced CDS position {best_spliced_position} in transcript {best_transcript}{Style.RESET_ALL}")
                positions_in_cds += 1
                
                # Show CDS regions for the transcript
                if best_transcript in cds_regions:
                    regions_str = ", ".join([f"{start}-{end}" for start, end in cds_regions[best_transcript]])
                    print(f"  CDS regions for {best_transcript}: {regions_str}")
                    
                    # Find which segment contains the position
                    for i, (start, end) in enumerate(cds_regions[best_transcript]):
                        if start <= position <= end:
                            print(f"  Position is in CDS segment {i+1}: {start}-{end}")
                            break
            else:
                # If not in CDS, check if it's in the overall range but in an intron
                if cds_start and cds_end and cds_start <= position <= cds_end:
                    print(f"  {Fore.YELLOW}⚠ Position {position} is within overall CDS range {cds_start}-{cds_end}, but in an intron{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.RED}✗ Position {position} is not in any CDS region for gene {gene_name}{Style.RESET_ALL}")
                
                # Show CDS regions for reference
                if transcript_ids:
                    for t_id in transcript_ids[:2]:  # Limit to first two transcripts
                        if t_id in cds_regions:
                            regions_str = ", ".join([f"{start}-{end}" for start, end in cds_regions[t_id]])
                            print(f"  CDS regions for {t_id}: {regions_str}")
            
            # Find the VCF file for this chromosome
            if clean_chrom in processed_chroms:
                vcf_file, display_chrom = processed_chroms[clean_chrom]
            else:
                vcf_file, display_chrom = find_vcf_file(clean_chrom, vcf_dir)
                processed_chroms[clean_chrom] = (vcf_file, display_chrom)
            
            if not vcf_file:
                print(f"  {Fore.RED}VCF file not found for chromosome {clean_chrom}{Style.RESET_ALL}")
                continue
                
            # Only show the chromosome header once per chromosome
            if clean_chrom not in processed_chroms:
                print(f"\n{Fore.BLUE}=== Using VCF file: {os.path.basename(vcf_file)} ==={Style.RESET_ALL}")
                
                # Check if VCF is indexed
                indexed = check_vcf_indexed(vcf_file)
                if not indexed:
                    print(f"  {Fore.YELLOW}Warning: VCF file is not indexed. Searches may be slow.{Style.RESET_ALL}")
            
            # Process this position
            total_positions += 1
            
            position_description = f"{display_chrom}:{position}"
            if best_spliced_position:
                position_description += f" (maps to spliced CDS position {best_spliced_position})"
                
            print(f"\n{Fore.CYAN}► Position {position_description} ({gene_name}){Style.RESET_ALL}")
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
        print(f"Positions in CDS regions: {positions_in_cds} ({positions_in_cds/total_positions*100:.1f}%)")
        print(f"Positions in VCF:")
        print(f"  Exact matches: {found_exact} ({found_exact/total_positions*100:.1f}%)")
        print(f"  Nearest position found: {found_nearby} ({found_nearby/total_positions*100:.1f}%)")
        print(f"  No matches found: {not_found} ({not_found/total_positions*100:.1f}%)")
        
        # Calculate correlation between CDS positions and VCF matches
        if positions_in_cds > 0:
            # Count positions that are in CDS and have exact VCF match
            cds_and_exact = sum(1 for i, row in valid_rows.iterrows() 
                              if row['Position'] in [pos for transcript in genomic_to_spliced.values() 
                                                    for pos in transcript.keys()])
            
            print(f"\nCorrelation between CDS regions and VCF matches:")
            print(f"  {cds_and_exact/positions_in_cds*100:.1f}% of positions in CDS regions had exact VCF matches")
        
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Get command line arguments if provided
    csv_file = "fixed_differences.csv"
    vcf_dir = "../vcfs"
    gtf_file = "gencode.v47.basic.annotation.gtf"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        vcf_dir = sys.argv[2]
    if len(sys.argv) > 3:
        gtf_file = sys.argv[3]
        
    validate_positions(csv_file, vcf_dir, gtf_file)
