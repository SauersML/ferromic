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

def extract_transcript_info_from_fullgeneid(field):
    """
    Extract transcript ID, gene ID, and position info from FullGeneID field.
    Format example: ABCC1_ENSG00000103222.20_ENST00000399410.8_chr16_start15949752_end16141277
    """
    if not isinstance(field, str):
        return None, None, None, None
        
    parts = field.split('_')
    
    # Initialize variables
    gene_name = None
    gene_id = None
    transcript_id = None
    start_pos = None
    
    for part in parts:
        if part.startswith("ENSG"):
            gene_id = part.split('.')[0]  # Remove version number
        elif part.startswith("ENST"):
            transcript_id = part.split('.')[0]  # Remove version number
        elif part.startswith("start"):
            start_pos = int(part[5:])  # Extract numeric part after "start"
    
    # First part is typically the gene name
    if len(parts) > 0:
        gene_name = parts[0]
            
    return gene_name, gene_id, transcript_id, start_pos

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
    # Extract just the chromosome part (e.g., "chr15" from "chr15_start24675868")
    clean_chrom = extract_chromosome(chrom_name)
    
    # Only show warning if significant cleaning was needed
    if clean_chrom != chrom_name and "_" in chrom_name:
        print(f"Note: Extracted {clean_chrom} from {chrom_name}")
    
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

def parse_gtf_for_cds_mappings(gtf_file):
    """Parse the GTF file to build CDS-to-genome coordinate maps for all transcripts.
    
    Returns:
        dict: A dictionary mapping transcript_ids to their CDS info
        {
            'transcript_id': {
                'chromosome': 'chr15',
                'gene_id': 'ENSG...',
                'gene_name': 'GENE1',
                'strand': '+',
                'cds_segments': [(start1, end1), (start2, end2), ...],  # Genomic coordinates
                'cds_length': total_length_of_cds
            }
        }
    """
    print(f"{Fore.BLUE}Parsing GTF file: {gtf_file}{Style.RESET_ALL}")
    
    # Store CDS information by transcript ID
    transcript_cds_map = {}
    genes_map = {}  # Map gene names to their transcripts
    
    try:
        # First pass: Collect all CDS regions by transcript
        with open(gtf_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 9 or fields[2] != 'CDS':
                    continue
                    
                # Extract basic information
                chromosome = fields[0]
                feature_type = fields[2]
                start = int(fields[3])  # 1-based inclusive
                end = int(fields[4])    # 1-based inclusive
                strand = fields[6]
                attributes = fields[8]
                
                # Parse attributes to get transcript_id, gene_id, gene_name
                transcript_id_match = re.search(r'transcript_id "([^"]+)"', attributes)
                gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
                gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
                
                if not transcript_id_match or not gene_id_match:
                    continue  # Skip if missing required IDs
                    
                transcript_id = transcript_id_match.group(1).split('.')[0]  # Remove version
                gene_id = gene_id_match.group(1).split('.')[0]  # Remove version
                gene_name = gene_name_match.group(1) if gene_name_match else "Unknown"
                
                # Initialize transcript entry if not exists
                if transcript_id not in transcript_cds_map:
                    transcript_cds_map[transcript_id] = {
                        'chromosome': chromosome,
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'strand': strand,
                        'cds_segments': []
                    }
                
                # Add this CDS segment
                transcript_cds_map[transcript_id]['cds_segments'].append((start, end))
                
                # Add to genes map
                if gene_name not in genes_map:
                    genes_map[gene_name] = []
                if transcript_id not in genes_map[gene_name]:
                    genes_map[gene_name].append(transcript_id)
                
        # Second pass: Sort CDS segments and calculate CDS length
        for transcript_id, info in transcript_cds_map.items():
            # Sort segments by start position
            info['cds_segments'].sort()
            
            # Calculate total CDS length
            cds_length = 0
            for start, end in info['cds_segments']:
                cds_length += (end - start + 1)  # +1 because end is inclusive
                
            info['cds_length'] = cds_length
            
            # For debugging: print transcripts with multiple CDS segments (multi-exon)
            if len(info['cds_segments']) > 1:
                print(f"  Multi-exon CDS: {transcript_id} ({info['gene_name']}) - {len(info['cds_segments'])} segments, strand {info['strand']}")
                for i, (start, end) in enumerate(info['cds_segments']):
                    print(f"    Segment {i+1}: {start}-{end} (length: {end-start+1})")
                
        print(f"  Parsed CDS information for {len(transcript_cds_map)} transcripts across {len(genes_map)} genes")
        return transcript_cds_map, genes_map
        
    except Exception as e:
        print(f"{Fore.RED}Error parsing GTF file: {str(e)}{Style.RESET_ALL}")
        return {}, {}

def convert_cds_to_genomic(transcript_id, cds_position, transcript_cds_map):
    """
    Convert a CDS position (relative to the start of the CDS) to its genomic position.
    
    Args:
        transcript_id: The transcript ID
        cds_position: Position within the CDS (1-based)
        transcript_cds_map: The CDS-to-genome mapping dictionary
        
    Returns:
        tuple: (chromosome, genomic_position) or (None, None) if conversion fails
    """
    if transcript_id not in transcript_cds_map:
        print(f"  {Fore.RED}Transcript {transcript_id} not found in GTF{Style.RESET_ALL}")
        return None, None
        
    info = transcript_cds_map[transcript_id]
    chromosome = info['chromosome']
    strand = info['strand']
    segments = info['cds_segments'].copy()  # List of (start, end) tuples in genomic coordinates
    
    # Validate CDS position
    if cds_position < 1 or cds_position > info['cds_length']:
        print(f"  {Fore.RED}CDS position {cds_position} out of range (1-{info['cds_length']}){Style.RESET_ALL}")
        return chromosome, None
    
    # Debug info
    print(f"  Converting CDS position {cds_position} for transcript {transcript_id}")
    print(f"  Strand: {strand}, Total CDS length: {info['cds_length']}")
    print(f"  Segments: {segments}")
    
    # For negative strand, we need to reverse the order of segments for coordinate calculation
    if strand == '-':
        segments.reverse()
        print(f"  Reversed segments for negative strand: {segments}")
    
    # For each segment, check if the CDS position falls within it
    current_pos = 1  # Start from 1 (1-based)
    
    for i, (start, end) in enumerate(segments):
        segment_length = end - start + 1
        segment_end_pos = current_pos + segment_length - 1
        
        print(f"  Segment {i+1}: {start}-{end} (length {segment_length})")
        print(f"  CDS range: {current_pos}-{segment_end_pos}")
        
        if current_pos <= cds_position <= segment_end_pos:
            # Position is in this segment
            offset = cds_position - current_pos
            
            if strand == '+':
                genomic_pos = start + offset
                print(f"  + strand mapping: CDS {cds_position} = genomic {genomic_pos} ({start} + {offset})")
            else:
                genomic_pos = end - offset
                print(f"  - strand mapping: CDS {cds_position} = genomic {genomic_pos} ({end} - {offset})")
                
            return chromosome, genomic_pos
            
        current_pos += segment_length
    
    print(f"  {Fore.RED}Failed to map CDS position {cds_position} to any segment{Style.RESET_ALL}")
    return chromosome, None

def validate_positions(csv_file="fixed_differences.csv", vcf_dir="../vcfs", gtf_file="../gencode.v47.basic.annotation.gtf"):
    """Validate positions from fixed_differences.csv against VCF files using GTF for coordinate mapping."""
    print(f"{Fore.BLUE}=== VCF Position Validator (With GTF Coordinate Mapping) ==={Style.RESET_ALL}")
    print(f"Checking positions from {csv_file} against VCF files in {vcf_dir}\n")
    print(f"Using GTF file {gtf_file} for CDS-to-genome coordinate mapping\n")
    
    # Make sure the CSV file exists
    if not os.path.exists(csv_file):
        print(f"{Fore.RED}Error: CSV file {csv_file} not found{Style.RESET_ALL}")
        return
        
    # Make sure the GTF file exists
    if not os.path.exists(gtf_file):
        print(f"{Fore.RED}Error: GTF file {gtf_file} not found{Style.RESET_ALL}")
        return
    
    # Parse the GTF file to build CDS-to-genome coordinate maps
    transcript_cds_map, genes_map = parse_gtf_for_cds_mappings(gtf_file)
    
    if not transcript_cds_map:
        print(f"{Fore.RED}Error: Failed to build CDS-to-genome maps from GTF{Style.RESET_ALL}")
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
            
            # Extract information from FullGeneID if available
            extracted_gene_name = None
            gene_id = None
            transcript_id = None
            cds_start_pos = None
            
            if full_gene_id and isinstance(full_gene_id, str):
                extracted_gene_name, gene_id, transcript_id, cds_start_pos = extract_transcript_info_from_fullgeneid(full_gene_id)
                print(f"  From FullGeneID: Gene={extracted_gene_name}, GeneID={gene_id}, TranscriptID={transcript_id}, CDSStart={cds_start_pos}")
            
            # Get the transcript ID - either from FullGeneID or by looking up gene name
            used_transcript_id = None
            used_transcript_info = None
            
            if transcript_id and transcript_id in transcript_cds_map:
                used_transcript_id = transcript_id
                used_transcript_info = transcript_cds_map[transcript_id]
                print(f"  Using transcript {used_transcript_id} from FullGeneID")
            elif gene_name in genes_map and genes_map[gene_name]:
                # Use the first transcript for this gene
                used_transcript_id = genes_map[gene_name][0]
                used_transcript_info = transcript_cds_map[used_transcript_id]
                print(f"  Using transcript {used_transcript_id} from gene name lookup")
            else:
                print(f"  {Fore.RED}Could not find transcript for gene {gene_name}{Style.RESET_ALL}")
            
            # Clean up chromosome field to get just the chromosome name for VCF lookup
            clean_chrom = extract_chromosome(chromosome)
            
            # Convert position to integer
            try:
                position = int(position)
            except (ValueError, TypeError):
                print(f"  {Fore.RED}Invalid position format: {position}{Style.RESET_ALL}")
                continue
                
            # Determine if this is a CDS position that needs coordinate conversion
            genomic_position = position
            is_converted = False
            
            if used_transcript_id and used_transcript_info:
                # This is a CDS position - convert to genomic coordinates
                _, converted_position = convert_cds_to_genomic(used_transcript_id, position, transcript_cds_map)
                
                if converted_position:
                    genomic_position = converted_position
                    is_converted = True
                    print(f"  {Fore.GREEN}Converted CDS position {position} → genomic position {genomic_position}{Style.RESET_ALL}")
                else:
                    print(f"  {Fore.RED}Failed to convert CDS position {position}{Style.RESET_ALL}")
            else:
                # Assume it's already a genomic position
                print(f"  Using position {position} as-is (no conversion)")
            
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
            
            position_description = f"{display_chrom}:{genomic_position}"
            if is_converted:
                position_description += f" (converted from CDS position {position})"
                
            print(f"\n{Fore.CYAN}► Position {position_description} ({gene_name}){Style.RESET_ALL}")
            print(f"  Fixed difference: Group0={Fore.GREEN}{g0_nuc}{Style.RESET_ALL}, Group1={Fore.RED}{g1_nuc}{Style.RESET_ALL}")
            
            # Check if position exists in VCF
            found, exact, nearest_pos, distance, vcf_line = find_position_in_vcf(display_chrom, genomic_position, vcf_file)
            
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
    gtf_file = "../gencode.v47.basic.annotation.gtf"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    if len(sys.argv) > 2:
        vcf_dir = sys.argv[2]
    if len(sys.argv) > 3:
        gtf_file = sys.argv[3]
        
    validate_positions(csv_file, vcf_dir, gtf_file)
