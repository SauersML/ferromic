# Ferromic

A Rust-based tool for population genetic analysis that calculates diversity statistics from VCF files, with support for haplotype-group-specific analyses and genomic regions.

## Overview

Ferromic processes genomic variant data from VCF files to calculate key population genetic statistics. It can analyze diversity metrics separately for different haplotype groups (0 and 1) as defined in a configuration file, making it particularly useful for analyzing regions with structural variants or any other genomic features where haplotypes can be classified into distinct groups.

## Features

- Efficient VCF processing using multi-threaded parallelization
- Calculate key population genetic statistics:
  - Nucleotide diversity (π)
  - Watterson's theta (θ)
  - Segregating sites counts
  - Allele frequencies
- Apply various filtering strategies:
  - Genotype quality (GQ) thresholds
  - Genomic masks (exclude regions)
  - Allowed regions (include only)
  - Multi-allelic site handling
  - Missing data management
- Extract coding sequences (CDS) from genomic regions using GTF annotations
- Generate PHYLIP format sequence files for phylogenetic analysis
- Create per-site diversity statistics for fine-grained analysis
- Support both individual region analysis and batch processing via configuration files

## Usage

```
cargo run --release --bin run_vcf -- [OPTIONS]
```

### Required Arguments

- `--vcf_folder <FOLDER>`: Directory containing VCF files
- `--reference <PATH>`: Path to reference genome FASTA file
- `--gtf <PATH>`: Path to GTF annotation file

### Optional Arguments

- `--chr <CHROMOSOME>`: Process a specific chromosome
- `--region <START-END>`: Process a specific region (1-based coordinates)
- `--config_file <FILE>`: Configuration file for batch processing multiple regions
- `--output_file <FILE>`: Output file path (default: output.csv)
- `--min_gq <INT>`: Minimum genotype quality threshold (default: 30)
- `--mask_file <FILE>`: BED file of regions to exclude
- `--allow_file <FILE>`: BED file of regions to include

## Example Command

```
cargo run --release --bin run_vcf -- \
    --vcf_folder ../vcfs \
    --config_file ../variants.tsv \
    --mask_file ../hardmask.bed \
    --reference ../hg38.no_alt.fa \
    --gtf ../hg38.knownGene.gtf
```

## Coordinate Systems

Ferromic handles different coordinate systems:
- VCF files: 1-based coordinates
- BED mask/allow files: 0-based, half-open intervals
- TSV config files: 1-based, inclusive coordinates
- GTF files: 1-based, inclusive coordinates

## Configuration File Format

The configuration file should be tab-delimited with these columns:
1. `seqnames`: Chromosome (with or without "chr" prefix)
2. `start`: Region start position (1-based, inclusive)
3. `end`: Region end position (1-based, inclusive)
4. `POS` and other columns: Additional information (ignored)
5. Sample columns: Each sample has a column with a genotype string in the format "0|0", "0|1", "1|0", or "1|1"

Where:
- "0" and "1" represent the two haplotype groups to be analyzed separately
- The "|" character indicates the phase separation between left and right haplotypes
- Genotypes with special formats (e.g., "0|1_lowconf") are included in unfiltered analyses but excluded from filtered analyses

## Output Files

### Main CSV Output

Contains summary statistics for each region with columns:
```
chr,region_start,region_end,0_sequence_length,1_sequence_length,0_sequence_length_adjusted,1_sequence_length_adjusted,0_segregating_sites,1_segregating_sites,0_w_theta,1_w_theta,0_pi,1_pi,0_segregating_sites_filtered,1_segregating_sites_filtered,0_w_theta_filtered,1_w_theta_filtered,0_pi_filtered,1_pi_filtered,0_num_hap_no_filter,1_num_hap_no_filter,0_num_hap_filter,1_num_hap_filter,inversion_freq_no_filter,inversion_freq_filter
```

Where:
- Values prefixed with "0_" are statistics for haplotype group 0
- Values prefixed with "1_" are statistics for haplotype group 1
- "sequence_length" is the raw length of the region
- "sequence_length_adjusted" accounts for masked regions
- "num_hap" columns indicate the number of haplotypes in each group
- Statistics with "_filtered" are calculated from strictly filtered data

### Per-site CSV Output

Contains position-specific diversity metrics:
```
relative_position,filtered_pi_chr_X_start_Y_end_Z_group_0,filtered_pi_chr_X_start_Y_end_Z_group_1,unfiltered_pi_chr_X_start_Y_end_Z_group_0,...
```

Where:
- "relative_position" is the 1-based position relative to the start of the region
- Column headers combine the statistic type, region, and haplotype group

### PHYLIP Files

Generated for each transcript that overlaps with the query region:
- File naming: `group_{0/1}_{transcript_id}_chr_{chromosome}_start_{start}_end_{end}_combined.phy`
- Contains aligned sequences (based on the reference genome with variants applied)
- Sample names in the PHYLIP files are constructed from sample names with "_L" or "_R" suffixes to indicate left or right haplotypes

## Implementation Details

- For PHYLIP files, if a CDS region overlaps with the query region (even partially), the entire transcript's coding sequence is included
- For diversity statistics (π and θ), only variants strictly within the region boundaries are used
- Different filtering approaches:
  - Unfiltered: Includes all valid genotypes, regardless of quality or exact format
  - Filtered: Excludes low-quality variants, masked regions, and non-standard genotypes
- Sequence length is adjusted for masked regions when calculating diversity statistics
- Multi-threading is implemented via Rayon for efficient processing
- Missing data is properly accounted for in diversity calculations
- Special values in results:
  - θ = 0: No segregating sites (no genetic variation)
  - θ = Infinity: Insufficient haplotypes or zero sequence length
  - π = 0: No nucleotide differences (genetic uniformity)
  - π = Infinity: Insufficient data