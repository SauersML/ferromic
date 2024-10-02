# VCF Statistics Calculator 📊

Welcome to the **VCF Statistics Calculator**, a Rust-based tool designed to compute **Watterson's Theta (θ)** and **Pi (π)** (and others, coming soon) for genomic regions defined in VCF (Variant Call Format) files.

---

## Table of Contents 📑

- [Features](#features)
- [Background](#background)
- [Installation 🛠️](#installation-️)
- [Usage 🚀](#usage-🚀)
  - [Command-Line Arguments](#command-line-arguments)
  - [Input Files](#input-files)
    - [VCF File 🧬](#vcf-file-️)
    - [TSV Configuration File 📋](#tsv-configuration-file-️)
    - [Mask File 🛡️](#mask-file-️)
  - [Output File 📈](#output-file-️)
- [Filtering Mechanisms 🔍](#filtering-mechanisms-️)
- [Progress Indicators 🎛️](#progress-indicators-️)
- [Common Warnings and Errors ⚠️](#common-warnings-and-errors-️)
- [Examples 🧪](#examples-️)
- [Contributing 🤝](#contributing-️)
- [License 📄](#license-️)
- [Contact 📬](#contact-️)

---

## Features ✨

- **Calculate Genetic Diversity Metrics**: Compute Watterson's Theta (θ) and Pi (π) for specified regions.
- **Haplotype Group Analysis**: Separate calculations for haplotypes with and without a structural variant class (such as inversions).
- **Flexible Input Handling**: Supports configuration via TSV files for multiple regions and haplotype groupings (e.g., by presence of structural variant).
- **Filtering**: Filter variants based on genotype quality (GQ) scores and predefined genomic masks.
- **Output**: Generates CSV files with statistical metrics for each genomic region.

---

## Background 🧬

This tool processes VCF files to calculate **Watterson's Theta (θ)** and **Pi (π)**, metrics for understanding genetic diversity within genomic regions. By using a TSV configuration file, users can define multiple regions and categorize haplotypes based on SV (e.g. inversion) statuses. This allows for distinguishing between inverted and non-inverted haplotypes across different regions and samples.

**Metrics**:
- **Watterson's Theta (θ)**: Based on the number of segregating (polymorphic) sites.
- **Pi (π)**: Measures nucleotide diversity, based on pairwise per-site nucleotide differences.

---

## Installation 🛠️

Make sure you have [Rust](https://www.rust-lang.org/tools/install) installed.

Clone the repository and build the project.

---

## Usage 🚀

### Command-Line Arguments

```bash
vcf_stats_calculator -v <VCF_FOLDER> \
                    -c <CONFIG_FILE> \
                    -o <OUTPUT_CSV> \
                    --min_gq <MIN_GQ> \
                    --mask_file <MASK_FILE> \
                    -h <CHR> \
                    -r <REGION>
```

**Parameters**:

- `-v`, `--vcf_folder`: **(Required)** Path to the directory containing VCF files.
- `-c`, `--config_file`: **(Optional)** Path to the TSV configuration file defining regions and haplotype groupings.
- `-o`, `--output_file`: **(Optional)** Path for the output CSV file containing statistical results. Defaults to `output.csv` if not specified.
- `--min_gq`: **(Optional)** Minimum genotype quality (GQ) Phred score for filtering variants. Defaults to `30`.
- `--mask_file`: **(Optional)** Path to the BED file specifying genomic regions to mask (filter out).
- `-h`, `--chr`: **(Optional)** Chromosome name to process when not using a config file.
- `-r`, `--region`: **(Optional)** Specific region to process within the chromosome, in the format `start-end` (e.g., `10732039-23685112`).

**Notes**:
- Either `--config_file` or both `--chr` and `--region` must be provided.
- When using `--config_file`, the tool can process multiple regions and haplotype groupings as defined in the TSV.
- When not using a config file, the tool will process the specified chromosome and region and output results to the console.

### Input Files

#### VCF File 🧬

- **Format**: [VCF v4.2](https://samtools.github.io/hts-specs/VCFv4.2.pdf)
- **Contents**: Variant data including positions, alleles, and genotype information for multiple samples.
- **Genotype Format**: Must include `GT` (genotype) and `GQ` (genotype quality) fields.

**Example**:
```vcf
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Phred scaled genotype quality computed by whatshap genotyping algorithm.">
#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT SAMPLE1 SAMPLE2
chr1 1500 . A T . PASS . GT:GQ 0|0:35 0|1:40
```

#### TSV Configuration File 📋

- **Purpose**: Defines multiple genomic regions and specifies haplotype groupings based on inversion statuses.
- **Structure**:
    - **Columns**:
        - `seqnames`: Chromosome name (e.g., `chr1`).
        - `start`: Start position of the region.
        - `end`: End position of the region.
        - **Sample Columns**: Genotype information for each sample in the format `0|1`, `1|0`, `0|0`, `1|1`, etc.

**Example**:
```tsv
seqnames	start	end	POS	orig_ID	verdict	categ	NA19434	HG00036	HG00191	...
chr1	13004251	13122531	13113384	chr1-13113384-INV-62181	pass	inv	1|1	1|1	1|1	...
```

**Notes**:
- Genotypes beyond the standard `0|0`, `0|1`, `1|0`, `1|1` (e.g., `0|1_lowconf`) will be used only for the "unfiltered" outputs.
- Haplotype groupings (presence or absence) are determined by the values in the genotype columns, indicating, e.g., inversion (`1`) or direct (`0`) haplotypes.

#### Mask File 🛡️

- **Format**: [BED](https://genome.ucsc.edu/FAQ/FAQformat.html#format1)
- **Contains**: Genomic regions to exclude from analysis (variants within the regions will betreated similarly to variants with low GQ scores).
- **Structure**:
    - Three columns: `chromosome`, `start`, `end`.

**Example**:
```bed
chr1	0	77102
chr1	88113	190752
...
```

---

### Output File 📈

- **Format**: CSV
- **Filename**: As specified by the `--output_file` parameter.
- **Headers**:
    ```
    chr,region_start,region_end,0_sequence_length,1_sequence_length,0_sequence_length_adjusted,1_sequence_length_adjusted,0_segregating_sites,1_segregating_sites,0_w_theta,1_w_theta,0_pi,1_pi,0_segregating_sites_filtered,1_segregating_sites_filtered,0_w_theta_filtered,1_w_theta_filtered,0_pi_filtered,1_pi_filtered,0_num_hap_no_filter,1_num_hap_no_filter,0_num_hap_filter,1_num_hap_filter,inversion_freq_no_filter,inversion_freq_filter
    ```
    
- **Column Descriptions**:
    - `chr`: Chromosome name.
    - `region_start`: Start position of the region.
    - `region_end`: End position of the region.
    - `0_sequence_length`: Total length of the sequence for haplotype group `0`.
    - `1_sequence_length`: Total length of the sequence for haplotype group `1`.
    - `0_sequence_length_adjusted`: Adjusted sequence length for haplotype group `0` after filtering.
    - `1_sequence_length_adjusted`: Adjusted sequence length for haplotype group `1` after filtering.
    - `0_segregating_sites`: Number of segregating sites (unfiltered) for haplotype group `0`.
    - `1_segregating_sites`: Number of segregating sites (unfiltered) for haplotype group `1`.
    - `0_w_theta`: Watterson's Theta (unfiltered) for haplotype group `0`.
    - `1_w_theta`: Watterson's Theta (unfiltered) for haplotype group `1`.
    - `0_pi`: Pi (unfiltered) for haplotype group `0`.
    - `1_pi`: Pi (unfiltered) for haplotype group `1`.
    - `0_segregating_sites_filtered`: Segregating sites for haplotype group `0`.
    - `1_segregating_sites_filtered`: Segregating sites for haplotype group `1`.
    - `0_w_theta_filtered`: Watterson's Theta for haplotype group `0`.
    - `1_w_theta_filtered`: Watterson's Theta for haplotype group `1`.
    - `0_pi_filtered`: Pi for haplotype group `0`.
    - `1_pi_filtered`: Pi for haplotype group `1`.
    - `0_num_hap_no_filter`: Number of haplotypes for group `0` before filtering.
    - `1_num_hap_no_filter`: Number of haplotypes for group `1` before filtering.
    - `0_num_hap_filter`: Number of haplotypes for group `0`.
    - `1_num_hap_filter`: Number of haplotypes for group `1`.
    - `inversion_freq_no_filter`: Allele frequency of inversion (1) before filtering.
    - `inversion_freq_filter`: Allele frequency of inversion (1).
    
- **Special Values**:
    - `θ = 0`: No segregating sites; no genetic variation observed.
    - `θ = Infinity (inf)`: Insufficient haplotypes or zero-length region; metrics undefined.
    - `π = 0`: No nucleotide differences.
    - `π = Infinity (inf)`: Insufficient data; metrics undefined.

---

## Filtering Mechanisms 🔍

### Genotype Quality (GQ) Filtering

- **Purpose**: Exclude variants with low genotype quality.
- **Mechanism**:
    - If any sample within a variant has a GQ score below the specified `--min_gq` threshold, the entire variant is excluded from **filtered** (but not unfiltered) analyses.
    - Variants passing the GQ filter are included in both **unfiltered** and **filtered** analyses.

### Genotype Matching

- **Purpose**: Only exact genotype matches (`0|0`, `0|1`, `1|0`, `1|1`) are included in **filtered** analyses.
- **Mechanism**:
    - Genotypes not strictly matching the four expected formats (e.g., `0|1_lowconf`) are considered missing data and excluded from **filtered** analyses.
    - **Unfiltered** analyses include all genotypes that can be parsed into valid formats based on the first three characters.

### Masking

- **Purpose**: Exclude entire genomic regions from analysis based on predefined masks.
- **Mechanism**:
    - Regions specified in the BED mask file are treated similarly to low GQ variants and excluded from **filtered** (but not unfiltered) analyses.
    - Sequence lengths are adjusted to account for masked regions in statistic calculations

---

## Common Warnings and Errors ⚠️

- **Missing Samples**: If certain samples defined in the configuration file are not found in the VCF, a warning is displayed with the missing samples.
  
- **Invalid Genotypes**: Genotypes not conforming to the expected formats (`0|0`, `0|1`, `1|0`, `1|1`) will be considered missing data. The number and percentage of invalid genotypes encountered will be shown.
  
- **Multi-allelic Sites**: Multi-allelic variants are not supported.
  
- **No Variants Found**: If no variants are found within the specified region or all variants are filtered out, a warning will be printed.

---

## Examples 🧪

### Running the Tool with All Parameters

```bash
cargo run --release --bin vcf_stats_calculator \
    -v ../vcfs \
    -c ../config/regions.tsv \
    -o ../results/output_stats.csv \
    --min_gq 30 \
    --mask_file ../masks/hardmask.hg38.v4_acroANDsdOnly.over99.bed
```

### Running the Tool Without a Configuration File

If you prefer to calculate statistics for a specific chromosome and region without using a configuration file, you can run the tool with the `--chr` and `--region` flags. **Note:** In this mode, the results will be printed to the console rather than written to a CSV file.

```bash
cargo run --release --bin vcf_stats_calculator \
    -v ../vcfs \
    -c chr22 \
    -r 10732039-23685112 \
    --min_gq 30 \
    --mask_file ../masks/hardmask.hg38.v4_acroANDsdOnly.over99.bed
```
