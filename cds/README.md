# CDS Analysis Pipeline

This directory contains the scripts and libraries for analyzing selective pressure on protein-coding genes within inverted regions. The pipeline estimates the ratio of nonsynonymous to synonymous substitution rates (dN/dS) per coding sequence, comparing direct and inverted orientations.

## Methodology

To estimate selective pressure, we calculate dN/dS ratios using the **codeml** program from the [PAML](http://abacus.gene.ucl.ac.uk/software/paml.html) software package.

### Filtering Criteria
Coding sequences are strictly filtered to ensure high-quality alignments and informative tree topologies:
*   **Divergence:** Excluded if median human-chimp divergence > 10% (to avoid misalignment).
*   **Haplotype Count:** Excluded if fewer than 3 haplotypes are present in either the Direct or Inverted group.
*   **Variation:** Excluded if fewer than 2 variable codons exist.
*   **Taxa Count:** Excluded if the phylogenetic tree has fewer than 4 taxa.
*   **Topology:** Excluded if there is not at least one pure internal node (descendants are all direct or all inverted) for each orientation.

### Statistical Models
For each qualifying coding sequence, we execute **codeml** using:
*   **Model:** Clade Model C (`model = 3`, `NSsites = 2`)
*   **Codon Frequencies:** F3x4 model (`CodonFreq = 2`)
*   **Runmode:** 0

Two models are compared via a Likelihood Ratio Test (LRT):
1.  **Null Model (H0):** dN/dS distribution between site classes is shared between all haplotypes (both Direct and Inverted are labeled as the same foreground partition).
2.  **Full Model (H1):** dN/dS is allowed to vary between groups (Direct and Inverted are labeled as distinct foreground partitions).

The resulting p-values are corrected for multiple testing using the Benjamini–Hochberg procedure.

## Scripts

### Data Preparation
*   **`axt_to_phy.py`**: Converts UCSC AXT alignments (Human vs. Chimp) into PHYLIP format to serve as outgroups.
*   **`combine_phy.py`**: Combines the user-generated haplotype PHYLIP files (from the main `ferromic` pipeline) with the Chimp outgroups.

### Analysis Core
*   **`pipeline_lib.py`**: The core library containing shared logic for:
    *   Running IQ-TREE to generate region-specific phylogenetic trees.
    *   Pruning trees for individual genes.
    *   Running PAML codeml with caching and error handling.
    *   Parsing results and generating figures.
*   **`iqtree_runner.py`**: Wrapper to execute IQ-TREE for a specific region.
*   **`codeml_runner.py`**: Wrapper to execute PAML codeml for a specific gene/region pair.

### Orchestration & CI
*   **`generate_gha_matrix.py`**: Generates a GitHub Actions matrix JSON to parallelize the analysis across many runners. It respects the allowlist in `data/inv_properties.tsv`.
*   **`omega_test.py`**: A local orchestrator script that can run the full pipeline (IQ-TREE + PAML) for testing or small-scale execution.

## Usage

This pipeline is primarily designed to run within the GitHub Actions workflow (`analysis_pipeline.yml`). However, individual components can be run locally if the environment is set up (requires `paml` and `iqtree` binaries).

### Data Preparation
```bash
python3 axt_to_phy.py
python3 combine_phy.py
```

### Running Analysis
To run the analysis logic locally (similar to the CI pipeline):
```bash
python3 omega_test.py
```
