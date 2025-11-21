# Supplemental PheWAS Outputs

This folder houses ancillary PheWAS figures and follow-up instructions. The lead image compares odds ratios derived from imputed inversion dosages with those obtained from tagging SNP dosages.

## Polygenic score control follow-up

To reproduce the polygenic score sensitivity analysis:

1. **Install the `gnomon` toolkit:**
   ```bash
   git clone https://github.com/SauersML/gnomon
   cd gnomon
   cargo build --release
   cd ..
   ```
2. **Download the microarray PLINK inputs:**
   ```bash
   gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
   ```
3. **Compute scores for the 17q21 region (example invocation):**
   ```bash
   ./gnomon/target/release/gnomon score "PGS004378 | chr17:45535159-46342045, PGS005198 | chr17:45535159-46342045, PGS004146 | chr17:45535159-46342045, PGS004229 | chr17:45535159-46342045, PGS004869 | chr17:45535159-46342045, PGS000507 | chr17:45535159-46342045" ./arrays
   ```
4. **Rename the output for downstream scripts:**
   ```bash
   cp arrays.sscore scores.tsv
   ```

Adjust the score list and genomic intervals as needed for other regions or panels.
