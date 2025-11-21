# Statistics Utilities

The `stats/` directory contains helper scripts for downstream analyses such as phenotype selection, Manhattan plot generation, and dosage inference support.

## Imputed dosage retrieval workflow

Use the following sequence to download the required variant lists, prepare inputs, and infer dosages from trained models. These commands can be executed in a clean working directory; each script is hosted in this repository to keep the workflow reproducible.

1. **Download the variant list from All of Us:**
   ```bash
   curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/stats/snv_list_acaf_download.py | python3
   ```
2. **Prepare PLINK-derived inputs for inference:**
   ```bash
   curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/imputation/prepare_data_for_infer.py | python3
   ```
3. **Fetch the PLS regression helper used by the inference script:**
   ```bash
   curl -O https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/imputation/pls_patch.py
   ```
4. **Run dosage inference with the trained models:**
   ```bash
   curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/imputation/infer_dosage.py | python3
   ```

After dosages are inferred, navigate to the `phewas/` directory for guidance on running association analyses against the predicted inversion genotypes.
