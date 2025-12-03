# Imputation Pipeline

This directory contains the implementation of an imputation pipeline designed to predict inversion genotype dosages from local SNP dosages. The pipeline uses Partial Least Squares (PLS) regression trained on a combination of real and synthetic diploid genomes to robustly infer inversion states without requiring phased data during inference.

## Overview

The goal of this pipeline is to impute the genotype dosage (0, 1, or 2 copies) of specific inversion alleles based on the dosage patterns of surrounding SNPs. This is particularly useful for large cohorts where direct inversion genotyping is difficult or expensive, but SNP data is readily available (e.g., from array genotyping or short-read sequencing).

## Quick start on AoU: imputed dosage retrieval workflow

Use the following sequence to download the required variant lists, prepare inputs, and infer dosages from trained models. These commands can be executed in a clean working directory.

1. **Download the variant list from All of Us:**
   ```bash
curl -s https://raw.githubusercontent.com/sauersml/ferromic/main/imputation/snv_list_acaf_download.py | python3
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

## Methodology

### Model Training (`linked.py`)
For each inversion locus, we train a specific imputation model:

1.  **Input Data**:
    -   **Inversion Dosages**: "Ground truth" inversion dosages from a reference phased dataset (e.g., Porubsky et al. 2022).
    -   **SNP Dosages**: Genotypes of SNPs within 50 kbp of the inversion breakpoints and within the inversion itself. Only SNPs with low missingness are used.

2.  **Synthetic Data Augmentation**:
    -   To improve model robustness and coverage of the haplotype space, we generate **synthetic diploid genomes**.
    -   Any two samples in the reference set contribute four haplotypes total, but only two observed diploid genomes. We construct the other eight possible diploid combinations to augment the training data.
    -   This allows the model to learn from haplotype combinations that may not be present in the limited reference set but could exist in the broader population.

3.  **Partial Least Squares (PLS) Regression**:
    -   We use PLS regression to model the relationship between the high-dimensional SNP data and the inversion dosage.
    -   The model predicts: `Inversion Dosage = Intercept + Σ (Component_k_Contribution)`
    -   This effectively reduces to a linear model: `Dosage = Intercept + Σ (SNP_Weight_i * SNP_Count_i)`.

4.  **Validation & Selection**:
    -   **Nested Cross-Validation**: Used to evaluate performance and select the optimal number of PLS components. Crucially, test genomes are never used to construct synthetic training genomes for the same fold to prevent data leakage.
    -   **Significance Testing**: Per-inversion models are tested against a dummy intercept-only model using a Wilcoxon signed-rank test to ensure they are learning real signal.
    -   **Component Selection**: The number of components is chosen to minimize mean squared error (MSE) via cross-validation.

5.  **Final Model**:
    -   The final model is refit using all available data (real + synthetic).
    -   The trained model (`.model.joblib`) and the list of SNPs used (`.snps.json`) are saved for inference.

### Inference
Inference is performed on target datasets (e.g., PLINK bed files) using the trained models. It does not require phased data.

## Workflow

The pipeline consists of three main stages: Training, Data Preparation, and Inference.

### 1. Training
Run `linked.py` to train models for all configured inversion loci. This script handles:
-   Loading VCF data.
-   Generating synthetic training data.
-   Running the nested cross-validation and PLS regression.
-   Saving the best models to `final_imputation_models/`.

```bash
python3 linked.py
```

### 2. Packaging (Optional)
Use `pack_models.py` to zip the trained models and metadata for easy distribution or transfer to an inference environment.

```bash
python3 pack_models.py -i final_imputation_models -o imputation_models_package.zip
```

### 3. Inference Preparation
Run `prepare_data_for_infer.py` to convert a target PLINK dataset into the specific genotype matrix format required by the inference script. This script:
-   Reads a PLINK `.bed` file.
-   Fetches the model manifest and SNP lists.
-   Extracts only the required SNPs for the models.
-   Handles allele flipping to match the training data reference.
-   Outputs memory-mapped numpy arrays (`.genotypes.npy`).

```bash
# Set environment variables for configuration
export PLINK_PREFIX="path/to/target_dataset"
export OUTPUT_DIR="genotype_matrices"
python3 prepare_data_for_infer.py
```

### 4. Inference
Run `infer_dosage.py` to generate inversion dosage predictions.
-   Loads the prepared genotype matrices and trained models.
-   Imputes missing SNP values (using column means).
-   Predicts inversion dosages.
-   Outputs a TSV file with the imputed dosages for all samples.

```bash
python3 infer_dosage.py
```

## Directory Structure & Scripts

*   **`linked.py`**: The core training script. Implements the PLS regression, synthetic data generation, and cross-validation logic.
*   **`infer_dosage.py`**: The inference engine. Applies trained models to new genotype data. Designed to be robust and memory-efficient.
*   **`prepare_data_for_infer.py`**: Pre-processing script that converts PLINK files into the efficient numpy format needed by `infer_dosage.py`. Handles SNP matching and allele alignment.
*   **`pack_models.py`**: Utility to verify and package trained models and their corresponding SNP metadata into a zip archive.
*   **`pls_patch.py`**: Contains the `PLSRegression` implementation used by `linked.py`. It is based on `sklearn`'s implementation but maintained locally to ensure stability or custom behavior.
*   **`tagging_snp_inversion_dosages.py`**: A specialized script for fetching specific tagging SNPs (e.g., for the 17q21 inversion) directly from Google Cloud Storage and generating hard calls based on strict unanimity.

## Dependencies
The pipeline requires a standard Python scientific stack:
-   `numpy`
-   `pandas`
-   `scikit-learn`
-   `scipy`
-   `joblib`
-   `cyvcf2` (for training from VCFs)
-   `bed-reader` (for reading PLINK files)
-   `tqdm` (for progress bars)
