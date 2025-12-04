# Phenome-wide Association Study (PheWAS) Pipeline

A production-grade, high-throughput pipeline for testing associations between structural variants (inversions) and thousands of phenotypes in large-scale cohorts (e.g., All of Us). This system is designed for robustness, resource efficiency, and statistical rigor, handling binary traits with complex covariate adjustments and ancestry-aware follow-up.

## Get the PheWAS code
```
# Mirror the repo's phewas/ directory locally
rm -rf -- phewas && mkdir -p phewas && \
curl -fsSL 'https://api.github.com/repos/SauersML/ferromic/git/trees/main?recursive=1' | \
python3 -c 'import sys, json, os, pathlib, urllib.request
root = "phewas/"
tree = json.load(sys.stdin)["tree"]
for it in tree:
    p = it.get("path","")
    if not (p.startswith(root) and it.get("type") == "blob"):
        continue
    dest = pathlib.Path(p)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/SauersML/ferromic/main/{p}"
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())'
```

## Methodology

### 1. Cohort & Input Data
*   **Cohort**: The pipeline is tuned for the **NIH All of Us Research Program (v8)** using Short-Read WGS (srWGS) data.
*   **Inversion Genotypes**: Input dosages are expected to be pre-imputed.
    *   *Inclusion Criteria*: Inversions are typically pre-filtered for imputation accuracy ($r^2 > 0.3$), low absolute error (Wilcoxon $p < 0.05$), and population differentiation ($F_{ST} > 0.5$).
    *   *Low Variance Filter*: The pipeline automatically skips inversions with insufficient dosage variance within the analysis cohort.
*   **Phenotype Map**: Phenotypes are defined using **phecodeX** (mapping ICD-9/10 codes to phecodes).
    *   *Heritability Filter*: The default phenotype list is restricted to traits with evidence of non-zero SNP heritability (combined $h^2 \ge 0.15$) in Pan-UKBB data, prioritizing biologically plausible associations.

### 2. Phenotype QC & Engineering
Before modeling, phenotypes undergo rigorous quality control and deduplication:
*   **Case Definition**: Participants with at least one mapped ICD code for the phecode.
*   **Control Definition**: Participants with **no** ICD codes in the entire exclusion category associated with the phecode (to prevent contamination from related conditions).
*   **Prevalence Cap**: Phenotypes with $>90,000$ cases are excluded as overly general (e.g., "Viral infection").
*   **Minimum Counts**: By default, phenotypes require $\ge 1,000$ cases and $\ge 1,000$ controls (configurable via `--min-cases-controls`).
*   **Deduplication**: To reduce multiple testing burden, highly correlated phenotypes are pruned. A phenotype is dropped if it shares $>70\%$ of its cases with another phenotype or has a binary correlation ($\phi$) $> 0.7$.
*   **Sex Stratification**: If $>99\%$ of cases belong to one sex, the analysis is automatically restricted to that sex, and the sex covariate is dropped.

### 3. Statistical Model (Stage 1: Discovery)
For every eligible phenotype-inversion pair, the pipeline fits a logistic regression model:

$$
\text{logit}(P(\text{case})) = \beta_0 + \beta_{inv} \cdot \text{Dosage} + \beta_{sex} \cdot \text{Sex} + \sum_{k=1}^{16} \gamma_k \text{PC}_k + \beta_{age} \cdot \text{Age} + \beta_{age^2} \cdot \text{Age}^2 + \boldsymbol{\theta} \cdot \text{Ancestry}
$$

*   **Covariates**:
    *   **Genetically Inferred Sex** (dropped if sex-restricted).
    *   **Principal Components (PCs)**: First 16 genetic PCs to control for fine-scale structure.
    *   **Age**: Centered age and squared centered age ($Age^2$) to capture non-linear effects.
    *   **Ancestry**: Categorical indicator variables ("afr", "amr", "mid", "sas", "eur") based on random forest classification.
*   **Inference Strategy**:
    1.  **Standard MLE**: Likelihood Ratio Test (LRT) comparing the full model to a null model (without dosage).
    2.  **Robust Fallback**: If MLE fails (e.g., due to perfect separation or convergence issues):
        *   **Firth Regression**: Penalized likelihood to handle separation.
        *   **Score Tests**: Rao score test or Parametric Bootstrap score test (if enabled) to estimate p-values without full model fitting.
*   **Multiple Testing**: P-values are corrected using the Benjamini–Hochberg (FDR 5%) procedure.

### 4. Ancestry-Aware Follow-up (Stage 2)
For significant associations (FDR < 0.05), the pipeline performs heterogeneity testing:
*   **Interaction Test**: A likelihood ratio test comparing a model with a `Dosage × Ancestry` interaction term against the base model.
    *   *Note*: If the interaction model is unstable, a robust Rao score test is used.
*   **Stratified Analysis**: The regression is run separately within each major ancestry group to estimate population-specific Odds Ratios (OR).

## Architecture

The codebase is organized into modular components:

*   **`run.py`**: The orchestration entry point. Manages data loading, parallel process pools, and results aggregation.
*   **`models.py`**: Core statistical engine. Implements logistic regression ladders (MLE $\to$ Ridge $\to$ Firth), score tests, and bootstrap logic.
*   **`pipes.py`**: Resource governor. Monitors system RAM/CPU and throttles task submission to prevent OOM (Out of Memory) crashes in containerized environments.
*   **`pheno.py`**: Data loader. Handles BigQuery fetching, parquet caching, and phenotype deduplication logic.
*   **`categories.py`**: Implements aggregate category-level tests (GBJ, GLS) to detect signal enrichment within disease groups.

## Usage

### Prerequisites
*   Python 3.9+
*   Google Cloud SDK (authenticated for BigQuery access)
*   Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `google-cloud-bigquery`, `psutil`

### Configuration
Key settings are controlled via environment variables or CLI arguments.

**Environment Variables:**
*   `WORKSPACE_CDR`: BigQuery dataset ID (e.g., `fc-aou-datasets-controlled.v8...`).
*   `GOOGLE_PROJECT`: GCP Project ID for billing/access.

**CLI Arguments:**
```bash
# Basic run (processes all inversions in config)
python3 -m phewas.run

# Run with custom sample size thresholds
python3 -m phewas.cli --min-cases-controls 500

# Restrict analysis to a specific population subset
python3 -m phewas.cli --pop-label "eur"

# Debug a single phenotype
python3 -m phewas.cli --pheno "Type_2_diabetes"
```

### Outputs
Results are saved to `phewas_results_<timestamp>.tsv` containing:
*   **Stats**: `OR`, `Beta`, `P_Value`, `Q_Value` (FDR).
*   **Counts**: `N_Cases`, `N_Controls`.
*   **Diagnostics**: `P_Source` (method used), `Model_Notes` (e.g., "sex_restricted").
*   **Follow-up**: `P_LRT_AncestryxDosage` and per-ancestry ORs (if significant).

Intermediate files (metadata, individual test JSONs) are stored in `phewas_cache/` to allow resuming interrupted runs.
