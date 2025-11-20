# Phenome-wide association (PheWAS) pipeline

The `phewas/` directory hosts a production-grade association testing pipeline designed to scan structural variants (specifically inversions) against thousands of phenotypes. It is built for high-throughput analysis on large cohorts, leveraging Python's multiprocessing and robust resource governance.

## Architecture

The pipeline is structured into several key components:

*   **Orchestration (`run.py`)**: The central supervisor that coordinates the entire workflow. It handles:
    *   Loading shared data (demographics, principal components, ancestry).
    *   Managing the lifecycle of inversion processing.
    *   Aggregating results from parallel workers.
    *   Running Stage-1 (Overall) and Stage-2 (Follow-up/Ancestry) analyses.
*   **Resource Governance (`pipes.py`)**: A dedicated module for managing system resources (RAM, CPU).
    *   `BudgetManager`: Tracks memory usage against a global budget (respecting container/cgroup limits).
    *   `MultiTenantGovernor`: Dynamically throttles task submission based on memory pressure and observed worker footprints.
    *   Handles shared memory segments for large covariate matrices to minimize overhead.
*   **Statistical Modeling (`models.py`)**: Implements the core statistical tests.
    *   **Stage 1 (Overall)**: Tests the association between an inversion and a phenotype across the entire cohort.
        *   **Model**: Logistic regression (MLE) with adjustment for covariates (Age, Sex, PCs).
        *   **Robustness**: Falls back to **Firth penalized regression** or **Score tests** (Rao/Bootstrap) when MLE fails (e.g., separation, low case counts).
        *   **Category Metrics**: Computes aggregated statistics (GBJ, GLS) for phenotype categories (e.g., "Circulatory system").
    *   **Stage 2 (Follow-up)**: Performed for significant hits from Stage 1.
        *   **Interaction**: Tests for `Inversion x Ancestry` interactions.
        *   **Stratification**: Performs per-ancestry association tests.
*   **Phenotype Engineering (`pheno.py`, `categories.py`)**:
    *   **Fetching**: Retrieves phenotype data from Google BigQuery or local Parquet caches.
    *   **Deduplication**: Identifies and removes highly correlated phenotypes (using Phi coefficient and case overlap) to reduce multiple testing burden.
    *   **Preprocessing**: Applies inclusion/exclusion criteria (min cases/controls, sex restrictions).

## Statistical Approach

The pipeline employs a two-stage design to balance computational efficiency with statistical rigor.

### Stage 1: Discovery (Overall)
For every inversion and every eligible phenotype:
1.  **Fit**: A logistic regression model is fitted: `Phenotype ~ Inversion + Age + Age^2 + Sex + PCs + Ancestry`.
2.  **Fallback**: If the standard MLE fit is unstable (e.g., due to perfect separation), the pipeline automatically attempts:
    *   **Firth Regression**: Penalized likelihood to reduce bias and handle separation.
    *   **Score Tests**: If model fitting remains difficult, a Rao score test or a parametric bootstrap score test is used to estimate p-values without full model convergence.
3.  **Aggregation**: Results are consolidated, and "Category" level p-values (GBJ, GLS) are computed to detect signal enrichment within disease groups.

### Stage 2: Refinement (Ancestry-Aware)
For phenotypes crossing the FDR threshold in Stage 1:
1.  **Interaction Test**: Tests if the inversion's effect varies by ancestry (`Inversion * Ancestry`).
2.  **Stratified Analysis**: Runs the regression separately within each major ancestry group to assess consistency and drive specific associations.

## Configuration

The pipeline is configured via a combination of environment variables and command-line arguments.

### Key Environment Variables
*   `WORKSPACE_CDR`: The BigQuery dataset ID for the cohort data.
*   `GOOGLE_PROJECT`: The Google Cloud project ID.
*   `FERROMIC_POPULATION_FILTER`: Restrict analysis to a specific population (default: "all").
*   `FERROMIC_PHENOTYPE_FILTER`: Restrict analysis to a single phenotype (for debugging).

### Command-Line Interface (CLI)
The pipeline can be launched and configured via `python3 -m phewas.cli`:

```bash
python3 -m phewas.cli [OPTIONS]
```

| Option | Description |
| :--- | :--- |
| `--min-cases-controls <N>` | Minimum number of cases and controls required for a phenotype to be tested. Overrides internal defaults. |
| `--pop-label <LABEL>` | Restrict analysis to participants with the specified population label (e.g., `eur`, `afr`). Matches ancestry labels from shared setup. |
| `--pheno <NAME>` | Run analysis for a single specific phenotype only. Useful for debugging or targeted replication. |

## Inputs and Outputs

### Inputs
*   **Genotypes**: Inversion dosages loaded from a TSV file (configured in `run.py`).
*   **Phenotypes**: Definitions and ICD codes loaded from a TSV URL.
*   **Covariates**: Demographics, PCs, and Ancestry data pulled from BigQuery/Storage.

### Outputs
*   **`phewas_results_<timestamp>.tsv`**: The master results file containing:
    *   `Phenotype`, `Category`, `Inversion`
    *   `P_Value`, `OR` (Odds Ratio), `Beta`, `N_Cases`, `N_Controls`
    *   `P_Source` (e.g., `lrt_mle`, `score_boot_firth`)
    *   Stage-2 results (if applicable): `P_LRT_AncestryxDosage`, per-ancestry OR/P-values.
*   **`category_summary.tsv`**: Aggregated statistics (GBJ, GLS) per disease category.
*   **Cache Directory (`phewas_cache/`)**: Stores intermediate Parquet files, metadata, and atomic results to allow resuming interrupted runs.

## Running the Pipeline

1.  **Setup**: Ensure you have the necessary Python dependencies (`numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `google-cloud-bigquery`).
2.  **Authentication**: Ensure you are authenticated with Google Cloud (`gcloud auth application-default login`) if accessing BigQuery.
3.  **Launch**:
    ```bash
    # Standard run
    python3 -m phewas.run

    # Run with custom thresholds
    python3 -m phewas.cli --min-cases-controls 200

    # Run for a single phenotype
    python3 -m phewas.cli --pheno "Type_2_diabetes"
    ```
4.  **Monitoring**: The pipeline prints detailed progress logs to stdout, including system resource usage (CPU/RAM) and worker status.

## Extras

The `phewas/extra/` directory contains supplementary scripts for specific follow-up tasks, such as custom control group analysis (`custom_control_followup.py`) or additional visualizations.

---
*Note: This pipeline relies on specific data structures in the All of Us Researcher Workbench or similar BigQuery-backed environments.*
