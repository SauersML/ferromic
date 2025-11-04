# Category-Level "Big Phenotype" PheWAS Test

## Overview

This extra test (`categories_test.py`) performs PheWAS analysis on **category-level composite phenotypes**.

Unlike the main PheWAS pipeline which tests individual phecodes, this test defines "BIG PHENOTYPES" at the disease category level:

- **CASE**: Individual has ANY ICD code (phecode) in that category
- **CONTROL**: Individual has NO ICD codes in that category

This is **distinct** from the category omnibus tests in `phewas/categories.py`, which aggregate p-values from individual phenotype tests. Here we define entirely new phenotypes and run standard logistic regression on them.

## Key Features

- **Reuses infrastructure**: Imports functions from main phewas code (`pheno.py`, `iox.py`, `models.py`, `run.py`)
- **Same controls**: 16 PCs, sex, age, age², ancestry categories (matching main phewas design)
- **Conservative**: FDR correction applied across all category tests
- **Single output**: TSV file with all results and statistical details

## Phenotype Definition

For each disease category in the phenotype definitions:

1. Collect ALL phecodes (ICD codes) belonging to that category
2. Mark individual as CASE if they have ANY code in the category
3. Mark as CONTROL otherwise
4. Apply same filters: minimum 1,000 cases, 1,000 controls

## Statistical Approach

- **Model**: Logistic regression
- **Covariates**:
  - Inversion dosage
  - Sex (genetic)
  - 16 principal components
  - Age (centered)
  - Age² (centered squared)
  - Ancestry category indicators (drop_first)
- **Inference**: Wald test on inversion coefficient
- **Multiple testing**: FDR (Benjamini-Hochberg) at α=0.05 across all category-inversion tests

## Usage

```bash
# From project root, in notebook/Vertex environment with dependencies:
python -m phewas.extra.categories_test
```

Or import and run programmatically:
```python
from phewas.extra import categories_test
categories_test.main()
```

## Output

Results saved to: `./phewas_cache/extras_categories/category_phewas_results.tsv`

### Output Columns

| Column | Description |
|--------|-------------|
| Category | Disease category name |
| Inversion | Inversion identifier |
| Beta | Log-odds coefficient for inversion |
| SE | Standard error of Beta |
| Z | Z-statistic (Beta/SE) |
| P_Value | Two-sided p-value from Wald test |
| OR | Odds ratio (exp(Beta)) |
| OR_CI95_Low | 95% CI lower bound for OR |
| OR_CI95_High | 95% CI upper bound for OR |
| N_Total | Total sample size |
| N_Cases | Number of cases |
| N_Controls | Number of controls |
| Converged | Whether GLM converged |
| LLF | Log-likelihood of fitted model |
| Q_FDR | FDR-adjusted q-value |

## Differences from Main PheWAS

| Aspect | Main PheWAS | Categories Test |
|--------|-------------|-----------------|
| Phenotype | Individual phecodes | Category composites |
| Definition | Specific ICD code set | ANY code in category |
| # Tests | ~1,500+ phecodes | ~20 categories |
| Post-hoc | Category omnibus after | Direct category tests |

## Expected Case Counts

Categories will have **LOTS** of cases because:
- Multiple phecodes contribute
- OR logic: ANY single code → CASE
- More inclusive than individual phenotypes

This is intentional and matches the user specification for "BIG PHENOTYPES".

## Requirements

Same as main PheWAS:
- Google Cloud environment (Vertex AI Workbench)
- Access to All of Us CDR
- Environment variables: `WORKSPACE_CDR`, `GOOGLE_PROJECT`
- Phenotype caches populated (via main PheWAS or prepass)

## Notes

- Conservative FDR control ensures valid inference despite high case counts
- Reuses phenotype caches from main PheWAS (no duplicate BigQuery queries)
- Normal control logic applies: controls cannot be cases of ANY phenotype in category
- Compatible with all inversions that pass variance filter in main PheWAS
