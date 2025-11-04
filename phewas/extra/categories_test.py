"""Category-level "Big Phenotype" PheWAS test.

This module tests category-level composite phenotypes where each category defines
a "BIG PHENOTYPE": if an individual has ANY ICD code (phecode) in that category,
they are a CASE for that category; otherwise they are a CONTROL.

This is distinct from the category omnibus tests in phewas.categories which aggregate
individual phenotype p-values. Here we define NEW phenotypes at the category level
and run standard logistic regression on them.

Conservative FDR control is applied across all category tests.
"""

import os
import sys
import time
import warnings
from typing import Optional, Dict, List

# Threading config (same as main phewas)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    if _k not in os.environ:
        os.environ[_k] = "1"

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats

# Import phewas modules - reuse as much as possible
from .. import pheno
from .. import iox as io
from .. import models
from .. import pipes
from .. import run as phewas_run


# Configuration - aligned with main phewas
CACHE_DIR = "./phewas_cache"
LOCK_DIR = os.path.join(CACHE_DIR, "locks")
OUTPUT_DIR = os.path.join(CACHE_DIR, "extras_categories")

# Use same phenotype definitions as main phewas
PHENOTYPE_DEFINITIONS_URL = phewas_run.PHENOTYPE_DEFINITIONS_URL

# Controls: 16 PCs, sex, age, age^2, ancestry categories (same as main phewas)
NUM_PCS = 16
FDR_ALPHA = 0.05

# Inversion dosages file
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"

# URIs for covariates (same as main phewas)
PCS_URI = phewas_run.PCS_URI
SEX_URI = phewas_run.SEX_URI
RELATEDNESS_URI = phewas_run.RELATEDNESS_URI

# Phenotype restrictions (conservative, matching main phewas logic)
MIN_CASES_FILTER = 1_000
MIN_CONTROLS_FILTER = 1_000

# Output suppression for convergence warnings
warnings.filterwarnings('ignore', category=sm.tools.sm_exceptions.ConvergenceWarning)


def _log(msg: str):
    """Simple timestamped logging."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def build_category_phenotypes(
    pheno_defs_df: pd.DataFrame,
    core_index: pd.Index,
    cdr_codename: str,
    cache_dir: str,
) -> Dict[str, np.ndarray]:
    """Build category-level "big phenotypes" where ANY code in category = case.

    Returns:
        Dictionary mapping category name -> boolean array (True = case, False = control)
        indexed to core_index.
    """
    _log("Building category-level phenotypes...")

    category_phenotypes = {}

    for category, group in pheno_defs_df.groupby("disease_category"):
        _log(f"  Processing category: {category} ({len(group)} phecodes)")

        # Collect all case IDs across all phecodes in this category
        category_case_ids = set()

        for _, row in group.iterrows():
            pheno_name = row["sanitized_name"]

            # Load case IDs from cache (reusing phewas cache infrastructure)
            try:
                case_ids = pheno._case_ids_cached(pheno_name, cdr_codename, cache_dir)
                if case_ids:
                    category_case_ids.update(str(pid) for pid in case_ids)
            except Exception as e:
                _log(f"    WARN: Failed to load cases for {pheno_name}: {e}")
                continue

        # Convert to boolean array indexed to core_index
        case_mask = np.zeros(len(core_index), dtype=bool)
        if category_case_ids:
            case_positions = core_index.get_indexer(list(category_case_ids))
            valid_positions = case_positions[case_positions >= 0]
            if valid_positions.size > 0:
                case_mask[valid_positions] = True

        n_cases = case_mask.sum()
        _log(f"    Category {category}: {n_cases:,} cases")

        # Store if meets minimum thresholds
        if n_cases >= MIN_CASES_FILTER:
            category_phenotypes[category] = case_mask
        else:
            _log(f"    SKIP: {category} has only {n_cases:,} cases (< {MIN_CASES_FILTER:,})")

    _log(f"Built {len(category_phenotypes)} category phenotypes")
    return category_phenotypes


def run_logistic_regression(
    y: np.ndarray,
    X: pd.DataFrame,
    inversion_col: str,
) -> Dict:
    """Run logistic regression for a single category phenotype.

    Args:
        y: Binary outcome (1=case, 0=control)
        X: Covariate matrix including inversion dosage
        inversion_col: Name of inversion column in X

    Returns:
        Dictionary with regression results (Beta, SE, P, OR, CI, N_cases, N_controls, etc.)
    """
    # Prepare design matrix with constant
    X_design = sm.add_constant(X, has_constant='add')

    # Fit logistic regression
    try:
        model = sm.GLM(y, X_design, family=sm.families.Binomial())
        result = model.fit()

        # Extract inversion coefficient
        inv_idx = list(X_design.columns).index(inversion_col)
        beta = float(result.params.iloc[inv_idx])
        se = float(result.bse.iloc[inv_idx])
        z_stat = beta / se if se > 0 else np.nan
        p_value = float(result.pvalues.iloc[inv_idx])

        # Compute OR and CI
        or_val = np.exp(beta)
        ci_low = np.exp(beta - 1.96 * se)
        ci_high = np.exp(beta + 1.96 * se)

        # Sample sizes
        n_cases = int(y.sum())
        n_controls = int((1 - y).sum())
        n_total = len(y)

        return {
            "Beta": beta,
            "SE": se,
            "Z": z_stat,
            "P_Value": p_value,
            "OR": or_val,
            "OR_CI95_Low": ci_low,
            "OR_CI95_High": ci_high,
            "N_Total": n_total,
            "N_Cases": n_cases,
            "N_Controls": n_controls,
            "Converged": bool(result.converged),
            "LLF": float(result.llf) if hasattr(result, 'llf') else np.nan,
        }

    except Exception as e:
        _log(f"    ERROR in logistic regression: {e}")
        return {
            "Beta": np.nan,
            "SE": np.nan,
            "Z": np.nan,
            "P_Value": np.nan,
            "OR": np.nan,
            "OR_CI95_Low": np.nan,
            "OR_CI95_High": np.nan,
            "N_Total": int((~np.isnan(y)).sum()),
            "N_Cases": int(np.nansum(y)),
            "N_Controls": int(np.nansum(1 - y)),
            "Converged": False,
            "LLF": np.nan,
        }


def test_categories_for_inversion(
    inversion_name: str,
    category_phenotypes: Dict[str, np.ndarray],
    core_df: pd.DataFrame,
    ancestry_dummies: pd.DataFrame,
) -> pd.DataFrame:
    """Run category tests for a single inversion.

    Args:
        inversion_name: Name of inversion column
        category_phenotypes: Dict of category -> case mask
        core_df: Core covariate dataframe (with inversion dosage)
        ancestry_dummies: Ancestry dummy variables

    Returns:
        DataFrame with one row per category test
    """
    _log(f"Testing {len(category_phenotypes)} categories for inversion {inversion_name}")

    # Build full covariate matrix
    # Controls: inversion + sex + 16 PCs + AGE_c + AGE_c_sq + ancestry dummies
    pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
    covariate_cols = [inversion_name, "sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]

    X_base = core_df[covariate_cols].copy()
    X_full = pd.concat([X_base, ancestry_dummies.reindex(X_base.index).fillna(0.0)], axis=1)

    # Test each category
    results = []
    for category, case_mask in category_phenotypes.items():
        _log(f"  Testing category: {category}")

        # Build outcome vector aligned to X_full
        y = case_mask.astype(float)

        # Apply allowed controls logic: controls cannot be cases of ANY phenotype in category
        # For now, use all non-cases as controls (following the user's spec)
        # Note: could refine this with pan-category exclusions like main phewas

        # Check minimum controls
        n_controls = int((~case_mask).sum())
        if n_controls < MIN_CONTROLS_FILTER:
            _log(f"    SKIP: Only {n_controls:,} controls (< {MIN_CONTROLS_FILTER:,})")
            continue

        # Run regression
        reg_result = run_logistic_regression(y, X_full, inversion_name)

        # Add metadata
        reg_result["Category"] = category
        reg_result["Inversion"] = inversion_name

        results.append(reg_result)

        _log(f"    Result: OR={reg_result['OR']:.4f}, P={reg_result['P_Value']:.3e}, "
             f"N={reg_result['N_Cases']}/{reg_result['N_Controls']}")

    return pd.DataFrame(results)


def main():
    """Main entry point for category-level extras test."""
    _log("="*80)
    _log("Category-Level 'Big Phenotype' PheWAS Test")
    _log("="*80)

    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCK_DIR, exist_ok=True)

    # Get environment
    try:
        cdr_dataset_id = os.environ["WORKSPACE_CDR"]
        gcp_project = os.environ["GOOGLE_PROJECT"]
    except KeyError as e:
        _log(f"ERROR: Missing required environment variable: {e}")
        sys.exit(1)

    cdr_codename = cdr_dataset_id.split(".")[-1]
    _log(f"CDR: {cdr_codename}")
    _log(f"Project: {gcp_project}")

    # Load phenotype definitions
    _log("Loading phenotype definitions...")
    pheno_defs_df = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)
    _log(f"Loaded {len(pheno_defs_df)} phenotype definitions")

    categories = pheno_defs_df["disease_category"].unique()
    _log(f"Found {len(categories)} unique categories: {sorted(categories)}")

    # Load shared covariates (reusing phewas infrastructure)
    _log("Loading shared covariates...")

    from google.cloud import bigquery
    bq_client = bigquery.Client(project=gcp_project)

    # Demographics
    demographics_cache_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet")
    demographics_df = io.get_cached_or_generate(
        demographics_cache_path,
        io.load_demographics_with_stable_age,
        bq_client=bq_client,
        cdr_id=cdr_dataset_id,
        lock_dir=LOCK_DIR,
    )

    # PCs
    pcs_cache = os.path.join(
        CACHE_DIR,
        f"pcs_{NUM_PCS}_{phewas_run._source_key(gcp_project, PCS_URI, NUM_PCS)}.parquet",
    )
    pc_df = io.get_cached_or_generate(
        pcs_cache,
        io.load_pcs,
        gcp_project,
        PCS_URI,
        NUM_PCS,
        validate_num_pcs=NUM_PCS,
        lock_dir=LOCK_DIR,
    )

    # Sex
    sex_cache = os.path.join(
        CACHE_DIR,
        f"genetic_sex_{phewas_run._source_key(gcp_project, SEX_URI)}.parquet",
    )
    sex_df = io.get_cached_or_generate(
        sex_cache,
        io.load_genetic_sex,
        gcp_project,
        SEX_URI,
        lock_dir=LOCK_DIR,
    )

    # Relatedness
    _log("Loading relatedness filter...")
    related_ids_to_remove = io.load_related_to_remove(
        gcp_project=gcp_project,
        RELATEDNESS_URI=RELATEDNESS_URI
    )

    # Merge covariates
    demographics_df.index = demographics_df.index.astype(str)
    pc_df.index = pc_df.index.astype(str)
    sex_df.index = sex_df.index.astype(str)

    shared_covariates_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
    shared_covariates_df = shared_covariates_df[~shared_covariates_df.index.isin(related_ids_to_remove)]

    _log(f"Shared covariates: {len(shared_covariates_df):,} participants")

    # Ancestry labels
    _log("Loading ancestry labels...")
    ancestry_cache = os.path.join(
        CACHE_DIR,
        f"ancestry_labels_{phewas_run._source_key(gcp_project, PCS_URI)}.parquet",
    )
    ancestry = io.get_cached_or_generate(
        ancestry_cache,
        io.load_ancestry_labels,
        gcp_project,
        LABELS_URI=PCS_URI,
        lock_dir=LOCK_DIR,
    )

    anc_series = ancestry.reindex(shared_covariates_df.index)["ANCESTRY"]
    anc_cat = pd.Categorical(anc_series)
    ancestry_dummies = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True, dtype=np.float32)
    ancestry_dummies.index = ancestry_dummies.index.astype(str)

    _log(f"Ancestry categories: {sorted(anc_cat.categories.tolist())}")

    # Populate phenotype caches
    _log("Populating phenotype caches...")
    try:
        pheno.populate_caches_prepass(
            pheno_defs_df,
            bq_client,
            cdr_dataset_id,
            shared_covariates_df.index,
            CACHE_DIR,
            cdr_codename
        )
    except Exception as e:
        _log(f"WARN: Cache prepass encountered error: {e}")

    # Build category phenotypes
    category_phenotypes = build_category_phenotypes(
        pheno_defs_df,
        shared_covariates_df.index,
        cdr_codename,
        CACHE_DIR,
    )

    if not category_phenotypes:
        _log("ERROR: No category phenotypes passed filters. Exiting.")
        sys.exit(1)

    # Load inversions
    _log("Loading inversion dosages...")
    dosages_path = phewas_run._find_upwards(INVERSION_DOSAGES_FILE)

    # Determine target inversions (use same logic as main phewas)
    # For simplicity, test all inversions that pass variance filter
    # User can override with TARGET_INVERSIONS if needed
    target_inversions = getattr(phewas_run, 'TARGET_INVERSIONS', None)
    if target_inversions is None:
        # Read header to get all inversions
        hdr = pd.read_csv(dosages_path, sep="\t", nrows=0).columns.tolist()
        id_candidates = {"SampleID", "sample_id", "person_id", "research_id", "participant_id", "ID"}
        id_col = next((c for c in hdr if c in id_candidates), None)
        available_inversions = set(hdr) - ({id_col} if id_col else set())
        target_inversions = available_inversions
        _log(f"Testing all {len(target_inversions)} inversions from dosages file")
    else:
        if isinstance(target_inversions, str):
            target_inversions = {target_inversions}
        _log(f"Testing {len(target_inversions)} specified inversions")

    # Run tests for each inversion
    all_results = []

    for inv_name in sorted(target_inversions):
        _log(f"\n{'='*80}")
        _log(f"Processing inversion: {inv_name}")
        _log(f"{'='*80}")

        try:
            # Load inversion dosages
            inversion_cache_path = os.path.join(
                CACHE_DIR,
                f"inversion_{models.safe_basename(inv_name)}_{phewas_run._source_key(dosages_path, inv_name)}.parquet"
            )

            inversion_df = io.get_cached_or_generate(
                inversion_cache_path,
                io.load_inversions,
                inv_name,
                dosages_path,
                validate_target=inv_name,
                lock_dir=LOCK_DIR,
            )
            inversion_df.index = inversion_df.index.astype(str)

        except io.LowVarianceInversionError as exc:
            _log(f"SKIP: Inversion {inv_name} has low variance (std={exc.std:.4f})")
            continue
        except Exception as e:
            _log(f"ERROR loading inversion {inv_name}: {e}")
            continue

        # Merge with covariates
        core_df = shared_covariates_df.join(inversion_df, how="inner")

        # Center age
        age_mean = core_df['AGE'].mean()
        core_df['AGE_c'] = core_df['AGE'] - age_mean
        core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2

        _log(f"Core data: {len(core_df):,} participants with inversion + covariates")

        # Subset category phenotypes to core_df index
        category_phenotypes_subset = {}
        for cat, case_mask in category_phenotypes.items():
            # Re-index to core_df
            case_mask_subset = np.zeros(len(core_df), dtype=bool)
            shared_idx = core_df.index.intersection(shared_covariates_df.index)
            if len(shared_idx) > 0:
                orig_positions = shared_covariates_df.index.get_indexer(shared_idx)
                new_positions = core_df.index.get_indexer(shared_idx)
                case_mask_subset[new_positions] = case_mask[orig_positions]
            category_phenotypes_subset[cat] = case_mask_subset

        # Test categories for this inversion
        inv_results = test_categories_for_inversion(
            inv_name,
            category_phenotypes_subset,
            core_df,
            ancestry_dummies,
        )

        if not inv_results.empty:
            all_results.append(inv_results)
            _log(f"Completed {len(inv_results)} category tests for {inv_name}")

    # Combine all results
    if not all_results:
        _log("ERROR: No results generated. Exiting.")
        sys.exit(1)

    combined_results = pd.concat(all_results, ignore_index=True)
    _log(f"\nTotal results: {len(combined_results)} category-inversion tests")

    # Apply FDR correction
    _log("Applying FDR correction...")
    valid_mask = pd.notna(combined_results["P_Value"]) & np.isfinite(combined_results["P_Value"])

    combined_results["Q_FDR"] = np.nan
    if valid_mask.sum() > 0:
        _, q_vals, _, _ = multipletests(
            combined_results.loc[valid_mask, "P_Value"],
            alpha=FDR_ALPHA,
            method="fdr_bh"
        )
        combined_results.loc[valid_mask, "Q_FDR"] = q_vals

    # Sort by p-value
    combined_results = combined_results.sort_values("P_Value", na_position='last')

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "category_phewas_results.tsv")
    combined_results.to_csv(output_path, sep="\t", index=False, na_rep="NA")
    _log(f"\nResults saved to: {output_path}")

    # Print summary
    _log("\n" + "="*80)
    _log("TOP RESULTS (by P-value)")
    _log("="*80)

    top_results = combined_results.head(20)
    for _, row in top_results.iterrows():
        _log(
            f"{row['Category']:30s} | {row['Inversion']:35s} | "
            f"OR={row['OR']:7.4f} | P={row['P_Value']:.3e} | Q={row['Q_FDR']:.3e} | "
            f"N={row['N_Cases']:6,}/{row['N_Controls']:6,}"
        )

    _log("\n" + "="*80)
    _log("Category-level PheWAS test complete!")
    _log("="*80)


if __name__ == "__main__":
    main()
