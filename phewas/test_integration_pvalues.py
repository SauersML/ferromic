"""Integration test that exercises the end-to-end PheWAS pipeline."""

from __future__ import annotations

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from stats import qq_plot

from phewas import run, pheno, models
from phewas.tests import (
    TEST_TARGET_INVERSION,
    TEST_CDR_CODENAME,
    temp_workspace,
    preserve_run_globals,
    make_realistic_followup_dataset,
    prime_all_caches_for_run,
    make_local_pheno_defs_tsv,
    write_tsv,
)

pytestmark = pytest.mark.timeout(300)


@pytest.fixture(params=[False, True], ids=["real_inversion", "null_inversion"])
def inversion_mode(request):
    """Fixture to run test with both real and null inversion."""
    return request.param


def _generate_dense_phenotypes(
    core_data: dict[str, pd.DataFrame],
    base_phenos: dict[str, dict[str, object]],
    *,
    extra: int = 48,
    seed: int = 20240320,
) -> dict[str, dict[str, object]]:
    """Return an expanded phenotype dictionary with many synthetic case profiles."""
    rng = np.random.default_rng(seed)
    phenos = {name: dict(payload) for name, payload in base_phenos.items()}

    person_index = core_data["demographics"].index.to_numpy()

    for idx in range(extra):
        target_cases = int(round(rng.uniform(0.25, 0.55) * len(person_index)))
        max_cases = max(len(person_index) - 1000, 1000)
        target_cases = min(max(target_cases, 1000), max_cases)
        selected = rng.choice(person_index, size=target_cases, replace=False)
        cases = set(selected)

        sanitized = f"Synthetic_{idx:04d}"
        phenos[sanitized] = {
            "disease": sanitized,
            "category": f"synthetic_category_{idx:04d}",
            "cases": cases,
        }

    return phenos


def test_pipeline_final_results_lambda_is_reasonable(inversion_mode):
    """Run the CLI pipeline, parse final results, and evaluate genomic inflation.
    
    Args:
        inversion_mode: If True, replace inversion dosages with random noise (no true associations)
    """
    null_inversion = inversion_mode
    with temp_workspace() as tmpdir, preserve_run_globals():
        core_data, base_phenos = make_realistic_followup_dataset(N=4200, seed=2025)
        phenos = _generate_dense_phenotypes(core_data, base_phenos, extra=97)

        cache_root = Path(tmpdir) / "phewas_cache"
        run.CACHE_DIR = str(cache_root)
        run.LOCK_DIR = str(cache_root / "locks")
        run.TARGET_INVERSIONS = [TEST_TARGET_INVERSION]
        run.NUM_PCS = core_data["pcs"].shape[1]
        run.MIN_NEFF_FILTER = 0
        run.MLE_REFIT_MIN_NEFF = 0
        run.FDR_ALPHA = 0.05
        run.LRT_SELECT_ALPHA = 0.1
        run.MIN_CASES_FILTER = 20
        run.MIN_CONTROLS_FILTER = 20
        pheno.MIN_CASES_FILTER = run.MIN_CASES_FILTER
        pheno.MIN_CONTROLS_FILTER = run.MIN_CONTROLS_FILTER
        run.MASTER_RESULTS_CSV = str(Path(tmpdir) / "final_results.tsv")
        run.INVERSION_DOSAGES_FILE = str(Path(tmpdir) / "dosages.tsv")
        run.PHENOTYPE_FILTER = None
        run.CLI_MIN_CASES_CONTROLS_OVERRIDE = run.MIN_CASES_FILTER

        prior_models_ctx = dict(models.CTX)
        models.CTX = {
            **models.CTX,
            "MIN_CASES_FILTER": run.MIN_CASES_FILTER,
            "MIN_CONTROLS_FILTER": run.MIN_CONTROLS_FILTER,
            "MIN_NEFF_FILTER": 0,
        }

        # Optionally replace inversion with null (random noise)
        if null_inversion:
            print("\n" + "="*70)
            print("NULL INVERSION MODE: Replacing inversion dosages with random noise")
            print("="*70 + "\n")
            rng = np.random.default_rng(seed=99999)
            null_dosages = rng.normal(0.0, 0.58, size=len(core_data["inversion_main"]))
            # Replace in core_data so it gets cached correctly
            core_data["inversion_main"] = pd.DataFrame(
                {TEST_TARGET_INVERSION: null_dosages},
                index=core_data["inversion_main"].index
            )
            dosages_df = core_data["inversion_main"].reset_index().rename(columns={"person_id": "SampleID"})
        else:
            dosages_df = (
                core_data["inversion_main"].reset_index().rename(columns={"person_id": "SampleID"})
            )
        write_tsv(run.INVERSION_DOSAGES_FILE, dosages_df)

        defs_df = prime_all_caches_for_run(
            core_data,
            phenos,
            TEST_CDR_CODENAME,
            TEST_TARGET_INVERSION,
            cache_dir=run.CACHE_DIR,
        )
        defs_path = make_local_pheno_defs_tsv(defs_df, tmpdir)
        run.PHENOTYPE_DEFINITIONS_URL = str(defs_path)

        original_prequeue = pheno._prequeue_should_run

        def _force_queue(
            pheno_info,
            core_index,
            allowed_mask_by_cat,
            sex_vec,
            *p_args,
            return_case_idx=False,
            **p_kwargs,
        ):
            outcome = original_prequeue(
                pheno_info,
                core_index,
                allowed_mask_by_cat,
                sex_vec,
                *p_args,
                return_case_idx=return_case_idx,
                **p_kwargs,
            )

            if return_case_idx:
                ok = bool(outcome[0]) if isinstance(outcome, tuple) else bool(outcome)
                if ok:
                    return outcome
            else:
                if bool(outcome[0]) if isinstance(outcome, tuple) else bool(outcome):
                    return outcome

            sanitized = (
                pheno_info.get("sanitized_name")
                or pheno_info.get("name")
            )
            if not sanitized:
                return outcome

            cache_path = Path(run.CACHE_DIR) / f"pheno_{sanitized}_{TEST_CDR_CODENAME}.parquet"
            if not cache_path.exists():
                return outcome

            try:
                case_frame = pd.read_parquet(cache_path, columns=["is_case"])
            except Exception:
                return outcome

            case_ids = case_frame.index[case_frame["is_case"] == 1].astype(str)
            if case_ids.empty:
                return outcome

            case_idx = core_index.get_indexer(case_ids)
            case_idx = case_idx[case_idx >= 0]
            if case_idx.size == 0:
                return outcome

            category = pheno_info.get("disease_category")
            allowed_mask = allowed_mask_by_cat.get(category)
            if allowed_mask is None:
                allowed_mask = np.ones(core_index.size, dtype=bool)

            ctrl_idx = np.setdiff1d(np.flatnonzero(allowed_mask), case_idx, assume_unique=False)
            if ctrl_idx.size < run.MIN_CONTROLS_FILTER:
                return outcome

            if return_case_idx:
                return True, case_idx.astype(np.int32)
            return True

        with contextlib.ExitStack() as stack:
            stack.callback(lambda prev=prior_models_ctx: setattr(models, "CTX", prev))
            stack.enter_context(patch("phewas.run.bigquery.Client", MagicMock()))
            stack.enter_context(patch("phewas.run.io.load_related_to_remove", return_value=set()))

            stack.enter_context(
                patch(
                    "phewas.run.io.load_pcs",
                    lambda *args, **kwargs: core_data["pcs"].iloc[:, : run.NUM_PCS],
                )
            )
            stack.enter_context(
                patch(
                    "phewas.run.io.load_genetic_sex",
                    lambda *args, **kwargs: core_data["sex"],
                )
            )
            stack.enter_context(
                patch(
                    "phewas.run.io.load_ancestry_labels",
                    lambda *args, **kwargs: core_data["ancestry"],
                )
            )
            stack.enter_context(
                patch(
                    "phewas.run.io.load_demographics_with_stable_age",
                    lambda *args, **kwargs: core_data["demographics"],
                )
            )
            stack.enter_context(
                patch("phewas.pheno.populate_caches_prepass", lambda *args, **kwargs: None)
            )
            stack.enter_context(
                patch("phewas.pheno._prequeue_should_run", _force_queue)
            )
            stack.enter_context(patch("phewas.pipes.POOL_PROCS_PER_INV", 1))
            stack.enter_context(patch("phewas.run.supervisor_main", lambda: run._pipeline_once()))

            from phewas import cli

            cli.main([])

        final_results = pd.read_csv(run.MASTER_RESULTS_CSV, sep="\t")
        assert not final_results.empty
        assert qq_plot.P_COL in final_results.columns

        parsed = pd.to_numeric(final_results[qq_plot.P_COL], errors="coerce").dropna()
        assert not parsed.empty, "Expected at least one valid p-value entry"

        lambda_gc = qq_plot.calculate_lambda_gc(parsed.to_numpy())
        
        # Calculate lambda for synthetic (null) phenotypes only
        synthetic_results = final_results[final_results['Phenotype'].str.startswith('Synthetic_')]
        synthetic_pvalues = pd.to_numeric(synthetic_results[qq_plot.P_COL], errors='coerce').dropna()
        lambda_gc_synthetic = qq_plot.calculate_lambda_gc(synthetic_pvalues.to_numpy()) if len(synthetic_pvalues) > 0 else np.nan
        
        mode_label = "NULL INVERSION (random noise)" if null_inversion else "REAL INVERSION (true associations)"
        print(f"\n{'='*70}")
        print(f"TEST MODE: {mode_label}")
        print(f"{'='*70}")
        print(f"GENOMIC INFLATION FACTOR (Lambda) - ALL: {lambda_gc:.6f}")
        print(f"GENOMIC INFLATION FACTOR (Lambda) - SYNTHETIC ONLY: {lambda_gc_synthetic:.6f}")
        print(f"Total associations tested: {len(final_results)}")
        print(f"  - Base phenotypes: {len(final_results) - len(synthetic_results)}")
        print(f"  - Synthetic phenotypes: {len(synthetic_results)}")
        print(f"Valid p-values: {len(parsed)}")
        print(f"Test threshold: < 1.1")
        print(f"Test passes: {lambda_gc < 1.1}")
        print(f"{'='*70}\n")
        
        # Analyze associations
        print(f"\n{'='*70}")
        print("ASSOCIATION ANALYSIS")
        print(f"{'='*70}")
        print(f"\nColumns in results: {list(final_results.columns)}")
        
        # Show top significant associations
        sig_threshold = 0.05
        sig_results = final_results[final_results[qq_plot.P_COL] < sig_threshold].copy()
        sig_results = sig_results.sort_values(qq_plot.P_COL)
        
        print(f"\nSignificant associations (p < {sig_threshold}): {len(sig_results)}")
        if len(sig_results) > 0:
            print("\nTop 10 most significant associations:")
            display_cols = [col for col in ['Phenotype', 'P_Value_x', 'OR', 'Beta', 'N_Total', 'N_Cases', 'N_Controls'] if col in sig_results.columns]
            print(sig_results[display_cols].head(10).to_string(index=False))
        
        # Check for base phenotypes specifically
        base_pheno_names = ['Metabolic_strong', 'Neuro_moderate', 'Inflammation_low']
        base_results = final_results[final_results['Phenotype'].isin(base_pheno_names)]
        if len(base_results) > 0:
            print(f"\n\nBase phenotypes (designed with inversion effects):")
            print(base_results[display_cols].to_string(index=False))
        
        # Summary statistics
        print(f"\n\nSummary of all {len(final_results)} associations:")
        print(f"  Bonferroni threshold (0.05/{len(final_results)}): {0.05/len(final_results):.2e}")
        bonf_sig = final_results[final_results[qq_plot.P_COL] < 0.05/len(final_results)]
        print(f"  Bonferroni significant: {len(bonf_sig)}")
        print(f"  Nominal significant (p<0.05): {len(sig_results)}")
        print(f"  Non-significant: {len(final_results) - len(sig_results)}")
        
        print(f"\n{'='*70}\n")
        
        assert np.isfinite(lambda_gc)
        assert lambda_gc < 1.1, f"Inflation factor {lambda_gc:.6f} should remain below nominal bound 1.1"
