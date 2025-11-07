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


def test_pipeline_final_results_lambda_is_reasonable():
    """Run the CLI pipeline, parse final results, and evaluate genomic inflation."""
    with temp_workspace() as tmpdir, preserve_run_globals():
        core_data, base_phenos = make_realistic_followup_dataset(N=4200, seed=2025)
        phenos = _generate_dense_phenotypes(core_data, base_phenos, extra=48)

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
        assert np.isfinite(lambda_gc)
        assert lambda_gc < 1.1, "Inflation factor should remain below nominal bound"
