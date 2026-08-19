"""Tests for the ancestry-stratified meta-analysis that consumes the two PC-source arms."""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "stats"))

import phewas_within_ancestry_meta as wam  # noqa: E402

INVERSION = "chr17-45585160-INV-706887"


def _write_run(directory, stamp, pop, arm, rows):
    name = f"phewas_results_{stamp}_pop-{pop}"
    if arm != "global":
        name += f"_pcs-{arm}"
    path = directory / f"{name}.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _row(beta, se, phenotype="Morbid_obesity", inversion=INVERSION):
    lo, hi = np.exp(beta - 1.959963984540054 * se), np.exp(beta + 1.959963984540054 * se)
    from scipy import stats as _st

    return {
        "Phenotype": phenotype,
        "Inversion": inversion,
        "Beta": beta,
        "OR": np.exp(beta),
        "CI_LO_OR": lo,
        "CI_HI_OR": hi,
        "P_Value": 2 * _st.norm.sf(abs(beta / se)),
        "N_Cases": 5000,
    }


@pytest.fixture()
def two_arm_dir(tmp_path):
    for pop, beta in (("eur", 0.10), ("afr", 0.12), ("amr", 0.08)):
        _write_run(tmp_path, "20260101000000", pop, "global", [_row(beta, 0.02)])
        _write_run(tmp_path, "20260101000001", pop, "within-ancestry", [_row(beta * 0.95, 0.02)])
    return tmp_path


def test_runs_are_discovered_with_their_arm(two_arm_dir):
    runs = wam.discover_runs(str(two_arm_dir))
    assert len(runs) == 6
    assert set(runs["arm"]) == {"global", "within-ancestry"}
    assert set(runs["population"]) == {"eur", "afr", "amr"}


def test_a_rerun_supersedes_the_earlier_stamp(two_arm_dir):
    _write_run(two_arm_dir, "20260601000000", "eur", "within-ancestry", [_row(0.5, 0.02)])
    runs = wam.discover_runs(str(two_arm_dir))
    row = runs[(runs["population"] == "eur") & (runs["arm"] == "within-ancestry")].iloc[0]
    # A stale run must not silently enter the meta-analysis alongside its replacement.
    assert row["stamp"] == "20260601000000"
    assert len(runs) == 6


def test_missing_results_directory_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="No stratified result tables"):
        wam.discover_runs(str(tmp_path))


def test_standard_error_comes_from_the_reported_interval():
    se = wam._standard_error(_row(0.2, 0.05))
    assert se == pytest.approx(0.05, rel=1e-6)


def test_standard_error_falls_back_to_the_p_value():
    row = _row(0.2, 0.05)
    row["CI_LO_OR"] = np.nan
    row["CI_HI_OR"] = np.nan
    assert wam._standard_error(row) == pytest.approx(0.05, rel=1e-3)


def test_meta_analysis_pools_across_populations(two_arm_dir):
    runs = wam.discover_runs(str(two_arm_dir))
    long = pd.concat(
        [wam.load_arm(runs, arm) for arm in ("global", "within-ancestry")], ignore_index=True
    )
    meta = wam.meta_analyse(long)

    assert set(meta["arm"]) == {"global", "within-ancestry"}
    fixed = meta[meta["arm"] == "global"].iloc[0]
    assert fixed["n_populations"] == 3
    # Equal standard errors, so the fixed-effect estimate is the mean of the three betas.
    assert fixed["beta_fixed"] == pytest.approx(np.mean([0.10, 0.12, 0.08]), rel=1e-6)
    assert fixed["se_fixed"] == pytest.approx(0.02 / np.sqrt(3), rel=1e-6)
    assert fixed["Locus"] == "17q21.31"


def test_comparison_reports_the_ratio_between_arms(two_arm_dir, tmp_path):
    runs = wam.discover_runs(str(two_arm_dir))
    long = pd.concat(
        [wam.load_arm(runs, arm) for arm in ("global", "within-ancestry")], ignore_index=True
    )
    meta = wam.meta_analyse(long)

    pooled = tmp_path / "pooled.tsv"
    pd.DataFrame([_row(0.10, 0.01)]).to_csv(pooled, sep="\t", index=False)

    comparison = wam.compare_arms(meta, str(pooled))
    row = comparison.iloc[0]
    # The fine-scale arm was generated at 95% of the control arm's effect.
    assert row["ratio_within_over_global_stratified"] == pytest.approx(0.95, rel=1e-6)
    assert bool(row["direction_preserved"])


def test_comparison_requires_both_arms(two_arm_dir):
    runs = wam.discover_runs(str(two_arm_dir))
    long = wam.load_arm(runs, "within-ancestry")
    meta = wam.meta_analyse(long)
    with pytest.raises(SystemExit, match="Both arms are required"):
        wam.compare_arms(meta, "/nonexistent/pooled.tsv")


def test_lambda_is_reported_per_population_and_arm(two_arm_dir):
    runs = wam.discover_runs(str(two_arm_dir))
    long = wam.load_arm(runs, "global")
    lam = wam.lambda_by_arm(long)
    assert set(lam["population"]) == {"eur", "afr", "amr"}
    assert (lam["arm"] == "global").all()
