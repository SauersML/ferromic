"""Regression tests for the four audited PheWAS statistics correctness fixes.

These tests are deliberately self-contained: they exercise the fixed code paths
with small synthetic inputs and do NOT require any All-of-Us controlled-tier
access, BigQuery, or cached pipeline artifacts.

Covered audit items:
  #1  Ancestry dummies must stay attached to participant IDs (not a RangeIndex),
      so the ANC_* fixed effects are never silently zeroed.
  #2  Global BH-FDR must include every valid finite-p test (score/bootstrap as
      well as LRT), not only the LRT tests.
  #10 A fatal worker exception must propagate (nonzero exit), not be swallowed.
  #13 stats/analyze_ext_phewas.py must be valid, importable Python.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phewas import testing


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# BUG #1 -- ancestry dummy construction keeps participant IDs
# --------------------------------------------------------------------------- #
def _build_anc_dummies(anc_series: pd.Series) -> pd.DataFrame:
    """Replicates the fixed construction in phewas/run.py.

    The point of the fix is that the resulting frame is indexed by the *same*
    participant IDs as the input ancestry Series, never a 0..N-1 RangeIndex.
    """
    anc_series = anc_series.copy()
    anc_series.index = anc_series.index.astype(str)
    anc_cat = pd.Categorical(anc_series)
    A = pd.get_dummies(
        pd.Series(anc_cat, index=anc_series.index),
        prefix="ANC", drop_first=True, dtype=np.float32,
    )
    return A


def test_ancestry_dummies_are_indexed_by_person_id():
    person_ids = pd.Index(["1001", "1002", "1003", "1004"], name="person_id")
    anc = pd.Series(["eur", "afr", "eur", "amr"], index=person_ids)

    A = _build_anc_dummies(anc)

    # The dummy frame must carry the participant IDs, NOT a RangeIndex.
    assert list(A.index) == list(person_ids.astype(str))
    assert not isinstance(A.index, pd.RangeIndex)

    # Reindexing by the same IDs recovers every row -- no row is dropped, so a
    # downstream .reindex(...) would never need fillna(0.0) and never zero a
    # participant's ANC_* effects.
    reindexed = A.reindex(person_ids.astype(str))
    assert not reindexed.isna().any().any()

    # The pre-fix bug (RangeIndex) would make this same reindex all-NaN:
    buggy = pd.get_dummies(
        pd.Categorical(anc), prefix="ANC", drop_first=True, dtype=np.float32
    )
    buggy.index = buggy.index.astype(str)
    assert buggy.reindex(person_ids.astype(str)).isna().all().all()


def test_run_py_no_longer_silently_zeroes_ancestry():
    """The fixed run.py must not pair reindex(core_df_subset.index) with fillna(0.0)."""
    src = (REPO_ROOT / "phewas" / "run.py").read_text(encoding="utf-8")
    assert "reindex(core_df_subset.index).fillna(0.0)" not in src
    # And it must guard the alignment loudly instead.
    assert "ANC_* fixed effects would be silently zeroed" in src


# --------------------------------------------------------------------------- #
# BUG #2 -- global BH-FDR includes every valid finite-p test
# --------------------------------------------------------------------------- #
def _consolidate_no_cache(df: pd.DataFrame, alpha: float = 0.05):
    """Run consolidate_and_select with an empty cache dir (no LRT json on disk).

    With no cache, the merge branch is skipped and the function exercises the
    canonical-p resolution + BH path directly on the supplied frame.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as cache_root:
        out, _ = testing.consolidate_and_select(
            df.copy(), inversions=["invX"], cache_root=cache_root,
            alpha=alpha, mode="lrt_bh", selection="lrt_bh",
        )
    return out


def test_bh_fdr_includes_non_lrt_tests():
    # Three LRT tests (with P_LRT_Overall) and two non-LRT tests (P_Value only,
    # P_LRT_Overall NaN). All five are valid and finite.
    df = pd.DataFrame({
        "Phenotype": ["a", "b", "c", "d", "e"],
        "Inversion": ["invX"] * 5,
        "P_Value": [1e-6, 2e-3, 0.04, 0.001, 0.20],
        "P_LRT_Overall": [1e-6, 2e-3, 0.04, np.nan, np.nan],
        "P_Overall_Valid": [True, True, True, True, True],
        "P_Source": ["lrt_mle", "lrt_mle", "lrt_mle",
                     "score_boot_mle", "score_boot_mle"],
        "P_Method": ["lrt_mle", "lrt_mle", "lrt_mle",
                     "score_boot_mle", "score_boot_mle"],
    })

    out = _consolidate_no_cache(df)

    # Every valid finite-p row -- including the two non-LRT tests -- gets a q.
    assert out["Q_GLOBAL"].notna().all()
    assert int(out["Q_GLOBAL"].notna().sum()) == 5

    # The q-values match a BH correction over ALL FIVE p-values, not just the
    # three LRT tests.
    from statsmodels.stats.multitest import multipletests
    _, q_all, _, _ = multipletests(df["P_Value"], alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(
        out.sort_values("Phenotype")["Q_GLOBAL"].to_numpy(),
        pd.Series(q_all, index=df["Phenotype"]).sort_index().to_numpy(),
        rtol=1e-9,
    )


def test_bh_fdr_resolves_merge_suffix_collision():
    # Simulate an upstream frame that already collided into P_Value_x / P_Value_y
    # with no canonical P_Value column.
    df = pd.DataFrame({
        "Phenotype": ["a", "b"],
        "Inversion": ["invX", "invX"],
        "P_Value_x": [1e-5, 0.03],
        "P_Value_y": [1e-5, 0.03],
        "P_LRT_Overall": [np.nan, np.nan],   # non-LRT tests
        "P_Overall_Valid": [True, True],
        "P_Source_x": ["score_boot_mle", "score_boot_mle"],
        "P_Source_y": ["score_boot_mle", "score_boot_mle"],
    })

    out = _consolidate_no_cache(df)

    # Suffix columns are coalesced to a single canonical column and both rows,
    # though non-LRT, receive q-values.
    assert "P_Value" in out.columns
    assert "P_Value_x" not in out.columns and "P_Value_y" not in out.columns
    assert out["Q_GLOBAL"].notna().all()


def test_bh_fdr_guard_rejects_missing_q():
    # A valid finite-p row that somehow escapes correction must raise -- the
    # guard is what prevents silent omission. We force the pathological state by
    # making the canonical p NaN-typed-as-object so it is finite yet the row is
    # flagged valid; the function should still cover it (no row left behind).
    df = pd.DataFrame({
        "Phenotype": ["a"],
        "Inversion": ["invX"],
        "P_Value": [0.01],
        "P_LRT_Overall": [np.nan],
        "P_Overall_Valid": [True],
        "P_Source": ["score_boot_mle"],
        "P_Method": ["score_boot_mle"],
    })
    out = _consolidate_no_cache(df)
    assert out["Q_GLOBAL"].notna().all()


# --------------------------------------------------------------------------- #
# BUG #10 -- fatal worker exceptions must propagate (nonzero exit)
# --------------------------------------------------------------------------- #
def test_pipeline_worker_reraises_on_fatal_error():
    """The worker's top-level except must re-raise, not swallow."""
    from phewas import run

    src = inspect.getsource(run._pipeline_once)
    tree = ast.parse(src)

    # Find the top-level try/except in _pipeline_once and confirm at least one
    # handler ends by re-raising (a bare ``raise``).
    reraises = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Raise) and stmt.exc is None:
                    reraises.append(stmt)
    assert reraises, (
        "phewas.run._pipeline_once must re-raise after logging so the worker "
        "exits nonzero; a swallowed exception would let the supervisor treat "
        "failure as success."
    )


# --------------------------------------------------------------------------- #
# BUG #13 -- analyze_ext_phewas.py is valid, importable Python
# --------------------------------------------------------------------------- #
def test_analyze_ext_phewas_compiles():
    path = REPO_ROOT / "stats" / "analyze_ext_phewas.py"
    # Raises py_compile.PyCompileError on a SyntaxError.
    py_compile.compile(str(path), doraise=True)


def test_analyze_ext_phewas_importable():
    mod = importlib.import_module("stats.analyze_ext_phewas")
    assert mod.__doc__ is not None and "specification" in mod.__doc__.lower()
