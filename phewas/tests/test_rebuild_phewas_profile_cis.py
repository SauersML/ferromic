"""Tests for stats/rebuild_phewas_profile_cis.py and the profile-CI guard.

The rebuild replaces the profile intervals that went through the defective
offset refit (fixed in c364c189) with the interval implied by Beta and the LRT
p-value. These tests pin down the arithmetic, the text-preserving table edit,
idempotency, agreement with the corrected profile code on simulated data, and
the guard that rejects collapsed profile bounds inside the pipeline.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
from statistics import NormalDist

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "stats" / "rebuild_phewas_profile_cis.py"

_spec = importlib.util.spec_from_file_location("rebuild_phewas_profile_cis", SCRIPT)
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------
def test_rebuilt_bounds_match_independent_formula():
    beta, p = -0.07285866447191801, 4.5e-6
    lo, hi = rb.rebuilt_bounds(beta, p)
    z = NormalDist().inv_cdf(1 - p / 2)
    se = abs(beta) / z
    assert lo == pytest.approx(math.exp(beta - 1.959963984540054 * se), rel=1e-12)
    assert hi == pytest.approx(math.exp(beta + 1.959963984540054 * se), rel=1e-12)
    # symmetric on the log scale, estimate at the centre
    assert math.log(lo) + math.log(hi) == pytest.approx(2 * beta, abs=1e-12)
    # the pipeline's own missing-CI fallback (phewas/run.py) uses 1.96; agree to 1e-4
    assert lo == pytest.approx(math.exp(beta - 1.96 * se), rel=1e-4)


def test_rebuilt_bounds_exclude_one_exactly_when_p_below_alpha():
    for beta in (-0.3, -0.05, 0.02, 0.4):
        for p in (0.049, 0.051, 1e-3, 0.5):
            lo, hi = rb.rebuilt_bounds(beta, p)
            excludes_one = hi < 1.0 or lo > 1.0
            assert excludes_one == (p < 0.05)


def test_rebuilt_bounds_reject_degenerate_inputs():
    with pytest.raises(ValueError):
        rb.rebuilt_bounds(0.0, 0.5)
    with pytest.raises(ValueError):
        rb.rebuilt_bounds(0.1, 0.0)
    with pytest.raises(ValueError):
        rb.rebuilt_bounds(0.1, 1.0)
    with pytest.raises(ValueError):
        rb.rebuilt_bounds(float("nan"), 0.1)


def test_fmt_ci_matches_pipeline_format():
    assert rb.fmt_ci(0.9292307913797571, 0.9508199943358401) == "0.929,0.951"
    assert rb.fmt_ci(0.0004, 1200.0) == "4.000e-04,1.200e+03"
    assert rb.fmt_ci(0.0, float("inf")) == "0.000,+inf"


# ---------------------------------------------------------------------------
# Text-preserving table rewrite
# ---------------------------------------------------------------------------
MAIN_HEADER = [
    "Phenotype", "Beta", "OR", "P_Source_x", "OR_CI95", "CI_Method", "CI_Sided", "CI_Label",
    "CI_Valid", "CI_LO_OR", "CI_HI_OR", "Inversion", "P_LRT_Overall",
    "CI_Method_DISPLAY", "OR_CI95_DISPLAY", "CI_LO_OR_DISPLAY", "CI_HI_OR_DISPLAY",
    "EUR_OR", "EUR_P", "EUR_P_Source", "EUR_CI_Method", "EUR_CI_LO_OR", "EUR_CI_HI_OR", "EUR_CI95",
]


def _row(**kw):
    base = {c: "" for c in MAIN_HEADER}
    base.update(kw)
    return [base[c] for c in MAIN_HEADER]


def _write_main(path, rows, newline="\r\n"):
    text = newline.join(["\t".join(MAIN_HEADER)] + ["\t".join(r) for r in rows]) + newline
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _main_rows():
    return [
        # LRT-backed profile row: must be rebuilt (overall and EUR block)
        _row(Phenotype="Lump", Beta="-0.07285866447191801", OR="0.929732224935175", P_Source_x="lrt_mle",
             OR_CI95="0.929,0.951", CI_Method="profile", CI_Sided="two", CI_Valid="True",
             CI_LO_OR="0.9292307913797571", CI_HI_OR="0.9508199943358401", Inversion="chr17",
             P_LRT_Overall="4.5e-06", CI_Method_DISPLAY="profile", OR_CI95_DISPLAY="0.929,0.951",
             CI_LO_OR_DISPLAY="0.9292307913797571", CI_HI_OR_DISPLAY="0.9508199943358401",
             EUR_OR="0.924793818106945", EUR_P="1.1374642671390772e-05", EUR_P_Source="lrt_mle",
             EUR_CI_Method="profile", EUR_CI_LO_OR="0.9235057878583991", EUR_CI_HI_OR="0.9493852902399251",
             EUR_CI95="0.924,0.949"),
        # Wald row: untouched
        _row(Phenotype="Nevi", Beta="-0.10711807083107701", OR="0.898", P_Source_x="lrt_mle",
             OR_CI95="0.870,0.928", CI_Method="wald_mle", CI_Sided="two", CI_Valid="True",
             CI_LO_OR="0.8700", CI_HI_OR="0.9280", Inversion="chr17", P_LRT_Overall="1e-10",
             CI_Method_DISPLAY="wald_mle", OR_CI95_DISPLAY="0.870,0.928",
             CI_LO_OR_DISPLAY="0.8700", CI_HI_OR_DISPLAY="0.9280"),
        # Firth penalised profile row: untouched (different, correct code path)
        _row(Phenotype="Rare", Beta="0.5", OR="1.648", P_Source_x="score_boot_mle",
             OR_CI95="1.1,2.4", CI_Method="profile_penalized", CI_Sided="two", CI_Valid="True",
             CI_LO_OR="1.1", CI_HI_OR="2.4", Inversion="chr6", P_LRT_Overall="nan"),
        # profile row whose p-value is not an LRT: left as is but reported
        _row(Phenotype="Score", Beta="0.2", OR="1.221", P_Source_x="score_chi2",
             OR_CI95="1.05,1.40", CI_Method="profile", CI_Sided="two", CI_Valid="True",
             CI_LO_OR="1.05", CI_HI_OR="1.40", Inversion="chr6", P_LRT_Overall="0.01"),
        # row with no model at all
        _row(Phenotype="Empty", Beta="nan", OR="nan", Inversion="chr8", P_LRT_Overall="nan"),
    ]


def test_rebuild_rewrites_only_lrt_backed_profile_rows(tmp_path):
    path = tmp_path / rb.MAIN_TABLE
    _write_main(path, _main_rows())
    before = path.read_bytes().decode().split("\r\n")

    reports, problems = rb.run(tmp_path, write=True, only={rb.MAIN_TABLE}, verbose=False)
    assert problems == []
    by_block = {(r.table, r.block): r for r in reports}
    overall = by_block[(str(path), "overall")]
    eur = by_block[(str(path), "EUR")]
    assert overall.rebuilt == 1 and eur.rebuilt == 1
    assert overall.skipped == {"profile interval without an LRT p-value (left as is)": 1}

    raw = path.read_bytes().decode()
    assert "\r\n" in raw and raw.endswith("\r\n")
    after = raw.split("\r\n")
    assert after[0] == before[0]
    # every row except the rebuilt one is byte-identical
    for i in (2, 3, 4, 5):
        assert after[i] == before[i]

    cells = dict(zip(MAIN_HEADER, after[1].split("\t")))
    lo, hi = rb.rebuilt_bounds(-0.07285866447191801, 4.5e-6)
    assert cells["CI_Method"] == "lrt_quadratic"
    assert float(cells["CI_LO_OR"]) == pytest.approx(lo, rel=1e-12)
    assert float(cells["CI_HI_OR"]) == pytest.approx(hi, rel=1e-12)
    assert cells["OR_CI95"] == rb.fmt_ci(lo, hi) == "0.901,0.959"
    assert cells["CI_Method_DISPLAY"] == "lrt_quadratic"
    assert cells["OR_CI95_DISPLAY"] == cells["OR_CI95"]
    assert cells["CI_LO_OR_DISPLAY"] == cells["CI_LO_OR"]
    assert cells["CI_HI_OR_DISPLAY"] == cells["CI_HI_OR"]
    # untouched cells of the rebuilt row
    assert cells["Beta"] == "-0.07285866447191801" and cells["CI_Valid"] == "True" and cells["CI_Sided"] == "two"
    # EUR block uses ln(EUR_OR) and EUR_P
    elo, ehi = rb.rebuilt_bounds(math.log(0.924793818106945), 1.1374642671390772e-05)
    assert cells["EUR_CI_Method"] == "lrt_quadratic"
    assert float(cells["EUR_CI_LO_OR"]) == pytest.approx(elo, rel=1e-12)
    assert float(cells["EUR_CI_HI_OR"]) == pytest.approx(ehi, rel=1e-12)
    assert cells["EUR_CI95"] == rb.fmt_ci(elo, ehi)


def test_rebuild_is_idempotent_and_check_mode_tracks_state(tmp_path):
    path = tmp_path / rb.MAIN_TABLE
    _write_main(path, _main_rows())
    reports, problems = rb.run(tmp_path, write=False, only={rb.MAIN_TABLE}, verbose=False)
    assert rb.pending(reports, problems)  # a rebuild is pending before the rewrite
    assert path.read_bytes() == path.read_bytes()

    rb.run(tmp_path, write=True, only={rb.MAIN_TABLE}, verbose=False)
    first = path.read_bytes()
    reports, problems = rb.run(tmp_path, write=True, only={rb.MAIN_TABLE}, verbose=False)
    assert path.read_bytes() == first
    assert all(r.rebuilt == 0 for r in reports)
    assert sum(r.already for r in reports) == 2
    reports, problems = rb.run(tmp_path, write=False, only={rb.MAIN_TABLE}, verbose=False)
    assert rb.pending(reports, problems) == []

    # tampering with a rebuilt bound is detected as drift
    lines = first.decode().split("\r\n")
    cells = lines[1].split("\t")
    cells[MAIN_HEADER.index("CI_LO_OR")] = "0.5"
    lines[1] = "\t".join(cells)
    path.write_bytes("\r\n".join(lines).encode())
    reports, problems = rb.run(tmp_path, write=False, only={rb.MAIN_TABLE}, verbose=False)
    assert any("no longer match" in issue for issue in rb.pending(reports, problems))


def test_rebuild_refuses_files_it_cannot_round_trip(tmp_path):
    path = tmp_path / rb.MAIN_TABLE
    _write_main(path, _main_rows(), newline="\n")
    path.write_bytes(path.read_bytes().rstrip(b"\n") + b"\n\n")  # blank trailing line breaks the grid
    reports, problems = rb.run(tmp_path, write=True, only={rb.MAIN_TABLE}, verbose=False)
    assert problems and "refusing to edit" in problems[0]
    assert all(r.rebuilt == 0 for r in reports)


def test_committed_tables_are_rebuilt():
    """CI guard: the repository tables must contain no LRT-backed profile interval."""
    if not (REPO / rb.MAIN_TABLE).exists():
        pytest.skip("committed PheWAS tables not present")
    reports, problems = rb.run(REPO, write=False, verbose=False)
    assert rb.pending(reports, problems) == []
    main = [r for r in reports if r.table == str(REPO / rb.MAIN_TABLE) and r.block == "overall"]
    assert main and main[0].already > 0


# ---------------------------------------------------------------------------
# Agreement with the corrected profile code
# ---------------------------------------------------------------------------
def test_rebuilt_interval_tracks_fixed_profile_interval():
    sm = pytest.importorskip("statsmodels.api")
    from scipy import stats as sp_stats
    from scipy.special import expit

    from phewas import models

    rng = np.random.default_rng(3)
    n = 20000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n), rng.binomial(2, 0.3, size=n)])
    beta = np.array([-2.0, 0.3, -0.2, -0.12])
    y = rng.binomial(1, expit(X @ beta)).astype(float)
    full = sm.Logit(y, X).fit(disp=0)
    red = sm.Logit(y, np.delete(X, 3, axis=1)).fit(disp=0)
    p = float(sp_stats.chi2.sf(2 * (full.llf - red.llf), 1))
    ci = models._profile_ci_beta(X, y, 3, full, kind="mle")
    assert ci["valid"]
    lo, hi = rb.rebuilt_bounds(float(full.params[3]), p)
    bh = float(full.params[3])
    for rebuilt, profile in ((math.log(lo), ci["lo"]), (math.log(hi), ci["hi"])):
        assert abs(rebuilt - profile) < 0.05 * abs(profile - bh)
    # the guard accepts the genuine profile interval
    ok, note = models._profile_ci_plausible(ci, bh, 2 * (full.llf - red.llf))
    assert ok and note == ""


# ---------------------------------------------------------------------------
# Pipeline guard
# ---------------------------------------------------------------------------
def test_guard_rejects_collapsed_profile_bound():
    from phewas import models

    bh, stat = -0.073, 21.1  # lump-or-mass-like row
    implied = 1.959964 * abs(bh) / math.sqrt(stat)
    good = {"lo": bh - implied, "hi": bh + implied, "valid": True}
    assert models._profile_ci_plausible(good, bh, stat)[0]
    collapsed = {"lo": bh - 0.0005, "hi": bh + 0.9 * implied, "valid": True}
    ok, note = models._profile_ci_plausible(collapsed, bh, stat)
    assert not ok and "lower half-width" in note
    collapsed_hi = {"lo": bh - implied, "hi": bh + 0.3 * implied, "valid": True}
    assert not models._profile_ci_plausible(collapsed_hi, bh, stat)[0]
    # one-sided boundary interval: the infinite side is not judged
    boundary = {"lo": -np.inf, "hi": bh + implied, "valid": True}
    assert models._profile_ci_plausible(boundary, bh, stat)[0]
    # weak statistic: check skipped
    assert models._profile_ci_plausible(collapsed, bh, 0.5)[0]
