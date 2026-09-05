"""The constrained logistic refit must maximize the likelihood at a fixed
coefficient, and the profile interval built on it must match a reference
profile likelihood. Regression test for the offset refit whose Newton step
ignored the offset (profile intervals in the PheWAS tables were too narrow on
the side away from zero)."""
import numpy as np
import pytest

statsmodels = pytest.importorskip("statsmodels")
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy import stats as sp_stats
from scipy.special import expit

from phewas import models


def _simulate(seed=1, n=20000):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        np.ones(n), rng.normal(size=n), rng.normal(size=n), rng.binomial(2, 0.3, size=n),
    ])
    beta = np.array([-2.0, 0.3, -0.2, -0.08])
    y = rng.binomial(1, expit(X @ beta)).astype(float)
    return X, y


def test_offset_refit_reaches_constrained_maximum():
    X, y = _simulate()
    X_red, x_t = np.delete(X, 3, axis=1), X[:, 3]
    for b0 in (-0.09, -0.14, 0.05):
        ours = models._logit_mle_refit_offset(X_red, y, offset=b0 * x_t)
        ref = sm.Logit(y, X_red, offset=b0 * x_t).fit(disp=0)
        assert ours.llf == pytest.approx(ref.llf, abs=1e-6)
        assert np.allclose(ours.params, ref.params, atol=1e-6)


def test_profile_interval_matches_reference_profile_likelihood():
    X, y = _simulate()
    fit = sm.Logit(y, X).fit(disp=0)
    ci = models._profile_ci_beta(X, y, 3, fit, kind="mle")
    assert ci["valid"] and ci["sided"] == "two"
    crit = sp_stats.chi2.ppf(0.95, 1)

    def dev(b0):
        con = sm.Logit(y, np.delete(X, 3, axis=1), offset=b0 * X[:, 3]).fit(disp=0)
        return 2.0 * (fit.llf - con.llf) - crit

    bh = fit.params[3]
    lo = brentq(dev, bh - 1.0, bh)
    hi = brentq(dev, bh, bh + 1.0)
    assert ci["lo"] == pytest.approx(lo, abs=2e-4)
    assert ci["hi"] == pytest.approx(hi, abs=2e-4)
    # the estimate sits well inside the interval on both sides
    assert (bh - ci["lo"]) / (ci["hi"] - bh) == pytest.approx(1.0, abs=0.25)
