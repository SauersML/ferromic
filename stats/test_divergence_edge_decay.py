"""Regression tests for stats/divergence_edge_decay.py.

Focus: da/dxy must be summarised PER CALLABLE BASE, not over variant sites only
(audit BUG #11). Invariant callable bases contribute 0 to divergence; uncallable
bases (NA in the per-base diversity track) are excluded from both numerator and
denominator.
"""

import numpy as np
import pytest

from stats.divergence_edge_decay import (
    _window_mean_per_base,
    _window_mean_snpcond,
    folded_spearman,
)

NA = np.nan


def test_per_base_includes_invariant_callable_bases_as_zero():
    """Region of 4 callable bases, one variant with dxy=0.8. The per-base mean is
    0.8 / 4 = 0.2 (the 3 invariant callable bases contribute 0). The
    SNP-conditioned mean is 0.8 (averaged over the single variant site)."""
    values = np.array([NA, 0.8, NA, NA])           # FST track: NA except variant
    callable_mask = np.array([True, True, True, True])
    valid = np.isfinite(values) & (values > 1e-12)  # variant-only

    per_base = _window_mean_per_base(values, callable_mask)
    snpcond = _window_mean_snpcond(values, valid)

    assert per_base == 0.2
    assert snpcond == 0.8
    # The bug inflated per-base divergence by the SNP-density factor (here 4x).
    assert snpcond > per_base


def test_per_base_excludes_uncallable_bases_from_denominator():
    """Of 5 bases, only 3 are callable (mask). One callable base is a variant
    (dxy=0.6). Per-base mean = 0.6 / 3 = 0.2; the 2 uncallable bases do not
    enlarge the denominator."""
    values = np.array([0.6, NA, NA, NA, NA])
    callable_mask = np.array([True, True, True, False, False])
    assert _window_mean_per_base(values, callable_mask) == pytest.approx(0.2)


def test_per_base_all_invariant_callable_is_zero_not_nan():
    """A window of callable bases with no variant has divergence exactly 0 per
    base (not NaN): invariant callable bases are real zero-divergence bases."""
    values = np.array([NA, NA, NA])
    callable_mask = np.array([True, True, True])
    assert _window_mean_per_base(values, callable_mask) == 0.0


def test_per_base_no_callable_bases_is_nan():
    values = np.array([NA, NA])
    assert np.isnan(_window_mean_per_base(values, np.array([False, False])))
    assert np.isnan(_window_mean_per_base(values, None))


def test_snpcond_matches_variant_mean():
    values = np.array([0.4, NA, 0.6, NA])
    valid = np.isfinite(values) & (values > 1e-12)
    assert _window_mean_snpcond(values, valid) == 0.5


def test_folded_spearman_per_base_zero_fills_invariant_callable():
    """With a callable mask the folded bin means divide by callable bases (a
    decaying density of variants on a fully-callable region yields a decreasing
    per-base divergence toward the centre -> negative rho). Without the mask
    (SNP-conditioned) the same variant *values* are constant, so no decay."""
    L = 220_000  # > 2 * MAX_DECAY_SPAN so the folded decay span is usable
    span = 100_000  # the usable folded span from each edge
    values = np.full(L, NA)
    callable_mask = np.ones(L, dtype=bool)
    # Variant DENSITY decays linearly from each edge to the centre while each
    # variant carries the SAME value (1.0). Per-base divergence therefore decays
    # with distance from the edge, but the SNP-conditioned mean (over variant
    # sites only) stays ~1.0 everywhere and misses the decay.
    rng = np.random.default_rng(0)
    for d in range(span):
        prob = 1.0 - d / span            # high near edge, ~0 near centre
        if rng.random() < prob:
            values[d] = 1.0              # left flank
            values[L - 1 - d] = 1.0      # right flank (mirrored)

    rho_pb, p_pb, bins_pb = folded_spearman(values, np.isfinite(values), callable_mask)
    rho_sc, p_sc, bins_sc = folded_spearman(values, np.isfinite(values), None)

    # Per-base divergence is high at the edges (dense variants/base) and ~0 in the
    # middle -> strongly decreasing with distance from edge -> negative rho.
    assert rho_pb is not None and rho_pb < -0.8
    # SNP-conditioned uses only variant sites, all of value 1.0 -> flat (rho ~ 0
    # or undefined constant); it does not capture the density decay.
    assert rho_sc is None or abs(rho_sc) < abs(rho_pb)
