"""Determinism, order-invariance, and feature-reproduction QC for the extractor and the
parsimony classifier (no external inputs).

The extractor tests run anywhere. The ``classify`` tests exercise the reference
recurrence classifier (IQ-TREE ML tree + Biopython Fitch parsimony, see
``simulations/refsim/refsim.py``) and are skipped where that toolchain is absent.
"""
import math
import os
import shutil

import numpy as np
import pytest

from recurrence.features import extract_features
from recurrence.parsimony import classify

_HAS_IQTREE = bool(os.environ.get("IQTREE_BIN")) or any(
    shutil.which(n) for n in ("iqtree2", "iqtree3", "iqtree"))
try:  # noqa: SIM105
    import Bio  # noqa: F401
    _HAS_BIO = True
except ImportError:
    _HAS_BIO = False

needs_reference_toolchain = pytest.mark.skipif(
    not (_HAS_IQTREE and _HAS_BIO),
    reason="reference classifier needs an iqtree2 binary and Biopython")

# Golden feature values for the committed small fixture (see conftest.small_fixture),
# computed from the extractor and pinned here so a change in feature math is caught.
GOLDEN = {
    "tree_n_events": 1.0,
    "log_tree_events": 0.6931471806,
    "one_minus_best_r2": 0.0,
    "one_minus_top3_r2": 0.0,
    "log_pi_ratio": 0.4054651081,
    "log_pi_inv": -1.3862943611,
    "disp_ratio_inv": 0.3,
    "pdist_sd": 0.3343415874,
    "pdist_excess": 1.1052631579,
    "log_pdist_ratio": 0.4054651081,
    "pdist_q80": 0.3333333333,
    "hudson_fst": 0.75,
    "log_n_sites": 1.9459101491,
}


def test_features_reproduce_on_fixture(small_fixture):
    G, labels = small_fixture
    # tree_n_events is supplied so this test covers the pop-gen feature maths
    # without requiring the IQ-TREE toolchain; the origin count itself is
    # covered by the reference-classifier tests below.
    f = extract_features(G, labels, tree_n_events=1)
    for k, want in GOLDEN.items():
        got = f[k]
        assert not (isinstance(got, float) and math.isnan(got)), f"{k} is NaN"
        assert abs(float(got) - want) < 1e-8, f"{k}: got {got!r} != golden {want!r}"


@needs_reference_toolchain
def test_classify_deterministic(small_fixture):
    G, labels = small_fixture
    vals = {classify(G, labels) for _ in range(5)}
    assert len(vals) == 1


@needs_reference_toolchain
def test_classify_order_invariant(small_fixture):
    G, labels = small_fixture
    base = classify(G, labels)
    rng = np.random.default_rng(12345)
    for _ in range(20):
        perm = rng.permutation(G.shape[0])
        assert classify(G[perm], labels[perm]) == base


@needs_reference_toolchain
def test_classify_counts_recurrent_topology():
    """Two independent inverted clades on distinct genotype backgrounds -> parsimony
    infers >= 2 orientation origins (the recurrent call)."""
    G = np.array([
        [1, 1, 1, 0, 0, 0],   # inverted clade A
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],   # inverted clade B (distant background)
        [0, 0, 0, 1, 1, 1],
        [1, 1, 0, 1, 1, 0],   # direct, intermediate between A and B
        [1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 1, 1],
        [1, 1, 0, 0, 1, 1],
    ], dtype=np.uint8)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert classify(G, labels) >= 2
