"""Shared fixtures for the recurrence reproduction/QC tests.

All tests run from committed inputs and recorded results (recurrence/data,
recurrence/results) and need only numpy/scipy/scikit-learn/pandas -- no msprime, no
network. They re-fit the classifier and re-score the real inversions from the committed
training set and assert the recorded reference tables reproduce.
"""
import numpy as np
import pytest


@pytest.fixture
def small_fixture():
    """A deterministic 8-haplotype x 6-site fixture, 4 inverted / 4 direct, with one
    site that perfectly tags orientation. Golden feature values are asserted in
    test_classifier.py."""
    G = np.array([
        [1, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 1],
    ], dtype=np.uint8)
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return G, labels
