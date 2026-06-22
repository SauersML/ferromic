"""Regression tests for BUG #5: inner-CV leakage of synthetic samples from real parents.

A synthetic genotype is built from two real parent haplotypes. If the synthetic
descendant lands in a different inner-CV fold than its parent (or shares a parent with a
row in another fold), the model-selection step leaks information. The fix retains both
parent indices for every synthetic row and groups each synthetic row with its real
parents (connected family components) so a grouped CV keeps them together.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from linked import build_family_groups, create_synthetic_data  # noqa: E402


def test_build_family_groups_real_only():
    # No synthetic rows: every real sample is its own family.
    real_idx = np.array([0, 1, 2, 3])
    groups = build_family_groups(real_idx, [])
    assert len(groups) == 4
    assert len(np.unique(groups)) == 4


def test_synthetic_row_shares_group_with_both_parents():
    # real rows: indices 0,1,2 ; one synthetic row from parents (0, 2)
    real_idx = np.array([0, 1, 2])
    parent_map = [[0, 2]]
    groups = build_family_groups(real_idx, parent_map)
    assert len(groups) == 4  # 3 real + 1 synth

    g_real = {int(real_idx[r]): groups[r] for r in range(len(real_idx))}
    synth_group = groups[3]

    # Synthetic shares the group of BOTH its parents, and the two parents are unioned.
    assert g_real[0] == g_real[2] == synth_group
    # The unrelated real sample 1 is in a different family.
    assert g_real[1] != synth_group


def test_parents_sharing_a_child_are_unioned_transitively():
    # Two synthetic rows: (0,1) and (1,2) link 0-1-2 into one family; 3 stays alone.
    real_idx = np.array([0, 1, 2, 3])
    parent_map = [[0, 1], [1, 2]]
    groups = build_family_groups(real_idx, parent_map)
    g_real = {int(real_idx[r]): groups[r] for r in range(len(real_idx))}
    assert g_real[0] == g_real[1] == g_real[2]
    assert g_real[3] != g_real[0]
    # Both synthetic rows belong to the shared 0-1-2 family.
    assert groups[4] == groups[5] == g_real[0]


def test_augmentation_mode_returns_parent_map():
    """BUG #5 core: augmentation mode must return a parent map aligned to X_synth,
    not discard parent IDs."""
    n = 4
    X_hap1 = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
    X_hap2 = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]])
    # gts: two homozygous-ref (class 0) and two homozygous-alt (class 2)
    raw_gts = pd.Series(["0|0", "1|1", "0|0", "1|1"])
    sample_indices = np.arange(n)
    confidence_mask = np.ones(n, dtype=bool)
    X_existing = X_hap1 + X_hap2

    X_synth, y_synth, parent_map = create_synthetic_data(
        X_hap1, X_hap2, raw_gts, sample_indices, confidence_mask, X_existing,
        target_counts=None,
    )
    assert X_synth is not None
    # Parent map must be present and aligned 1:1 with synthetic rows (was discarded before fix).
    assert parent_map is not None
    assert len(parent_map) == len(X_synth) == len(y_synth)
    valid = set(sample_indices.tolist())
    for pair in parent_map:
        assert len(pair) == 2
        for p in pair:
            assert int(p) in valid


def test_no_parent_and_descendant_split_across_inner_folds():
    """End-to-end grouping property: with the family-group labels, a grouped splitter
    never places a synthetic descendant and one of its real parents in different folds."""
    from sklearn.model_selection import GroupKFold

    real_idx = np.array([0, 1, 2, 3, 4, 5])
    parent_map = [[0, 1], [2, 3], [4, 5], [0, 1]]
    groups = build_family_groups(real_idx, parent_map)

    n_real = len(real_idx)
    pos_of_real = {int(real_idx[r]): r for r in range(n_real)}

    n_splits = min(3, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    X = np.zeros((len(groups), 2))
    for _, test_idx in gkf.split(X, groups=groups):
        test_set = set(test_idx.tolist())
        for s, (pa, pb) in enumerate(parent_map):
            synth_row = n_real + s
            pa_row, pb_row = pos_of_real[int(pa)], pos_of_real[int(pb)]
            # If the synthetic row is in this test fold, both parents must be too,
            # and vice versa (never split across folds).
            assert (synth_row in test_set) == (pa_row in test_set) == (pb_row in test_set), (
                f"parent/descendant split across folds for synth {s}"
            )
