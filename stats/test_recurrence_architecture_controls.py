"""Regression tests for stats/recurrence_architecture_controls.py.

Focus: pooled SNP density must count the UNION of unique segregating positions
across orientations, not seg0 + seg1 (audit BUG #12). A position segregating in
BOTH orientations was previously counted twice, inflating the background
covariate with the cross-orientation sharing that co-varies with the process
under test.
"""

import numpy as np

from stats.recurrence_architecture_controls import union_segregating_from_tracks


# np.nan marks an uncallable base; a finite value > 0 marks a segregating base.
NA = np.nan


def test_union_excludes_double_counting_of_shared_positions():
    """Positions 0 and 1 segregate in both groups (shared); position 2
    segregates only in group0; position 3 only in group1. The union is 4 unique
    segregating positions, whereas seg0+seg1 = 3 + 3 = 6 double-counts the two
    shared sites."""
    g0 = [0.5, 0.5, 0.5, 0.0]
    g1 = [0.5, 0.5, 0.0, 0.5]
    seg0 = 3
    seg1 = 3
    assert union_segregating_from_tracks(g0, g1) == 4
    assert seg0 + seg1 == 6  # the previous (buggy) pooled count


def test_union_equals_sum_when_no_sharing():
    """With disjoint segregating positions the union equals the sum (no
    double-counting to correct)."""
    g0 = [0.5, 0.0, 0.0]
    g1 = [0.0, 0.5, 0.5]
    assert union_segregating_from_tracks(g0, g1) == 3  # == seg0(1) + seg1(2)


def test_uncallable_and_zero_bases_are_not_segregating():
    """NaN (uncallable) and exactly-zero (callable monomorphic) bases never
    count toward the segregating union."""
    g0 = [NA, 0.0, 0.5, NA]
    g1 = [0.0, NA, 0.0, NA]
    # Only position 2 (group0 pi>0) segregates.
    assert union_segregating_from_tracks(g0, g1) == 1


def test_position_segregating_in_one_group_when_other_uncallable():
    """A position segregating in group1 still counts once even if group0 is
    uncallable (NaN) there."""
    g0 = [NA, NA]
    g1 = [0.5, 0.0]
    assert union_segregating_from_tracks(g0, g1) == 1


def test_length_mismatch_raises():
    try:
        union_segregating_from_tracks([0.5], [0.5, 0.5])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched track lengths")
