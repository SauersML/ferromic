"""Orientation parsimony classifier: neighbor-joining tree on Hamming distances +
Fitch small-parsimony minimum number of orientation state changes.

This is the tree-based recurrence signal of Porubsky et al. 2022 / the manuscript
pipeline: map inverted/direct orientation onto haplotype-tree tips and count the
minimum number of orientation flips. ``classify`` returns that count; a count >= 2
is the parsimony "recurrent" call.

The identical logic lives in ``simulations/flux/flux_sim.py`` (which also carries the
msprime demographies). It is duplicated here, dependency-light (numpy only, no
msprime), so the recurrence classifier and its reproduction tests can be imported
and run without the coalescent-simulation stack. Keep the two copies in sync.
"""
from __future__ import annotations

import sys

import numpy as np


def hamming_matrix(G):
    """(n, n) Hamming-distance count matrix over the 0/1 site columns of G."""
    n = G.shape[0]
    if G.shape[1] == 0:
        return np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = (G[i] != G).sum(axis=1)
    return D


def neighbor_joining(D):
    """Return (root_id, children, leafset) of an NJ tree (rooted arbitrarily)
    built from the distance matrix ``D``. Leaf ids are 0..n-1."""
    n = D.shape[0]
    nodes = list(range(n))                # active node ids
    Dm = {(i, j): D[i, j] for i in range(n) for j in range(n)}
    children = {i: [] for i in range(n)}  # internal node -> child ids
    leafset = set(range(n))
    next_id = n
    while len(nodes) > 2:
        m = len(nodes)
        r = {i: sum(Dm[(i, j)] for j in nodes if j != i) for i in nodes}
        best = None
        for a_idx in range(m):
            i = nodes[a_idx]
            for b_idx in range(a_idx + 1, m):
                j = nodes[b_idx]
                q = (m - 2) * Dm[(i, j)] - r[i] - r[j]
                if best is None or q < best[0]:
                    best = (q, i, j)
        _, i, j = best
        u = next_id
        next_id += 1
        children[u] = [i, j]
        for k in nodes:
            if k == i or k == j:
                continue
            Dm[(u, k)] = Dm[(k, u)] = 0.5 * (Dm[(i, k)] + Dm[(j, k)] - Dm[(i, j)])
        nodes = [k for k in nodes if k != i and k != j] + [u]
    root = next_id
    children[root] = list(nodes)
    return root, children, leafset


def fitch_score(root, children, leafset, label_of):
    """Min number of state changes (Fitch small-parsimony) for a binary trait."""
    score = [0]
    sys.setrecursionlimit(100000)

    def post(u):
        if u in leafset:
            return {label_of[u]}
        sets = [post(c) for c in children[u]]
        inter = set.intersection(*sets) if sets else set()
        if inter:
            return inter
        score[0] += 1
        return set.union(*sets) if sets else set()

    post(root)
    return score[0]


def classify(G, labels):
    """Inferred number of orientation origin events (parsimony score).

    ``G``: (n_hap, n_site) 0/1 haplotype x biallelic-site matrix.
    ``labels``: (n_hap,) orientation in {0 = direct, 1 = inverted}.
    Returns the Fitch minimum number of orientation state changes; a value >= 2 is
    the parsimony recurrent call.

    Deterministic and order-invariant: the result is a pure function of the
    {(haplotype, orientation)} multiset. neighbor_joining resolves equal-criterion
    ties by list position, so rows are first reordered into a canonical order
    (lexicographic by genotype, then by orientation label) before building the
    tree; permuting the input rows therefore cannot change the returned count.
    """
    n = G.shape[0]
    if G.shape[1] == 0:
        # no segregating sites: tree unresolved -> single change at best
        return 1
    labels = np.asarray(labels)
    order = sorted(range(n), key=lambda i: (tuple(int(x) for x in G[i]), int(labels[i])))
    Gc = G[order]
    lab = [int(labels[i]) for i in order]
    D = hamming_matrix(Gc)
    root, children, leafset = neighbor_joining(D)
    label_of = {i: lab[i] for i in range(n)}
    return fitch_score(root, children, leafset, label_of)
