#!/usr/bin/env python
"""
Between-orientation flux sweep for the structured-coalescent recurrence classifier.

Reproduces the msprime structured-coalescent model that backs the manuscript's
recurrent-inversion power analysis (hsiehphLab/inversionSimulation,
scripts/recurrentINV_m1.2pop.py) and adds a *between-orientation flux* term
(gene conversion / double crossover) as symmetric migration between
opposite-orientation demes.

Classifier (faithful to Porubsky et al. 2022 / the repo pipeline):
  haplotype tree  ->  map inversion orientation (inverted/direct) onto tips
  ->  minimum number of orientation state-changes by Fitch parsimony.
  Inferred #events = parsimony score.  Classifier calls "RECURRENT" if score >= 2.

The repo builds an ML tree with IQ-TREE; here we build a neighbor-joining tree
on Hamming distances over segregating sites.  The parsimony score is a function
of tree *topology* only, and for these clean coalescent genealogies NJ recovers
the same orientation-clade structure as ML, so the homoplasy count is preserved.

Fixed params (manuscript Methods / repo):
  Ne (N_a) = 6000, mu = 1.25e-8, generation_time = 25 y, locus = 200 kbp.
Demographic deme sizes follow the repo (P_D = N_a, P_I = 0.1 N_a, etc.).
"""
import sys, math, argparse
import numpy as np
import msprime

N_A = 6000
MU = 1.25e-8
GEN_TIME = 25
SEQ_LEN = 200_000

# ---------------------------------------------------------------------------
# Demographies
# ---------------------------------------------------------------------------
# Both share the repo's nested split structure.  The crucial difference:
#   SINGLE  : all inverted samples descend from ONE inverted deme -> the
#             inversion arose once (monophyletic inverted clade expected).
#   RECURRENT (3 events): inverted samples drawn from two independently
#             derived inverted demes (P1_I, P2_I) that each acquired the
#             inverted orientation on separate direct backgrounds, exactly as
#             in recurrentINV_m1.2pop.py -> polyphyletic inverted lineages.
#
# Flux term `m_flux` is symmetric migration between OPPOSITE-orientation demes
# (I<->D).  Within-orientation migration `m_within` matches the repo's
# mig_const (default 1e-8, the manifest's "lowM").


def _times_gen(t01_23_y, t0_1_y, t2_3_y):
    return (t01_23_y / GEN_TIME, t0_1_y / GEN_TIME, t2_3_y / GEN_TIME)


def demography_recurrent(t01_23_y, t0_1_y, t2_3_y, m_within, m_flux, rng):
    """Repo's 3-event recurrent topology + between-orientation flux."""
    T01_23, T0_1, T2_3 = _times_gen(t01_23_y, t0_1_y, t2_3_y)
    de = msprime.Demography()
    de.add_population(name="P_I", initial_size=0.1 * N_A)
    de.add_population(name="P_D", initial_size=N_A)
    de.add_population(name="P0_D", initial_size=0.01 * N_A)
    de.add_population(name="P1_I", initial_size=0.1 * N_A)
    de.add_population(name="P2_I", initial_size=0.1 * N_A)
    de.add_population(name="P3_D", initial_size=N_A)
    de.add_population(name="Pa_I", initial_size=0.1 * N_A)
    de.add_population(name="Pa_D", initial_size=N_A)
    de.add_population(name="P00", initial_size=N_A)

    # within-orientation migration (repo mig_const)
    de.set_symmetric_migration_rate(["P0_D", "P3_D"], m_within)
    de.set_symmetric_migration_rate(["P1_I", "P2_I"], m_within)
    # BETWEEN-orientation flux (the new term): every I deme <-> every D deme
    if m_flux > 0:
        for i in ("P1_I", "P2_I"):
            for d in ("P0_D", "P3_D"):
                de.set_symmetric_migration_rate([i, d], m_flux)
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)

    fI = rng.integers(0, 11) / 10
    fD = rng.integers(0, 11) / 10
    de.add_admixture(time=1e-5, derived="P_I", ancestral=["P1_I", "P2_I"],
                     proportions=[fI, 1 - fI])
    de.add_admixture(time=1e-5, derived="P_D", ancestral=["P0_D", "P3_D"],
                     proportions=[fD, 1 - fD])
    de.add_population_split(time=T2_3, derived=["P2_I", "P3_D"], ancestral="Pa_D")
    de.add_population_split(time=T0_1, derived=["P0_D", "P1_I"], ancestral="Pa_I")
    de.add_population_split(time=T01_23, derived=["Pa_I", "Pa_D"], ancestral="P00")
    de.sort_events()
    return de


def demography_single(t_inv_y, m_within, m_flux):
    """Single-origin: one inverted deme split from the direct background at
    t_inv, then both coalesce into the ancestral direct population.  Inversion
    arose exactly once."""
    T_inv = t_inv_y / GEN_TIME
    de = msprime.Demography()
    de.add_population(name="P_I", initial_size=0.1 * N_A)
    de.add_population(name="P_D", initial_size=N_A)
    de.add_population(name="P00", initial_size=N_A)
    if m_flux > 0:
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)
    de.add_population_split(time=T_inv, derived=["P_I", "P_D"], ancestral="P00")
    de.sort_events()
    return de


# ---------------------------------------------------------------------------
# Simulate one locus -> genotype matrix + orientation labels
# ---------------------------------------------------------------------------
def simulate(scenario, inv_freq, sample_hap, rho, m_within, m_flux,
             times, seed):
    rng = np.random.default_rng(seed)
    n_inv = max(2, int(round(sample_hap * inv_freq)))
    n_dir = sample_hap - n_inv
    # round to diploid sample sets like the repo (ploidy=2)
    n_inv_ind = max(1, round(n_inv / 2))
    n_dir_ind = max(1, round(n_dir / 2))

    if scenario == "single":
        de = demography_single(times["t_inv"], m_within, m_flux)
        samples = [msprime.SampleSet(n_inv_ind, population="P_I", ploidy=2),
                   msprime.SampleSet(n_dir_ind, population="P_D", ploidy=2)]
    else:
        de = demography_recurrent(times["t01_23"], times["t0_1"], times["t2_3"],
                                  m_within, m_flux, rng)
        samples = [msprime.SampleSet(n_inv_ind, population="P_I", ploidy=2),
                   msprime.SampleSet(n_dir_ind, population="P_D", ploidy=2)]

    ts = msprime.sim_ancestry(samples=samples, demography=de,
                              sequence_length=SEQ_LEN, recombination_rate=rho,
                              random_seed=seed)
    mts = msprime.sim_mutations(ts, rate=MU, random_seed=seed)
    G = mts.genotype_matrix().T  # haplotypes x sites (0/1)
    n_hap_inv = 2 * n_inv_ind
    labels = np.array([1] * n_hap_inv + [0] * (G.shape[0] - n_hap_inv))
    return G, labels


# ---------------------------------------------------------------------------
# Haplotype tree (NJ on Hamming distance) + Fitch parsimony of orientation
# ---------------------------------------------------------------------------
def hamming_matrix(G):
    n = G.shape[0]
    if G.shape[1] == 0:
        return np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = (G[i] != G).sum(axis=1)
    return D


class _Node:
    __slots__ = ("children", "state")
    def __init__(self):
        self.children = []
        self.state = None


def neighbor_joining(D):
    """Return root _Node of an unrooted NJ tree (rooted arbitrarily) whose
    leaves carry index i (0..n-1) in .state placeholder via leaf map."""
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
        u = next_id; next_id += 1
        children[u] = [i, j]
        for k in nodes:
            if k == i or k == j:
                continue
            Dm[(u, k)] = Dm[(k, u)] = 0.5 * (Dm[(i, k)] + Dm[(j, k)] - Dm[(i, j)])
        nodes = [k for k in nodes if k != i and k != j] + [u]
    # join the final (<=2) remaining into a root
    root = next_id
    children[root] = list(nodes)
    return root, children, leafset


def fitch_score(root, children, leafset, label_of):
    """Min number of state changes (small-parsimony Fitch) for binary trait."""
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
    """Return inferred #events (parsimony score). >=2 => recurrent call."""
    n = G.shape[0]
    if G.shape[1] == 0:
        # no segregating sites: tree unresolved -> single change at best
        return 1
    D = hamming_matrix(G)
    root, children, leafset = neighbor_joining(D)
    label_of = {i: int(labels[i]) for i in range(n)}
    return fitch_score(root, children, leafset, label_of)


if __name__ == "__main__":
    # quick smoke test
    times = dict(t01_23=250000, t0_1=100000, t2_3=50000, t_inv=100000)
    for sc in ("single", "recurrent"):
        G, lab = simulate(sc, 0.1, 240, 1e-8, 1e-8, 0.0, times, seed=1)
        print(sc, "sites=", G.shape[1], "events=", classify(G, lab))
