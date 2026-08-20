"""Age-anchored version of the balancing test: plug in a known split age.

The envelope test (stats/inversion_selection_envelope.py) measures the age of
the arrangement split from sequence divergence and normalizes it by the LOCAL
direct-class diversity, pi_dir. That normalization is what broke at
chr7:70.96Mb: a hidden ancient lineage split inside the direct class inflated
the clock and manufactured a signal. It is also the reason every p-value there
inherits whatever is wrong with pi_dir at that locus.

For 17q21.31 we do not have to estimate the age at all. It is one of the best
characterised polymorphisms in the human genome and long-read work dates the
H1/H2 coalescence independently of anything in our data. So this script asks
the question in the direction the external estimate allows:

    given that the two arrangements last shared an ancestor T years ago, how
    probable is it that a NEUTRAL allele that old is still segregating at the
    observed frequency today?

Under neutrality age and frequency are welded together, and the weld is what
gets tested: a neutral variant that old should long since have fixed or been
lost, not be sitting at 11%. Same conditioned-branch null as the envelope test
-- neutral coalescent, candidate branches with exactly k of n descendants,
weighted by branch length -- but the statistic is the mean cross-arrangement
COALESCENCE TIME in absolute units, so no local clock enters.

WHAT THIS DOES NOT DO: it does not escape needing a clock, it relocates the
clock. pi_dir is replaced by an effective population size, because a coalescent
time in generations only becomes a probability once you know N_e. That is a
real assumption and the answer is sensitive to it, so nothing here reports a
single p-value: the output is a grid over N_e, and the reader sees where the
conclusion changes. What it DOES buy is independence from the one quantity that
demonstrably fails at some loci.

Output: results/inversion_age_test.tsv + a printed grid.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inversion_selection_envelope import parse_demography  # noqa: E402

RNG_SEED = 31337


def cross_coalescence_null(n, k, rng, n_cand=200_000, max_trees=3_000_000,
                           tmap=None):
    """Neutral distribution of mean cross-arrangement coalescence time.

    Same conditioning as the envelope null: plain Kingman genealogies, every
    branch subtending EXACTLY k of the n tips is a candidate arrangement
    origin, weighted by its length (mutation opportunity). For each candidate
    the statistic is the mean coalescence time over all (carrier,
    non-carrier) pairs, in units of 2*N_e generations -- the same quantity
    sequence divergence between the arrangements measures, but here left in
    time units instead of being converted to a divergence and divided by a
    local clock.

    Returns (T_cross, T_within_carrier, weights, tree_id, n_trees)."""
    Tx, Tw, W, Tid = [], [], [], []
    trees = 0
    while len(Tx) < n_cand and trees < max_trees:
        trees += 1
        alive = {i: ([i], 0.0) for i in range(n)}
        # pair_time[i][j] is filled when i and j first share an ancestor
        coal = np.zeros((n, n))
        branches = []
        candidates = []
        t = 0.0
        next_id = n
        ids = list(range(n))
        while len(ids) > 1:
            m = len(ids)
            t += rng.exponential(2.0 / (m * (m - 1)))
            i = int(rng.integers(m))
            j = int(rng.integers(m - 1))
            if j >= i:
                j += 1
            a, b = ids[i], ids[j]
            ma, ta = alive.pop(a)
            mb, tb = alive.pop(b)
            # every cross pair between the two merging clades coalesces now
            ia = np.fromiter(ma, dtype=np.int64, count=len(ma))
            ib = np.fromiter(mb, dtype=np.int64, count=len(mb))
            coal[np.ix_(ia, ib)] = t
            coal[np.ix_(ib, ia)] = t
            for mem, t0 in ((ma, ta), (mb, tb)):
                branches.append((mem, t0, t))
                if len(mem) == k:
                    candidates.append(len(branches) - 1)
            alive[next_id] = (ma + mb, t)
            ids = [x for x in ids if x not in (a, b)]
            ids.append(next_id)
            next_id += 1
        if not candidates:
            continue
        # demography: coalescence times are simulated in standard time, then
        # mapped. Topology is demography-invariant; only the times move.
        cmat = tmap(coal) if tmap is not None else coal
        for ci in candidates:
            mem, h_child, h_par = branches[ci]
            carriers = np.fromiter(mem, dtype=np.int64, count=len(mem))
            mask = np.zeros(n, dtype=bool)
            mask[carriers] = True
            others = np.nonzero(~mask)[0]
            Tx.append(float(cmat[np.ix_(carriers, others)].mean()))
            if len(carriers) > 1:
                sub = cmat[np.ix_(carriers, carriers)]
                iu = np.triu_indices(len(carriers), 1)
                Tw.append(float(sub[iu].mean()))
            else:
                Tw.append(np.nan)
            lo, hi = (tmap(np.array([h_child, h_par])) if tmap is not None
                      else (h_child, h_par))
            W.append(float(hi - lo) if tmap is not None
                     else float(h_par - h_child))
            Tid.append(trees - 1)
    return (np.array(Tx), np.array(Tw), np.array(W),
            np.array(Tid, dtype=np.int64), trees)


def weighted_tail(stat, obs, weights, tree_id, rng, n_boot=1000):
    """P(stat >= obs), branch-length weighted, with a tree-level bootstrap CI."""
    hit = stat >= obs
    tot = weights.sum()
    if tot <= 0:
        return float("nan"), float("nan"), float("nan"), 0
    ess = float(tot ** 2 / (weights ** 2).sum())
    p_raw = float(weights[hit].sum() / tot)
    p = (p_raw * ess + 1.0) / (ess + 1.0)
    nt = int(tree_id.max()) + 1
    wt = np.bincount(tree_id, weights=weights, minlength=nt)
    st = np.bincount(tree_id, weights=weights * hit, minlength=nt)
    keep = wt > 0
    wt, st = wt[keep], st[keep]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        c = rng.integers(0, len(wt), len(wt))
        s = wt[c].sum()
        boots[b] = st[c].sum() / s if s > 0 else np.nan
    boots = boots[np.isfinite(boots)]
    boots = (boots * ess + 1.0) / (ess + 1.0)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots)
              else (np.nan, np.nan))
    return p, float(lo), float(hi), int(hit.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir")
    ap.add_argument("--n", type=int, default=80, help="sampled haplotypes")
    ap.add_argument("--k", type=int, default=9, help="inverted haplotypes")
    ap.add_argument("--age-years", type=float, default=2.38e6,
                    help="externally dated arrangement split, in years")
    ap.add_argument("--generation-time", type=float, default=30.0)
    ap.add_argument("--ne-grid", default="5000,7500,10000,12500,15000,20000,"
                                        "25000,30000",
                    help="effective sizes to report across; the answer depends "
                         "on this, so it is shown rather than chosen")
    ap.add_argument("--n-cand", type=int, default=200_000)
    ap.add_argument("--demography", default="const")
    ap.add_argument("--out", default="results/inversion_age_test.tsv")
    a = ap.parse_args()
    if a.workdir:
        os.chdir(a.workdir)
        os.makedirs("results", exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)
    tmap, _ = parse_demography(a.demography)
    Tx, Tw, W, Tid, trees = cross_coalescence_null(
        a.n, a.k, rng, n_cand=a.n_cand,
        tmap=None if a.demography in ("const", "", None) else tmap)
    print(f"null: {len(Tx)} candidate branches with exactly k={a.k} of "
          f"n={a.n}, from {trees} trees; demography={a.demography}")
    print(f"cross-arrangement coalescence time under neutrality "
          f"(units of 2*Ne generations):")
    qs = np.percentile(Tx, [50, 90, 95, 99, 99.9])
    print(f"  median {qs[0]:.3f}   90th {qs[1]:.3f}   95th {qs[2]:.3f}   "
          f"99th {qs[3]:.3f}   99.9th {qs[4]:.3f}   max {Tx.max():.3f}")
    print(f"  (for scale: whole-sample TMRCA averages "
          f"{2 * (1 - 1.0 / a.n):.3f} in these units)")

    rows = []
    print(f"\nexternal split age {a.age_years / 1e6:.2f} Myr at "
          f"{a.generation_time:.0f} yr/generation "
          f"= {a.age_years / a.generation_time:.0f} generations\n")
    print(f"{'Ne':>8}{'unit (Myr)':>12}{'age in units':>14}{'p':>10}"
          f"{'95% CI':>20}{'n_tail':>9}")
    for ne_s in a.ne_grid.split(","):
        ne = float(ne_s)
        unit_gen = 2.0 * ne
        unit_yr = unit_gen * a.generation_time
        obs_units = (a.age_years / a.generation_time) / unit_gen
        p, lo, hi, ntail = weighted_tail(Tx, obs_units, W, Tid, rng)
        rows.append({"n": a.n, "k_inv": a.k, "age_years": a.age_years,
                     "generation_time": a.generation_time, "Ne": ne,
                     "unit_years": unit_yr, "age_in_units": obs_units,
                     "p_age": p, "p_age_lo": lo, "p_age_hi": hi,
                     "n_tail": ntail, "null_candidates": len(Tx),
                     "demography": a.demography})
        print(f"{ne:>8.0f}{unit_yr / 1e6:>12.3f}{obs_units:>14.3f}"
              f"{p:>10.5f}   [{lo:.5f}, {hi:.5f}]{ntail:>9d}")

    out = pd.DataFrame(rows)
    if a.workdir:
        out.to_csv(a.out, sep="\t", index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
