#!/usr/bin/env python
"""Concordance between 4-fold-site pi and the other pi measures (Reviewer 1).

Reviewer 1 asked for diversity at 4-fold degenerate sites as the most reliable
neutral estimate. The response letter quotes Spearman correlations between
orientation *differences* -- whole-locus vs 4-fold, and 4-fold vs whole-CDS --
which ``four_fold_pi.py`` computes the inputs for but never records. This script
computes them from the committed per-inversion table so the numbers are
reproducible from committed data alone, with no alignment inputs required.

Beyond the rank correlations it records three further concordance statistics:

* sign agreement -- how often the two measures put the orientation difference
  on the same side of zero, against a fair-coin null;
* level correlation -- pi measured the two ways across every locus-orientation
  observation, not only their within-locus differences;
* a noise ceiling for each difference correlation -- the Spearman correlation
  expected if the noisier measure agreed with the other *perfectly* and
  differed only by site-sampling noise, given each locus's actual 4-fold site
  count. A locus has few 4-fold sites, so its 4-fold pi is a noisy estimate
  and the attainable correlation is bounded well below 1; the ceiling says how
  much of the shortfall from 1 is explained by sampling noise alone. Sites are
  simulated as independent Bernoulli draws per site, i.e. the sampling noise of
  a single haplotype pair; averaging over the many (correlated) pairs in the
  real estimator can only shrink the noise, so the simulated ceiling is
  conservative (a lower bound on the attainable correlation).

Orientation difference for a measure m is ``m_inverted - m_direct`` per locus.
A locus contributes only when both orientations actually have 4-fold sites
(``pi_fourfold_*`` is 0, not blank, when they do not). Two subsets are
reported: ``all_with_fourfold`` (46 loci) and ``recurrence_classified``
(26 loci -- the further restriction to a consensus recurrence call that the
paired tests in ``four_fold_pi.py`` apply, and the set the response letter
quotes).

Input : data/four_fold_pi_by_inversion.tsv
Output: data/four_fold_pi_correlations.tsv
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_IN = os.path.join(REPO, "data", "four_fold_pi_by_inversion.tsv")
DEFAULT_OUT = os.path.join(REPO, "data", "four_fold_pi_correlations.tsv")

PAIRS = [
    ("wholeLocus", "fourfold",
     "whole-locus vs 4-fold orientation difference"),
    ("fourfold", "wholeCDS",
     "4-fold vs whole-CDS orientation difference"),
    ("wholeLocus", "wholeCDS",
     "whole-locus vs whole-CDS orientation difference"),
]

# Ceilings are simulated for the comparisons in which one side is the 4-fold
# estimate; the other measure supplies the "truth" ranks.
CEILING_TRUTH = {"wholeLocus": "pi_wholeLocus", "wholeCDS": "pi_wholeCDS"}

N_SIM = 4000
SEED = 2026


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _delta(row, measure):
    inv = _f(row.get(f"pi_{measure}_inverted"))
    dir_ = _f(row.get(f"pi_{measure}_direct"))
    if inv is None or dir_ is None:
        return None
    return inv - dir_


def spearman(x, y):
    """Spearman rho with average ranks, and a two-sided t-approximation p-value."""
    from scipy import stats

    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def sign_agreement(xs, ys):
    """Loci where both differences are nonzero and share a sign, vs a fair coin."""
    from scipy import stats

    signed = [(x, y) for x, y in zip(xs, ys) if x != 0 and y != 0]
    if len(signed) < 3:
        return None
    k = sum(1 for x, y in signed if (x > 0) == (y > 0))
    p = stats.binomtest(k, len(signed), 0.5, alternative="two-sided").pvalue
    return k, len(signed), float(p)


def noise_ceiling(subset, truth_prefix, rho_obs, rng):
    """Spearman ceiling under perfect agreement plus 4-fold site-sampling noise.

    Truth per locus and orientation is the other measure's pi. Each simulated
    4-fold estimate draws Binomial(L, pi)/L over that locus-orientation's real
    4-fold site count L -- single-pair noise, which overstates the noise of the
    real many-pair estimator, so the ceiling is conservative.
    """
    truths, sites = [], []
    for r in subset:
        rec = []
        for ori in ("direct", "inverted"):
            pi_t = _f(r.get(f"{truth_prefix}_{ori}"))
            L = _f(r.get(f"fourfold_sites_{ori}"))
            if pi_t is None or not L or L <= 0:
                rec = None
                break
            rec.append((min(max(pi_t, 0.0), 1.0), int(L)))
        if rec is not None:
            truths.append(rec)
    if len(truths) < 3:
        return None
    d_truth = [inv[0] - dir_[0] for dir_, inv in truths]

    rhos = np.empty(N_SIM)
    for s in range(N_SIM):
        d_sim = []
        for (pi_d, L_d), (pi_i, L_i) in truths:
            hat_d = rng.binomial(L_d, pi_d) / L_d
            hat_i = rng.binomial(L_i, pi_i) / L_i
            d_sim.append(hat_i - hat_d)
        rhos[s], _ = spearman(d_truth, d_sim)
    lo, med, hi = np.percentile(rhos, [2.5, 50, 97.5])
    frac_below_obs = float(np.mean(rhos <= rho_obs))
    return len(truths), float(med), float(lo), float(hi), frac_below_obs


def build(in_path=DEFAULT_IN, out_path=DEFAULT_OUT):
    with open(in_path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    # A locus only carries 4-fold information when both orientations actually
    # have 4-fold sites; pi_fourfold_* is 0 rather than blank when they do not.
    usable = [r for r in rows
              if (_f(r.get("fourfold_sites_direct")) or 0) > 0
              and (_f(r.get("fourfold_sites_inverted")) or 0) > 0]
    # The paired tests in four_fold_pi.py are further restricted to loci with a
    # consensus recurrence call, which is the set the response letter quotes.
    classified = [r for r in usable if str(r.get("recurrence", "")).strip() != ""]

    subsets = [("all_with_fourfold", usable),
               ("recurrence_classified", classified)]

    rng = np.random.default_rng(SEED)
    out = []

    def emit(subset_name, a, b, comparison, n, statistic, value, p):
        out.append({
            "subset": subset_name,
            "measure_x": a, "measure_y": b, "comparison": comparison,
            "n_loci": n, "statistic": statistic,
            "value": f"{value:.6f}",
            "p_value": "" if p is None else f"{p:.6g}",
        })

    for subset_name, subset in subsets:
        for a, b, label in PAIRS:
            pairs = [(_delta(r, a), _delta(r, b)) for r in subset]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 3:
                print(f"skipping {subset_name}/{label}: {len(pairs)} usable loci")
                continue
            xs = [q[0] for q in pairs]
            ys = [q[1] for q in pairs]
            rho, p = spearman(xs, ys)
            emit(subset_name, a, b, label, len(pairs), "spearman_rho", rho, p)

            sa = sign_agreement(xs, ys)
            if sa is not None:
                k, n, p_sign = sa
                emit(subset_name, a, b,
                     f"{label}: loci agreeing in sign ({k} of {n})",
                     n, "sign_agreement_fraction", k / n, p_sign)

            other = a if b == "fourfold" else (b if a == "fourfold" else None)
            if other in CEILING_TRUTH:
                ceiling = noise_ceiling(subset, CEILING_TRUTH[other], rho, rng)
                if ceiling is not None:
                    n_c, med, lo, hi, frac = ceiling
                    emit(subset_name, a, b,
                         f"{label}: ceiling if agreement were perfect apart "
                         f"from 4-fold site-sampling noise "
                         f"(95% range {lo:.3f}-{hi:.3f})",
                         n_c, "noise_ceiling_median_rho", med, frac)

        # Levels, not differences: every locus-orientation observation.
        for a, b in (("wholeLocus", "fourfold"), ("wholeCDS", "fourfold")):
            obs = []
            for r in subset:
                for ori in ("direct", "inverted"):
                    if (_f(r.get(f"fourfold_sites_{ori}")) or 0) <= 0:
                        continue
                    x = _f(r.get(f"pi_{a}_{ori}"))
                    y = _f(r.get(f"pi_{b}_{ori}"))
                    if x is not None and y is not None:
                        obs.append((x, y))
            if len(obs) < 3:
                continue
            rho, p = spearman([q[0] for q in obs], [q[1] for q in obs])
            emit(subset_name, a, b,
                 f"{a} vs {b} pi levels across locus-orientation observations",
                 len(obs), "spearman_rho_levels", rho, p)

    # Split-half self-agreement of the 4-fold measure, if the alignment pass
    # has produced it (stats/four_fold_pi.py without --from-table).
    split_path = os.path.join(os.path.dirname(in_path), "four_fold_split_half.tsv")
    if os.path.exists(split_path):
        with open(split_path, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                emit(r["subset"], "fourfold_half_A", "fourfold_half_B",
                     f"{r['description']} (95% range {float(r['ci_lo']):.3f}-"
                     f"{float(r['ci_hi']):.3f}, {r['n_replicates']} random splits)",
                     int(r["n_loci"]), f"split_half_{r['statistic']}",
                     float(r["median"]), None)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]), delimiter="\t")
        w.writeheader()
        w.writerows(out)

    print(f"{len(rows)} loci in table -> {out_path}\n")
    for r in out:
        print(f"{r['subset']:22s} {r['statistic']:26s} n={r['n_loci']:3d}  "
              f"value={float(r['value']):.3f}  p={r['p_value'] or 'NA'}  "
              f"| {r['comparison']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", default=DEFAULT_IN)
    ap.add_argument("--out", dest="out_path", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    build(args.in_path, args.out_path)


if __name__ == "__main__":
    main()
