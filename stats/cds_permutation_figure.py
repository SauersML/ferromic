#!/usr/bin/env python
"""Exact permutation null for the CDS recurrence x orientation interaction.

``stats/robust_cds_reanalysis.py`` reports the interaction as an exact
Welch-studentised permutation over every assignment of the recurrence labels
(all C(26,7) of them), but records only the resulting p-value. This redraws that
same enumeration to show the null it is measured against, with the observed
statistic marked.

The data are rebuilt by importing that module, so the null here is the null
there; the recomputed two-sided p is checked against the recorded value in
``data/robust_cds_reanalysis_results.tsv`` and the script fails if they differ.

    python stats/cds_permutation_figure.py [-o data/cds_permutation_null.png]
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from stats import robust_cds_reanalysis as rcr  # noqa: E402
from stats._figstyle import NEUTRAL, SINGLE, apply as _apply_style  # noqa: E402

DATA = os.path.join(_ROOT, "data")
RESULTS = os.path.join(DATA, "robust_cds_reanalysis_results.tsv")
RECORDED_METHOD = "Exact Welch-studentised recurrence-label permutation"


def inversion_deltas():
    """The per-inversion Inverted - Direct differences the tests act on."""
    from pathlib import Path

    df, _ = rcr.load_matched_data(Path(_ROOT))
    return rcr.inversion_summary(rcr.build_paired_cds(df))


def sign_flip_null(values):
    """(null t values, observed t) for the exact sign-flip test -- 2^n flips."""
    x = np.asarray(values, float)
    t_obs = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(x))))
    perm = signs * x[None, :]
    ses = perm.std(axis=1, ddof=1) / math.sqrt(len(x))
    return perm.mean(axis=1) / ses, t_obs


def label_permutation_null(inv):
    """(null t values, observed t) for the recurrence-label permutation."""
    x = inv["delta"].to_numpy(float)
    g = inv["consensus"].to_numpy(int)

    a, b = x[g == 0], x[g != 0]
    n_a, n_b = len(a), len(b)
    t_obs = ((a.mean() - b.mean())
             / math.sqrt(a.var(ddof=1) / n_a + b.var(ddof=1) / n_b))

    total, total_ss = float(x.sum()), float(np.dot(x, x))
    t_null = np.empty(math.comb(len(x), n_a))
    for k, inds in enumerate(itertools.combinations(range(len(x)), n_a)):
        sel = x[list(inds)]
        sa, ssa = float(sel.sum()), float(np.dot(sel, sel))
        sb, ssb = total - sa, total_ss - ssa
        va = max((ssa - sa * sa / n_a) / (n_a - 1), 0.0)
        vb = max((ssb - sb * sb / n_b) / (n_b - 1), 0.0)
        se = math.sqrt(va / n_a + vb / n_b)
        t_null[k] = (sa / n_a - sb / n_b) / se if se > 0 else 0.0
    return t_null, float(t_obs)


def _check(p_two, method):
    """The recomputed p must equal the one already recorded for that test."""
    row = pd.read_csv(RESULTS, sep="\t").query("method == @method").iloc[0]
    if not math.isclose(p_two, float(row.p_two_sided), rel_tol=1e-9,
                        abs_tol=1e-12):
        raise SystemExit(f"{method}: recomputed p {p_two} != {row.p_two_sided}")


def _panel(ax, t_null, t_obs, method, title, bins):
    tol = rcr._tol(t_obs)
    p_two = float(np.mean(np.abs(t_null) >= abs(t_obs) - tol))
    _check(p_two, method)

    edges = np.histogram_bin_edges(t_null, bins=bins)
    ax.hist(t_null, bins=edges, color=NEUTRAL, alpha=0.35, zorder=2)
    ax.hist(t_null[np.abs(t_null) >= abs(t_obs) - tol], bins=edges,
            color=SINGLE, alpha=0.9, zorder=3)
    ax.axvline(t_obs, color=SINGLE, lw=1.8, zorder=4)
    ax.annotate(f"t = {t_obs:.2f}\np = {p_two:.3f}", xy=(t_obs, 0),
                xytext=(-8, 26), textcoords="offset points", ha="right",
                fontsize=8.5, color="#333333")
    ax.set_xlabel("studentised t under the null")
    ax.set_ylabel(f"assignments (of {len(t_null):,})")
    ax.set_title(title, loc="left")


def make_figure(path):
    _apply_style()
    inv = inversion_deltas()
    single = inv.loc[inv.consensus == 0, "delta"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    _panel(axes[0], *sign_flip_null(single),
           method="Primary: paired inversion-level mean; exact sign-flip t",
           title="Single-event orientation effect, sign flips", bins=24)
    _panel(axes[1], *label_permutation_null(inv), method=RECORDED_METHOD,
           title="Recurrence x orientation, label permutation", bins=90)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out",
                    default=os.path.join(DATA, "cds_permutation_null.png"))
    args = ap.parse_args(argv)
    make_figure(args.out)


if __name__ == "__main__":
    main()
