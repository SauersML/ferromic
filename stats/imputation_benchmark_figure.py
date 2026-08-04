#!/usr/bin/env python
"""6q24.1 (HsInv0284) imputation benchmark: imputed against experimental dosage.

Reads only the recorded benchmark output of ``imputation/benchmark_hsinv0284.py``
(``data/imputation_benchmark_HsInv0284.tsv`` and its ``_summary`` table), so every
number shown is one the benchmark already wrote.

Left: imputed vs experimentally measured dosage per sample, coloured by
superpopulation. Right: carrier frequency by superpopulation, which is why the
benchmark is dominated by AFR -- the inversion is essentially absent elsewhere.

    python stats/imputation_benchmark_figure.py [-o data/imputation_benchmark_HsInv0284.png]
"""
from __future__ import annotations

import argparse
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

from stats._figstyle import CATEGORICAL, NEUTRAL, apply as _apply_style  # noqa: E402

DATA = os.path.join(_ROOT, "data")
CALLS = os.path.join(DATA, "imputation_benchmark_HsInv0284.tsv")
SUMMARY = os.path.join(DATA, "imputation_benchmark_HsInv0284_summary.tsv")

POPS = ["AFR", "EAS", "EUR", "SAS"]
JITTER = 0.045          # dosages are integers on the truth axis; spread them


def panel_scatter(ax, calls, summary):
    """Each superpopulation gets its own column within a truth dosage.

    486 of the 517 samples share experimental dosage 0; drawn on one x they
    overplot completely and whichever group is painted last hides the rest.
    """
    rng = np.random.default_rng(0)
    offsets = np.linspace(-0.26, 0.26, len(POPS))
    for pop, colour, off in zip(POPS, CATEGORICAL, offsets):
        m = calls.Superpopulation == pop
        ax.scatter(calls.experimental_dosage[m] + off
                   + rng.uniform(-JITTER, JITTER, int(m.sum())),
                   calls.imputed_dosage[m], s=14, color=colour, alpha=0.6,
                   linewidths=0, zorder=3, label=pop)
    for t in (0, 1, 2):
        ax.plot([t - 0.32, t + 0.32], [t, t], color=NEUTRAL, lw=0.9, ls=":",
                zorder=1)

    row = summary[summary.group == "ALL"].iloc[0]
    ax.text(0.03, 0.97, f"$r^2$ = {row.r2:.3f}\nconcordance = {row.concordance:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#333333")

    ax.set_xticks([0, 1, 2])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-0.08, 2.15)
    ax.set_xlabel("experimental dosage")
    ax.set_ylabel("imputed dosage")
    ax.legend(loc="lower right", fontsize=7.5, handletextpad=0.2)


def panel_frequency(ax, calls):
    """Measured against imputed frequency, per superpopulation.

    Not a population frequency estimate -- these are the benchmark samples, and
    every carrier in them is AFR. The panel is here to show that the imputation
    reproduces that, including the slight over-call in AFR.
    """
    def freq(column):
        return (calls.groupby("Superpopulation")[column]
                .apply(lambda s: s.sum() / (2 * len(s))).reindex(POPS))

    n = calls.groupby("Superpopulation").size().reindex(POPS)
    x = np.arange(len(POPS))
    ax.bar(x - 0.19, freq("experimental_dosage").values, width=0.36,
           color=CATEGORICAL[2], zorder=3, label="experimental")
    ax.bar(x + 0.19, freq("imputed_dosage").values, width=0.36,
           color=CATEGORICAL[4], zorder=3, label="imputed")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nn = {int(k)}" for p, k in zip(POPS, n)])
    ax.set_ylabel("inverted allele frequency\nin benchmark samples")
    ax.legend(loc="upper right", fontsize=7.5, handletextpad=0.4)


def make_figure(path):
    _apply_style()
    calls = pd.read_csv(CALLS, sep="\t")
    summary = pd.read_csv(SUMMARY, sep="\t")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), width_ratios=[1.3, 1.0])
    panel_scatter(axes[0], calls, summary)
    panel_frequency(axes[1], calls)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default=os.path.join(
        DATA, "imputation_benchmark_HsInv0284.png"))
    args = ap.parse_args(argv)
    make_figure(args.out)


if __name__ == "__main__":
    main()
