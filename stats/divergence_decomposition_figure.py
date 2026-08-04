#!/usr/bin/env python
"""D_xy decomposed into within-orientation diversity and net divergence.

D_xy = mean(pi) + d_a, so two inversion classes can reach the same absolute
divergence with completely different composition. Reads only
``data/divergence_da_dxy_by_type.tsv``, the recorded per-locus output of
``stats/divergence_da_dxy_by_type.py``; loci without a finite d_a are dropped,
exactly as that script's tests do.

    python stats/divergence_decomposition_figure.py [-o data/divergence_decomposition.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from stats._figstyle import RECURRENT, SINGLE, apply as _apply_style  # noqa: E402

DATA = os.path.join(_ROOT, "data")
TABLE = os.path.join(DATA, "divergence_da_dxy_by_type.tsv")

SCALE = 1e3             # everything is reported in units of 10^-3
PI_COLOUR = "#C9C9C9"   # the shared component, common to both classes


def make_figure(path):
    _apply_style()
    d = pd.read_csv(TABLE, sep="\t").dropna(subset=["da", "dxy"])
    d["mean_pi"] = (d.hudson_pi_hap_group_0 + d.hudson_pi_hap_group_1) / 2
    order = ["Single-event", "Recurrent"]
    means = d.groupby("category")[["mean_pi", "da", "dxy"]].mean() * SCALE
    n = d.category.value_counts()

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, cat in enumerate(order):
        colour = SINGLE if cat == "Single-event" else RECURRENT
        pi, da, dxy = means.loc[cat, ["mean_pi", "da", "dxy"]]
        ax.barh(i, pi, color=PI_COLOUR, height=0.55, zorder=3)
        ax.barh(i, da, left=pi, color=colour, height=0.55, zorder=3)
        ax.text(pi / 2, i, f"{pi:.2f}", ha="center", va="center", fontsize=8.5,
                color="#333333", zorder=4)
        ax.text(pi + da / 2, i, f"{da:.2f}", ha="center", va="center",
                fontsize=8.5, color="white", zorder=4)
        ax.text(dxy + 0.015, i, f"{dxy:.2f}", va="center", fontsize=8.5,
                color="#333333")

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}\nn = {int(n[c])}" for c in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, float(means.dxy.max()) * 1.12)
    ax.set_xlabel("divergence between orientations ($\\times 10^{-3}$)")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=PI_COLOUR, label="mean $\\pi$"),
        plt.Rectangle((0, 0), 1, 1, color=SINGLE, label="$d_a$, single-event"),
        plt.Rectangle((0, 0), 1, 1, color=RECURRENT, label="$d_a$, recurrent")],
        loc="lower right", fontsize=7.5, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out",
                    default=os.path.join(DATA, "divergence_decomposition.png"))
    args = ap.parse_args(argv)
    make_figure(args.out)


if __name__ == "__main__":
    main()
