#!/usr/bin/env python
"""Figure for the SD-architecture recurrence classification.

Reads only ``data/recurrence_sd_calls.tsv``, the recorded output of
``stats/recurrence_sd_architecture.py``, so the figure carries no numbers of its
own. Nothing here comes from simulation: both panels are assembly-derived
flanking-repeat architecture against the consensus labels.

   the 93 loci in flanking inverted-repeat size x identity, coloured by the
   consensus label, shaded by what the a-priori hard rule calls them
   (recurrent iff >= 10 kbp and >= 95% identity);
   the orientation x recurrence diversity interaction under either label set.

    python stats/recurrence_sd_figure.py [-o data/recurrence_sd_figure.png]
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

from stats._figstyle import RECURRENT, SINGLE, apply as _apply_style  # noqa: E402

DATA = os.path.join(_ROOT, "data")
CALLS = os.path.join(DATA, "recurrence_sd_calls.tsv")
SUMMARY = os.path.join(DATA, "recurrence_sd_summary.tsv")

MIN_IDENTITY = 95.0     # the NAHR thresholds of the hard rule, fixed a priori
MIN_SIZE_KBP = 10.0
JITTER = 0.34           # half-width of the score panel's jitter, in x units


def panel_architecture(ax):
    calls = pd.read_csv(CALLS, sep="\t")
    # The rule partitions the plane, so shade both sides in the colour of the
    # call they carry rather than labelling one corner in words.
    ax.add_patch(plt.Rectangle((-1e3, -1e3), 2e3, 2e3, color=SINGLE,
                               alpha=0.07, zorder=0))
    ax.add_patch(plt.Rectangle((MIN_SIZE_KBP, MIN_IDENTITY), 1e3, 1e3,
                               color=RECURRENT, alpha=0.16, zorder=0))
    ax.axhline(MIN_IDENTITY, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
    ax.axvline(MIN_SIZE_KBP, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
    for label, colour in ((0.0, SINGLE), (1.0, RECURRENT)):
        m = calls.consensus == label
        ax.scatter(calls.sd_size_kbp[m], calls.sd_identity_pct[m], s=26,
                   color=colour, alpha=0.85, zorder=3)
    ax.set_xlim(-2, 68)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("flanking inverted-repeat size (kbp)")
    ax.set_ylabel("repeat identity (%)")
    ax.set_title("SD architecture", loc="left")
    # Shading is what the rule calls, points are the consensus labels it is
    # checked against; a point on the wrong shade is a reclassification.
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=SINGLE, alpha=0.07,
                      label="rule: single-event"),
        plt.Rectangle((0, 0), 1, 1, color=RECURRENT, alpha=0.16,
                      label="rule: recurrent"),
        plt.Line2D([], [], ls="", marker="o", color=SINGLE,
                   label="consensus single-event"),
        plt.Line2D([], [], ls="", marker="o", color=RECURRENT,
                   label="consensus recurrent")],
        loc="lower right", fontsize=7.5, handletextpad=0.4,
        labelspacing=0.35)


def panel_interaction(ax):
    """The orientation x recurrence interaction under either label set."""
    summary = pd.read_csv(SUMMARY, sep="\t")

    def row(label_set):
        q = f"[{label_set}] Interaction (difference between those two)"
        r = summary.loc[summary.quantity == q].iloc[0]
        return float(r.value), float(r.p)

    sets = [("consensus", "consensus"), ("SD hard rule (primary)", "SD rule")]
    # Deliberately NOT the recurrence palette used in the other two panels:
    # there green/orange mean single-event/recurrent, here the bars distinguish
    # which CLASSIFIER produced the labels, and reusing the same two colours for
    # a different contrast reads as if the bars were the two recurrence classes.
    classifier_colours = ("#4C5B8A", "#B07AA1")
    for i, (key, name) in enumerate(sets):
        value, p = row(key)
        ax.bar(i, value, width=0.55, color=classifier_colours[i],
               alpha=0.9, zorder=3)
        ax.text(i, value + 0.12, f"{value:.2f}$\\times$\np = {p:.1g}",
                ha="center", va="bottom", fontsize=8, color="#333333")

    ax.axhline(1.0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([n for _, n in sets])
    ax.set_xlim(-0.6, len(sets) - 0.4)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("single-event / recurrent\norientation effect")
    ax.set_title("Diversity interaction", loc="left")


def make_figure(path):
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8),
                             width_ratios=[1.35, 0.85])
    panel_architecture(axes[0])
    panel_interaction(axes[1])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out",
                    default=os.path.join(DATA, "recurrence_sd_figure.png"))
    args = ap.parse_args(argv)
    make_figure(args.out)


if __name__ == "__main__":
    main()
