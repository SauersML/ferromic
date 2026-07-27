"""One figure style for the whole paper, so panels read as a single system.

Before this, each script picked its own colours -- ``tab:blue``/``tab:green``/
``tab:red`` in the simulation report, ``#0072B2`` in the PAML plots, matplotlib
defaults elsewhere -- so the same distinction was drawn in a different colour
depending on which figure you were looking at. This module fixes the mapping once.

Palette
-------
Okabe-Ito, which is colourblind-safe by construction, restricted to the six steps
that pass the categorical checks against a light surface (lightness band, chroma
floor, adjacent-pair CVD separation, normal-vision floor). Worst adjacent pair is
ΔE 9.6 under deuteranopia and 16.4 under normal vision.

The semantic assignments preserve the conventions already in the manuscript, so
existing panels do not change meaning:

  single-event  green   #009E73     (Fig. 3B already uses green for single-event)
  recurrent     orange  #E69F00     (and orange for recurrent)
  direct        blue    #0072B2
  inverted      vermil. #D55E00

Three of the steps fall below 3:1 contrast against a light surface, so they must
never be the only cue: every figure using them carries a legend or direct labels,
and the underlying numbers are in the supplementary tables. That is the required
relief for the contrast warning, not an oversight.

Usage::

    from stats._figstyle import apply, SINGLE, RECURRENT, DIRECT, INVERTED
    apply()
"""
from __future__ import annotations

# Okabe-Ito, in fixed order. Never cycle past the end -- an extra category folds
# into "Other", small multiples, or a second encoding channel.
CATEGORICAL = ["#009E73", "#E69F00", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9"]

SINGLE = "#009E73"      # single-event inversions
RECURRENT = "#E69F00"   # recurrent inversions
DIRECT = "#0072B2"      # direct orientation
INVERTED = "#D55E00"    # inverted orientation
NEUTRAL = "#666666"     # reference lines, unclassified points

RECURRENCE_COLORS = {
    "single": SINGLE, "single-event": SINGLE, "Single-event": SINGLE, 0: SINGLE,
    "recurrent": RECURRENT, "Recurrent": RECURRENT, 1: RECURRENT,
}
ORIENTATION_COLORS = {
    "direct": DIRECT, "Direct": DIRECT, 0: DIRECT,
    "inverted": INVERTED, "Inverted": INVERTED, 1: INVERTED,
}


def apply(*, base_size: float = 9.0, dpi: int = 300) -> None:
    """Set rcParams. Safe to call more than once; changes nothing but style."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "legend.frameon": False,
        # Recessive frame: data should be the darkest thing on the page.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
    })


def recurrence_color(value, default: str = NEUTRAL) -> str:
    return RECURRENCE_COLORS.get(value, default)


def orientation_color(value, default: str = NEUTRAL) -> str:
    return ORIENTATION_COLORS.get(value, default)
