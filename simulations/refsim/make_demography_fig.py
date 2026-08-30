#!/usr/bin/env python
"""Draw the two demographies of ``refsim.py`` with demesdraw.

The models come straight from ``refsim.demography`` (upstream's 9-deme
recurrent model) and ``refsim.demography_single`` (the one-divergence
single-event model), so the picture cannot drift from the simulated model.
Tube widths are N_e, arrows are the model's own migration rates.

The recurrent panel omits ``P_I`` / ``P_D``: they exist for the 1e-5 generations
of the sampling pulse and carry no history, only the admixture proportions
f_I / f_D. Gene flux is drawn between opposite-orientation populations for
every interval in which both exist.

    python make_demography_fig.py [-o demography_model.png] [--depth young]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import demes  # noqa: E402
import demesdraw  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import refsim  # noqa: E402

try:
    from stats._figstyle import DIRECT, INVERTED, apply as _apply_style
except Exception:                                          # pragma: no cover
    DIRECT, INVERTED = "#0072B2", "#D55E00"

    def _apply_style():
        pass

M_CONST = 1e-8         # upstream's within-orientation migration
M_FLUX = 1e-7          # a mid-sweep flux rate, drawn so the term is visible
ANCESTRAL = "#BBBBBB"

# The model's deme names are positional; these are what they mean.
# Every deme carries an orientation except the root: upstream's P_a_I / P_a_D
# are its "Ancestral INV / DIR group". Between-orientation flux moves lineages
# between the two, so orientation is a state a lineage can leave, not a label
# fixed at birth.
NAMES = {
    "P0_D": "direct_1", "P1_I": "inverted_1",
    "P2_I": "inverted_2", "P3_D": "direct_2",
    "Pa_I": "ancestral_inverted", "Pa_D": "ancestral_direct",
    "P00": "ancestral_unoriented",
    "P_I": "inverted", "P_D": "direct",
}
# Ancestry is legible from the drawing, so the ancestors only need their
# orientation; the deme names themselves must stay unique for demes.
DISPLAY = {"ancestral_inverted": "inverted", "ancestral_direct": "direct",
           "ancestral_unoriented": "ancestral"}
GAP = 9000             # x spacing; wider than the largest deme (N_a = 6000)

# Left to right, so the two ancestry pairs (P0_D+P1_I, P2_I+P3_D) sit together.
POS_RECURRENT = {"direct_1": 0, "ancestral_inverted": 0.5 * GAP,
                 "inverted_1": GAP, "ancestral_unoriented": 1.5 * GAP,
                 "inverted_2": 2 * GAP, "ancestral_direct": 2.5 * GAP,
                 "direct_2": 3 * GAP}
# The single-event panel is upstream's own ``singleINV_m1.py``: two populations
# and one divergence. This is not a simplification of the recurrent model -- it
# is a separate script, with its own demography, and it is what produced the
# manuscript's single-event false-positive rates. For this revision, the
# orientation-changing child keeps upstream's N_a/100 (refsim.SINGLE_INV_FRACTION); the direct
# child and the ancestor are N_a. Direct lineages never enter the inverted deme.
POS_SINGLE = {"inverted": 0, "ancestral_unoriented": 0.5 * GAP, "direct": GAP}


def _colours(graph):
    return {d.name: (INVERTED if "inverted" in d.name else
                     DIRECT if "direct" in d.name else ANCESTRAL)
            for d in graph.demes}


def _relabel(graph, generation_time=refsim.GENERATION_TIME):
    """Readable deme names, and time in years rather than generations."""
    d = graph.asdict()

    def scale(obj):
        if isinstance(obj, dict):
            return {k: (v * generation_time
                        if k in ("time", "start_time", "end_time")
                        and isinstance(v, (int, float)) else scale(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [scale(v) for v in obj]
        return obj

    d = scale(d)
    d["time_units"] = "years"
    d["generation_time"] = generation_time
    for deme in d["demes"]:
        deme["name"] = NAMES.get(deme["name"], deme["name"])
        deme["ancestors"] = [NAMES.get(a, a) for a in deme.get("ancestors", [])]
    for m in d.get("migrations", []):
        if "demes" in m:
            m["demes"] = [NAMES.get(x, x) for x in m["demes"]]
        for k in ("source", "dest"):
            if k in m:
                m[k] = NAMES.get(m[k], m[k])
    return demes.Graph.fromdict(d)


def _drop(graph, names):
    """The graph without ``names`` -- and without anything referring to them."""
    d = graph.asdict()
    d["demes"] = [x for x in d["demes"] if x["name"] not in names]
    d["migrations"] = [m for m in d.get("migrations", [])
                       if not (set(m.get("demes", [])) & names)
                       and m.get("source") not in names
                       and m.get("dest") not in names]
    d["pulses"] = [p for p in d.get("pulses", [])
                   if p.get("dest") not in names
                   and not (set(p.get("sources", [])) & names)]
    return demes.Graph.fromdict(d)


def graphs(depth_name, single_model="constant", inv_freq=0.10, m_flux=M_FLUX):
    depth = refsim.TIME_DEPTHS[depth_name]
    if single_model == "growth":
        de = refsim.demography_recurrent_growth(
            depth["t01_23"], depth["t0_1"], depth["t2_3"], inv_freq, 0.5, 0.5,
            m_const=M_CONST, m_flux=m_flux)
    else:
        de = refsim.demography(depth["t01_23"], depth["t0_1"], depth["t2_3"],
                               M_CONST, 0.5, 0.5, m_flux=m_flux)
    # demes rejects migration over the pulse demes' 1e-5-generation lifetime.
    de.set_symmetric_migration_rate(["P_I", "P_D"], 0)
    recurrent = _relabel(_drop(de.to_demes(), {"P_I", "P_D"}))
    # The single-event arm is its own model, not a subgraph of the recurrent one.
    # ``constant`` is upstream's ``singleINV_m1.py``, whose inverted deme sits at
    # N_a/100 for the whole of its history. ``growth`` is the model in which the
    # inversion instead rises from a single haplotype to ``inv_freq``, so the
    # inverted tube tapers to a point at the inversion event and the frequency is
    # a property of the demography rather than of the sampling.
    if single_model == "growth":
        single = _relabel(refsim.demography_growth(
            depth["t_inv"], inv_freq, m_flux=m_flux).to_demes())
    else:
        single = _relabel(
            refsim.demography_single(depth["t_inv"], m_flux=m_flux).to_demes())
    return recurrent, single


def make_figure(path, depth_name="young", single_model="constant",
                inv_freq=0.10, panels="both", m_flux=M_FLUX):
    _apply_style()
    recurrent, single = graphs(depth_name, single_model, inv_freq, m_flux)

    if single_model == "growth":
        single_title = "Single-event, frequency trajectory"
        recurrent_title = "Recurrent, frequency trajectory"
    else:
        single_title = "Single-event"
        recurrent_title = "Recurrent"
    spec = [(recurrent, POS_RECURRENT, recurrent_title, 3.0),
            (single, POS_SINGLE, single_title, 2.2)]
    if panels == "single":
        spec = spec[1:]
    elif panels == "recurrent":
        spec = spec[:1]
    # A lone panel keeps its own title, so the panel title is the figure's
    # subject rather than one half of a comparison.
    widths = [w for *_, w in spec]
    fig_w = 11.0 if len(spec) == 2 else 6.2
    fig_h = 4.6
    fig, axes = plt.subplots(1, len(spec), figsize=(fig_w, fig_h),
                             width_ratios=widths, sharey=True, squeeze=False)
    axes = list(axes[0])
    # The recurrent model sets the time axis when it is drawn; on its own the
    # single-event model sets its own.
    max_time = 1.3 * max(d.start_time for g, *_ in spec for d in g.demes
                         if d.start_time != float("inf"))
    for ax, (graph, pos, title, _w) in zip(axes, spec):
        demesdraw.tubes(graph, ax=ax, colours=_colours(graph),
                        positions={k: v for k, v in pos.items()},
                        num_lines_per_migration=1, labels="xticks-mid",
                        max_time=max_time, title=title, seed=1)
        ax.set_ylabel("")
        # NAMES are python identifiers because demes requires them to be.
        pretty = (lambda t: DISPLAY.get(t, t.replace("_", " ")))
        for t in ax.texts:
            t.set_text(pretty(t.get_text()))
        ax.set_xticklabels([pretty(t.get_text())
                            for t in ax.get_xticklabels()])
    axes[0].set_ylabel("thousands of years ago")
    axes[0].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    axes[0].set_ylim(0, 1.08 * max(a.get_ylim()[1] for a in axes))

    fig.legend(handles=[Patch(facecolor=INVERTED, label="inverted"),
                        Patch(facecolor=DIRECT, label="direct"),
                        Patch(facecolor=ANCESTRAL, label="ancestral")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    stem, extension = os.path.splitext(path)
    if extension.lower() != ".pdf":
        fig.savefig(stem + ".pdf")
    plt.close(fig)
    print("wrote", path)
    if extension.lower() != ".pdf":
        print("wrote", stem + ".pdf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out",
                    default=os.path.join(_HERE, "demography_model.png"))
    ap.add_argument("--depth", default="young",
                    choices=sorted(refsim.TIME_DEPTHS))
    ap.add_argument("--single-model", default="constant",
                    choices=["constant", "growth"],
                    help="single-event panel: upstream's constant-size two-deme "
                         "model, or the sensitivity model in which the inversion "
                         "rises from one haplotype to --inv-freq")
    ap.add_argument("--inv-freq", type=float, default=0.10)
    ap.add_argument("--panels", default="both",
                    choices=["both", "single", "recurrent"],
                    help="draw both demographies, or one on its own")
    ap.add_argument("--m-flux", type=float, default=M_FLUX,
                    help="between-orientation flux drawn as arrows; 0 leaves "
                         "them off, for a figure about the demography alone")
    args = ap.parse_args(argv)
    make_figure(args.out, args.depth, args.single_model, args.inv_freq,
                args.panels, args.m_flux)


if __name__ == "__main__":
    main()
