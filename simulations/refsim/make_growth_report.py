#!/usr/bin/env python
"""Tables and figure for the frequency-trajectory model (Reviewer 1).

Two things have to be shown together, because either alone invites the obvious
objection. First, that the constant-size model really is inert in inversion
frequency and age -- so the criticism is not a matter of degree. Second, that
replacing it with a model in which the inversion rises in frequency from a single
haplotype,
and accumulates diversity while it does, leaves the recurrence classification
where it was -- so the criticism does not overturn the inference.

Panel A measures nucleotide diversity within the inverted haplotypes under both
models. This is a coalescent-only measurement: ancestries and mutations, no
alignment and no tree inference, so it runs in seconds and does not need the
production grid.

Panel B is the false-positive rate from ``run_grid.py --task growth``, which does
run the full upstream classifier.

    python make_growth_report.py out/growth_balanced.csv [--outdir .]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refsim  # noqa: E402

FREQS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
DEPTH_ORDER = ["very_recent", "recent", "young", "old"]
DEPTH_LABEL = {"old": "Old", "young": "Young", "recent": "Recent",
               "very_recent": "Very recent"}
DEPTH_COLOUR = {"old": "#2f5f9f", "young": "#d9b13c",
                "recent": "#8c8c8c", "very_recent": "#a4413a"}
MODEL_LABEL = {"single": "constant size", "single_growth": "trajectory",
               "recurrent": "constant size", "recurrent_growth": "trajectory"}
# The single-event arms give a false-positive rate; the recurrent arms give
# power. Both are the same underlying quantity -- the rate at which the
# classifier calls recurrence -- so both are read off ``call_recurrent``.
SINGLE_ARMS = ("single", "single_growth")
RECUR_ARMS = ("recurrent", "recurrent_growth")


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def diversity_curves(reps=60, seed0=9_500_000):
    """Mean pi within inverted haplotypes, both models, every depth x frequency."""
    import msprime

    out = {}
    for depth in DEPTH_ORDER:
        t_inv = refsim.TIME_DEPTHS[depth]["t_inv"]
        for model in ("single", "single_growth"):
            vals = []
            for x0 in FREQS:
                n_inv = max(2, int(240 * x0) // 2)
                n_dir = (240 - int(240 * x0)) // 2
                de = (refsim.demography_growth(t_inv, x0)
                      if model == "single_growth"
                      else refsim.demography_single(t_inv))
                acc = []
                for r in range(reps):
                    ts = msprime.sim_ancestry(
                        samples=[msprime.SampleSet(n_inv, population="P_I",
                                                   ploidy=2),
                                 msprime.SampleSet(n_dir, population="P_D",
                                                   ploidy=2)],
                        demography=de, sequence_length=refsim.SEQ_LENGTH,
                        recombination_rate=0.0, random_seed=seed0 + r)
                    mts = msprime.sim_mutations(ts, rate=refsim.MU,
                                                random_seed=seed0 + r)
                    acc.append(mts.diversity(
                        sample_sets=[list(range(2 * n_inv))]).item())
                vals.append(statistics.fmean(acc))
            out[(depth, model)] = vals
    return out


def load_grid(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("error"):
                continue
            rows.append(dict(scenario=r["scenario"], depth=r["depth"],
                             rho=float(r["rho"]), freq=float(r["inv_freq"]),
                             call=int(r["call_recurrent"]),
                             events=int(float(r["tree_n_events"]))))
    return rows


def fpr(rows, **kw):
    sel = [r for r in rows if all(r[k] == v for k, v in kw.items())]
    k = sum(r["call"] for r in sel)
    lo, hi = wilson_ci(k, len(sel))
    return k, len(sel), (k / len(sel) if sel else float("nan")), lo, hi


def write_md(div, rows, path, rrows=()):
    with open(path, "w") as fh:
        fh.write("# Inversion frequency as a parameter of the model\n\n")
        fh.write("The constant-size single-event model fixes the inverted class at "
                 "`N_a / 100`\nregardless of the inversion's frequency or age, so "
                 "frequency enters only through\nthe number of haplotypes drawn. "
                 "The trajectory model instead starts the\ninversion as one "
                 "haplotype at `t_inv` and grows it to its observed frequency,\n"
                 "at rate `ln(2 N_a x_0) / T` per generation, with the direct "
                 "class taking the\nremainder so the two orientations conserve "
                 "`N_a`.\n\n")

        fh.write("## Nucleotide diversity (pi, mean pairwise differences per site)\n"
                 "within inverted haplotypes, x 1e-4\n\n")
        fh.write("| depth | model | " + " | ".join(f"{f:.0%}" for f in FREQS)
                 + " | fold 1% to 50% |\n")
        fh.write("|---" * (len(FREQS) + 3) + "|\n")
        for depth in DEPTH_ORDER:
            for model in ("single", "single_growth"):
                v = div[(depth, model)]
                fold = v[-1] / v[0] if v[0] else float("nan")
                fh.write(f"| {DEPTH_LABEL[depth]} | {MODEL_LABEL[model]} | "
                         + " | ".join(f"{1e4 * x:.2f}" for x in v)
                         + f" | {fold:.1f} |\n")

        fh.write("\n## False-positive rate, full classifier\n\n")
        fh.write("| model | rate | 95% CI | calls / loci |\n|---|---|---|---|\n")
        for model in ("single", "single_growth"):
            k, n, p, lo, hi = fpr(rows, scenario=model)
            fh.write(f"| {MODEL_LABEL[model]} | {p:.4f} | {lo:.4f}-{hi:.4f} | "
                     f"{k}/{n} |\n")
        a = [r for r in rows if r["scenario"] == "single"]
        b = [r for r in rows if r["scenario"] == "single_growth"]
        k1, n1 = sum(r["call"] for r in a), len(a)
        k2, n2 = sum(r["call"] for r in b), len(b)
        pp = (k1 + k2) / (n1 + n2)
        se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) if 0 < pp < 1 else 0.0
        z = (k2 / n2 - k1 / n1) / se if se else 0.0
        fh.write(f"\nTwo-proportion z = {z:.2f}, p = "
                 f"{math.erfc(abs(z) / math.sqrt(2)):.4f}.\n")

        fh.write("\n## False-positive rate by inversion frequency\n\n")
        fh.write("| model | " + " | ".join(f"{f:.0%}" for f in FREQS) + " |\n")
        fh.write("|---" * (len(FREQS) + 1) + "|\n")
        for model in ("single", "single_growth"):
            cells = []
            for f in FREQS:
                k, n, p, lo, hi = fpr(rows, scenario=model, freq=f)
                cells.append(f"{p:.4f} ({lo:.3f}-{hi:.3f})")
            fh.write(f"| {MODEL_LABEL[model]} | " + " | ".join(cells) + " |\n")

        if rrows:
            fh.write("\n## Power to detect recurrence, full classifier\n\n")
            fh.write("| model | power | 95% CI | calls / loci |\n|---|---|---|---|\n")
            for model in RECUR_ARMS:
                k, n, p, lo, hi = fpr(rrows, scenario=model)
                fh.write(f"| {MODEL_LABEL[model]} | {p:.4f} | {lo:.4f}-{hi:.4f}"
                         f" | {k}/{n} |\n")
            a = [r for r in rrows if r["scenario"] == RECUR_ARMS[0]]
            b = [r for r in rrows if r["scenario"] == RECUR_ARMS[1]]
            k1, n1 = sum(r["call"] for r in a), len(a)
            k2, n2 = sum(r["call"] for r in b), len(b)
            pp = (k1 + k2) / (n1 + n2)
            se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
            z = (k2 / n2 - k1 / n1) / se if se else 0.0
            fh.write(f"\nTwo-proportion z = {z:.2f}, p = "
                     f"{math.erfc(abs(z) / math.sqrt(2)):.4f}.\n")
            fh.write("\n## Power by inversion frequency\n\n")
            fh.write("| model | " + " | ".join(f"{f:.0%}" for f in FREQS) + " |\n")
            fh.write("|---" * (len(FREQS) + 1) + "|\n")
            for model in RECUR_ARMS:
                cells = []
                for f in FREQS:
                    k, n, p, lo, hi = fpr(rrows, scenario=model, freq=f)
                    cells.append(f"{p:.3f} ({lo:.2f}-{hi:.2f})")
                fh.write(f"| {MODEL_LABEL[model]} | " + " | ".join(cells) + " |\n")
    print("wrote", path)


def plot_model(path):
    """The model itself: what the trajectory does that a constant size cannot.

    Left, the inversion's frequency path from a single founding haplotype to its
    present value, which is the thing the constant-size model omits. Right, the
    inverted effective size today against frequency -- flat at N_a/100 under the
    constant-size model, spanning fifty-fold under the trajectory. Every series is
    directly labelled, so identity never rests on colour alone.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped figure")
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.8, 4.3))
    x0 = 0.10
    for depth in DEPTH_ORDER:
        t_inv = refsim.TIME_DEPTHS[depth]["t_inv"]
        t_gen = t_inv / refsim.GENERATION_TIME
        alpha = refsim.growth_rate(t_inv, x0)
        ts = [t_gen * i / 400 for i in range(401)]
        xs = [x0 * math.exp(-alpha * t) for t in ts]
        yrs = [t * refsim.GENERATION_TIME / 1000.0 for t in ts]
        ax_a.plot(yrs, xs, lw=1.8, color=DEPTH_COLOUR[depth],
                  label=DEPTH_LABEL[depth])
    ax_a.axhline(1.0 / (2 * refsim.N_A), color="#999999", lw=0.8, ls=":")
    ax_a.annotate("one haplotype", xy=(0, 1.0 / (2 * refsim.N_A)),
                  xytext=(2, 4), textcoords="offset points", fontsize=7,
                  color="#666666")
    ax_a.set_yscale("log")
    ax_a.set_xlabel("thousands of years ago")
    ax_a.set_ylabel("inversion frequency")
    ax_a.set_title("Rise in frequency of a 10% inversion")
    ax_a.legend(title="Inversion age", frameon=False, fontsize=8,
                title_fontsize=8, loc="lower left")
    # Time runs into the past, so the axis is reversed: the oldest event sits on
    # the left and the present on the right, and the trajectory reads as a rise.
    ax_a.set_xlim(540, 0)
    ax_a.set_ylim(5e-5, 0.25)

    pub = [refsim.N_A / 100] * len(FREQS)
    traj = [refsim.N_A * f for f in FREQS]
    ax_b.plot(FREQS, pub, lw=1.8, ls=":", marker="o", ms=4, color="#333333",
              label="constant size")
    ax_b.plot(FREQS, traj, lw=1.8, ls="-", marker="o", ms=4, color="#333333",
              label="frequency trajectory")
    ax_b.set_ylim(30, 6000)
    ax_b.legend(frameon=False, fontsize=9, loc="upper left")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xticks(FREQS)
    ax_b.set_xticklabels([f"{f:.0%}" for f in FREQS])
    ax_b.set_xlabel("inversion frequency")
    ax_b.set_ylabel("inverted effective population size today")
    ax_b.set_title("What frequency controls")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("wrote", path)


def plot_ratio(div, path, empirical=0.26):
    """Diversity in inverted relative to direct haplotypes.

    The manuscript reports a 74% reduction in single-event inverted haplotypes,
    i.e. a ratio of 0.26; it is drawn as a reference line because the trajectory
    model predicts that quantity rather than imposing it.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped figure")
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    base = {d: div[(d, "single")] for d in DEPTH_ORDER}
    for depth in DEPTH_ORDER:
        pi_i = div[(depth, "single_growth")]
        # Direct-haplotype diversity is the same under both models, so the
        # constant-size inverted curve at 50% is a usable common denominator only if
        # taken from the same run; use the direct-class expectation 4 N_a mu L.
        pi_d = 4 * refsim.N_A * refsim.MU
        ax.plot(FREQS, [v / pi_d for v in pi_i], lw=1.8, marker="o", ms=4,
                color=DEPTH_COLOUR[depth], label=DEPTH_LABEL[depth])
        ax.plot(FREQS, [v / pi_d for v in base[depth]], lw=1.4, ls=":",
                color=DEPTH_COLOUR[depth], alpha=0.7)
    ax.axhline(empirical, color="#666666", lw=0.9, ls="--")
    ax.annotate("observed in single-event inversions (74% reduction)",
                xy=(FREQS[0], empirical), xytext=(0, 5),
                textcoords="offset points", fontsize=7.5, color="#444444")
    ax.set_xscale("log")
    ax.set_xticks(FREQS)
    ax.set_xticklabels([f"{f:.0%}" for f in FREQS])
    ax.set_xlabel("inversion frequency")
    ax.set_ylabel(r"nucleotide diversity $\pi$, inverted / direct")
    ax.set_title("Reduced diversity is predicted, not imposed")
    ax.legend(title="Inversion age", frameon=False, fontsize=8,
              title_fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("wrote", path)


def _draw_recurrent_demography(ax, depth="young", inv_freq=0.10):
    """The recurrent trajectory demography, into an existing axes."""
    import demesdraw
    import make_demography_fig as mdf

    graph, _single = mdf.graphs(depth, single_model="growth",
                                inv_freq=inv_freq, m_flux=0.0)
    demesdraw.tubes(graph, ax=ax, colours=mdf._colours(graph),
                    positions=dict(mdf.POS_RECURRENT),
                    num_lines_per_migration=0, labels="xticks-mid",
                    max_time=1.25 * max(d.start_time for d in graph.demes
                                        if d.start_time != float("inf")),
                    seed=1)
    pretty = lambda t: mdf.DISPLAY.get(t, t.replace("_", " "))
    for t in ax.texts:
        t.set_text(pretty(t.get_text()))
    ax.set_xticklabels([pretty(t.get_text()) for t in ax.get_xticklabels()],
                       fontsize=7.5)
    ax.set_ylabel("thousands of years ago")
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))


def plot_recurrent_summary(rrows, path, depth="young", inv_freq=0.10):
    """Everything the recurrent trajectory run produced, in one figure.

    (A) the model, across the full width because the three tapering tubes are
    the point of it; (B) power against frequency and (C) power against the age
    of the first event, beneath.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except Exception:
        print("matplotlib unavailable; skipped figure")
        return

    fig = plt.figure(figsize=(11.4, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32,
                          wspace=0.24)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    try:
        _draw_recurrent_demography(ax_a, depth, inv_freq)
        ax_a.set_title("Model: each event begins as one haplotype")
    except Exception as exc:                                # pragma: no cover
        ax_a.text(0.5, 0.5, f"demography unavailable\n{exc}", ha="center",
                  va="center", fontsize=7, transform=ax_a.transAxes)

    for model, ls in zip(RECUR_ARMS, (":", "-")):
        y, lo, hi = [], [], []
        for f in FREQS:
            k, n, p, l, h = fpr(rrows, scenario=model, freq=f)
            y.append(p); lo.append(p - l); hi.append(h - p)
        ax_b.errorbar(FREQS, y, yerr=[lo, hi], ls=ls, marker="o", ms=4, lw=1.8,
                      elinewidth=0.9, capsize=2.5, color="#333333",
                      label=MODEL_LABEL[model])
    ax_b.set_xscale("log")
    ax_b.set_xticks(FREQS)
    ax_b.set_xticklabels([f"{f:.0%}" for f in FREQS])
    ax_b.set_xlabel("inversion frequency")
    ax_b.set_ylabel("power to detect recurrence")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title("Power against frequency")
    ax_b.legend(frameon=False, fontsize=9, loc="lower right")

    xs = range(len(DEPTH_ORDER))
    for model, ls in zip(RECUR_ARMS, (":", "-")):
        y, lo, hi = [], [], []
        for d in DEPTH_ORDER:
            k, n, p, l, h = fpr(rrows, scenario=model, depth=d)
            y.append(p); lo.append(p - l); hi.append(h - p)
        ax_c.errorbar(list(xs), y, yerr=[lo, hi], ls=ls, marker="o", ms=4,
                      lw=1.8, elinewidth=0.9, capsize=2.5, color="#333333",
                      label=MODEL_LABEL[model])
    ax_c.set_xticks(list(xs))
    ax_c.set_xticklabels([DEPTH_LABEL[d] for d in DEPTH_ORDER])
    ax_c.set_xlabel("age of the first inversion event")
    ax_c.set_ylabel("power to detect recurrence")
    ax_c.set_ylim(0, 1.0)
    ax_c.set_title("Power against age")
    ax_c.legend(frameon=False, fontsize=9, loc="lower right")

    for ax, letter in zip((ax_a, ax_b, ax_c), "ABC"):
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-38, 14), textcoords="offset points",
                    fontsize=13, fontweight="bold", va="top")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


def plot_power(rrows, path):
    """Power against inversion frequency, on its own.

    Two series, so a legend carries identity and the curves are also directly
    labelled. Wilson intervals rather than the normal approximation, which is
    what the near-boundary cells need.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped figure")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for model, ls in zip(RECUR_ARMS, (":", "-")):
        y, lo, hi = [], [], []
        for f in FREQS:
            k, n, p, l, h = fpr(rrows, scenario=model, freq=f)
            y.append(p)
            lo.append(p - l)
            hi.append(h - p)
        ax.errorbar(FREQS, y, yerr=[lo, hi], ls=ls, marker="o", ms=4.5, lw=1.8,
                    elinewidth=0.9, capsize=2.5, color="#333333",
                    label=MODEL_LABEL[model])
    ax.set_xscale("log")
    ax.set_xticks(FREQS)
    ax.set_xticklabels([f"{f:.0%}" for f in FREQS])
    ax.set_xlabel("inversion frequency")
    ax.set_ylabel("power to detect recurrence")
    ax.set_title("Recurrent inversions, three events")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print("wrote", path)


def plot(div, rows, path, rrows=()):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        print("matplotlib unavailable; skipped figure")
        return

    n_panels = 3 if rrows else 2
    fig, axs = plt.subplots(1, n_panels, figsize=(4.9 * n_panels, 4.5))
    ax_a, ax_b = axs[0], axs[1]
    for depth in DEPTH_ORDER:
        for model, ls in (("single", ":"), ("single_growth", "-")):
            ax_a.plot(FREQS, [1e4 * v for v in div[(depth, model)]],
                      ls=ls, marker="o", ms=3.5, lw=1.6,
                      color=DEPTH_COLOUR[depth], alpha=0.9)
    ax_a.set_xscale("log")
    ax_a.set_xticks(FREQS)
    ax_a.set_xticklabels([f"{f:.0%}" for f in FREQS])
    ax_a.set_xlabel("inversion frequency")
    ax_a.set_ylabel("nucleotide diversity $\\pi$ within inverted\nhaplotypes ($\\times10^{-4}$ per site)")
    ax_a.set_title("Diversity accumulated along the trajectory")

    for model, ls in (("single", ":"), ("single_growth", "-")):
        y, lo, hi = [], [], []
        for f in FREQS:
            k, n, p, l, h = fpr(rows, scenario=model, freq=f)
            y.append(p)
            lo.append(p - l)
            hi.append(h - p)
        ax_b.errorbar(FREQS, y, yerr=[lo, hi], ls=ls, marker="o", ms=3.5,
                      lw=1.6, elinewidth=0.8, capsize=2, color="#333333")
    ax_b.axhline(0.05, color="#999999", lw=0.8, ls=":", zorder=0)
    ax_b.annotate("5%", xy=(FREQS[0], 0.05), xytext=(0, 4),
                  textcoords="offset points", fontsize=7, color="#666666")
    ax_b.set_xscale("log")
    ax_b.set_xticks(FREQS)
    ax_b.set_xticklabels([f"{f:.0%}" for f in FREQS])
    ax_b.set_xlabel("inversion frequency")
    ax_b.set_ylabel("false-positive rate")
    ax_b.set_title("Recurrence false-positive rate")
    ax_b.set_ylim(-0.004, 0.06)

    if rrows:
        ax_c = axs[2]
        for model, ls in zip(RECUR_ARMS, (":", "-")):
            y, lo, hi = [], [], []
            for f in FREQS:
                k, n, p, l, h = fpr(rrows, scenario=model, freq=f)
                y.append(p)
                lo.append(p - l)
                hi.append(h - p)
            ax_c.errorbar(FREQS, y, yerr=[lo, hi], ls=ls, marker="o", ms=3.5,
                          lw=1.6, elinewidth=0.8, capsize=2, color="#333333")
        ax_c.set_xscale("log")
        ax_c.set_xticks(FREQS)
        ax_c.set_xticklabels([f"{f:.0%}" for f in FREQS])
        ax_c.set_xlabel("inversion frequency")
        ax_c.set_ylabel("power")
        ax_c.set_title("Power to detect recurrence")
        ax_c.set_ylim(0, 1.0)

    handles = [Line2D([], [], color=DEPTH_COLOUR[d], lw=1.6,
                      label=DEPTH_LABEL[d]) for d in DEPTH_ORDER]
    handles += [Line2D([], [], color="#333333", lw=1.6, ls=":",
                       label="constant size"),
                Line2D([], [], color="#333333", lw=1.6, ls="-",
                       label="frequency trajectory")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(path, dpi=200)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("grid", help="single-event grid (run_grid.py --task growth)")
    ap.add_argument("--recurrent-grid", default=None,
                    help="recurrent grid (run_grid.py --task rgrowth); adds the "
                         "power panel and table")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--reps", type=int, default=60)
    args = ap.parse_args(argv)
    rows = load_grid(args.grid)
    rrows = load_grid(args.recurrent_grid) if args.recurrent_grid else []
    div = diversity_curves(args.reps)
    os.makedirs(args.outdir, exist_ok=True)
    write_md(div, rows, os.path.join(args.outdir, "growth_results.md"), rrows)
    plot_model(os.path.join(args.outdir, "growth_model.png"))
    plot(div, rows, os.path.join(args.outdir, "growth_frequency.png"), rrows)
    plot_ratio(div, os.path.join(args.outdir, "growth_diversity_ratio.png"))
    if rrows:
        plot_power(rrows, os.path.join(args.outdir, "growth_power.png"))
        plot_recurrent_summary(
            rrows, os.path.join(args.outdir, "growth_recurrent_summary.png"))
    print(f"{len(rows)} single-event loci, {len(rrows)} recurrent loci")


if __name__ == "__main__":
    main()
