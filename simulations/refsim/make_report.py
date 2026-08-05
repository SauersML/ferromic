#!/usr/bin/env python
"""Aggregate the per-replicate reference-pipeline output into the flux tables/figure.

Input: the per-locus CSVs written by ``run_grid.py --task flux`` (one per shard).
Output: ``flux_results.csv`` (one row per grid cell), ``flux_results.md``, and
``flux_fpr_power.png``.

Every rate here is the upstream classifier's: ``call_recurrent`` is
``minMutHomoplasy >= 2`` on the IQ-TREE ML tree. For ``scenario = single`` the
recurrent-call rate is the false-positive rate; for ``scenario = recurrent`` it
is the detection rate. The ``m_flux = 0`` column is upstream's own model with no
between-orientation exchange, so it is the reference FPR / power that the
manuscript's Fig. 1G reports; the remaining columns are this repository's flux
extension.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
from collections import defaultdict

DEPTH_ORDER = ["very_recent", "recent", "young", "old"]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    The normal approximation is useless here -- several cells are at 0/60, where
    it gives a zero-width interval. Wilson stays inside [0, 1] and keeps a
    sensible width at the boundaries, which is where these rates live. The
    manuscript quotes its own false-positive rate as "4%, 95% C.I.: 0-8%", so a
    point estimate alone cannot be compared with it.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def scenario_names(cs):
    """(single-origin scenario, recurrent scenario) as they appear in the data.

    The single-origin arm has been called ``single`` and ``single_upstream`` at
    different times; hard-coding either silently empties half the report, so take
    the name from the rows rather than assuming it.
    """
    present = {c["scenario"] for c in cs}
    single = sorted(present - {"recurrent"})
    return (single[0] if single else "single"), "recurrent"


def flux_values(cs):
    """The flux column, taken from the data rather than hard-coded, so the same
    report builds the main sweep and the extreme-flux extension."""
    return sorted({c["m_flux"] for c in cs})


def load(patterns):
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("error"):
                        continue
                    rows.append(dict(
                        scenario=r["scenario"], depth=r["depth"],
                        rho=float(r["rho"]), m_flux=float(r["m_flux"]),
                        n_events=int(float(r["tree_n_events"])),
                        call=int(r["call_recurrent"]),
                        n_sites=int(r["n_sites"]),
                        seed=int(r["seed"]),
                    ))
    return rows


def cells(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["depth"], r["rho"], r["m_flux"])].append(r)
    out = []
    for (sc, depth, rho, m), rs in by.items():
        ev = [r["n_events"] for r in rs]
        hist = defaultdict(int)
        for e in ev:
            hist[e] += 1
        k = sum(r["call"] for r in rs)
        lo, hi = wilson_ci(k, len(rs))
        out.append(dict(
            scenario=sc, depth=depth, rho=rho, m_flux=m, reps=len(rs),
            n_called=k,
            recurrent_call_rate=k / len(rs),
            ci_low=lo, ci_high=hi,
            mean_events=statistics.fmean(ev),
            median_events=statistics.median(ev),
            mean_n_sites=statistics.fmean([r["n_sites"] for r in rs]),
            events_hist=dict(sorted(hist.items())),
        ))
    return sorted(out, key=lambda c: (c["scenario"],
                                      DEPTH_ORDER.index(c["depth"]),
                                      c["rho"], c["m_flux"]))


def write_csv(cs, path):
    cols = ["scenario", "depth", "rho", "m_flux", "reps", "n_called",
            "recurrent_call_rate", "ci_low", "ci_high",
            "mean_events", "median_events", "mean_n_sites"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in cs:
            w.writerow([c[k] if not isinstance(c[k], float) else round(c[k], 6)
                        for k in cols])
    print("wrote", path)


def _grid(cs, sc, rho, depth, FLUX):
    return [next((c for c in cs if c["scenario"] == sc and c["depth"] == depth
                  and c["rho"] == rho and abs(c["m_flux"] - m) < 1e-30), None)
            for m in FLUX]


def endpoint_trend(rows, scenario):
    """Two-proportion z test of the lowest against the highest flux level.

    The per-cell rates are noisy at 60 replicates, so the marginal comparison
    across all (depth x rho) cells is what carries any real trend.
    """
    fluxes = sorted({r["m_flux"] for r in rows})
    lo = [r for r in rows if r["scenario"] == scenario and r["m_flux"] == fluxes[0]]
    hi = [r for r in rows if r["scenario"] == scenario and r["m_flux"] == fluxes[-1]]
    k1, n1 = sum(r["call"] for r in lo), len(lo)
    k2, n2 = sum(r["call"] for r in hi), len(hi)
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2)) if p_pool not in (0, 1) else 0.0
    z = (k2 / n2 - k1 / n1) / se if se > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return dict(m_lo=fluxes[0], m_hi=fluxes[-1], rate_lo=k1 / n1, rate_hi=k2 / n2,
                n_lo=n1, n_hi=n2, z=z, p=p)


def write_md(cs, path, rows=None):
    FLUX = flux_values(cs)
    rhos = sorted({c["rho"] for c in cs})
    with open(path, "w") as fh:
        fh.write("# Between-orientation flux sweep — reference classifier\n\n")
        fh.write("Recurrence is called exactly as in `hsiehphLab/inversionSimulation`:\n"
                 "IQ-TREE ML tree over the full-length haplotype alignment, outgroup\n"
                 "collapsed, Fitch parsimony on the orientation trait, recurrent iff\n"
                 "`minMutHomoplasy >= 2`. The `m=0` column is the upstream model itself.\n")
        sc_single, sc_recur = scenario_names(cs)
        for sc, metric in ((sc_single, "false-positive rate"),
                           (sc_recur, "detection rate")):
            fh.write(f"\n## {sc} scenario — {metric}\n")
            for rho in rhos:
                fh.write(f"\n**rho = {rho:.0e}**\n\n")
                fh.write("| depth | " + " | ".join(f"m={m:.0e}" for m in FLUX) + " |\n")
                fh.write("|" + "---|" * (len(FLUX) + 1) + "\n")
                for depth in DEPTH_ORDER:
                    row = _grid(cs, sc, rho, depth, FLUX)
                    if not any(row):
                        continue
                    fh.write(f"| {depth} | " + " | ".join(
                        (f"{c['recurrent_call_rate']:.3f} "
                         f"({c['ci_low']:.2f}-{c['ci_high']:.2f})") if c else "—"
                        for c in row) + " |\n")
        fh.write("\n## Marginal over the nine (depth x rho) cells\n\n")
        fh.write("| scenario | " + " | ".join(f"m={m:.0e}" for m in FLUX) + " |\n")
        fh.write("|" + "---|" * (len(FLUX) + 1) + "\n")
        for sc in (sc_single, sc_recur):
            vals = []
            for m in FLUX:
                sel = [c for c in cs if c["scenario"] == sc
                       and abs(c["m_flux"] - m) < 1e-30]
                if not sel:
                    vals.append("—")
                    continue
                k = sum(c["n_called"] for c in sel)
                n = sum(c["reps"] for c in sel)
                lo, hi = wilson_ci(k, n)
                vals.append(f"{k / n:.3f} ({lo:.3f}-{hi:.3f})")
            fh.write(f"| {sc} | " + " | ".join(vals) + " |\n")

        if rows:
            fh.write("\n## Lowest against highest flux, pooled over all cells\n\n")
            fh.write("| scenario | rate at m_lo | rate at m_hi | z | p |\n|---|---|---|---|---|\n")
            for sc in (sc_single, sc_recur):
                t = endpoint_trend(rows, sc)
                fh.write(f"| {sc} | {t['rate_lo']:.4f} (n={t['n_lo']}) | "
                         f"{t['rate_hi']:.4f} (n={t['n_hi']}) | {t['z']:.2f} | {t['p']:.4f} |\n")
    print("wrote", path)


def plot(cs, path):
    """Two panels: false-positive rate and power, against flux.

    Colour encodes inversion age and line style encodes recombination rate, so
    the two factors are separable at a glance rather than nine indistinguishable
    series. Age is an ordered quantity, so it gets a one-hue light-to-dark ramp
    (older = darker) rather than three unrelated hues, which would also collide
    with the recurrence/orientation colours used elsewhere in the paper. Both
    legends name their variable in words -- a bare rho reads as "p" to anyone who
    has not just read the Methods.

    The 5% reference line appears only on the false-positive panel, where it
    means something.
    """
    FLUX = flux_values(cs)
    import os
    import sys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    try:
        from stats._figstyle import apply as _apply_style
        _apply_style()
    except Exception:
        pass
    # Fig. 1G's own palette, unaltered, so a reader moving between the two
    # figures maps the depth classes without relearning them: Old blue, Young
    # gold, Recent gray.
    depth_color = {"old": "#2f5f9f", "young": "#d9b13c", "recent": "#8c8c8c",
                   "very_recent": "#a4413a"}
    # The names are Fig. 1G's, and so are the split times behind them.
    # Bare names, no times: the recurrent panel has three split times per class
    # and the single-origin panel has one, so any number here is wrong for one
    # of the two panels. Both parameterisations belong in the caption.
    depth_label = {"old": "Old", "young": "Young", "recent": "Recent",
                   "very_recent": "VeryRecent"}

    def _pow10(v):
        """`1e-08` is machine notation; a figure should say 10^-8."""
        if v == 0:
            return "0"
        return f"$10^{{{round(math.log10(v))}}}$"

    rhos = sorted({c["rho"] for c in cs})
    rho_style = {r: st for r, st in zip(rhos, ["-", "--", ":"])}
    xticks = [max(m, 3e-10) for m in FLUX]
    xlabels = [_pow10(m) for m in FLUX]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.7), sharey=True)
    sc_single, sc_recur = scenario_names(cs)
    for ax, sc, title, ylab in (
            (axes[0], sc_single, "Single-origin false-positive rate", "rate"),
            (axes[1], sc_recur, "Recurrent power", "")):
        for depth in DEPTH_ORDER:
            for rho in rhos:
                row = _grid(cs, sc, rho, depth, FLUX)
                if not any(row):
                    continue
                y = [c["recurrent_call_rate"] if c else float("nan")
                     for c in row]
                lo = [(c["recurrent_call_rate"] - c["ci_low"]) if c else 0.0
                      for c in row]
                hi = [(c["ci_high"] - c["recurrent_call_rate"]) if c else 0.0
                      for c in row]
                ax.errorbar(xticks, y, yerr=[lo, hi],
                            marker="o", ms=3.5, lw=1.6,
                            elinewidth=0.8, capsize=2, alpha=0.9,
                            ls=rho_style.get(rho, "-"),
                            color=depth_color.get(depth, "#666666"))
        ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("between-orientation flux $m$\n(per lineage per generation)")
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
        if ylab:
            ax.set_ylabel(ylab)
    # The 5% line is a false-positive benchmark; it says nothing about power.
    axes[0].axhline(0.05, color="#999999", lw=0.8, ls=":", zorder=0)
    axes[0].annotate("5%", xy=(xticks[0], 0.05), xytext=(0, 4),
                     textcoords="offset points", fontsize=7, color="#666666")

    # Both keys apply to both panels, so they go in one box centred under the
    # figure. Split into two legends sitting left and right they read as though
    # each belonged to the panel above it.
    def _header(text):
        return Line2D([], [], ls="", marker="", label=text)

    age = [Line2D([], [], color=depth_color[d], lw=2, label=depth_label[d])
           for d in ("old", "young", "recent", "very_recent")
           if any(c["depth"] == d for c in cs)]
    rec = [Line2D([], [], color="#444444", lw=1.6, ls=rho_style[r],
                  label=_pow10(r))
           for r in rhos]
    # Column-major fill, so column one is the age key and column two the rate.
    handles = ([_header("Inversion age")] + age
               + [_header("Recombination rate\n(per bp per generation)")] + rec)
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.01), frameon=True, framealpha=1.0,
               edgecolor="#CCCCCC", columnspacing=2.4, handlelength=2.4,
               borderpad=0.8, labelspacing=0.5)
    fig.tight_layout(rect=(0, 0.23, 1, 1.0))
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="*", default=["out/flux_shard*.csv"])
    ap.add_argument("--prefix", default="flux",
                   help="output basename stem (use 'extreme' for the extension grid)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args(argv)

    rows = load(args.inputs or ["out/flux_shard*.csv"])
    if not rows:
        raise SystemExit("no input rows found")
    cs = cells(rows)
    os.makedirs(args.outdir, exist_ok=True)
    write_csv(cs, os.path.join(args.outdir, f"{args.prefix}_results.csv"))
    write_md(cs, os.path.join(args.outdir, f"{args.prefix}_results.md"), rows=rows)
    with open(os.path.join(args.outdir, f"sweep_{args.prefix}.json"), "w") as fh:
        json.dump(cs, fh, indent=2)
    try:
        plot(cs, os.path.join(args.outdir, f"{args.prefix}_fpr_power.png"))
    except ImportError:
        print("matplotlib unavailable; skipped figure")
    print(f"{len(rows)} replicates -> {len(cs)} cells")


if __name__ == "__main__":
    main()
