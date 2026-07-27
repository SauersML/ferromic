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
import os
import statistics
from collections import defaultdict

DEPTH_ORDER = ["recent", "young", "old"]


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
        out.append(dict(
            scenario=sc, depth=depth, rho=rho, m_flux=m, reps=len(rs),
            recurrent_call_rate=sum(r["call"] for r in rs) / len(rs),
            mean_events=statistics.fmean(ev),
            median_events=statistics.median(ev),
            mean_n_sites=statistics.fmean([r["n_sites"] for r in rs]),
            events_hist=dict(sorted(hist.items())),
        ))
    return sorted(out, key=lambda c: (c["scenario"],
                                      DEPTH_ORDER.index(c["depth"]),
                                      c["rho"], c["m_flux"]))


def write_csv(cs, path):
    cols = ["scenario", "depth", "rho", "m_flux", "reps", "recurrent_call_rate",
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


def write_md(cs, path):
    FLUX = flux_values(cs)
    rhos = sorted({c["rho"] for c in cs})
    with open(path, "w") as fh:
        fh.write("# Between-orientation flux sweep — reference classifier\n\n")
        fh.write("Recurrence is called exactly as in `hsiehphLab/inversionSimulation`:\n"
                 "IQ-TREE ML tree over the full-length haplotype alignment, outgroup\n"
                 "collapsed, Fitch parsimony on the orientation trait, recurrent iff\n"
                 "`minMutHomoplasy >= 2`. The `m=0` column is the upstream model itself.\n")
        for sc, metric in (("single", "false-positive rate"),
                           ("recurrent", "detection rate")):
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
                        f"{c['recurrent_call_rate']:.3f}" if c else "—"
                        for c in row) + " |\n")
        fh.write("\n## Marginal over the nine (depth x rho) cells\n\n")
        fh.write("| scenario | " + " | ".join(f"m={m:.0e}" for m in FLUX) + " |\n")
        fh.write("|" + "---|" * (len(FLUX) + 1) + "\n")
        for sc in ("single", "recurrent"):
            vals = []
            for m in FLUX:
                sel = [c for c in cs if c["scenario"] == sc
                       and abs(c["m_flux"] - m) < 1e-30]
                vals.append(f"{statistics.fmean([c['recurrent_call_rate'] for c in sel]):.3f}"
                            if sel else "—")
            fh.write(f"| {sc} | " + " | ".join(vals) + " |\n")
    print("wrote", path)


def plot(cs, path):
    FLUX = flux_values(cs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xticks = [max(m, 3e-10) for m in FLUX]
    xlabels = ["0", "1e-9", "1e-8", "1e-7", "1e-6"]
    styles = {0.0: "-", 1e-8: "--", 1e-6: ":"}
    colors = {"recent": "tab:blue", "young": "tab:green", "old": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sc, title, ylab in (
            (axes[0], "single", "Single-origin: false-positive rate", "FPR"),
            (axes[1], "recurrent", "Recurrent: detection rate", "power")):
        for depth in DEPTH_ORDER:
            for rho in sorted({c["rho"] for c in cs}):
                row = _grid(cs, sc, rho, depth, FLUX)
                if not any(row):
                    continue
                ax.plot(xticks,
                        [c["recurrent_call_rate"] if c else float("nan") for c in row],
                        marker="o", ls=styles.get(rho, "-"),
                        color=colors.get(depth, "k"),
                        label=f"{depth}, rho={rho:.0e}")
        ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("between-orientation flux m (per lineage per generation)")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title)
        ax.axhline(0.05, color="gray", lw=0.7, ls=":")
        ax.legend(fontsize=7)
    fig.suptitle("Between-orientation flux and the Porubsky et al. recurrence classifier\n"
                 "(IQ-TREE ML tree + Fitch parsimony, hsiehphLab/inversionSimulation)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
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
    write_md(cs, os.path.join(args.outdir, f"{args.prefix}_results.md"))
    with open(os.path.join(args.outdir, f"sweep_{args.prefix}.json"), "w") as fh:
        json.dump(cs, fh, indent=2)
    try:
        plot(cs, os.path.join(args.outdir, f"{args.prefix}_fpr_power.png"))
    except ImportError:
        print("matplotlib unavailable; skipped figure")
    print(f"{len(rows)} replicates -> {len(cs)} cells")


if __name__ == "__main__":
    main()
