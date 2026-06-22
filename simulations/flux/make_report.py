#!/usr/bin/env python
"""Build results table (CSV + markdown) and figures from sweep JSON files."""
import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLUX = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows.extend(json.load(fh))
    # de-dup on (scenario,depth,rho,m_flux): keep the one with most reps
    best = {}
    for r in rows:
        k = (r["scenario"], r["depth"], r["rho"], r["m_flux"])
        if k not in best or r["reps"] > best[k]["reps"]:
            best[k] = r
    return list(best.values())


def write_csv(rows, path):
    # recurrent_call_rate is the unconditional detection rate (all reps); the
    # *_conditional column excludes recurrent reps where one inverted origin was
    # unsampled (fI in {0,1}), i.e. recurrence was not genealogically observable;
    # n_endpoint is how many such reps were excluded.
    cols = ["scenario", "depth", "rho", "m_flux", "reps",
            "recurrent_call_rate", "recurrent_call_rate_conditional", "n_endpoint",
            "mean_events", "median_events"]
    with open(path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in sorted(rows, key=lambda x: (x["scenario"], x["depth"],
                                             x["rho"], x["m_flux"])):
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def write_md(rows, path):
    with open(path, "w") as fh:
        for sc in ("single", "recurrent"):
            metric = "FPR (recurrent-call rate)" if sc == "single" else \
                     "Recurrent detection rate (unconditional)"
            fh.write(f"\n### {sc.upper()} scenario — {metric}\n\n")
            depths = sorted({r["depth"] for r in rows if r["scenario"] == sc})
            rhos = sorted({r["rho"] for r in rows if r["scenario"] == sc})
            for rho in rhos:
                fh.write(f"\n**rho = {rho:.0e}**\n\n")
                fh.write("| depth | " +
                         " | ".join(f"m={m:.0e}" for m in FLUX) + " |\n")
                fh.write("|" + "---|" * (len(FLUX) + 1) + "\n")
                for depth in depths:
                    cells = []
                    for m in FLUX:
                        match = [r for r in rows if r["scenario"] == sc and
                                 r["depth"] == depth and r["rho"] == rho and
                                 abs(r["m_flux"] - m) < 1e-30]
                        if match:
                            cells.append(f"{match[0]['recurrent_call_rate']:.2f}")
                        else:
                            cells.append("—")
                    fh.write(f"| {depth} | " + " | ".join(cells) + " |\n")


def plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xticks = [max(m, 3e-10) for m in FLUX]  # 0 -> placeholder for log axis
    xlabels = ["0", "1e-9", "1e-8", "1e-7", "1e-6"]
    for ax, sc, title, ylab in [
        (axes[0], "single", "Single-event: false-positive rate", "FPR"),
        (axes[1], "recurrent", "Recurrent: detection rate", "Recurrent detection rate"),
    ]:
        depths = sorted({r["depth"] for r in rows if r["scenario"] == sc})
        rhos = sorted({r["rho"] for r in rows if r["scenario"] == sc})
        styles = {0.0: "-", 1e-8: "--", 1e-6: ":"}
        colors = {"recent": "tab:blue", "young": "tab:green", "old": "tab:red"}
        for depth in depths:
            for rho in rhos:
                ys = []
                for m in FLUX:
                    match = [r for r in rows if r["scenario"] == sc and
                             r["depth"] == depth and r["rho"] == rho and
                             abs(r["m_flux"] - m) < 1e-30]
                    ys.append(match[0]["recurrent_call_rate"] if match else np.nan)
                ax.plot(xticks, ys, marker="o",
                        ls=styles.get(rho, "-"),
                        color=colors.get(depth, "k"),
                        label=f"{depth}, rho={rho:.0e}")
        ax.set_xscale("log")
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
        ax.set_xlabel("between-orientation flux m (per lineage per gen)")
        ax.set_ylabel(ylab)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title)
        ax.axhline(0.05, color="gray", lw=0.7, ls=":")
        ax.legend(fontsize=7)
    fig.suptitle("Effect of between-orientation flux (gene conversion / double crossover)\n"
                 "on the structured-coalescent recurrence classifier", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    paths = sys.argv[1:] or glob.glob("sweep_*.json")
    rows = load(paths)
    write_csv(rows, "flux_results.csv")
    write_md(rows, "flux_results_tables.md")
    plot(rows, "flux_fpr_power.png")
    print("rows:", len(rows))
