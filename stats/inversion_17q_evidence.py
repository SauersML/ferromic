"""The 17q21.31 evidence panel: the neutral envelope, and where the locus sits.

Every figure so far has shown p-values -- summaries of a comparison the reader
never sees. This draws the comparison itself: the distribution of
cross-arrangement divergence that neutral drift produces for an arrangement at
exactly this frequency, with the observed value marked on it. If the observed
value sits inside the bulk there is no result, and the figure says so
immediately; the p-value is then just the area to its right.

Panels (each self-contained, greyscale, no annotation text):

  1. neutral distribution of B = d_cross / pi_dir for k of n, with B_obs
  2. the same null under each published human demography, as curves, so the
     spread across histories is visible rather than tabulated
  3. sensitivity of the tail to the assumed effective size, from the
     age-anchored test (stats/inversion_age_test.py) if its table is present

Draws are written out so the figure can be redrawn without re-simulating.

Output: results/inversion_17q_null.tsv.gz, data/inversion_17q_evidence.png
"""

import argparse
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inversion_selection_envelope import (  # noqa: E402
    conditioned_branch_null, envelope_p, parse_demography)

RNG_SEED = 20260817


def draw_null(n, k, pi_dir_abs, n_cand, demography, seed):
    rng = np.random.default_rng(seed)
    tmap, _ = parse_demography(demography)
    A, B, W, Tid, trees = conditioned_branch_null(
        n, k, pi_dir_abs, rng, n_cand=n_cand,
        tmap=None if demography in ("const", "", None) else tmap)
    return A, B, W, Tid, trees, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--pi-dir-abs", type=float, default=306.08772635814887,
                    help="observed absolute direct-class diversity at 17q")
    ap.add_argument("--b-obs", type=float, default=11.580179652719712)
    ap.add_argument("--n-cand", type=int, default=200_000)
    ap.add_argument("--demographies", default="",
                    help="TSV from stats/human_demographies.py; adds panel 2")
    ap.add_argument("--age-table", default="results/inversion_age_test.tsv")
    ap.add_argument("--out-draws", default="results/inversion_17q_null.tsv.gz")
    ap.add_argument("--out-png", default="repo/data/inversion_17q_evidence.png")
    a = ap.parse_args()
    os.chdir(a.workdir)
    os.makedirs("results", exist_ok=True)

    print(f"drawing neutral null: n={a.n}, k={a.k}, {a.n_cand} candidates",
          flush=True)
    A, B, W, Tid, trees, rng = draw_null(a.n, a.k, a.pi_dir_abs, a.n_cand,
                                         "const", RNG_SEED)
    res = envelope_p(B, a.b_obs, W, Tid, "upper", rng)
    print(f"  {len(B)} candidates from {trees} trees; "
          f"p={res['p']:.5f} [{res['lo']:.5f}, {res['hi']:.5f}] "
          f"tail={res['n_tail']} ess={res['ess']:.0f}")

    with gzip.open(a.out_draws, "wt") as fh:
        fh.write("B\tweight\ttree_id\n")
        for b, w, t in zip(B, W, Tid):
            fh.write(f"{b:.6g}\t{w:.6g}\t{t}\n")
    print(f"  wrote {a.out_draws}")

    # ---- extra histories -----------------------------------------------
    curves = []
    if a.demographies and os.path.exists(a.demographies):
        import csv
        rows = [r for r in csv.DictReader(open(a.demographies), delimiter="\t")
                if r["spec"].startswith("pw:")]
        pooled = [r for r in rows if r["config"].startswith("pooled")]
        print(f"  {len(pooled)} multi-ancestry histories for panel 2",
              flush=True)
        for i, r in enumerate(pooled):
            Bd, Wd = draw_null(a.n, a.k, a.pi_dir_abs, 40_000,
                               r["spec"], RNG_SEED + 101 + i)[1:3]
            curves.append((r["model"], Bd, Wd))

    # ---- figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GRAY, LINE = "#1a1a1a", "#8a8a8a", "#dddddd"
    ncol = 2 if not curves else 3
    have_age = os.path.exists(a.age_table)
    ncol = ncol + (1 if have_age else 0) - (0 if curves else 1)
    fig, axes = plt.subplots(1, max(ncol, 2),
                             figsize=(4.4 * max(ncol, 2), 3.6), dpi=200)
    axes = np.atleast_1d(axes)

    def style(ax):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#999999")
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors="#666666", labelsize=8.5, width=0.8)

    # panel 1: the null, weighted, with the observation
    ax = axes[0]
    lb = np.log10(np.clip(B, 1e-3, None))
    bins = np.linspace(lb.min(), max(lb.max(), np.log10(a.b_obs) + 0.05), 70)
    ax.hist(lb, bins=bins, weights=W, color=GRAY, lw=0)
    ax.axvline(np.log10(a.b_obs), color=INK, lw=1.8)
    ax.set_xlabel("cross-orientation divergence / direct diversity",
                  fontsize=9.5, color="#333333")
    ax.set_ylabel("neutral density", fontsize=9.5, color="#333333")
    ticks = [0.3, 1, 3, 10, 30]
    ax.set_xticks([np.log10(t) for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    style(ax)

    # panel 2: tail across histories
    idx = 1
    if curves:
        ax = axes[idx]
        idx += 1
        for _nm, Bd, Wd in curves:
            s = np.argsort(Bd)
            bs, ws = Bd[s], Wd[s]
            surv = 1.0 - np.cumsum(ws) / ws.sum()
            ax.plot(np.clip(bs, 1e-3, None), np.clip(surv, 1e-6, 1),
                    color=GRAY, lw=0.9, alpha=0.85)
        s = np.argsort(B)
        surv = 1.0 - np.cumsum(W[s]) / W.sum()
        ax.plot(np.clip(B[s], 1e-3, None), np.clip(surv, 1e-6, 1),
                color=INK, lw=1.6)
        ax.axvline(a.b_obs, color=INK, lw=1.8, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.4)
        ax.set_xlabel("cross-orientation divergence / direct diversity",
                      fontsize=9.5, color="#333333")
        ax.set_ylabel("neutral probability of exceeding", fontsize=9.5,
                      color="#333333")
        style(ax)

    # panel 3: effective-size sensitivity from the age-anchored test
    if have_age and idx < len(axes):
        import csv
        ar = list(csv.DictReader(open(a.age_table), delimiter="\t"))
        if ar:
            ax = axes[idx]
            ne = [float(r["Ne"]) for r in ar]
            pv = [float(r["p_age"]) for r in ar]
            lo = [float(r["p_age_lo"]) for r in ar]
            hi = [float(r["p_age_hi"]) for r in ar]
            ax.plot(ne, pv, color=INK, lw=1.6, marker="o", ms=4)
            ax.fill_between(ne, lo, hi, color=GRAY, alpha=0.35, lw=0)
            ax.axhline(0.05, color=INK, lw=1.0, ls=(0, (4, 3)))
            ax.set_yscale("log")
            ax.set_xlabel("assumed effective population size", fontsize=9.5,
                          color="#333333")
            ax.set_ylabel("p (age-anchored)", fontsize=9.5, color="#333333")
            style(ax)
            idx += 1

    for j in range(idx, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(a.out_png, facecolor="white")
    print(f"  wrote {a.out_png}")


if __name__ == "__main__":
    main()
