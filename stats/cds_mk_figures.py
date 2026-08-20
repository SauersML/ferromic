"""Minimal white-background figures for the arrangement-conditional MK battery.

Reads data/cds_mk_loci.tsv and data/cds_mk_sfs.tsv, writes to data/:
  cds_mk_fractions.png  per-locus protein-changing fraction, direct- vs
                        inverted-born mutations; dot area = mutation count;
                        large dark point = all loci pooled; y = x reference
  cds_mk_sfs.png        ECDF of derived allele frequency, N (dark) vs S
                        (light), one panel per arrangement of origin
  cds_mk_global.png     pooled protein-changing fraction with 95% Wilson CI:
                        inverted-born, direct-born, and the 17q21.31
                        arrangement-fixed substitutions

Marks only: no titles, no legends, no in-plot annotations.
"""

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
INK, GRAY, LINE = "#1a1a1a", "#b0b0b0", "#dddddd"
RECUR, SINGLE = "#e69f00", "#009e73"

loci = list(csv.DictReader(open(os.path.join(DATA, "cds_mk_loci.tsv")),
                           delimiter="\t"))
for r in loci:
    for k in ("N_inv", "S_inv", "N_dir", "S_dir", "N_fix", "S_fix"):
        r[k] = int(r[k])


def wilson(k, n):
    if n == 0:
        return (0, 0, 1)
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#666666", labelsize=8, width=0.8)


# ---- 1. per-locus fractions, direct- vs inverted-born ----------------------
fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=200)
ax.plot([0, 1], [0, 1], color=LINE, lw=1, zorder=1)
for r in loci:
    ni, si, nd, sd = r["N_inv"], r["S_inv"], r["N_dir"], r["S_dir"]
    if (ni + si) == 0 or (nd + sd) == 0:
        continue
    tot = ni + si + nd + sd
    c = RECUR if r["recurrence"] == "recurrent" else SINGLE
    ax.scatter(nd / (nd + sd), ni / (ni + si), s=14 + tot * 1.6,
               color=c, alpha=0.75, lw=0, zorder=2)
Ni = sum(r["N_inv"] for r in loci); Si = sum(r["S_inv"] for r in loci)
Nd = sum(r["N_dir"] for r in loci); Sd = sum(r["S_dir"] for r in loci)
ax.scatter(Nd / (Nd + Sd), Ni / (Ni + Si), s=90, color=INK, zorder=3, lw=0)
ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
ax.set_xlabel("protein-changing fraction, direct-born mutations",
              fontsize=9, color="#444444")
ax.set_ylabel("protein-changing fraction, inverted-born mutations",
              fontsize=9, color="#444444")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(DATA, "cds_mk_fractions.png"), facecolor="white")

# ---- 2. SFS: derived-frequency ECDF, N vs S, per origin --------------------
sfs_path = os.path.join(DATA, "cds_mk_sfs.tsv")
if os.path.exists(sfs_path):
    sfs = list(csv.DictReader(open(sfs_path), delimiter="\t"))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), dpi=200, sharey=True)
    for ax, origin in zip(axes, ("inv", "dir")):
        for cls, col, lw in (("S", GRAY, 1.6), ("N", INK, 1.8)):
            fr = sorted(float(r["freq"]) for r in sfs
                        if r["origin"] == origin and r["cls"] == cls)
            if not fr:
                continue
            ys = [i / len(fr) for i in range(1, len(fr) + 1)]
            ax.step([0] + fr, [0] + ys, where="post", color=col, lw=lw)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"derived frequency, {'inverted' if origin=='inv' else 'direct'}-born",
                      fontsize=9, color="#444444")
        style(ax)
    axes[0].set_ylabel("cumulative fraction of mutations",
                       fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "cds_mk_sfs.png"), facecolor="white")

# ---- 3. pooled fractions with Wilson CIs -----------------------------------
fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=200)
fix17 = next((r for r in loci if r["inv_id"].startswith("17:")), None)
rows = [("inverted-born", Ni, Ni + Si),
        ("direct-born", Nd, Nd + Sd)]
if fix17 and (fix17["N_fix"] + fix17["S_fix"]) > 0:
    rows.append(("17q21.31 fixed", fix17["N_fix"],
                 fix17["N_fix"] + fix17["S_fix"]))
for i, (lab, k, n) in enumerate(rows):
    p, lo, hi = wilson(k, n)
    y = len(rows) - 1 - i
    ax.plot([lo, hi], [y, y], color=GRAY, lw=2.4, solid_capstyle="round",
            zorder=2)
    ax.scatter([p], [y], s=52, color=INK, zorder=3, lw=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9, color="#444444")
ax.set_xlim(0, 1)
ax.set_xlabel("protein-changing fraction of mutations (95% CI)",
              fontsize=9, color="#444444")
style(ax)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(os.path.join(DATA, "cds_mk_global.png"), facecolor="white")

print("wrote MK figures to data/")
