"""Minimal white-background figures for the CDS-vs-intron selection test.

Reads data/cds_selection_intron_control.tsv and data/cds_selection_placement.tsv
(the calibrated inference; the label-permutation p-values are invalid, see
stats/cds_selection_placement_test.py), writes three PNGs to data/:
  cds_selection_shares.png  share_dir vs share_inv, y=x reference
  cds_selection_S.png       rank-ordered S with zero line
  cds_selection_pvals.png   permutation p vs placement p, log-log, y=x

Marks only -- no titles, no legends, no in-plot text. All points gray: no gene
survives the calibrated placement test (min BY q = 1.0). The p-p panel shows
the permutation test's anti-conservatism directly.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
VERM, BLUE, GRAY, LINE = "#d55e00", "#0072b2", "#b0b0b0", "#dddddd"

rows = [r for r in csv.DictReader(
    open(os.path.join(DATA, "cds_selection_intron_control.tsv")), delimiter="\t")
    if r["status"] == "OK"]
for r in rows:
    for k in ("share_dir", "share_inv", "S", "q_by_selection", "q_by_two_sided",
              "cds_diffs_inv", "intron_diffs_inv"):
        r[k] = float(r[k])

place = {(r["gene_name"], r["inv_id"]): r for r in csv.DictReader(
    open(os.path.join(DATA, "cds_selection_placement.tsv")), delimiter="\t")}

def color(r):
    p = place.get((r["gene_name"], r["inv_id"]))
    if p and float(p["q_by_place"]) < 0.05 and r["S"] > 0:
        return VERM
    if p and float(p["q_by_place_two"]) < 0.05 and r["S"] < 0:
        return BLUE
    return GRAY

def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#666666", labelsize=8, width=0.8)

# ---- 1. shares scatter ----
fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=200)
lim = max(max(r["share_dir"] for r in rows), max(r["share_inv"] for r in rows)) * 1.08
ax.plot([0, lim], [0, lim], color=LINE, lw=1, zorder=1)
for r in rows:
    c = color(r)
    ax.scatter(r["share_dir"], r["share_inv"], s=26 if c != GRAY else 14,
               color=c, alpha=0.9 if c != GRAY else 0.55, lw=0, zorder=2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("CDS share of differences, direct", fontsize=9, color="#444444")
ax.set_ylabel("CDS share of differences, inverted", fontsize=9, color="#444444")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(DATA, "cds_selection_shares.png"), facecolor="white")

# ---- 2. rank-ordered S ----
fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=200)
srt = sorted(rows, key=lambda r: r["S"])
ax.axhline(0, color=LINE, lw=1, zorder=1)
for i, r in enumerate(srt):
    c = color(r)
    ax.scatter(i, r["S"], s=26 if c != GRAY else 12,
               color=c, alpha=0.9 if c != GRAY else 0.55, lw=0, zorder=2)
ax.set_xlabel("genes, ranked", fontsize=9, color="#444444")
ax.set_ylabel("S", fontsize=9, color="#444444")
ax.set_xticks([])
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(DATA, "cds_selection_S.png"), facecolor="white")

# ---- 3. permutation p vs placement p (the calibration failure) ----
fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
lo = 1e-5
ax.plot([lo, 1], [lo, 1], color=LINE, lw=1, zorder=1)
for r in rows:
    p = place.get((r["gene_name"], r["inv_id"]))
    if not p:
        continue
    ax.scatter(max(float(r["p_selection_onesided"]), lo),
               max(float(p["p_place_onesided"]), lo),
               s=14, color=GRAY, alpha=0.6, lw=0, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, 1.3); ax.set_ylim(lo, 1.3)
ax.set_xlabel("permutation p (invalid)", fontsize=9, color="#444444")
ax.set_ylabel("placement p (calibrated)", fontsize=9, color="#444444")
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(DATA, "cds_selection_pvals.png"), facecolor="white")

print("wrote 3 PNGs to data/")
