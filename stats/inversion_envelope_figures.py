"""Minimal white figures for the inversion-selection envelope test.

Writes to data/:

  inversion_envelope_AB.png    every testable locus: inverted-class relative
                               age (A) vs cross-arrangement depth (B, log).
                               Vermilion = nominal sweep whose pi_dir clock is
                               flagged untrustworthy; gold = the balancing
                               locus; open marks = clock-flagged.
  inversion_envelope_val.png   validation: false-positive / power rates per
                               scenario on both axes, dashed line at 0.05.
  inversion_envelope_demog.png 17q21.31's p_balance under every published human
                               demography, with Monte Carlo CIs. The point of
                               the figure is the SPREAD: one p-value from one
                               assumed history is not a result.
  inversion_envelope_mc.png    the same locus's p_balance as Monte Carlo
                               resolution increases -- why the originally
                               reported values were noise.

These are EXPLANATORY figures, so unlike the manuscript panels they carry
titles, legends and plain-language labels. Nobody can read "OOA+Archaic_5R19
YRI" or "+-5%n window, 20k cand" without the source code in front of them, so
model ids are rendered as author-year, population codes as the populations they
name, and every colour is defined in a legend on the figure itself.
"""

import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")
INK, GRAY, LINE = "#1a1a1a", "#b0b0b0", "#dddddd"
VERM, GOLD, GREEN = "#d55e00", "#e69f00", "#009e73"
BLUE = "#0072b2"

# Locus panel source. This is the single-origin-only table: the one-branch null
# does not describe recurrent or unclassified inversions, so those loci are
# never scored and never appear here. The old whole-callset screens
# (inversion_envelope_1kg{,_exactk,_win05}.tsv,
# inversion_selection_envelope.tsv) were deleted for exactly that reason -- do
# not reintroduce them as a fallback.
AB_SOURCES = ["inversion_envelope_single_origin.tsv"]


def first_existing(names):
    """Path of the first name that exists, or (None, None). Missing inputs skip
    their panel rather than killing the run: the panels are independent, and the
    single-origin table is produced by a long job."""
    for nm in names:
        p = os.path.join(DATA, nm)
        if os.path.exists(p):
            return p, nm
    return None, None


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def fnum(row, *keys, default=float("nan")):
    """First present, parseable numeric field among keys."""
    for k in keys:
        v = row.get(k, "")
        if v not in ("", "nan", "NaN", None):
            try:
                return float(v)
            except ValueError:
                continue
    return default


# stdpopsim model ids -> the paper everyone actually cites
MODEL_NAMES = {
    "OutOfAfrica_3G09": "Gutenkunst 2009",
    "OutOfAfrica_2T12": "Tennessen 2012",
    "Africa_1T12": "Tennessen 2012 (Africa)",
    "AmericanAdmixture_4B18": "Browning 2018",
    "OutOfAfricaArchaicAdmixture_5R19": "Ragsdale 2019",
    "OutOfAfricaExtendedNeandertalAdmixturePulse_3I21": "Iasi 2021",
    "Zigzag_1S14": "Schiffels 2014 (zigzag)",
    "AncientEurasia_9K19": "Kamm 2019",
    "PapuansOutOfAfrica_10J19": "Jacobs 2019",
    "AshkSub_7G19": "Gladstein 2019",
    "OutOfAfrica_4J17": "Jouganous 2017",
    "Africa_1B08": "Boyko 2008",
    "CONSTANT": "constant population size",
}

# 1000G / stdpopsim population codes -> what they are
POP_NAMES = {
    "YRI": "Yoruba", "CEU": "European", "CHB": "Han Chinese",
    "JPT": "Japanese", "AFR": "African", "EUR": "European",
    "ASIA": "East Asian", "ADMIX": "admixed American",
    "African_Americans": "African-American", "generic": "generic",
    "Mbuti": "Mbuti", "Sardinian": "Sardinian", "Han": "Han Chinese",
    "ME": "Middle Eastern", "J": "Jewish",
    "WAJ": "West Ashkenazi", "EAJ": "East Ashkenazi", "Papuan": "Papuan",
}
AFRICAN = {"YRI", "AFR", "Mbuti", "African_Americans"}


def pop_label(cfg):
    """'pooled:YRI+CEU+CHB' -> 'several ancestries'; 'CEU' -> 'European only'."""
    if cfg == "reference":
        return ""
    if cfg.startswith("pooled:"):
        n = len(cfg[7:].split("+"))
        return f"several ancestries ({n})"
    return f"{POP_NAMES.get(cfg, cfg)} only"


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#666666", labelsize=8, width=0.8)


# ---- 1. real loci: A vs B ---------------------------------------------------
ab_path, ab_name = first_existing(AB_SOURCES)
rows = [r for r in read_tsv(ab_path) if r.get("status") == "OK"] if ab_path \
    else []
if not rows:
    print(f"  locus panel skipped (need one of {AB_SOURCES})")
fig, ax = plt.subplots(figsize=(5.6, 4.8), dpi=200)
ax.axvline(1, color=LINE, lw=1, zorder=1)
ax.axhline(1, color=LINE, lw=1, zorder=1)
for r in rows:
    A = fnum(r, "A", "A_ageratio")
    B = fnum(r, "B", "B_crossdepth", "B_sojourn")
    ps = fnum(r, "p_sweep", default=1.0)
    pb = fnum(r, "p_balance", default=1.0)
    if not (np.isfinite(A) and np.isfinite(B)):
        continue
    # a nominal sweep whose clock is flagged is NOT a candidate: it is the
    # known failure mode, so it gets an open mark rather than a filled one.
    flagged = bool(r.get("clock_flag", "").strip())
    if ps < 0.05:
        c, s = VERM, 46
    elif pb < 0.05:
        c, s = GOLD, 46
    else:
        c, s = GRAY, 16
    ax.scatter(max(A, 0.004), B, s=s,
               facecolors="none" if flagged and c is not GRAY else c,
               edgecolors=c if flagged and c is not GRAY else "none",
               linewidths=1.4 if flagged else 0,
               color=None if (flagged and c is not GRAY) else c,
               alpha=0.9 if c is not GRAY else 0.55, zorder=2)
if rows:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("inverted diversity / direct diversity", fontsize=9.5,
                  color="#333333")
    ax.set_ylabel("cross-orientation divergence / direct diversity",
                  fontsize=9.5, color="#333333")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "inversion_envelope_AB.png"),
                facecolor="white")
    print(f"  locus panel from {ab_name}: {len(rows)} loci")
plt.close(fig)

# ---- 2. validation rates ----------------------------------------------------
SCEN = [(1, "neutral"),
        (2, "neutral + growth"),
        (3, "neutral + structure"),
        (6, "neutral + flux"),
        (4, "true sweep"),
        (5, "true balancing")]
def rates_from_log(path):
    """Rejection rates per scenario from a rescore log.

    The rescore that re-ran these simulations under the corrected estimator
    printed its rates before it learned to save them, and the log is the
    authoritative output of that run. Parsing it keeps the figure on the
    numbers the current test actually produces instead of the superseded
    scored TSVs. Once a rescore writes its own TSVs this path stops being
    used, because the file loader below is preferred when it is newer."""
    out, cur = {}, None
    with open(path) as fh:
        for line in fh:
            m = re.match(r"=== scenario (\d+):", line.strip())
            if m:
                cur = int(m.group(1))
                out.setdefault(cur, {})
                continue
            m = re.match(r"(p_sweep|p_balance): reject@0\.05 ([0-9.]+)",
                         line.strip())
            if m and cur is not None:
                out[cur][m.group(1)] = float(m.group(2))
    return {k: (v["p_sweep"], v["p_balance"]) for k, v in out.items()
            if {"p_sweep", "p_balance"} <= set(v)}


rates = {}
log_path = os.path.join(DATA, "valrescore_rates.log")
if os.path.exists(log_path):
    rates = rates_from_log(log_path)
    print(f"  validation rates from rescore log: scenarios {sorted(rates)}")
for num, _ in SCEN:
    if num in rates:
        continue
    p_scen = os.path.join(DATA, f"invsel_scored_scen{num}.tsv")
    if not os.path.exists(p_scen):
        continue
    rs = read_tsv(p_scen)
    ps = [float(r["p_sweep"]) for r in rs]
    pb = [float(r["p_balance"]) for r in rs]
    rates[num] = (sum(p < 0.05 for p in ps) / len(ps),
                  sum(p < 0.05 for p in pb) / len(pb))
SCEN = [(num, lab) for num, lab in SCEN if num in rates]
if not SCEN:
    print("  validation panel skipped (no rates available)")

# Simple greyscale bars. Null scenarios mid grey, true-selection scenarios
# black, 5% reference line dashed. No colour, and NO annotation text on the
# plot: caveats belong in the caption and the prose, not stamped on the axes.
fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), dpi=200, sharey=True)
PANEL = ["sweep test", "balancing test"]
for ax, which in zip(axes, (0, 1)):
    for i, (num, label) in enumerate(SCEN):
        y = len(SCEN) - 1 - i
        val = rates[num][which]
        truth = (which == 0 and num == 4) or (which == 1 and num == 5)
        ax.barh(y, val, height=0.62, color=INK if truth else "#8a8a8a",
                zorder=2)
        # park the number clear of the 5% reference line, else short bars
        # collide with it
        ax.annotate(f"{val:.0%}", (max(val, 0.075), y),
                    textcoords="offset points", xytext=(5, 0), va="center",
                    fontsize=8.5, color="#333333")
    ax.axvline(0.05, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=3)
    # divider: above it the answer should be "no", below it "yes"
    ax.axhline(1.5, color="#cccccc", lw=0.9, zorder=1)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=9)
    ax.set_title(PANEL[which], fontsize=10, color=INK, loc="left", pad=6)
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(SCEN)))
axes[0].set_yticklabels([s[1] for s in reversed(SCEN)], fontsize=9,
                        color="#333333")
for ax in axes:
    ax.set_xlabel("declared selection (p < 0.05)", fontsize=9.5,
                  color="#333333")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "inversion_envelope_val.png"),
            facecolor="white")

# ---- 3. 17q21.31 under every published human demography ---------------------
dem_path = os.path.join(DATA, "demog_scan_17q.tsv")
if os.path.exists(dem_path):
    dr = [r for r in read_tsv(dem_path)]
    # Single-deme models emit a "pooled" config identical to the single-deme
    # one. Identical spec string == identical history, so collapse them and
    # keep the plainly-named row; the figure must not imply two histories
    # agreed when only one was run twice.
    seen, dedup = {}, []
    for r in dr:
        s = r.get("spec", "")
        if s in seen:
            if seen[s]["config"].startswith("pooled"):
                seen[s].update(r)          # prefer the non-pooled label
            continue
        seen[s] = r
        dedup.append(r)
    if len(dedup) != len(dr):
        print(f"  demography panel: collapsed {len(dr) - len(dedup)} duplicate "
              f"histories (single-deme models)")
    dr = dedup
    dr.sort(key=lambda r: float(r["p_balance"]))

    def short(model, cfg):
        name = MODEL_NAMES.get(model, model)
        pop = pop_label(cfg)
        return f"{name} — {pop}" if pop else name

    def kind(cfg):
        if cfg.startswith("pooled") or cfg == "reference":
            return "pooled"
        if cfg in AFRICAN:
            return "african"
        return "ooa"

    colors = {"pooled": GOLD, "african": GREEN, "ooa": GRAY}
    n = len(dr)
    fig, ax = plt.subplots(figsize=(8.2, 0.23 * n + 2.0), dpi=200)
    ax.axvline(0.05, color=VERM, lw=1.3, ls=(0, (5, 3)), zorder=1)
    ax.axvline(0.01, color="#999999", lw=1.0, ls=(0, (3, 3)), zorder=1)
    for i, r in enumerate(dr):
        y = n - 1 - i
        p = float(r["p_balance"])
        lo, hi = float(r["p_balance_lo"]), float(r["p_balance_hi"])
        cfg = r["config"]
        c = VERM if r["model"] == "CONSTANT" else colors[kind(cfg)]
        empty = int(float(r["n_tail_balance"])) == 0
        ax.plot([max(lo, 1e-5), max(hi, 1e-5)], [y, y], color=c, lw=1.6,
                alpha=0.55, solid_capstyle="round", zorder=2)
        # empty tail => p is a resolution floor, not an estimate: hollow mark
        ax.scatter([max(p, 1e-5)], [y], s=34,
                   facecolors="none" if empty else c,
                   edgecolors=c, linewidths=1.3, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([short(r["model"], r["config"]) for r in reversed(dr)],
                       fontsize=7.6, color="#333333")
    ax.set_xscale("log")
    ax.set_xlim(8e-6, 0.45)
    ax.set_ylim(-1.0, n - 0.2)
    ax.set_xlabel("p (balancing) for 17q21.31", fontsize=10, color="#333333")
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "inversion_envelope_demog.png"),
                facecolor="white")
    p_all = np.array([float(r["p_balance"]) for r in dr])
    print(f"  demography panel: {n} histories, p in "
          f"[{p_all.min():.2g}, {p_all.max():.2g}], "
          f"{(p_all < 0.05).sum()} below 0.05")
else:
    print("  demography panel skipped (no demog_scan_17q.tsv)")

# ---- 4. Monte Carlo resolution: why the first numbers were noise ------------
# Each row is the SAME statistic on the SAME data; only the null's resolution
# and conditioning differ. The originally reported values had no CI at all,
# which is the actual finding here.
# The first row is the originally reported number. It has no interval because
# none was ever computed -- that absence is the finding, so the row is drawn
# with a bare mark rather than given a fabricated one.
MC = [("as first reported", 0.0719, None, None, -1)]
mc_files = [("corrected", "inversion_envelope_single_origin.tsv"),
            ("corrected, 10x longer", "inversion_envelope_1kg_deep17q.tsv")]
for label, fn in mc_files:
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        continue
    for r in read_tsv(p):
        if r.get("inv_id") == "17:45585159-46292045" and r.get("status") == "OK":
            MC.append((label, float(r["p_balance"]),
                       float(r["p_balance_lo"]), float(r["p_balance_hi"]),
                       int(float(r["n_tail_balance"]))))
if MC:
    fig, ax = plt.subplots(figsize=(7.6, 0.95 * len(MC) + 1.9), dpi=200)
    ax.axvline(0.05, color=VERM, lw=1.3, ls=(0, (5, 3)), zorder=1)
    for i, (label, p, lo, hi, tail) in enumerate(MC):
        y = len(MC) - 1 - i
        c = BLUE if "10x" in label else (VERM if tail == -1 else INK)
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=c, lw=3.0, alpha=0.45,
                    solid_capstyle="round", zorder=2)
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.11, y + 0.11], color=c, lw=1.6,
                        alpha=0.75, zorder=2)
        ax.scatter([p], [y], s=64, color=c, lw=0, zorder=3)
    ax.set_ylim(-0.7, len(MC) - 0.25)
    ax.set_yticks(range(len(MC)))
    ax.set_yticklabels([m[0] for m in reversed(MC)], fontsize=9,
                       color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("p (balancing) for 17q21.31, with 95% interval",
                  fontsize=10, color="#333333")
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "inversion_envelope_mc.png"),
                facecolor="white")
    print(f"  MC panel: {len(MC)} resolutions "
          f"(tail counts {[m[4] for m in MC]})")

# ---- 5. the size histories the neutral trees are drawn under ----------------
# What the null actually uses. Each published model, reduced to the relative
# population size nu(t) that reproduces its pairwise coalescence rate for a
# given sampling configuration; the neutral trees are simulated in standard
# coalescent time and their node heights mapped through this. nu > 1 means
# coalescence is slower than at the present (a larger past population), nu < 1
# faster (a bottleneck).
dem_spec = os.path.join(DATA, "human_demographies.tsv")
if os.path.exists(dem_spec):
    def steps_from_spec(spec):
        """'pw:t1,nu1;t2,nu2;...' -> arrays for a step plot."""
        ts, nus = [], []
        for part in spec[3:].split(";"):
            if not part.strip():
                continue
            t_right, nu_left = part.split(",")
            ts.append(float(t_right))
            nus.append(float(nu_left))
        xs, ys, t_prev = [], [], 1e-6
        for t_right, nu in zip(ts, nus):
            xs += [max(t_prev, 1e-6), max(t_right, 1e-6)]
            ys += [nu, nu]
            t_prev = t_right
        return np.array(xs), np.array(ys)

    dspec = [r for r in read_tsv(dem_spec) if r["spec"].startswith("pw:")]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    ax.axhline(1.0, color=LINE, lw=1, zorder=1)
    pooled_n = 0
    for r in dspec:
        xs, ys = steps_from_spec(r["spec"])
        is_pooled = r["config"].startswith("pooled")
        pooled_n += is_pooled
        ax.plot(xs, ys, lw=1.6 if is_pooled else 0.7,
                color=INK if is_pooled else "#bbbbbb",
                alpha=0.9 if is_pooled else 0.7,
                zorder=3 if is_pooled else 2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 12)
    ax.set_xlabel("time into the past (coalescent units)", fontsize=9.5,
                  color="#333333")
    ax.set_ylabel("relative population size", fontsize=9.5, color="#333333")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(DATA, "inversion_envelope_histories.png"),
                facecolor="white")
    print(f"  history panel: {len(dspec)} size histories "
          f"({pooled_n} multi-ancestry)")

print("wrote envelope figures to data/")
