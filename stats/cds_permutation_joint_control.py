"""Direct multiplicity control, calibration, and power for the per-gene CDS test.

stats/per_gene_cds_permutation.py tests each gene marginally and then corrects
with BH/BY, which assume a dependence structure. Here the dependence is not
assumed but reproduced: orientation labels are shuffled ONCE PER LOCUS and every
gene in that locus is recomputed under the same shuffle (genes in one inversion
share their haplotypes -- 71 of the 164 tests sit in 8p23.1 alone -- while
different loci are independent). Each row of the resulting R x G null matrix is
a complete null dataset with the true joint dependence. From that one object:

  1. DIRECT CONTROL of the observed data
     - Westfall-Young min-p FWER-adjusted p-values
     - plug-in permutation FDR q-values (expected false discoveries at each
       threshold, read off the joint null; monotonised)
  2. CALIBRATION of the whole pipeline, treating each null row as the observed
     data: realised per-test type-I error at nominal alpha (aggregate and by
     inverted-group size), and realised family-wise error of BH / BY / WY at
     q < 0.05. Under the global null FDR equals FWER, so these numbers read
     directly as "is the pipeline conservative or anti-conservative".
  3. POWER under a founder-effect alternative: the inverted group is replaced by
     f = round(w * k_inv) copies of one randomly drawn haplotype plus
     k_inv - f draws from the locus pool, w in {0.25, 0.5, 0.75, 1.0} (w = 1 is
     the single-origin scenario the manuscript posits for single-event
     inversions). Simulated statistics are referred to the gene's own
     permutation null; power is reported at nominal 0.05, Bonferroni, and the
     Westfall-Young family threshold.

Inputs: as stats/per_gene_cds_permutation.py (set CDS_PHY_DIR).
Outputs:
  data/cds_permutation_joint_control.tsv   per-gene: joint p, WY p, direct-FDR q
  data/cds_permutation_calibration.tsv     calibration summary rows
  data/cds_permutation_power.tsv           per gene x w: achieved delta, power
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd

_STATS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_STATS_DIR)
_DATA_DIR = os.path.join(_REPO_DIR, "data")
sys.path.insert(0, _STATS_DIR)

from per_gene_cds_permutation import (  # noqa: E402
    FNAME_RE, read_phy, resolve_phy_dir, recurrence_labels, TESTS_TSV)

OUT_JOINT = os.path.join(_DATA_DIR, "cds_permutation_joint_control.tsv")
OUT_CALIB = os.path.join(_DATA_DIR, "cds_permutation_calibration.tsv")
OUT_POWER = os.path.join(_DATA_DIR, "cds_permutation_power.tsv")
OUT_MDE = os.path.join(_DATA_DIR, "cds_permutation_mde.tsv")
OUT_POWER_DELTA = os.path.join(_DATA_DIR, "cds_permutation_power_bydelta.tsv")
OUT_POWER_OBS = os.path.join(_DATA_DIR, "cds_permutation_power_observed.tsv")
# Plot-resolution grid for the power curve; the reported quantities are the
# curve itself and MDE80, not these values.
DELTA_GRID = tuple(round(0.05 * i, 2) for i in range(1, 15))  # 0.05 .. 0.70
OUT_POWER_CURVE = os.path.join(_DATA_DIR, "cds_permutation_power_curve.pdf")
OUT_FIG = os.path.join(_DATA_DIR, "cds_permutation_joint_control.pdf")
JACKKNIFE_TSV = os.path.join(_DATA_DIR, "gene_inversion_direct_inverted.tsv")

MIN_GROUP = 4   # exclude genes whose smaller orientation group has < 4
                 # haplotypes: their permutation null cannot reach p < 0.05
                 # even under a maximal effect (see the power section), so they
                 # only dilute the family. Group size is fixed under the null,
                 # making this valid independent filtering (Bourgon 2010).
R_DRAWS = 100_000
CHUNK = 25_000
POWER_B = 2_000
FOUNDER_W = (0.25, 0.50, 0.75, 1.00)
ALPHA = 0.05
RNG_SEED = 2026


# ---------------------------------------------------------------- data loading

def load_loci():
    """Group the tested genes by locus, with a shared haplotype order per locus."""
    tests = pd.read_csv(TESTS_TSV, sep="\t")
    phy_dir = resolve_phy_dir()
    files = {}
    for fn in os.listdir(phy_dir):
        m = FNAME_RE.match(fn)
        if m:
            files[(m["grp"], m["gene"], m["enst"], m["chrom"],
                   int(m["is"]), int(m["ie"]))] = os.path.join(phy_dir, fn)

    loci = {}
    for _, r in tests.iterrows():
        ch, c = r["inv_id"].split(":")
        s, e = (int(x) for x in c.split("-"))
        base = (r["gene_name"], r["transcript_id"], f"chr{ch}", s, e)
        seq_dir = read_phy(files[("0", *base)])
        seq_inv = read_phy(files[("1", *base)])
        L = loci.setdefault(r["inv_id"], {"orient": {}, "genes": []})
        for h in seq_dir:
            assert L["orient"].setdefault(h, 0) == 0, (r["inv_id"], h)
        for h in seq_inv:
            assert L["orient"].setdefault(h, 1) == 1, (r["inv_id"], h)
        L["genes"].append({"name": r["gene_name"], "inv_id": r["inv_id"],
                           "seqs": {**seq_dir, **seq_inv}})

    for inv_id, L in loci.items():
        L["haps"] = sorted(L["orient"])
        pos = {h: i for i, h in enumerate(L["haps"])}
        L["inv_idx"] = np.array([pos[h] for h, o in L["orient"].items() if o == 1])
        L["k_inv"] = len(L["inv_idx"])
        L["n"] = len(L["haps"])
        kept = []
        for g in L["genes"]:
            class_of = {}
            cls = np.full(L["n"], -1, dtype=np.int64)
            for h, seq in g["seqs"].items():
                cls[pos[h]] = class_of.setdefault(seq, len(class_of))
            g["cls"] = cls                       # -1 where gene lacks the haplotype
            g["n_classes"] = len(class_of)
            valid = cls >= 0
            k_inv_g = int(np.isin(np.where(valid)[0], L["inv_idx"]).sum())
            k_dir_g = int(valid.sum()) - k_inv_g
            if g["n_classes"] < 2:
                # Monomorphic CDS: every haplotype carries the identical
                # sequence, so delta == 0 and p == 1 under every relabeling.
                # The criterion is a function of the pooled data only
                # (label-invariant), so excluding these genes is valid
                # independent filtering (Bourgon 2010) exactly like the
                # MIN_GROUP filter above; keeping them only pads the BH
                # denominator with degenerate p = 1 tests. Genes with two
                # classes but zero observed delta (fixed differences) are
                # NOT excluded: their statistic varies under relabeling.
                continue
            if min(k_inv_g, k_dir_g) >= MIN_GROUP:
                kept.append(g)
        L["genes"] = kept
    dropped = {inv: len(L["genes"]) for inv, L in loci.items() if not L["genes"]}
    loci = {inv: L for inv, L in loci.items() if L["genes"]}
    n_kept = sum(len(L["genes"]) for L in loci.values())
    print(f"independent filter (min group >= {MIN_GROUP}): kept {n_kept} genes; "
          f"loci dropped entirely: {len(dropped)}")
    return loci


# ------------------------------------------------------- statistic per draw set

def deltas_for_membership(gene, inv_member):
    """|inv_member| is a bool matrix (B, n_locus): True = labelled inverted.

    Returns Delta = p_ident(inverted) - p_ident(direct) per row (NaN when a
    group has < 2 haplotypes for this gene)."""
    cls, K = gene["cls"], gene["n_classes"]
    valid = cls >= 0
    onehot = np.zeros((len(cls), K), dtype=np.float32)
    onehot[valid, cls[valid]] = 1.0
    m = inv_member.astype(np.float32)
    counts_inv = np.rint(m @ onehot).astype(np.int64)
    total = onehot.sum(axis=0).astype(np.int64)
    counts_dir = total[None, :] - counts_inv
    k_inv = counts_inv.sum(axis=1)
    k_dir = counts_dir.sum(axis=1)
    ident_inv = (counts_inv * (counts_inv - 1) // 2).sum(axis=1)
    ident_dir = (counts_dir * (counts_dir - 1) // 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_inv = ident_inv / (k_inv * (k_inv - 1) / 2)
        p_dir = ident_dir / (k_dir * (k_dir - 1) / 2)
    out = p_inv - p_dir
    out[(k_inv < 2) | (k_dir < 2)] = np.nan
    return out


def observed_delta(gene, L):
    member = np.zeros((1, L["n"]), dtype=bool)
    member[0, L["inv_idx"]] = True
    return float(deltas_for_membership(gene, member)[0])


# ----------------------------------------------------------------- joint null

def build_joint_null(loci, rng):
    """R_DRAWS locus-level relabellings; returns |Delta| matrix (R, G) and genes."""
    genes = [g for L in loci.values() for g in L["genes"]]
    order = {id(g): j for j, g in enumerate(genes)}
    null_abs = np.empty((R_DRAWS, len(genes)), dtype=np.float64)
    for inv_id, L in loci.items():
        n, k = L["n"], L["k_inv"]
        for start in range(0, R_DRAWS, CHUNK):
            b = min(CHUNK, R_DRAWS - start)
            noise = rng.random((b, n))
            idx = np.argpartition(noise, k - 1, axis=1)[:, :k]
            member = np.zeros((b, n), dtype=bool)
            np.put_along_axis(member, idx, True, axis=1)
            for g in L["genes"]:
                null_abs[start:start + b, order[id(g)]] = np.abs(
                    deltas_for_membership(g, member))
        print(f"  null built: {inv_id} ({len(L['genes'])} genes)", flush=True)
    return null_abs, genes


def tail_p(sorted_null, values):
    """(1 + #null >= v) / (R + 1), vectorised over values; NaN -> 1."""
    v = np.atleast_1d(np.asarray(values, dtype=np.float64))
    n = len(sorted_null)
    cnt = n - np.searchsorted(sorted_null, v - 1e-9, side="left")
    out = (cnt + 1) / (n + 1)
    return np.where(np.isnan(v), 1.0, out)


# ------------------------------------------------------------------- figure

def _spread(ys, min_gap):
    """Nudge label positions apart, preserving order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def make_figure(path=OUT_FIG):
    """A: fate of the 20 previously reported genes (dumbbell, genes on y).
       B: volcano of the filtered 130-gene family under direct FDR control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from _figstyle import apply, RECURRENCE_COLORS, NEUTRAL
    apply(base_size=15)
    ALPHA = 0.05
    GREY, FAINT = "#8a8a8a", "#c8c8c8"
    CAP = 12.0                     # x position representing jackknife q = 0

    joint = pd.read_csv(OUT_JOINT, sep="\t")
    jack = pd.read_csv(JACKKNIFE_TSV, sep="\t")
    jack["key"] = jack.gene_name + "|" + jack.inv_id
    joint["key"] = joint.gene_name + "|" + joint.inv_id
    prev = jack[jack.q_value < ALPHA].copy()          # the 20 published genes
    newq = joint.set_index("key")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13.6, 6.6), width_ratios=[0.85, 1.3])
    fig.patch.set_facecolor("white")
    for ax in (axA, axB):
        ax.set_facecolor("white")

    # ---------- Panel A: before/after slope for the 20 previous genes ----------
    CAPY = 12.0
    nlq = lambda q: CAPY if q <= 0 else min(-np.log10(q), CAPY)
    rowsA = []
    for _, r in prev.iterrows():
        if r["key"] in newq.index:
            j = newq.loc[r["key"]]
            rowsA.append((r.gene_name, nlq(r.q_value), nlq(j.direct_fdr_q),
                          j.recurrence, j.direct_fdr_q < ALPHA, False))
        else:
            rowsA.append((r.gene_name, nlq(r.q_value), 0.0, None, False, True))

    for name, y0, y1, rec, alive, removed in rowsA:
        c = RECURRENCE_COLORS.get(rec, NEUTRAL) if rec else FAINT
        axA.plot([0, 1], [y0, y1], lw=2.4 if alive else 1.0, color=c,
                 alpha=0.95 if alive else 0.45, zorder=3 if alive else 2,
                 solid_capstyle="round")
        axA.scatter([0], [y0], s=34, color=c, alpha=0.95 if alive else 0.45,
                    zorder=4)
        if removed:
            axA.scatter([1], [0], marker="x", s=52, color=FAINT,
                        linewidth=1.8, zorder=4)
        else:
            axA.scatter([1], [y1], s=48, color=c,
                        edgecolor="black" if alive else "none", linewidth=0.8,
                        alpha=1.0 if alive else 0.45, zorder=4)

    conf = [(n, y1) for n, _, y1, _, alive, _ in rowsA if alive]
    lab_y = _spread(np.array([y for _, y in conf]), 0.62)
    for (name, y1), ly in zip(conf, lab_y):
        axA.plot([1.03, 1.09], [y1, ly], lw=0.6, color=FAINT, zorder=1,
                 clip_on=False)
        axA.annotate(name, (1.11, ly), va="center", fontsize=12.5,
                     fontweight="bold", annotation_clip=False)
    axA.axhline(-np.log10(ALPHA), ls="--", lw=1.0, color=NEUTRAL, zorder=0)
    axA.annotate("0.05", (-0.13, -np.log10(ALPHA)), fontsize=11.5,
                 color=NEUTRAL, ha="right", va="center", annotation_clip=False)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["before\n(jackknife)", "after\n(permutation)"])
    axA.set_yticks([0, 2, 4, 6, 8, 10, CAPY])
    axA.set_yticklabels(["0", "2", "4", "6", "8", "10", "q$\\,$=$\\,$0"])
    axA.set_xlim(-0.18, 1.44)
    axA.set_ylim(-0.55, CAPY + 0.4)
    axA.set_ylabel("$-\\log_{10}\\,q$")
    axA.set_title("A", loc="left", fontweight="bold")

    # ---------- Panel B ----------
    d = joint.copy()
    floor = d.loc[d.joint_p > 0, "joint_p"].min() / 2
    d["nlp"] = -np.log10(d.joint_p.clip(lower=floor))
    sig = d.direct_fdr_q < ALPHA
    col = d.recurrence.map(RECURRENCE_COLORS).fillna(NEUTRAL)
    axB.scatter(d.delta[~sig], d.nlp[~sig], s=44, c=col[~sig], alpha=0.35,
                edgecolor="none", zorder=2)
    axB.scatter(d.delta[sig], d.nlp[sig], s=74, c=col[sig], edgecolor="black",
                linewidth=0.8, zorder=4)
    axB.axvline(0, lw=0.8, color=NEUTRAL, zorder=1)
    t_line = -np.log10(d.loc[sig, "joint_p"].max())
    axB.axvline  # noqa: B018 (no-op; keeps line spacing readable)
    axB.axhline(t_line, ls="--", lw=1.0, color=NEUTRAL, zorder=1)
    ymax = d.nlp.max()
    axB.set_xlim(d.delta.min() - 0.28, d.delta.max() + 0.22)
    axB.set_ylim(-0.3, ymax * 1.13)
    axB.annotate("FDR $q=0.05$",
                 (axB.get_xlim()[0] + 0.02, t_line + 0.09),
                 fontsize=12, color=NEUTRAL, ha="left", va="bottom")
    named = d[sig]
    dx = (axB.get_xlim()[1] - axB.get_xlim()[0]) * 0.026
    for side in (-1, 1):
        part = named[np.sign(named.delta).replace(0, 1) == side]
        if part.empty:
            continue
        for (_, r), ly in zip(part.iterrows(),
                              _spread(part.nlp.to_numpy(), ymax * 0.068)):
            axB.annotate(r.gene_name, xy=(r.delta, r.nlp),
                         xytext=(r.delta + side * dx, ly), textcoords="data",
                         fontsize=12, va="center",
                         ha="left" if side > 0 else "right",
                         arrowprops=dict(arrowstyle="-", lw=0.6,
                                         color=FAINT, shrinkA=0, shrinkB=3))
    axB.set_xlabel("$\\Delta$ pair identity (inverted $-$ direct)")
    axB.set_ylabel("$-\\log_{10}\\,p$")
    axB.set_title("B", loc="left", fontweight="bold")
    axB.legend(handles=[
        Line2D([], [], marker="o", ls="",
               color=RECURRENCE_COLORS["single-event"], markersize=9,
               label="single-event"),
        Line2D([], [], marker="o", ls="",
               color=RECURRENCE_COLORS["recurrent"], markersize=9,
               label="recurrent"),
    ], loc="upper right", frameon=False, fontsize=12.5,
        handletextpad=0.25, borderaxespad=0.2)

    fig.tight_layout(w_pad=2.6)
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, facecolor="white",
                bbox_inches="tight")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------- main

def main():
    if "--plot-only" in sys.argv:
        make_figure()
        return
    rng = np.random.default_rng(RNG_SEED)
    labels = recurrence_labels()
    loci = load_loci()
    n_genes = sum(len(L["genes"]) for L in loci.values())
    print(f"{len(loci)} loci, {n_genes} genes; building joint null "
          f"(R = {R_DRAWS:,}) ...")
    null_abs, genes = build_joint_null(loci, rng)
    G = len(genes)
    nan_frac = float(np.isnan(null_abs).mean())
    null_abs = np.nan_to_num(null_abs, nan=-np.inf)   # undefined draw: never a hit
    sorted_null = np.sort(null_abs, axis=0)

    # per-draw marginal p of every null draw within its own gene column
    # (rank among all R draws; used for calibration and the WY min-p reference)
    ranks = np.empty_like(null_abs, dtype=np.float64)
    for j in range(G):
        col = null_abs[:, j]
        ord_ = np.argsort(col, kind="stable")
        # p = #{>= value}/R with ties counted together
        sorted_col = col[ord_]
        cnt_ge = len(col) - np.searchsorted(sorted_col, col - 1e-9, side="left")
        ranks[:, j] = cnt_ge / len(col)
    minp_rows = ranks.min(axis=1)

    # ---- observed statistics, joint marginal p, WY, direct FDR ----
    obs = []
    for L in loci.values():
        for g in L["genes"]:
            obs.append(observed_delta(g, L))
    obs = np.asarray(obs)
    obs_p = np.array([tail_p(sorted_null[:, j], abs(obs[j]))[0] for j in range(G)])
    wy_p = np.array([(1 + int((minp_rows <= p).sum())) / (R_DRAWS + 1)
                     for p in obs_p])

    order = np.argsort(obs_p)
    fdr = np.empty(G)
    for i in order:
        t = obs_p[i]
        v_hat = float((ranks <= t).sum()) / R_DRAWS      # E[#false disc at t]
        s_obs = int((obs_p <= t).sum())
        fdr[i] = min(1.0, v_hat / max(s_obs, 1))
    running = 1.0                                         # monotonise (step-up)
    for i in np.argsort(-obs_p):
        running = min(running, fdr[i])
        fdr[i] = running

    joint = pd.DataFrame({
        "gene_name": [g["name"] for g in genes],
        "inv_id": [g["inv_id"] for g in genes],
        "recurrence": [labels.get(g["inv_id"], "unknown") for g in genes],
        "k_inverted": [loci[g["inv_id"]]["k_inv"] for g in genes],
        "delta": obs,
        "joint_p": obs_p,
        "wy_fwer_p": wy_p,
        "direct_fdr_q": fdr,
    }).sort_values(["direct_fdr_q", "joint_p"])

    # ---- minimum detectable effect per gene (no generative model needed) ----
    # The permutation null alone fixes the instrument's resolution: the
    # smallest |delta| that would reach a given p-value target at each gene.
    # Reporting this next to the observed |delta| answers "what could this
    # test ever have detected?" without inventing an alternative.
    def mde_at(p_target: float) -> np.ndarray:
        # tail_p counts ties as >=, so the detectable value is the null
        # quantile at rank ceil((1 - p_target) * R): the smallest value v
        # with #{null >= v}/R <= p_target.
        k = int(np.ceil((1.0 - p_target) * R_DRAWS))
        k = min(max(k, 0), R_DRAWS - 1)
        return sorted_null[k, :] + 1e-12

    fdr_thresholds = joint.loc[joint["direct_fdr_q"] < ALPHA, "joint_p"]
    p_fdr_cut = float(fdr_thresholds.max()) if len(fdr_thresholds) else ALPHA / G
    mde = pd.DataFrame({
        "gene_name": [g["name"] for g in genes],
        "inv_id": [g["inv_id"] for g in genes],
        "k_inverted": [loci[g["inv_id"]]["k_inv"] for g in genes],
        "observed_abs_delta": np.abs(obs),
        "mde_abs_delta_nominal05": mde_at(ALPHA),
        "mde_abs_delta_fdr": mde_at(p_fdr_cut),
        "detectable_effect_exists_nominal05": mde_at(ALPHA) < 1.0,
    }).sort_values("mde_abs_delta_fdr")
    mde.to_csv(OUT_MDE, sep="\t", index=False)
    print(f"Wrote {OUT_MDE}")
    med = float(np.median(mde["mde_abs_delta_fdr"]))
    print(f"median minimum detectable |delta| at the FDR threshold: {med:.3f}")
    joint.to_csv(OUT_JOINT, sep="\t", index=False)

    print(f"\nundefined-draw fraction in null: {nan_frac:.2e}")
    print("\n=== DIRECT CONTROL (observed data) ===")
    print(f"  direct FDR q < {ALPHA}: {int((joint.direct_fdr_q < ALPHA).sum())} genes")
    print(f"  WY FWER p < {ALPHA}:    {int((joint.wy_fwer_p < ALPHA).sum())} genes")
    print(joint[joint.direct_fdr_q < ALPHA]
          [["gene_name", "inv_id", "delta", "joint_p", "wy_fwer_p", "direct_fdr_q"]]
          .to_string(index=False))

    # ---- calibration: every null row is a null dataset with true dependence ----
    print("\n=== CALIBRATION (pipeline on pure-null data) ===")
    k_inv_arr = joint.set_index([joint.index])  # noqa: F841 (clarity only)
    kvec = np.array([loci[g["inv_id"]]["k_inv"] for g in genes])
    calib_rows = []
    for nominal in (0.05, 0.01):
        realized = float((ranks <= nominal).mean())
        calib_rows.append({"metric": f"per_test_typeI_at_{nominal}",
                           "nominal": nominal, "realized": realized})
        print(f"  per-test type-I at alpha={nominal}: realized {realized:.4f} "
              f"({'conservative' if realized <= nominal else 'ANTI-conservative'})")
    for kcut, lab in ((3, "k_inv=3"), (10, "k_inv<=10")):
        sel = kvec <= kcut
        if not sel.any():
            continue
        realized = float((ranks[:, sel] <= 0.05).mean())
        calib_rows.append({"metric": f"per_test_typeI_at_0.05_{lab}",
                           "nominal": 0.05, "realized": realized})
        print(f"  per-test type-I at 0.05, {lab} genes (n={int(sel.sum())}): "
              f"{realized:.4f}")

    hs = ranks.astype(np.float64)
    hs.sort(axis=1)                                  # row-sorted p-values
    thresh_bh = ALPHA * np.arange(1, G + 1) / G
    c_g = np.log(G) + np.euler_gamma + 1 / (2 * G)
    thresh_by = thresh_bh / c_g
    fwer_bh = float((hs <= thresh_bh).any(axis=1).mean())
    fwer_by = float((hs <= thresh_by).any(axis=1).mean())
    sorted_minp = np.sort(minp_rows)
    t_wy = sorted_minp[max(0, int(ALPHA * R_DRAWS) - 1)]
    while t_wy > sorted_minp[0] and float((minp_rows <= t_wy).mean()) > ALPHA:
        t_wy = np.nextafter(t_wy, 0)
        t_wy = sorted_minp[np.searchsorted(sorted_minp, t_wy, side="right") - 1]
    fwer_wy = float((minp_rows <= t_wy).mean())
    # Calibration of the plug-in FDR procedure itself: treat every null row as
    # the observed data, run the same estimator (V_hat from the pooled table,
    # step-up with right-cummin monotonisation), and count discoveries. Under
    # the global null every discovery is false, so P(any discovery) is both the
    # FWER and the realised FDR of the procedure.
    G_ = ranks.shape[1]
    pooled = np.sort(ranks.astype(np.float32).ravel())
    row_sorted = np.sort(ranks.astype(np.float32), axis=1)
    v_counts = np.searchsorted(pooled, row_sorted.ravel(), side="right")
    v_hat_rows = (v_counts.reshape(row_sorted.shape) / R_DRAWS)
    fdr_rows = v_hat_rows / np.arange(1, G_ + 1)[None, :]
    fdr_rows = np.minimum.accumulate(fdr_rows[:, ::-1], axis=1)[:, ::-1]
    disc = fdr_rows <= ALPHA
    # n_disc: largest rank whose monotonised q <= alpha (step-up rule)
    n_disc = np.where(disc.any(axis=1),
                      disc.shape[1] - np.argmax(disc[:, ::-1], axis=1), 0)
    fwer_fdr = float((n_disc > 0).mean())
    mean_false = float(n_disc.mean())
    mc_se = float(np.sqrt(fwer_fdr * (1 - fwer_fdr) / R_DRAWS))
    verdict = ("conservative" if fwer_fdr <= ALPHA
               else "calibrated (within Monte Carlo error)"
               if fwer_fdr <= ALPHA + 2 * mc_se else "ANTI-conservative")
    calib_rows.append({"metric": "family_anyFalseDiscovery_directFDR_at_0.05",
                       "nominal": ALPHA, "realized": fwer_fdr})
    calib_rows.append({"metric": "mean_false_discoveries_directFDR_at_0.05",
                       "nominal": np.nan, "realized": mean_false})
    print(f"  direct-FDR procedure on null data: P(any discovery) = "
          f"{fwer_fdr:.4f} (+/- {mc_se:.4f} MC), mean false discoveries = "
          f"{mean_false:.4f} ({verdict})")
    for lab, val in (("BH", fwer_bh), ("BY", fwer_by), ("WY", fwer_wy)):
        calib_rows.append({"metric": f"family_FWER_{lab}_at_0.05",
                           "nominal": ALPHA, "realized": val})
        print(f"  family-wise error of {lab} at q<0.05 on null data: {val:.4f} "
              f"({'conservative' if val <= ALPHA else 'ANTI-conservative'})")
    pd.DataFrame(calib_rows).to_csv(OUT_CALIB, sep="\t", index=False)

    # ---- power under founder-effect alternatives ----
    sig_p = obs_p[fdr < ALPHA]
    t_fdr = float(sig_p.max()) if len(sig_p) else ALPHA / G
    print(f"\n=== POWER (founder model, B = {POWER_B} per gene per w; "
          f"thresholds: nominal {ALPHA}, FDR cutoff {t_fdr:.2e}, "
          f"WY t* = {t_wy:.2e}, Bonferroni {ALPHA/G:.2e}) ===")
    power_rows = []
    for j, g in enumerate(genes):
        L = loci[g["inv_id"]]
        cls, valid = g["cls"], g["cls"] >= 0
        pool = cls[valid]
        hap_ids = np.where(valid)[0]
        k_inv = int((np.isin(hap_ids, L["inv_idx"])).sum())
        dir_ids = hap_ids[~np.isin(hap_ids, L["inv_idx"])]
        k_dir = len(dir_ids)
        if k_inv < 2 or k_dir < 2:
            continue
        cnt_dir = np.bincount(cls[dir_ids], minlength=g["n_classes"])
        p_dir = float((cnt_dir * (cnt_dir - 1) // 2).sum()
                      / (k_dir * (k_dir - 1) / 2))
        for w in FOUNDER_W:
            f = max(1, int(round(w * k_inv)))
            founders = pool[rng.integers(0, len(pool), POWER_B)]
            sim = np.empty((POWER_B, k_inv), dtype=np.int64)
            sim[:, :f] = founders[:, None]
            if k_inv > f:
                sim[:, f:] = pool[rng.integers(0, len(pool),
                                               (POWER_B, k_inv - f))]
            row = np.repeat(np.arange(POWER_B), k_inv)
            counts = np.zeros((POWER_B, g["n_classes"]), dtype=np.int64)
            np.add.at(counts, (row, sim.ravel()), 1)
            p_inv = ((counts * (counts - 1) // 2).sum(axis=1)
                     / (k_inv * (k_inv - 1) / 2))
            d_sim = np.abs(p_inv - p_dir)
            p_sim = tail_p(sorted_null[:, j], d_sim)
            power_rows.append({
                "gene_name": g["name"], "inv_id": g["inv_id"],
                "k_inverted": k_inv, "founder_w": w,
                "mean_abs_delta": float(np.mean(np.abs(p_inv - p_dir))),
                "power_nominal": float((p_sim <= ALPHA).mean()),
                "power_fdr": float((p_sim <= t_fdr).mean()),
                "power_bonferroni": float((p_sim <= ALPHA / G).mean()),
                "power_wy": float((p_sim <= t_wy).mean()),
            })
    power = pd.DataFrame(power_rows)
    power.to_csv(OUT_POWER, sep="\t", index=False)

    power["k_group"] = pd.cut(power.k_inverted, [0, 3, 10, 30, 100],
                              labels=["3", "4-10", "11-30", ">30"])
    summ = (power.groupby(["k_group", "founder_w"], observed=True)
            [["mean_abs_delta", "power_nominal", "power_fdr", "power_wy"]]
            .mean().round(3))
    print(summ.to_string())

    # ---- power indexed by the true contrast size |delta| -------------------
    # The founder-w alternative above is one specific biological scenario and
    # only ever makes the inverted group MORE identical. Here the alternative
    # is defined directly on the statistic: the synthetic inverted group's
    # expected pair identity is p_dir +/- delta, in whichever directions the
    # gene can express (identity lives in [0, 1]). Upward contrasts mix the
    # empirical class pool with a point mass on a randomly drawn founder
    # class; downward contrasts give haplotypes private singleton classes
    # ("each lineage carries its own mutations"). Real direct group, real
    # pool, real null and thresholds - only the effect is synthetic, and its
    # size is the x-axis.
    print(f"\n=== POWER BY TRUE CONTRAST |delta| (B = {POWER_B} per gene per "
          f"delta per direction; same thresholds) ===")
    delta_rows = []
    for j, g in enumerate(genes):
        L = loci[g["inv_id"]]
        cls, valid = g["cls"], g["cls"] >= 0
        pool = cls[valid]
        hap_ids = np.where(valid)[0]
        k_inv = int((np.isin(hap_ids, L["inv_idx"])).sum())
        dir_ids = hap_ids[~np.isin(hap_ids, L["inv_idx"])]
        k_dir = len(dir_ids)
        if k_inv < 2 or k_dir < 2:
            continue
        cnt_dir = np.bincount(cls[dir_ids], minlength=g["n_classes"])
        p_dir = float((cnt_dir * (cnt_dir - 1) // 2).sum()
                      / (k_dir * (k_dir - 1) / 2))
        n_pool = len(pool)
        cnt_pool = np.bincount(pool, minlength=g["n_classes"])
        freq = cnt_pool / n_pool
        F_pool = float((freq ** 2).sum())
        npairs_inv = k_inv * (k_inv - 1) / 2
        for delta in DELTA_GRID:
            for direction, h in (("up", p_dir + delta), ("down", p_dir - delta)):
                if not (0.0 <= h <= 1.0):
                    continue
                if h >= F_pool:
                    # point-mass mixture: identity(q) = a q^2 + b q + F_pool
                    founders = pool[rng.integers(0, n_pool, POWER_B)]
                    f_c = freq[founders]
                    a = 1.0 - 2.0 * f_c + F_pool          # > 0 (pool not clonal)
                    b = 2.0 * (f_c - F_pool)
                    disc = np.maximum(b * b - 4.0 * a * (F_pool - h), 0.0)
                    q = np.clip((-b + np.sqrt(disc)) / (2.0 * a), 0.0, 1.0)
                    take = rng.random((POWER_B, k_inv)) < q[:, None]
                    sim = pool[rng.integers(0, n_pool, (POWER_B, k_inv))]
                    sim[take] = np.broadcast_to(founders[:, None],
                                                sim.shape)[take]
                    row = np.repeat(np.arange(POWER_B), k_inv)
                    counts = np.zeros((POWER_B, g["n_classes"]), dtype=np.int64)
                    np.add.at(counts, (row, sim.ravel()), 1)
                else:
                    # private singletons: identity(r) = (1 - r)^2 * F_pool
                    r = 1.0 - np.sqrt(h / F_pool)
                    keep = rng.random((POWER_B, k_inv)) >= r
                    sim = pool[rng.integers(0, n_pool, (POWER_B, k_inv))]
                    counts = np.zeros((POWER_B, g["n_classes"]), dtype=np.int64)
                    rows_kept = np.repeat(np.arange(POWER_B), k_inv)[keep.ravel()]
                    np.add.at(counts, (rows_kept, sim[keep]), 1)
                p_inv = ((counts * (counts - 1) // 2).sum(axis=1) / npairs_inv)
                d_sim = np.abs(p_inv - p_dir)
                p_sim = tail_p(sorted_null[:, j], d_sim)
                delta_rows.append({
                    "gene_name": g["name"], "inv_id": g["inv_id"],
                    "k_inverted": k_inv, "delta": delta,
                    "direction": direction, "target_identity": h,
                    "p_direct": p_dir,
                    "mean_abs_delta": float(d_sim.mean()),
                    "power_nominal": float((p_sim <= ALPHA).mean()),
                    "power_fdr": float((p_sim <= t_fdr).mean()),
                })
    pdelta = pd.DataFrame(delta_rows)
    pdelta.to_csv(OUT_POWER_DELTA, sep="\t", index=False)

    # ---- power at each gene's OWN observed contrast (DIAGNOSTIC ONLY) ------
    # WARNING: this is post-hoc ("observed") power - approximately a monotone
    # transform of the observed p-value (Hoenig & Heisey 2001, Am Stat), and
    # inflated by winner's curse for the significant genes. NEVER report these
    # numbers as the power analysis. The reportable quantities are the
    # power curve over the fixed grid of hypothesized contrasts (OUT_POWER_DELTA) and the
    # minimum detectable effect (OUT_MDE). This table exists only as an
    # internal cross-check that detected genes sit in the detectable regime.
    obs_rows = []
    for j, g in enumerate(genes):
        L = loci[g["inv_id"]]
        cls, valid = g["cls"], g["cls"] >= 0
        pool = cls[valid]
        hap_ids = np.where(valid)[0]
        k_inv = int((np.isin(hap_ids, L["inv_idx"])).sum())
        dir_ids = hap_ids[~np.isin(hap_ids, L["inv_idx"])]
        k_dir = len(dir_ids)
        if k_inv < 2 or k_dir < 2:
            continue
        cnt_dir = np.bincount(cls[dir_ids], minlength=g["n_classes"])
        p_dir = float((cnt_dir * (cnt_dir - 1) // 2).sum()
                      / (k_dir * (k_dir - 1) / 2))
        n_pool = len(pool)
        cnt_pool = np.bincount(pool, minlength=g["n_classes"])
        freq = cnt_pool / n_pool
        F_pool = float((freq ** 2).sum())
        npairs_inv = k_inv * (k_inv - 1) / 2
        h = float(np.clip(p_dir + obs[j], 0.0, 1.0))    # observed p_inv
        if h >= F_pool:
            founders = pool[rng.integers(0, n_pool, POWER_B)]
            f_c = freq[founders]
            a = 1.0 - 2.0 * f_c + F_pool
            b = 2.0 * (f_c - F_pool)
            disc = np.maximum(b * b - 4.0 * a * (F_pool - h), 0.0)
            q = np.clip((-b + np.sqrt(disc)) / (2.0 * a), 0.0, 1.0)
            take = rng.random((POWER_B, k_inv)) < q[:, None]
            sim = pool[rng.integers(0, n_pool, (POWER_B, k_inv))]
            sim[take] = np.broadcast_to(founders[:, None], sim.shape)[take]
            row = np.repeat(np.arange(POWER_B), k_inv)
            counts = np.zeros((POWER_B, g["n_classes"]), dtype=np.int64)
            np.add.at(counts, (row, sim.ravel()), 1)
        else:
            r = 1.0 - np.sqrt(h / F_pool)
            keep = rng.random((POWER_B, k_inv)) >= r
            sim = pool[rng.integers(0, n_pool, (POWER_B, k_inv))]
            counts = np.zeros((POWER_B, g["n_classes"]), dtype=np.int64)
            rows_kept = np.repeat(np.arange(POWER_B), k_inv)[keep.ravel()]
            np.add.at(counts, (rows_kept, sim[keep]), 1)
        p_inv = ((counts * (counts - 1) // 2).sum(axis=1) / npairs_inv)
        d_sim = np.abs(p_inv - p_dir)
        p_sim = tail_p(sorted_null[:, j], d_sim)
        obs_rows.append({
            "gene_name": g["name"], "inv_id": g["inv_id"],
            "k_inverted": k_inv,
            "observed_delta": float(obs[j]),
            "observed_abs_delta": float(abs(obs[j])),
            "target_identity": h, "p_direct": p_dir,
            "mean_abs_delta_sim": float(d_sim.mean()),
            "power_nominal": float((p_sim <= ALPHA).mean()),
            "power_fdr": float((p_sim <= t_fdr).mean()),
            "significant_observed": bool(fdr[j] < ALPHA),
        })
    pobs = pd.DataFrame(obs_rows).sort_values("observed_abs_delta",
                                              ascending=False)
    pobs.to_csv(OUT_POWER_OBS, sep="\t", index=False)
    sig = pobs[pobs.significant_observed]
    print("\n=== POWER AT THE OBSERVED CONTRASTS ===")
    print(pobs.head(16)[["gene_name", "observed_delta", "power_nominal",
                         "power_fdr"]].round(3).to_string(index=False))
    print(f"13 significant genes: power_nominal mean {sig.power_nominal.mean():.3f} "
          f"median {sig.power_nominal.median():.3f} min {sig.power_nominal.min():.3f}; "
          f"power_fdr mean {sig.power_fdr.mean():.3f} "
          f"median {sig.power_fdr.median():.3f} min {sig.power_fdr.min():.3f}")
    by_dir = (pdelta.groupby(["delta", "direction"])
              [["mean_abs_delta", "power_nominal", "power_fdr"]]
              .agg(["mean", "count"]).round(3))
    print(by_dir.to_string())
    per_gene_curve = (pdelta.groupby(["gene_name", "delta"])
                      [["power_nominal", "power_fdr"]].mean().reset_index())
    combined = (per_gene_curve.groupby("delta")
                [["power_nominal", "power_fdr"]]
                .agg(["mean", "count"]).round(3))
    print("\ncombined (per-gene mean over feasible directions; count = genes"
          " where the effect can exist):")
    print(combined.to_string())

    # ---- MDE80: smallest true difference each gene detects with 80% power --
    # Interpolated on the gene's own curve; NaN when the curve never reaches
    # 80% (those genes are counted, not dropped).
    def _mde80(gcurve, col):
        gcurve = gcurve.sort_values("delta")
        y = gcurve[col].to_numpy()
        x = gcurve["delta"].to_numpy()
        above = np.nonzero(y >= 0.8)[0]
        if len(above) == 0:
            return np.nan
        i = above[0]
        if i == 0 or y[i] == y[i - 1]:
            return float(x[i])
        return float(x[i - 1] + (0.8 - y[i - 1]) / (y[i] - y[i - 1])
                     * (x[i] - x[i - 1]))

    mde80 = per_gene_curve.groupby("gene_name").apply(
        lambda gg: pd.Series({
            "mde80_nominal": _mde80(gg, "power_nominal"),
            "mde80_fdr": _mde80(gg, "power_fdr"),
        }), include_groups=False).reset_index()
    mde_tbl = pd.read_csv(OUT_MDE, sep="\t").merge(mde80, on="gene_name",
                                                    how="left")
    mde_tbl.to_csv(OUT_MDE, sep="\t", index=False)
    n80n = int(mde_tbl["mde80_nominal"].notna().sum())
    n80f = int(mde_tbl["mde80_fdr"].notna().sum())
    print(f"\nMDE80 nominal: median {mde_tbl.mde80_nominal.median():.3f} "
          f"(defined for {n80n}/{len(mde_tbl)} genes; the rest never reach "
          f"80% power in either direction)")
    print(f"MDE80 FDR:     median {mde_tbl.mde80_fdr.median():.3f} "
          f"(defined for {n80f}/{len(mde_tbl)} genes)")

    # ---- power-curve figure (marks and short axis labels only) -------------
    import matplotlib.pyplot as plt
    curve = (per_gene_curve.groupby("delta")
             [["power_nominal", "power_fdr"]].mean())
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(curve.index, curve["power_nominal"], color="#3b5b92", lw=1.8,
            marker="o", ms=3.5, label="\u03b1 = 0.05")
    ax.plot(curve.index, curve["power_fdr"], color="#c26d2b", lw=1.8,
            marker="s", ms=3.2, label="FDR threshold")
    ax.axhline(0.8, color="#999999", lw=0.8, ls=(0, (3, 3)))
    hit_sizes = np.abs(obs[fdr < ALPHA])
    ax.plot(hit_sizes, np.full_like(hit_sizes, -0.035), marker="|", ls="none",
            color="#444444", ms=8, clip_on=False)
    ax.set_xlim(0, float(max(DELTA_GRID)))
    ax.set_ylim(-0.06, 1.02)
    ax.set_xlabel("True difference in CDS pair identity")
    ax.set_ylabel("Power")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_POWER_CURVE)
    fig.savefig(OUT_POWER_CURVE.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_POWER_CURVE}")
    print(f"\nWrote {OUT_JOINT}\nWrote {OUT_CALIB}\nWrote {OUT_POWER}\n"
          f"Wrote {OUT_POWER_DELTA}\nWrote {OUT_POWER_OBS}")
    make_figure()


if __name__ == "__main__":
    main()
