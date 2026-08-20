"""Refit the conservation model on entire gene sequences (stage 2).

stats/gene_span_conservation.py produced, for every gene x inversion tested in
the CDS conservation analysis, the proportion of within-orientation haplotype
pairs that are identical across the whole transcript span (exons + introns +
UTRs) and across the CDS alone, from the same whole-locus alignments.

This script answers the editorial question directly:

  1. REFIT.  The published model
         prop ~ C(consensus) * C(phy_group) + log_m + log_L + log_k
     (binomial, freq_weights = n_pairs, cluster-robust by inversion) is refit
     with prop = whole-gene identity instead of CDS identity, log_m = log gene
     span. Reported with the parametric Wald p AND the inversion-level
     permutation p that the CDS version was re-referred to.

  2. WAS IT POWER?  The whole gene span is 10-100x longer than the CDS and its
     identity proportion is far from the ceiling that the CDS statistic sits
     against, so if the CDS test failed only for lack of sites the gene-span
     refit should recover the effect. Power for both statistics is computed on
     the same exact sign-flip footing (inversion = independent unit).

  3. IS IT CODING-SPECIFIC?  The reviewer's alternative explanation is that
     inverted haplotypes are identical everywhere because they are young and
     low-diversity, not because coding sequence is conserved. That predicts the
     gene-span effect should be as large as, or larger than, the CDS effect,
     and that CDS identity should carry no signal once the gene-span background
     is conditioned on. Both are tested here.

Input:  data/gene_span_conservation.tsv
Output: data/gene_span_conservation_model.tsv   (all estimates, one row per test)
        data/gene_span_conservation.pdf/.png    (figure)
        printed report
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STATS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_STATS_DIR)
_DATA = os.path.join(_REPO_DIR, "data")

IN_TSV = os.path.join(_DATA, "gene_span_conservation.tsv")
OUT_TSV = os.path.join(_DATA, "gene_span_conservation_model.tsv")
OUT_PDF = os.path.join(_DATA, "gene_span_conservation.pdf")
OUT_PNG = os.path.join(_DATA, "gene_span_conservation.png")
PLACEBO_TSV = os.path.join(_DATA, "gene_span_conservation_placebo.tsv")
PLACEBO_P_TSV = os.path.join(_DATA, "gene_span_conservation_placebo_p.tsv")

N_PERM = 2_000        # matches the 2,000 inversion-label shuffles used for CDS
N_PERM_PLACEBO = 500  # per placebo draw; 50 draws x 500 refits is the budget
MAX_EXACT = 16        # enumerate all 2^n sign flips up to this many inversions
RNG_SEED = 2026

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

COL_SINGLE = "#2c7fb8"
COL_RECUR = "#e6802e"


# ------------------------------------------------------------------ long form
def to_long(df, kind):
    """One row per gene x orientation for the requested statistic."""
    pre = "gene" if kind == "gene" else "cds"
    length = df["span_bp"] if kind == "gene" else df["cds_bp"]
    out = []
    for grp, phy in (("direct", 0), ("inverted", 1)):
        out.append(pd.DataFrame({
            "gene_name": df["gene_name"],
            "inv_id": df["inv_id"],
            "recurrence": df["recurrence"],
            "consensus": (df["recurrence"] == "recurrent").astype(int),
            "phy_group": phy,
            "prop": df[f"{pre}_prop_identical_{grp}"],
            "n": df[f"{pre}_pairs_{grp}"],
            "k": df[f"k_{grp}"],
            "m": length,
            "L": df["inv_bp"],
            "pi": df[f"{pre}_pi_{grp}"],
        }))
    long = pd.concat(out, ignore_index=True)
    long = long[long["n"] > 0].dropna(subset=["prop"]).copy()
    long["log_m"] = np.log(long["m"].clip(lower=1))
    long["log_L"] = np.log(long["L"].clip(lower=1))
    long["log_k"] = np.log(long["k"].clip(lower=1))
    return long


def fit_glm(df):
    formula = "prop ~ C(consensus) * C(phy_group) + log_m + log_L + log_k"
    model = smf.glm(formula, data=df, family=sm.families.Binomial(),
                    freq_weights=df["n"])
    return model.fit(cov_type="cluster",
                     cov_kwds={"groups": df["inv_id"]}, scale=None)


def orientation_within_single(res):
    """Wald z and p for the direct->inverted step inside single-event loci."""
    name = "C(phy_group)[T.1]"
    if name not in res.params.index:
        return np.nan, np.nan
    b = res.params[name]
    se = res.bse[name]
    z = b / se
    return z, 2 * stats.norm.sf(abs(z))


def interaction_stat(res):
    name = "C(consensus)[T.1]:C(phy_group)[T.1]"
    if name not in res.params.index:
        return np.nan, np.nan
    z = res.params[name] / res.bse[name]
    return z, 2 * stats.norm.sf(abs(z))


def permute_labels(long, rng, n_perm=N_PERM):
    """Shuffle orientation labels ONCE PER INVERSION and refit.

    Genes inside one inversion share haplotypes, so the exchangeable unit is
    the inversion, not the gene. Within an inversion the two orientation labels
    are swapped or not (a sign flip), which preserves every gene's pair of
    values and the covariate structure exactly.
    """
    inv_ids = sorted(long["inv_id"].unique())
    flip_col = {}
    for inv in inv_ids:
        flip_col[inv] = long["inv_id"].to_numpy() == inv

    z_single = np.empty(n_perm)
    z_inter = np.empty(n_perm)
    base = long.copy()
    for b in range(n_perm):
        pg = base["phy_group"].to_numpy().copy()
        for inv in inv_ids:
            if rng.random() < 0.5:
                sel = flip_col[inv]
                pg[sel] = 1 - pg[sel]
        base["phy_group"] = pg
        try:
            r = fit_glm(base)
            z_single[b] = orientation_within_single(r)[0]
            z_inter[b] = interaction_stat(r)[0]
        except Exception:
            z_single[b] = np.nan
            z_inter[b] = np.nan
    return z_single, z_inter


_SIGN_CACHE = {}


def sign_matrix(n):
    """All 2^n sign vectors when that is small, else a fixed random subsample."""
    if n not in _SIGN_CACHE:
        if n <= MAX_EXACT:
            s = np.array(list(itertools.product([-1.0, 1.0], repeat=n)))
        else:
            s = np.random.default_rng(RNG_SEED).choice(
                [-1.0, 1.0], size=(1 << MAX_EXACT, n))
        _SIGN_CACHE[n] = s
    return _SIGN_CACHE[n]


def exact_signflip(deltas):
    """Two-sided sign-flip p for the mean of per-inversion differences."""
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan, 0
    obs = d.mean()
    means = sign_matrix(n) @ d / n
    p = float((np.abs(means) >= abs(obs) - 1e-12).mean())
    return obs, p, n


def signflip_power(n_units, sd_units, effect, alpha=0.05, n_sim=4_000, seed=7):
    """Power of the sign-flip test for a given true effect, fully vectorised."""
    if n_units < 2 or not np.isfinite(sd_units) or sd_units <= 0:
        return np.nan
    rng = np.random.default_rng(seed)
    S = sign_matrix(n_units)
    hits = 0
    chunk = max(1, int(4e6 // max(S.shape[0], 1)))
    done = 0
    while done < n_sim:
        b = min(chunk, n_sim - done)
        D = rng.normal(effect, sd_units, (b, n_units))
        obs = np.abs(D.mean(axis=1))
        means = np.abs(D @ S.T) / n_units          # (b, n_signs)
        frac = (means >= obs[:, None] - 1e-12).mean(axis=1)
        hits += int((frac <= alpha).sum())
        done += b
    return hits / n_sim


def effect_for_power(n_units, sd_units, target=0.8, hi=1.0):
    """Smallest true effect reaching `target` power, by bisection."""
    if n_units < 2 or not np.isfinite(sd_units) or sd_units <= 0:
        return None
    if signflip_power(n_units, sd_units, hi, n_sim=2_000) < target:
        return None
    lo = 0.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if signflip_power(n_units, sd_units, mid, n_sim=2_000) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def per_inversion_delta(long):
    """Mean (inverted - direct) identity per inversion, averaged over genes."""
    w = long.pivot_table(index=["inv_id", "recurrence", "gene_name"],
                         columns="phy_group", values="prop")
    w = w.dropna()
    w.columns = ["direct", "inverted"]
    w["delta"] = w["inverted"] - w["direct"]
    per_inv = (w.reset_index()
               .groupby(["inv_id", "recurrence"])["delta"]
               .agg(["mean", "size"])
               .reset_index()
               .rename(columns={"mean": "delta", "size": "n_genes"}))
    return per_inv, w.reset_index()


def report_block(name, long, rng, rows):
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    res = fit_glm(long)
    z_s, p_s = orientation_within_single(res)
    z_i, p_i = interaction_stat(res)
    print(f"GLM  n_rows={len(long)}  n_inversions={long['inv_id'].nunique()}")
    print(f"  orientation within single-event : z={z_s:+.3f}  Wald p={p_s:.4g}")
    print(f"  recurrence x orientation        : z={z_i:+.3f}  Wald p={p_i:.4g}")

    zs_null, zi_null = permute_labels(long, rng, N_PERM)
    ok_s = np.isfinite(zs_null)
    ok_i = np.isfinite(zi_null)
    pp_s = ((np.abs(zs_null[ok_s]) >= abs(z_s) - 1e-12).sum() + 1) / (ok_s.sum() + 1)
    pp_i = ((np.abs(zi_null[ok_i]) >= abs(z_i) - 1e-12).sum() + 1) / (ok_i.sum() + 1)
    print(f"  inversion-level permutation p (orientation|single) = {pp_s:.4g}"
          f"  [{int(ok_s.sum())} valid draws]")
    print(f"  inversion-level permutation p (interaction)        = {pp_i:.4g}")

    per_inv, per_gene = per_inversion_delta(long)
    for rec in ("single-event", "recurrent"):
        d = per_inv.loc[per_inv["recurrence"] == rec, "delta"]
        obs, p, n = exact_signflip(d.to_numpy())
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else np.nan
        print(f"  paired inversion-level, {rec:<13}: mean delta = "
              f"{100 * obs:+.2f} points, exact sign-flip p = {p:.4g} (n = {n} "
              f"inversions, between-inversion SD = {100 * sd:.2f} points)")
        rows.append(dict(statistic=name, analysis=f"paired_inversion_{rec}",
                         estimate=obs, p_value=p, n_units=n, sd_units=sd))
        if np.isfinite(sd) and n > 1:
            pw = signflip_power(n, sd, abs(obs))
            need = effect_for_power(n, sd)
            floor_p = 2.0 / (1 << n) if n <= MAX_EXACT else 0.0
            print(f"      power at the observed effect = {100 * pw:.1f}%; "
                  f"effect needed for 80% power = "
                  f"{'%.1f points' % (100 * need) if need else 'not reachable'}"
                  f"; smallest attainable p at n={n} is {floor_p:.4g}")
            rows.append(dict(statistic=name, analysis=f"power_{rec}",
                             estimate=pw, p_value=np.nan, n_units=n,
                             sd_units=need))

    rows.append(dict(statistic=name, analysis="glm_orientation_single",
                     estimate=z_s, p_value=p_s, n_units=long["inv_id"].nunique(),
                     sd_units=np.nan))
    rows.append(dict(statistic=name, analysis="perm_orientation_single",
                     estimate=z_s, p_value=pp_s, n_units=long["inv_id"].nunique(),
                     sd_units=np.nan))
    rows.append(dict(statistic=name, analysis="glm_interaction",
                     estimate=z_i, p_value=p_i, n_units=long["inv_id"].nunique(),
                     sd_units=np.nan))
    rows.append(dict(statistic=name, analysis="perm_interaction",
                     estimate=z_i, p_value=pp_i, n_units=long["inv_id"].nunique(),
                     sd_units=np.nan))
    return per_inv, per_gene


def main():
    if not os.path.exists(IN_TSV):
        sys.exit(f"missing {IN_TSV}; run stats/gene_span_conservation.py first")
    df = pd.read_csv(IN_TSV, sep="\t")
    df = df[df["status"] == "OK"].copy()
    print(f"{len(df)} gene x inversion tests, "
          f"{df['inv_id'].nunique()} inversions "
          f"({df.groupby('recurrence')['inv_id'].nunique().to_dict()})")
    print(f"median gene span {df['span_bp'].median():,.0f} bp vs median CDS "
          f"{df['cds_bp'].median():,.0f} bp "
          f"({df['span_bp'].median() / df['cds_bp'].median():.1f}x longer)")

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    long_cds = to_long(df, "cds")
    long_gene = to_long(df, "gene")

    inv_cds, gene_cds = report_block("CDS only (published statistic)",
                                     long_cds, rng, rows)
    inv_gene, gene_gene = report_block("ENTIRE GENE SPAN (exons+introns+UTRs)",
                                       long_gene, rng, rows)

    # ---- is any of it coding-specific once the gene background is held? ----
    print(f"\n{'=' * 72}\nCODING-SPECIFICITY: CDS identity given the gene-span "
          f"background\n{'=' * 72}")
    merged = long_cds.merge(
        long_gene[["gene_name", "inv_id", "phy_group", "prop", "pi"]],
        on=["gene_name", "inv_id", "phy_group"], suffixes=("_cds", "_gene"))
    merged = merged.dropna(subset=["prop_cds", "prop_gene"])
    eps = 1e-3
    merged["logit_gene"] = np.log((merged["prop_gene"] + eps) /
                                  (1 - merged["prop_gene"] + eps))
    formula = ("prop_cds ~ C(consensus) * C(phy_group) + logit_gene "
               "+ log_m + log_L + log_k")

    def fit_cond(d):
        return smf.glm(formula, data=d, family=sm.families.Binomial(),
                       freq_weights=d["n"]).fit(
            cov_type="cluster", cov_kwds={"groups": d["inv_id"]})

    res = fit_cond(merged)
    z_s, p_s = orientation_within_single(res)
    z_i, p_i = interaction_stat(res)
    print(f"  orientation within single-event, adjusted for whole-gene "
          f"identity: z={z_s:+.3f}  Wald p={p_s:.4g}")
    print(f"  recurrence x orientation, adjusted                     : "
          f"z={z_i:+.3f}  Wald p={p_i:.4g}")
    print(f"  whole-gene background coefficient: "
          f"{res.params['logit_gene']:+.3f} (p={res.pvalues['logit_gene']:.3g})")

    # The Wald p from this GLM inherits the same problem as the published one:
    # it treats each CDS as independent when the exchangeable unit is the
    # inversion. Refer both statistics to the same inversion-level label
    # permutation before believing them.
    inv_ids = sorted(merged["inv_id"].unique())
    sel_by_inv = {inv: merged["inv_id"].to_numpy() == inv for inv in inv_ids}
    zs_null = np.empty(N_PERM)
    zi_null = np.empty(N_PERM)
    base = merged.copy()
    for b in range(N_PERM):
        pg = base["phy_group"].to_numpy().copy()
        for inv in inv_ids:
            if rng.random() < 0.5:
                s = sel_by_inv[inv]
                pg[s] = 1 - pg[s]
        base["phy_group"] = pg
        try:
            r2 = fit_cond(base)
            zs_null[b] = orientation_within_single(r2)[0]
            zi_null[b] = interaction_stat(r2)[0]
        except Exception:
            zs_null[b] = np.nan
            zi_null[b] = np.nan
    ok_s, ok_i = np.isfinite(zs_null), np.isfinite(zi_null)
    pp_s = ((np.abs(zs_null[ok_s]) >= abs(z_s) - 1e-12).sum() + 1) / (ok_s.sum() + 1)
    pp_i = ((np.abs(zi_null[ok_i]) >= abs(z_i) - 1e-12).sum() + 1) / (ok_i.sum() + 1)
    print(f"  inversion-level permutation p (orientation|single, adjusted) "
          f"= {pp_s:.4g}")
    print(f"  inversion-level permutation p (interaction, adjusted)        "
          f"= {pp_i:.4g}")
    rows.append(dict(statistic="CDS | gene-span background",
                     analysis="perm_orientation_single", estimate=z_s,
                     p_value=pp_s, n_units=merged["inv_id"].nunique(),
                     sd_units=np.nan))
    rows.append(dict(statistic="CDS | gene-span background",
                     analysis="perm_interaction", estimate=z_i, p_value=pp_i,
                     n_units=merged["inv_id"].nunique(), sd_units=np.nan))
    rows.append(dict(statistic="CDS | gene-span background",
                     analysis="glm_orientation_single", estimate=z_s,
                     p_value=p_s, n_units=merged["inv_id"].nunique(),
                     sd_units=np.nan))
    rows.append(dict(statistic="CDS | gene-span background",
                     analysis="glm_interaction", estimate=z_i, p_value=p_i,
                     n_units=merged["inv_id"].nunique(), sd_units=np.nan))

    # ---- placebo calibration of that conditional test ---------------------
    # The conditional permutation above still assumes the two orientation
    # groups are exchangeable within a locus, which is exactly the assumption
    # that failed for the label-permutation version of the CDS test. Rather
    # than argue about it, calibrate: for each gene, relocate the CDS block
    # structure to random intronic positions inside the same gene span
    # (same site count, same blocks, same haplotypes, no coding sequence), and
    # push each such pseudo-CDS through the identical pipeline. If the pipeline
    # is calibrated, about 5% of placebo draws should reach p < 0.05.
    placebo_p = None
    if os.path.exists(PLACEBO_P_TSV):
        # already calibrated; re-reading is 15 minutes cheaper than redoing it
        placebo_p = pd.read_csv(PLACEBO_P_TSV, sep="\t")
        print(f"\n{'=' * 72}\nPLACEBO CALIBRATION (cached from "
              f"{os.path.basename(PLACEBO_P_TSV)})\n{'=' * 72}")
        a5 = float((placebo_p["p_orientation"] < 0.05).mean())
        b5 = float((placebo_p["p_interaction"] < 0.05).mean())
        print(f"  {len(placebo_p)} placebo draws; realised type-I error at "
              f"alpha = 0.05: orientation {100 * a5:.1f}%, "
              f"interaction {100 * b5:.1f}% (nominal 5%)")
        rows.append(dict(statistic="placebo (pseudo-CDS in introns)",
                         analysis="realised_type1_orientation", estimate=a5,
                         p_value=np.nan, n_units=len(placebo_p),
                         sd_units=np.nan))
        rows.append(dict(statistic="placebo (pseudo-CDS in introns)",
                         analysis="realised_type1_interaction", estimate=b5,
                         p_value=np.nan, n_units=len(placebo_p),
                         sd_units=np.nan))
    elif os.path.exists(PLACEBO_TSV):
        print(f"\n{'=' * 72}\nPLACEBO CALIBRATION of the conditional test\n"
              f"{'=' * 72}")
        pl = pd.read_csv(PLACEBO_TSV, sep="\t")
        gene_lookup = long_gene.set_index(
            ["gene_name", "inv_id", "phy_group"])[["prop", "n"]]
        draws = sorted(pl["draw"].unique())
        obs_s, obs_i = [], []
        for dnum in draws:
            sub = pl[pl["draw"] == dnum]
            recs = []
            for grp, phy in (("direct", 0), ("inverted", 1)):
                recs.append(pd.DataFrame({
                    "gene_name": sub["gene_name"],
                    "inv_id": sub["inv_id"],
                    "recurrence": sub["recurrence"],
                    "consensus": (sub["recurrence"] == "recurrent").astype(int),
                    "phy_group": phy,
                    "prop_cds": sub[f"prop_identical_{grp}"],
                    "n": sub[f"pairs_{grp}"],
                    "k": sub[f"k_{grp}"],
                    "m": sub["n_sites"],
                    "L": sub["inv_bp"],
                }))
            pdf_ = pd.concat(recs, ignore_index=True)
            pdf_ = pdf_[pdf_["n"] > 0].dropna(subset=["prop_cds"])
            pdf_["log_m"] = np.log(pdf_["m"].clip(lower=1))
            pdf_["log_L"] = np.log(pdf_["L"].clip(lower=1))
            pdf_["log_k"] = np.log(pdf_["k"].clip(lower=1))
            pdf_ = pdf_.join(gene_lookup, on=["gene_name", "inv_id", "phy_group"],
                             rsuffix="_gene").dropna(subset=["prop"])
            if pdf_["inv_id"].nunique() < 4:
                continue
            pdf_["logit_gene"] = np.log((pdf_["prop"] + eps) /
                                        (1 - pdf_["prop"] + eps))
            try:
                r0 = fit_cond(pdf_)
                z0 = orientation_within_single(r0)[0]
                zi0 = interaction_stat(r0)[0]
            except Exception:
                continue
            ids = sorted(pdf_["inv_id"].unique())
            sel = {i: pdf_["inv_id"].to_numpy() == i for i in ids}
            zs = np.empty(N_PERM_PLACEBO)
            zi = np.empty(N_PERM_PLACEBO)
            b_ = pdf_.copy()
            for b in range(N_PERM_PLACEBO):
                pg = b_["phy_group"].to_numpy().copy()
                for i in ids:
                    if rng.random() < 0.5:
                        s = sel[i]
                        pg[s] = 1 - pg[s]
                b_["phy_group"] = pg
                try:
                    rr = fit_cond(b_)
                    zs[b] = orientation_within_single(rr)[0]
                    zi[b] = interaction_stat(rr)[0]
                except Exception:
                    zs[b] = zi[b] = np.nan
            ks, ki = np.isfinite(zs), np.isfinite(zi)
            obs_s.append(((np.abs(zs[ks]) >= abs(z0) - 1e-12).sum() + 1) /
                         (ks.sum() + 1))
            obs_i.append(((np.abs(zi[ki]) >= abs(zi0) - 1e-12).sum() + 1) /
                         (ki.sum() + 1))
            print(f"  draw {dnum:>2}: placebo permutation p "
                  f"orientation={obs_s[-1]:.3f} interaction={obs_i[-1]:.3f}",
                  flush=True)
        if obs_s:
            a5 = float(np.mean(np.asarray(obs_s) < 0.05))
            b5 = float(np.mean(np.asarray(obs_i) < 0.05))
            print(f"\n  {len(obs_s)} placebo draws")
            print(f"  realised type-I error at alpha = 0.05: "
                  f"orientation {100 * a5:.1f}%, interaction {100 * b5:.1f}% "
                  f"(nominal 5%)")
            print(f"  median placebo p: orientation {np.median(obs_s):.3f}, "
                  f"interaction {np.median(obs_i):.3f}")
            rows.append(dict(statistic="placebo (pseudo-CDS in introns)",
                             analysis="realised_type1_orientation",
                             estimate=a5, p_value=np.nan, n_units=len(obs_s),
                             sd_units=np.nan))
            rows.append(dict(statistic="placebo (pseudo-CDS in introns)",
                             analysis="realised_type1_interaction",
                             estimate=b5, p_value=np.nan, n_units=len(obs_i),
                             sd_units=np.nan))
            placebo_p = pd.DataFrame({"draw": draws[:len(obs_s)],
                                      "p_orientation": obs_s,
                                      "p_interaction": obs_i})
            placebo_p.to_csv(PLACEBO_P_TSV, sep="\t", index=False)

    # per-inversion delta correlation between CDS and gene span
    m = inv_cds.merge(inv_gene, on=["inv_id", "recurrence"],
                      suffixes=("_cds", "_gene"))
    r, p = stats.pearsonr(m["delta_cds"], m["delta_gene"])
    rho, prho = stats.spearmanr(m["delta_cds"], m["delta_gene"])
    print(f"\n  per-inversion CDS vs whole-gene orientation effect: "
          f"Pearson r={r:.3f} (p={p:.3g}), Spearman rho={rho:.3f} (p={prho:.3g})"
          f"  [n={len(m)} inversions]")
    rows.append(dict(statistic="CDS vs gene-span per-inversion effect",
                     analysis="pearson_r", estimate=r, p_value=p,
                     n_units=len(m), sd_units=np.nan))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nWrote {OUT_TSV}")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3))

    # panel A: group means, CDS vs gene span
    ax = axes[0]
    labels = ["Single\ndirect", "Single\ninverted", "Recur\ndirect",
              "Recur\ninverted"]
    x = np.arange(4)
    for src, longd, mk, ls in (("CDS", long_cds, "o", "-"),
                               ("Whole gene", long_gene, "s", "--")):
        vals, errs = [], []
        for cons, phy in ((0, 0), (0, 1), (1, 0), (1, 1)):
            v = longd[(longd["consensus"] == cons) &
                      (longd["phy_group"] == phy)]["prop"]
            vals.append(v.mean())
            errs.append(v.std(ddof=1) / max(np.sqrt(len(v)), 1))
        ax.errorbar(x, vals, yerr=errs, marker=mk, ls=ls, capsize=3,
                    label=src, color="#333333" if src == "CDS" else "#cc3311")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Proportion of identical haplotype pairs")
    ax.set_title("A  Refit on entire gene sequences", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.02)

    # panel B: per-inversion orientation effect, CDS vs gene span
    ax = axes[1]
    for rec, col in (("single-event", COL_SINGLE), ("recurrent", COL_RECUR)):
        sel = m["recurrence"] == rec
        ax.scatter(100 * m.loc[sel, "delta_cds"], 100 * m.loc[sel, "delta_gene"],
                   s=42, color=col, alpha=.85, edgecolor="white",
                   label=f"{rec} (n={int(sel.sum())})")
    lim = [-100, 100]
    ax.plot(lim, lim, color="#999999", lw=.8, ls=":")
    ax.axhline(0, color="#cccccc", lw=.8)
    ax.axvline(0, color="#cccccc", lw=.8)
    ax.set_xlabel("CDS effect (inverted - direct, points)")
    ax.set_ylabel("Whole-gene effect (points)")
    ax.set_title(f"B  Same effect, two windows (r = {r:.2f})", loc="left",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    pad = 5
    ax.set_xlim(100 * m["delta_cds"].min() - pad, 100 * m["delta_cds"].max() + pad)
    ax.set_ylim(100 * m["delta_gene"].min() - pad, 100 * m["delta_gene"].max() + pad)

    # panel C: per-inversion deltas by recurrence, whole gene
    ax = axes[2]
    for i, (rec, col) in enumerate((("single-event", COL_SINGLE),
                                    ("recurrent", COL_RECUR))):
        d = 100 * inv_gene.loc[inv_gene["recurrence"] == rec, "delta"]
        xj = np.random.default_rng(1).normal(i, .06, len(d))
        ax.scatter(xj, d, s=40, color=col, alpha=.8, edgecolor="white")
        ax.hlines(d.mean(), i - .22, i + .22, color="black", lw=2)
    ax.axhline(0, color="#cccccc", lw=.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Single-event", "Recurrent"])
    ax.set_ylabel("Whole-gene identity,\ninverted - direct (points)")
    ax.set_title("C  Inversion is the unit", loc="left", fontsize=11)

    # panel D: what a coding-free window of the same size does in the same test
    ax = axes[3]
    if placebo_p is not None and len(placebo_p):
        bins = np.linspace(0, 1, 21)
        ax.hist(placebo_p["p_orientation"], bins=bins, color="#7f7f7f",
                alpha=.75, label="orientation")
        ax.hist(placebo_p["p_interaction"], bins=bins, histtype="step",
                lw=1.6, color="#cc3311", label="interaction")
        ax.axvline(0.05, color="#222222", ls=":", lw=1.2)
        ax.axhline(len(placebo_p) / 20, color="#999999", ls="--", lw=1,
                   label="uniform expectation")
        a5 = 100 * float((placebo_p["p_orientation"] < 0.05).mean())
        ax.set_xlabel("permutation p from a pseudo-CDS\n"
                      "(same size, intronic position)")
        ax.set_ylabel(f"placebo draws (of {len(placebo_p)})")
        ax.set_title(f"D  The test itself, calibrated\n"
                     f"{a5:.0f}% reject at $\\alpha$ = 0.05", loc="left",
                     fontsize=11)
        ax.legend(frameon=False, fontsize=7)
    else:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
