"""Genomic-control inflation for the inversion PheWAS.

Reviewer 3 (major comment 4) asks for positive evidence that the reported
associations are not produced by residual population structure. The direct,
quantitative answer available from the summary statistics is the genomic
control factor lambda_GC, computed three ways:

  1. PER INVERSION, across all phecodes tested (the conventional lambda: the
     null-dominated bulk of ~1,089 tests sets the scale, and a handful of true
     signals cannot move a median). lambda ~ 1 means the test statistics are
     calibrated; lambda >> 1 is the signature of uncorrected stratification.

  2. PER PHENOTYPE, across the inversions tested for it -- the request being
     that the points entering each lambda are the different inversions. Under
     the null the median of k = 7 chi-square(1) draws has a known sampling
     distribution, so the observed spread of per-phenotype lambdas is compared
     against that reference rather than against 1.0 alone. Phenotypes whose
     lambda sits above the null envelope are the ones where several inversions
     move together, which is what shared ancestry stratification would do.

  3. PER PHENOTYPE CATEGORY, because the reviewer names autoimmune, obesity
     and cognitive phenotypes specifically.

Then the number that actually matters for the manuscript's claims: every
reported association is re-tested after dividing its chi-square by the lambda
of its own inversion (Devlin & Roeder genomic control), and the family-wise
Benjamini-Hochberg correction is redone on the corrected p-values. An
association that survives its own inversion's inflation factor is not
explained by whatever stratification that lambda measures.

Finally, each inversion's imputed dosage is decomposed into between- and
within-ancestry variance (dosage F_ST across the All of Us genetic-ancestry
groups, from the published per-population frequencies). This says how strong
an ancestry proxy each inversion is, i.e. how much stratification risk it
carries in the first place.

Inputs:  data/phewas_results.tsv
         data/phewas v2 - categories.tsv          (phenotype -> category)
         data/inversion_population_frequencies.tsv
Outputs: data/phewas_lambda_gc.tsv                (per inversion)
         data/phewas_lambda_by_phenotype.tsv      (per phenotype)
         data/phewas_lambda_significant_hits.tsv  (GC-corrected hits)
         data/phewas_lambda_gc.pdf/.png
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_STATS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_STATS_DIR)
_DATA = os.path.join(_REPO_DIR, "data")

IN_PHEWAS = os.path.join(_DATA, "phewas_results.tsv")
IN_CATEGORIES = os.path.join(_DATA, "phewas v2 - categories.tsv")
IN_POPFREQ = os.path.join(_DATA, "inversion_population_frequencies.tsv")

OUT_INV = os.path.join(_DATA, "phewas_lambda_gc.tsv")
OUT_PHENO = os.path.join(_DATA, "phewas_lambda_by_phenotype.tsv")
OUT_HITS = os.path.join(_DATA, "phewas_lambda_significant_hits.tsv")
OUT_PDF = os.path.join(_DATA, "phewas_lambda_gc.pdf")
OUT_PNG = os.path.join(_DATA, "phewas_lambda_gc.png")

CHI2_MEDIAN = stats.chi2.ppf(0.5, 1)      # 0.4549364
N_BOOT = 4_000
N_NULL = 200_000
RNG_SEED = 2026

# short display names for the seven tested inversions
INV_LABEL = {
    "chr17-45585160-INV-706887": "17q21.31",
    "chr8-7301025-INV-5297356": "8p23.1",
    "chr10-79542902-INV-674513": "10q22.3",
    "chr6-167181003-INV-209976": "6q27",
    "chr6-141867315-INV-29159": "6q24.1",
    "chr4-33098029-INV-7075": "4q13.1",
    "chr12-46897663-INV-16289": "12q13.11",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def lambda_gc(pvals):
    """Devlin-Roeder genomic control factor from a vector of p-values."""
    p = np.asarray(pvals, float)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]
    if len(p) == 0:
        return np.nan, 0
    chi = stats.chi2.isf(p, 1)
    return float(np.median(chi) / CHI2_MEDIAN), len(p)


def lambda_ci(pvals, rng, n_boot=N_BOOT, blocks=None):
    """Bootstrap interval for lambda.

    Phecodes are not independent -- they share cases, and whole families of them
    (all the dermatological terms, say) move together. Resampling phecodes one at
    a time therefore understates the uncertainty in the median. When `blocks` is
    supplied, the resampling is done over blocks (phenotype categories) and then
    over phecodes within the drawn blocks, which keeps the within-category
    correlation intact and widens the interval to something honest.
    """
    p = np.asarray(pvals, float)
    keep = np.isfinite(p) & (p > 0) & (p <= 1)
    p = p[keep]
    if len(p) < 10:
        return np.nan, np.nan
    chi = stats.chi2.isf(p, 1)
    if blocks is None:
        idx = rng.integers(0, len(chi), size=(n_boot, len(chi)))
        lam = np.median(chi[idx], axis=1) / CHI2_MEDIAN
        return float(np.percentile(lam, 2.5)), float(np.percentile(lam, 97.5))

    blk = np.asarray(blocks, dtype=object)[keep]
    groups = [np.nonzero(blk == b)[0] for b in pd.unique(blk)]
    n_blk = len(groups)
    # With as many blocks as observations the blocking says nothing, and with
    # fewer than three there is nothing to resample over.
    if n_blk < 3 or n_blk > 0.5 * len(chi):
        return lambda_ci(p, rng, n_boot, blocks=None)
    # Cluster bootstrap: draw whole blocks with replacement. Resampling within
    # the drawn blocks as well would break the correlation the blocking exists
    # to preserve.
    chi_by_block = [chi[g] for g in groups]
    lam = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_blk, size=n_blk)
        lam[i] = np.median(np.concatenate([chi_by_block[j] for j in pick])) \
            / CHI2_MEDIAN
    return float(np.percentile(lam, 2.5)), float(np.percentile(lam, 97.5))


_RANK_NULL_CACHE = {}


def _rank_null_joint(rank_matrix, seed, n_draw=20_000):
    """Joint null for the vector of per-inversion mean ranks.

    Permuting the inversion labels within each phenotype gives, per draw, all
    k mean ranks at once. Keeping the whole vector matters: ranks are
    compositional, so the k statistics are negatively dependent, and order
    statistics taken from the joint null carry that dependence where marginal
    quantiles would not.
    """
    key = (rank_matrix.shape, seed, n_draw)
    if key not in _RANK_NULL_CACHE:
        rng = np.random.default_rng(seed)
        k = rank_matrix.shape[1]
        out = np.empty((n_draw, k))
        for b in range(n_draw):
            out[b] = rng.permuted(rank_matrix, axis=1).mean(axis=0)
        _RANK_NULL_CACHE[key] = out
    return _RANK_NULL_CACHE[key]


def _rank_null(rank_matrix, seed, n_draw=20_000):
    """Marginal null for a single inversion's mean rank."""
    return _rank_null_joint(rank_matrix, seed, n_draw)[:, 0]


def _num(v):
    """Scalar float or NaN, tolerant of blanks and strings."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def phenotype_categories(path):
    """phenotype -> category, expanded from the semicolon lists in the file."""
    if not os.path.exists(path):
        return {}
    cat = pd.read_csv(path, sep="\t")
    mapping = {}
    for _, r in cat.iterrows():
        phenos = str(r.get("Phenotypes", "") or "")
        for ph in phenos.split(";"):
            ph = ph.strip()
            if ph:
                mapping.setdefault(ph, r["Category"])
    return mapping


def dosage_fst(path):
    """Between-ancestry variance share of imputed dosage, per inversion.

    Weir-Cockerham-style ratio of variance components computed from the
    per-population inverted-allele frequencies and sample sizes: the fraction
    of total allelic variance that sits between genetic-ancestry groups.
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    df = df[df["Population"].str.upper() != "ALL"].copy()
    out = {}
    for inv, g in df.groupby("Inversion"):
        n = g["N"].to_numpy(float)
        p = g["Allele_Freq"].to_numpy(float)
        ok = np.isfinite(n) & np.isfinite(p) & (n > 0)
        n, p = n[ok], p[ok]
        if len(p) < 2:
            continue
        pbar = np.sum(n * p) / np.sum(n)
        if pbar <= 0 or pbar >= 1:
            continue
        var_between = np.sum(n * (p - pbar) ** 2) / np.sum(n)
        out[inv] = float(var_between / (pbar * (1 - pbar)))
    return out


def main():
    if not os.path.exists(IN_PHEWAS):
        sys.exit(f"missing {IN_PHEWAS}")
    rng = np.random.default_rng(RNG_SEED)

    df = pd.read_csv(IN_PHEWAS, sep="\t", low_memory=False)
    pcol = "P_LRT_Overall" if "P_LRT_Overall" in df.columns else "P_Value_x"
    df["p"] = pd.to_numeric(df[pcol], errors="coerce")
    df["q_global"] = pd.to_numeric(df.get("Q_GLOBAL"), errors="coerce")
    df["n_cases"] = pd.to_numeric(df.get("N_Cases"), errors="coerce")
    df = df[np.isfinite(df["p"]) & (df["p"] > 0) & (df["p"] <= 1)].copy()
    df["inv_label"] = df["Inversion"].map(INV_LABEL).fillna(df["Inversion"])
    df["chi2"] = stats.chi2.isf(df["p"], 1)
    print(f"{len(df):,} inversion x phecode tests, "
          f"{df['Inversion'].nunique()} inversions, "
          f"{df['Phenotype'].nunique()} phecodes")

    fst = dosage_fst(IN_POPFREQ)
    cats = phenotype_categories(IN_CATEGORIES)
    df["category"] = df["Phenotype"].map(cats)
    print(f"category labels attached to {int(df['category'].notna().sum()):,} rows "
          f"({df['category'].nunique()} categories)")

    # ------------------------------------------------------------ 1. per inversion
    print("\n" + "=" * 78)
    print("1. lambda_GC per inversion, across all phecodes")
    print("=" * 78)
    rows = []
    for inv, g in df.groupby("Inversion"):
        lam, n = lambda_gc(g["p"])
        # blocks are phenotype categories; uncategorised phecodes each
        # form their own block, which is the conservative reading
        # Uncategorised phecodes go into one block, which treats them as
        # maximally correlated -- the conservative choice for an interval.
        blk = g["category"].fillna("uncategorised")
        lo, hi = lambda_ci(g["p"], rng, blocks=blk)
        # sensitivity: drop the associations that reached family significance,
        # and drop rare phecodes whose asymptotics are least trustworthy
        lam_nosig, _ = lambda_gc(g.loc[~(g["q_global"] < 0.05), "p"])
        lam_common, n_common = lambda_gc(g.loc[g["n_cases"] >= 1000, "p"])
        rows.append(dict(
            inversion=inv, label=INV_LABEL.get(inv, inv), n_tests=n,
            lambda_gc=lam, lambda_lo=lo, lambda_hi=hi,
            lambda_excl_significant=lam_nosig,
            lambda_ncases_ge1000=lam_common, n_tests_ncases_ge1000=n_common,
            dosage_fst_between_ancestry=fst.get(inv, np.nan),
            n_family_significant=int((g["q_global"] < 0.05).sum()),
        ))
        print(f"  {INV_LABEL.get(inv, inv):<10} lambda = {lam:5.3f} "
              f"[{lo:.3f}, {hi:.3f}]   excl. sig {lam_nosig:5.3f}   "
              f"n>=1000 cases {lam_common:5.3f}   "
              f"dosage F_ST {fst.get(inv, float('nan')):.3f}   "
              f"({n} tests, {int((g['q_global'] < 0.05).sum())} significant)")
    inv_tab = pd.DataFrame(rows).sort_values("lambda_gc", ascending=False)
    inv_tab.to_csv(OUT_INV, sep="\t", index=False)
    lam_by_inv = dict(zip(inv_tab["inversion"], inv_tab["lambda_gc"]))
    print(f"\n  overall lambda across all {len(df):,} tests: "
          f"{lambda_gc(df['p'])[0]:.3f}")

    # --------------------------------------------------------- 2. per phenotype
    print("\n" + "=" * 78)
    print("2. lambda_GC per phenotype, points = the inversions tested for it")
    print("=" * 78)
    prows = []
    for ph, g in df.groupby("Phenotype"):
        lam, n = lambda_gc(g["p"])
        prows.append(dict(phenotype=ph, n_inversions=n, lambda_gc=lam,
                          category=g["category"].dropna().iloc[0]
                          if g["category"].notna().any() else np.nan,
                          min_p=float(g["p"].min()),
                          n_cases=float(g["n_cases"].median())))
    pheno_tab = pd.DataFrame(prows).sort_values("lambda_gc", ascending=False)

    k = int(pheno_tab["n_inversions"].median())
    null_lam = np.median(stats.chi2.rvs(1, size=(N_NULL, k),
                                        random_state=RNG_SEED), axis=1) / CHI2_MEDIAN
    lo95, hi95 = np.percentile(null_lam, [2.5, 97.5])
    thresh = np.percentile(null_lam, 100 * (1 - 0.05 / len(pheno_tab)))
    pheno_tab["null_p"] = [float((null_lam >= l).mean()) for l in pheno_tab["lambda_gc"]]
    pheno_tab["null_q_bh"] = bh(pheno_tab["null_p"].to_numpy())
    pheno_tab.to_csv(OUT_PHENO, sep="\t", index=False)

    print(f"  k = {k} inversions per phenotype; under the null the per-phenotype "
          f"lambda has median {np.median(null_lam):.3f}, 95% range "
          f"[{lo95:.3f}, {hi95:.3f}]")
    print(f"  observed: median {pheno_tab['lambda_gc'].median():.3f}, "
          f"95% range [{pheno_tab['lambda_gc'].quantile(.025):.3f}, "
          f"{pheno_tab['lambda_gc'].quantile(.975):.3f}]")
    n_exceed = int((pheno_tab["lambda_gc"] > thresh).sum())
    print(f"  phenotypes above the Bonferroni null envelope "
          f"(lambda > {thresh:.2f}): {n_exceed} of {len(pheno_tab)}")
    print(f"  phenotypes with BH q < 0.05 against the null: "
          f"{int((pheno_tab['null_q_bh'] < 0.05).sum())}")
    print("\n  top 10 phenotypes by lambda:")
    print(pheno_tab.head(10)[["phenotype", "lambda_gc", "min_p", "null_p",
                              "null_q_bh"]].to_string(index=False))

    # ------------------------------------------------------------ 3. by category
    print("\n" + "=" * 78)
    print("3. lambda_GC by phenotype category")
    print("=" * 78)
    if df["category"].notna().any():
        crows = []
        for cat, g in df[df["category"].notna()].groupby("category"):
            lam, n = lambda_gc(g["p"])
            crows.append(dict(category=cat, n_tests=n, lambda_gc=lam))
        ctab = pd.DataFrame(crows).sort_values("lambda_gc", ascending=False)
        print(ctab.to_string(index=False))
    else:
        ctab = pd.DataFrame()
        print("  (no category labels available)")

    # ------------------------------- 4. genomic control on the reported hits
    print("\n" + "=" * 78)
    print("4. reported associations after genomic control")
    print("=" * 78)
    hits = df[df["q_global"] < 0.05].copy()
    if len(hits) == 0:
        print("  no family-significant associations in this table")
    hits["lambda_used"] = hits["Inversion"].map(lam_by_inv)
    hits["chi2_gc"] = hits["chi2"] / hits["lambda_used"].clip(lower=1.0)
    hits["p_gc"] = stats.chi2.sf(hits["chi2_gc"], 1)

    # redo the joint BH on GC-corrected p over the whole family
    allp = df.copy()
    allp["lambda_used"] = allp["Inversion"].map(lam_by_inv)
    allp["p_gc"] = stats.chi2.sf(
        allp["chi2"] / allp["lambda_used"].clip(lower=1.0), 1)
    allp["q_gc"] = bh(allp["p_gc"].to_numpy())
    hits = hits.merge(allp[["Phenotype", "Inversion", "p_gc", "q_gc"]],
                      on=["Phenotype", "Inversion"], how="left",
                      suffixes=("", "_all"))
    hits = hits.sort_values("p")
    keep = ["Phenotype", "inv_label", "Inversion", "OR", "n_cases", "p",
            "q_global", "lambda_used", "p_gc", "q_gc", "category"]
    keep = [c for c in keep if c in hits.columns]
    hits[keep].to_csv(OUT_HITS, sep="\t", index=False)
    if len(hits):
        n_sur = int((hits["q_gc"] < 0.05).sum())
        print(f"  {len(hits)} associations reached BH q < 0.05; "
              f"{n_sur} still do after genomic control "
              f"({100 * n_sur / len(hits):.0f}%)")
        show = hits[keep].head(25).copy()
        for c in ("p", "q_global", "p_gc", "q_gc"):
            if c in show:
                show[c] = show[c].map(lambda v: f"{v:.3g}")
        print(show.to_string(index=False))

    # ---- 4b. the calibration test that actually has power ------------------
    #
    # A QQ plot of one inversion's 1,053 phecode p-values against a uniform is
    # the wrong picture: phecodes are not independent replicates, they share
    # cases and move in families, so the expected line and its band are both
    # wrong. The independent replicates here are the INVERSIONS -- seven loci on
    # six chromosomes, tested against the same phenotype.
    #
    # So stratify by phenotype. Within each phenotype, rank the seven inversions
    # by p-value. Under the null each inversion is equally likely to hold any
    # rank, so its 1,053 ranks should be uniform on 1..7 with mean 4. That
    # removes the phenotype main effect -- the entire source of the correlation
    # -- and uses every test rather than a single median, which is why it has
    # far more power than lambda.
    print("\n" + "=" * 78)
    print("4b. stratified calibration: rank the inversions within each phenotype")
    print("=" * 78)
    wide = df.pivot_table(index="Phenotype", columns="Inversion", values="p")
    wide = wide.dropna()
    k_inv = wide.shape[1]
    print(f"  {len(wide)} phenotypes tested against all {k_inv} inversions")
    ranks = wide.rank(axis=1, method="average")
    exp_q = (ranks - 0.5) / k_inv           # within-phenotype expected quantile

    rank_rows = []
    for inv in wide.columns:
        r = ranks[inv].to_numpy()
        mean_rank = float(r.mean())
        # chi-square goodness of fit of the rank distribution against uniform
        counts = np.array([(np.round(r) == j).sum() for j in range(1, k_inv + 1)],
                          dtype=float)
        expected = len(r) / k_inv
        chi2_gof = float(((counts - expected) ** 2 / expected).sum())
        p_gof = float(stats.chi2.sf(chi2_gof, k_inv - 1))
        # exact null for the mean rank, by permuting inversion labels within
        # phenotype: this is the same object the ranks came from, so residual
        # phenotype correlation is carried along rather than assumed away
        # Under exchangeability every inversion's mean rank has the same null
        # distribution, so one permutation null serves all seven; it is computed
        # once and cached rather than redone per locus.
        null_means = _rank_null(ranks.to_numpy(), RNG_SEED)
        centre = null_means.mean()
        n_draw = len(null_means)
        p_mean = float(((np.abs(null_means - centre) >=
                         abs(mean_rank - centre)).sum() + 1) / (n_draw + 1))
        rank_rows.append(dict(
            inversion=inv, label=INV_LABEL.get(inv, inv),
            mean_rank=mean_rank, expected_mean_rank=(k_inv + 1) / 2,
            chi2_gof=chi2_gof, p_uniform_ranks=p_gof,
            p_mean_rank_permutation=p_mean,
            frac_rank1=float((np.round(r) == 1).mean())))
        print(f"  {INV_LABEL.get(inv, inv):<10} mean rank {mean_rank:.3f} "
              f"(null {(k_inv + 1) / 2:.1f})  top-ranked in "
              f"{100 * float((np.round(r) == 1).mean()):.1f}% of phenotypes  "
              f"chi2 GoF p = {p_gof:.3g}  permutation p "
              f"{'< ' + format(1.0 / (len(null_means) + 1), '.1g') if p_mean <= 2.0 / (len(null_means) + 1) else '= ' + format(p_mean, '.3g')}")
    rank_tab = pd.DataFrame(rank_rows).sort_values("mean_rank")
    rank_tab.to_csv(os.path.join(_DATA, "phewas_inversion_rank_calibration.tsv"),
                    sep="\t", index=False)
    print("\n  Ranks are compositional -- they sum to a constant within each")
    print("  phenotype -- so two loci taking the top places necessarily pushes")
    print("  the rest down. The test says which loci carry the signal; it cannot")
    print("  by itself say whether that signal is pleiotropy or inflation.")
    glob_chi = float(rank_tab["chi2_gof"].sum())
    print(f"  global departure from uniform ranks: chi2 = {glob_chi:.1f} on "
          f"{k_inv * (k_inv - 1)} df, p = "
          f"{stats.chi2.sf(glob_chi, k_inv * (k_inv - 1)):.3g}")

    # The discriminating question, now asked with the powerful statistic rather
    # than with lambda: does an inversion's tendency to out-rank the others
    # track how strongly its dosage separates ancestry groups? Stratification
    # says yes; locus-specific biology says no.
    fst_series = pd.Series(fst)
    merged_rank = rank_tab.set_index("inversion").join(
        fst_series.rename("fst"), how="inner").dropna(subset=["fst"])
    if len(merged_rank) >= 4:
        rho_r, p_rho_r = stats.spearmanr(merged_rank["fst"],
                                         merged_rank["mean_rank"])
        print(f"  mean rank vs between-ancestry dosage F_ST: "
              f"Spearman rho = {rho_r:+.3f} (p = {p_rho_r:.3g}); "
              f"stratification predicts a NEGATIVE rho "
              f"(ancestry-informative loci ranking first)")
        rank_tab["dosage_fst_between_ancestry"] = merged_rank["fst"].reindex(
            rank_tab["inversion"]).to_numpy()
        rank_tab.to_csv(
            os.path.join(_DATA, "phewas_inversion_rank_calibration.tsv"),
            sep="\t", index=False)

    # ------------------- 5. is the inflation an ancestry effect at all? -------
    print("\n" + "=" * 78)
    print("5. does inflation track how ancestry-informative the inversion is?")
    print("=" * 78)
    sub = inv_tab.dropna(subset=["dosage_fst_between_ancestry"])
    if len(sub) >= 4:
        rho, prho = stats.spearmanr(sub["dosage_fst_between_ancestry"],
                                    sub["lambda_gc"])
        r, pr = stats.pearsonr(sub["dosage_fst_between_ancestry"],
                               sub["lambda_gc"])
        print(f"  lambda vs between-ancestry dosage F_ST across "
              f"{len(sub)} inversions: Spearman rho = {rho:+.3f} (p = {prho:.3g}), "
              f"Pearson r = {r:+.3f} (p = {pr:.3g})")
        print("  stratification predicts a POSITIVE relationship: the inversions "
              "that best proxy\n  ancestry should be the inflated ones.")

    # --------------- 6. within-ancestry meta-analysis of the reported hits ----
    print("\n" + "=" * 78)
    print("6. multi-ancestry estimate vs within-ancestry meta-analysis")
    print("=" * 78)
    print("  A within-ancestry estimate cannot be confounded by frequency or")
    print("  prevalence differences BETWEEN ancestry groups. If stratification")
    print("  drives a signal, the inverse-variance meta-analysis of the")
    print("  ancestry-specific effects shrinks toward the null relative to the")
    print("  pooled multi-ancestry effect.")
    anc = ["EUR", "AFR", "AMR", "EAS", "SAS", "MID"]
    mrows = []
    for _, r in df.iterrows():
        betas, ses, used = [], [], []
        for a in anc:
            orr = _num(r.get(f"{a}_OR"))
            lo = _num(r.get(f"{a}_CI_LO_OR"))
            hi = _num(r.get(f"{a}_CI_HI_OR"))
            pa = _num(r.get(f"{a}_P"))
            if not np.isfinite(orr) or orr <= 0:
                continue
            b = np.log(orr)
            se = np.nan
            if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
                se = (np.log(hi) - np.log(lo)) / (2 * 1.959964)
            elif np.isfinite(pa) and 0 < pa < 1 and b != 0:
                # recover the SE from the stratum's own p-value; several strata
                # report a p and an OR but no interval
                z = stats.norm.isf(pa / 2)
                if z > 0:
                    se = abs(b) / z
            if not np.isfinite(se) or se <= 0:
                continue
            betas.append(b)
            ses.append(se)
            used.append(a)
        if len(betas) < 2:
            continue
        b = np.asarray(betas)
        w = 1.0 / np.asarray(ses) ** 2
        b_meta = float(np.sum(w * b) / np.sum(w))
        se_meta = float(np.sqrt(1.0 / np.sum(w)))
        z_meta = b_meta / se_meta
        p_meta = 2 * stats.norm.sf(abs(z_meta))
        Q = float(np.sum(w * (b - b_meta) ** 2))
        p_Q = stats.chi2.sf(Q, len(b) - 1)
        I2 = max(0.0, 100 * (Q - (len(b) - 1)) / Q) if Q > 0 else 0.0
        # A fixed-effect meta-analysis assumes one true effect across ancestry
        # groups. Eight of these associations reject that, so the random-effects
        # estimate is the one to quote for them: DerSimonian-Laird tau-squared,
        # which widens the interval by exactly the excess heterogeneity.
        k_anc = len(b)
        c_dl = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2 = max(0.0, (Q - (k_anc - 1)) / c_dl) if c_dl > 0 else 0.0
        w_re = 1.0 / (np.asarray(ses) ** 2 + tau2)
        b_re = float(np.sum(w_re * b) / np.sum(w_re))
        se_re = float(np.sqrt(1.0 / np.sum(w_re)))
        p_re = 2 * stats.norm.sf(abs(b_re / se_re))
        or_pooled = _num(r.get("OR"))
        b_pooled = np.log(or_pooled) if np.isfinite(or_pooled) and or_pooled > 0 \
            else np.nan
        mrows.append(dict(
            Phenotype=r["Phenotype"], inv_label=r["inv_label"],
            Inversion=r["Inversion"], p_multiancestry=r["p"],
            q_global=r["q_global"], or_multiancestry=r.get("OR"),
            beta_multiancestry=b_pooled,
            beta_within_ancestry_meta=b_meta, se_within_ancestry_meta=se_meta,
            p_within_ancestry_meta=p_meta,
            beta_within_ancestry_random=b_re, se_within_ancestry_random=se_re,
            p_within_ancestry_random=p_re, tau2=tau2,
            shrinkage_ratio=(b_meta / b_pooled) if np.isfinite(b_pooled)
            and b_pooled != 0 else np.nan,
            cochran_Q=Q, p_heterogeneity=p_Q, I2=I2,
            n_ancestries=len(b), ancestries=",".join(used)))
    meta = pd.DataFrame(mrows)
    if len(meta):
        meta = meta.sort_values("p_multiancestry")
        meta.to_csv(os.path.join(_DATA, "phewas_within_ancestry_meta.tsv"),
                    sep="\t", index=False)
        sig = meta[meta["q_global"] < 0.05]
        print(f"\n  {len(meta)} associations have >=2 ancestry-specific estimates; "
              f"{len(sig)} of them are family-significant")
        if len(sig):
            print(f"  median shrinkage of the within-ancestry effect relative to "
                  f"the pooled effect: {sig['shrinkage_ratio'].median():.3f} "
                  f"(1.0 = no shrinkage, 0 = signal fully explained by "
                  f"between-ancestry differences)")
            print(f"  same direction as pooled: "
                  f"{int((sig['shrinkage_ratio'] > 0).sum())}/{len(sig)}")
            print(f"  within-ancestry meta p < 0.05: "
                  f"{int((sig['p_within_ancestry_meta'] < 0.05).sum())}/{len(sig)}"
                  f"  (random effects: "
                  f"{int((sig['p_within_ancestry_random'] < 0.05).sum())}/{len(sig)})")
            print(f"  median random-effects shrinkage: "
                  f"{(sig['beta_within_ancestry_random'] / sig['beta_multiancestry']).median():.3f}")
            print(f"  significant between-ancestry heterogeneity (p_Q < 0.05): "
                  f"{int((sig['p_heterogeneity'] < 0.05).sum())}/{len(sig)}")
            show = sig[["Phenotype", "inv_label", "or_multiancestry",
                        "beta_multiancestry", "beta_within_ancestry_meta",
                        "shrinkage_ratio", "p_within_ancestry_meta",
                        "I2", "p_heterogeneity", "n_ancestries"]].head(20).copy()
            for c in ("p_within_ancestry_meta", "p_heterogeneity"):
                show[c] = show[c].map(lambda v: f"{v:.3g}")
            for c in ("beta_multiancestry", "beta_within_ancestry_meta",
                      "shrinkage_ratio", "or_multiancestry", "I2"):
                show[c] = show[c].map(lambda v: f"{v:.3f}")
            print(show.to_string(index=False))
    else:
        print("  no ancestry-stratified estimates available in this table")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(18.2, 9.2))
    gs = fig.add_gridspec(2, 4, hspace=.42, wspace=.34)

    # --- A: a QQ whose points are the INVERSIONS. The seven loci sit on six
    # --- chromosomes and are the independent replicates here; phecodes are not,
    # --- so they cannot supply a calibrated null. Each inversion contributes one
    # --- statistic -- its mean rank across the phenome, standardised against the
    # --- joint permutation null -- and the seven are plotted against the null's
    # --- own order statistics, which carry the compositional dependence between
    # --- them that a marginal normal quantile would ignore.
    ax = fig.add_subplot(gs[0, :3])
    null_joint = _rank_null_joint(ranks.to_numpy(), RNG_SEED)
    null_z = (null_joint - null_joint.mean()) / null_joint.std()
    null_sorted = np.sort(null_z, axis=1)
    exp_order = null_sorted.mean(axis=0)
    band_lo = np.percentile(null_sorted, 2.5, axis=0)
    band_hi = np.percentile(null_sorted, 97.5, axis=0)

    obs_z = ((rank_tab.set_index("inversion")["mean_rank"]
              .reindex(wide.columns).to_numpy() - null_joint.mean())
             / null_joint.std())
    order_idx = np.argsort(obs_z)
    obs_sorted = obs_z[order_idx]
    labels_sorted = [INV_LABEL.get(c, c) for c in
                     np.asarray(wide.columns)[order_idx]]

    ax.fill_between(exp_order, band_lo, band_hi, color="#dddddd", lw=0,
                    label="95% permutation band")
    lim = [min(exp_order.min(), obs_sorted.min()) - .6,
           max(exp_order.max(), obs_sorted.max()) + .6]
    ax.plot(lim, lim, color="#555555", lw=1)
    ax.scatter(exp_order, obs_sorted, s=70, color="#C2601F", zorder=4,
               edgecolor="white", linewidth=.8)
    for xe, yo, lab in zip(exp_order, obs_sorted, labels_sorted):
        ax.annotate(lab, (xe, yo), textcoords="offset points", xytext=(8, -3),
                    fontsize=9)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("expected standardised mean rank (permutation order statistic)")
    ax.set_ylabel("observed standardised mean rank")
    ax.set_title("A  QQ over inversions, stratified by phenotype\n"
                 "each point is one inversion; negative = ranks ahead of the "
                 "others", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # --- B: the rank distribution, which is what panel A is testing
    ax = fig.add_subplot(gs[0, 3])
    inv_order = list(wide.columns)
    palette = ["#C2601F", "#3B5BA5", "#2A7360", "#8C5AA8", "#B0913B",
               "#4E8FA8", "#A8342A"]
    width = 0.8 / len(inv_order)
    xs = np.arange(1, k_inv + 1)
    for i, (inv, col) in enumerate(zip(inv_order, palette)):
        r = np.round(ranks[inv].to_numpy())
        frac = np.array([(r == j).mean() for j in xs])
        ax.bar(xs + (i - len(inv_order) / 2) * width, frac, width=width,
               color=col, label=INV_LABEL.get(inv, inv))
    ax.axhline(1 / k_inv, color="#222222", ls="--", lw=1.2)
    ax.text(k_inv - .2, 1 / k_inv * 1.04, "uniform", fontsize=8, ha="right")
    ax.set_xticks(xs)
    ax.set_xlabel("rank among the inversions (1 = smallest p)")
    ax.set_ylabel("fraction of phenotypes")
    ax.set_title("B  Rank distribution", loc="left", fontsize=11.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    # --- C: lambda computed DOWN the phenotypes -> one value per inversion
    ax = fig.add_subplot(gs[1, 0])
    sub_c = sub.sort_values("lambda_gc")
    ypos = np.arange(len(sub_c))
    ax.errorbar(sub_c["lambda_gc"], ypos,
                xerr=[sub_c["lambda_gc"] - sub_c["lambda_lo"],
                      sub_c["lambda_hi"] - sub_c["lambda_gc"]],
                fmt="o", ms=6, lw=1.1, color="#3B5BA5", ecolor="#9fb2d4",
                capsize=2)
    ax.axvline(1.0, color="#444444", ls=":", lw=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(sub_c["label"], fontsize=8)
    ax.set_xlabel("$\\lambda_{GC}$ over that inversion's 1,053 phenotypes",
                  fontsize=9)
    ax.set_title("C  Inflation attributed to the locus\none value per inversion",
                 fontsize=11.5, loc="left")

    # --- D: lambda computed ACROSS the inversions -> one value per phenotype
    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, max(3.5, pheno_tab["lambda_gc"].max() * 1.02), 45)
    ax.hist(null_lam, bins=bins, density=True, color="#cccccc",
            label=f"expected with no inflation (k={k})")
    ax.hist(pheno_tab["lambda_gc"], bins=bins, density=True, histtype="step",
            color="#cc3311", lw=1.6, label="observed")
    ax.axvline(thresh, color="#444444", ls=":", lw=1)
    ax.set_xlabel("$\\lambda_{GC}$ over that phenotype's 7 inversions",
                  fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_title(f"D  Inflation attributed to the phenotype\none value per "
                 f"phenotype; {n_exceed} of {len(pheno_tab)} exceed Bonferroni",
                 fontsize=11.5, loc="left")
    ax.legend(frameon=False, fontsize=7)

    # --- E: is the locus-level inflation an ancestry effect?
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(sub["dosage_fst_between_ancestry"], sub["lambda_gc"], s=55,
               color="#3B5BA5", edgecolor="white", zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(r["label"], (r["dosage_fst_between_ancestry"],
                                 r["lambda_gc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(1.0, color="#999999", ls=":", lw=1)
    ax.set_xlabel("between-ancestry dosage $F_{ST}$", fontsize=9)
    ax.set_ylabel("$\\lambda_{GC}$ (locus-level)", fontsize=9)
    ax.set_title(f"E  Is the locus inflation an ancestry effect?\n"
                 f"Spearman $\\rho$ = {rho:+.2f} (p = {prho:.2f}); "
                 f"stratification predicts $\\rho$ < 0 with $\\lambda$ high",
                 fontsize=11.5, loc="left")

    # --- E: within- vs cross-ancestry effects for the reported hits
    ax = fig.add_subplot(gs[1, 3])
    if len(meta):
        sgn = meta[meta["q_global"] < 0.05]
        ax.errorbar(sgn["beta_multiancestry"], sgn["beta_within_ancestry_meta"],
                    yerr=1.96 * sgn["se_within_ancestry_meta"], fmt="o",
                    ms=5, lw=.7, color="#cc3311", ecolor="#e8a99a",
                    capsize=0, alpha=.9)
        lim = np.nanmax(np.abs(np.r_[sgn["beta_multiancestry"],
                                     sgn["beta_within_ancestry_meta"]])) * 1.25
        ax.plot([-lim, lim], [-lim, lim], color="#888888", lw=.8)
        ax.axhline(0, color="#cccccc", lw=.8)
        ax.axvline(0, color="#cccccc", lw=.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_xlabel("pooled multi-ancestry $\\log$OR", fontsize=9)
    ax.set_ylabel("within-ancestry meta $\\log$OR", fontsize=9)
    ax.set_title("F  Reported hits: within- vs\ncross-ancestry effect",
                 fontsize=11.5, loc="left")

    fig.suptitle("Genomic-control calibration of the inversion PheWAS",
                 fontsize=13, y=.995)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"\nWrote {OUT_INV}\n      {OUT_PHENO}\n      {OUT_HITS}\n"
          f"      {OUT_PDF} / {OUT_PNG}")


if __name__ == "__main__":
    main()
