import os
import sys
import csv
import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit, logit
from scipy.stats import norm

# ---------- helpers ----------

CAT_LABELS = {
    (1,0): "Recurrent/Direct",
    (1,1): "Recurrent/Inverted",
    (0,0): "Single/Direct",
    (0,1): "Single/Inverted",
}

def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    # enforce monotone non-decreasing from the right
    for i in range(m-2, -1, -1):
        q[i] = min(q[i], q[i+1])
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out.tolist()

def read_first_n_sites_from_pairs(filename: str) -> int:
    path = f"pairs_CDS__{filename}.tsv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing per-file pairs TSV: {path}")
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr, None)
        if not header or "n_sites" not in header:
            raise ValueError(f"'n_sites' column missing in {path}")
        idx = header.index("n_sites")
        row = next(rdr, None)
        if row is None:
            raise ValueError(f"No data rows in {path}")
        return int(row[idx])

def load_data() -> pd.DataFrame:
    path = "cds_identical_proportions.tsv"
    if not os.path.exists(path):
        sys.exit("ERROR: cds_identical_proportions.tsv not found.")
    df = pd.read_csv(path, sep="\t", dtype=str)

    # types
    to_int = ["consensus","phy_group","n_sequences","n_pairs","n_identical_pairs","inv_start","inv_end","inv_exact_match"]
    for c in to_int:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # exact-only
    if "inv_exact_match" not in df.columns:
        sys.exit("ERROR: inv_exact_match column missing (needed for exact-only).")
    df = df[df["inv_exact_match"] == 1].copy()
    # keep informative
    df = df[(df["n_pairs"] > 0) & (df["n_identical_pairs"].notna()) & (df["n_sequences"] >= 2)]
    if df.empty:
        sys.exit("No rows after exact-only & n_pairs>0 filter.")

    # ids
    if "transcript_id" not in df.columns:
        sys.exit("ERROR: transcript_id missing.")
    df["cds_id"] = df["transcript_id"].astype(str)
    df["inv_id"] = df.apply(lambda r: f"{r['chr']}:{int(r['inv_start'])}-{int(r['inv_end'])}", axis=1)

    # counts & covariates
    df["y"] = df["n_identical_pairs"].astype(int)
    df["n"] = df["n_pairs"].astype(int)
    df["prop"] = df["y"] / df["n"]

    # inversion length
    df["inv_len"] = (df["inv_end"].astype(int) - df["inv_start"].astype(int)).abs() + 1
    df["log_L"] = np.log(df["inv_len"])
    df["log_k"] = np.log(df["n_sequences"].astype(int))

    # n_sites from per-file pairs
    ns = []
    for fn in df["filename"].astype(str):
        ns.append(read_first_n_sites_from_pairs(fn))
    df["n_sites"] = ns
    df["log_m"] = np.log(df["n_sites"].astype(int))

    return df

def equal_weight_per_inversion(df: pd.DataFrame) -> pd.Series:
    counts = df["inv_id"].value_counts()
    w = df["inv_id"].map(lambda x: 1.0 / counts[x])
    return w / w.sum()

def fe_and_cov(res):
    return res.params, res.cov_params()

def build_L_for_category(fe: pd.Series, cons: int, grp: int,
                         mean_log_m: float = 0.0, mean_log_L: float = 0.0, mean_log_k: float = 0.0) -> pd.Series:
    # Formula uses treatment coding:
    # Intercept, C(consensus)[T.1], C(phy_group)[T.1], C(consensus)[T.1]:C(phy_group)[T.1], and covariates if present.
    L = pd.Series(0.0, index=fe.index)
    if "Intercept" in L.index: L["Intercept"] = 1.0
    if cons == 1 and "C(consensus)[T.1]" in L.index: L["C(consensus)[T.1]"] = 1.0
    if grp == 1 and "C(phy_group)[T.1]" in L.index: L["C(phy_group)[T.1]"] = 1.0
    if cons == 1 and grp == 1 and "C(consensus)[T.1]:C(phy_group)[T.1]" in L.index:
        L["C(consensus)[T.1]:C(phy_group)[T.1]"] = 1.0
    if "log_m" in L.index: L["log_m"] = mean_log_m
    if "log_L" in L.index: L["log_L"] = mean_log_L
    if "log_k" in L.index: L["log_k"] = mean_log_k
    return L

def emms_and_pairs(res, df: pd.DataFrame, include_covariates: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fe, cov = fe_and_cov(res)
    # standardization: equal weight per inversion
    w = equal_weight_per_inversion(df)
    mean_log_m = float((w * df["log_m"]).sum()) if include_covariates else 0.0
    mean_log_L = float((w * df["log_L"]).sum()) if include_covariates else 0.0
    mean_log_k = float((w * df["log_k"]).sum()) if include_covariates else 0.0

    cats = [(1,0),(1,1),(0,0),(0,1)]
    Ls, zhat, emm_rows = {}, {}, []
    for c,g in cats:
        L = build_L_for_category(fe, c, g, mean_log_m, mean_log_L, mean_log_k)
        z = float(np.dot(L.values, fe.values))
        v = float(L.values @ cov.values @ L.values.T)
        se_z = math.sqrt(max(v, 0.0))
        p = expit(z)
        lcl = expit(z - 1.96*se_z)
        ucl = expit(z + 1.96*se_z)
        emm_rows.append({
            "consensus": c,
            "phy_group": g,
            "category": CAT_LABELS[(c,g)],
            "z_hat": z, "se_z": se_z,
            "p_hat": p, "p_lcl95": lcl, "p_ucl95": ucl
        })
        Ls[(c,g)] = L
        zhat[(c,g)] = z
    emm = pd.DataFrame(emm_rows).sort_values(["consensus","phy_group"], ascending=[False,True])

    # pairwise contrasts + BH-FDR
    from itertools import combinations
    rows, pvals = [], []
    for a,b in combinations(cats, 2):
        La, Lb = Ls[a], Ls[b]
        Ld = La - Lb
        dz = float(np.dot(Ld.values, fe.values))
        vd = float(Ld.values @ cov.values @ Ld.values.T)
        se = math.sqrt(max(vd,0.0))
        z = dz/se if se>0 else np.nan
        p = 2*(1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        pvals.append(p)

        pa, pb = expit(zhat[a]), expit(zhat[b])
        dP = pa - pb
        gvec = (pa*(1-pa))*La.values - (pb*(1-pb))*Lb.values
        var_dP = float(gvec @ cov.values @ gvec.T)
        se_dP = math.sqrt(max(var_dP,0.0))
        rows.append({
            "A": CAT_LABELS[a], "B": CAT_LABELS[b],
            "diff_logit": dz, "se_logit": se, "z_value": z, "p_value": p,
            "diff_prob": dP, "diff_prob_lcl95": dP - 1.96*se_dP, "diff_prob_ucl95": dP + 1.96*se_dP
        })
    qvals = bh_fdr([r["p_value"] for r in rows])
    for r,q in zip(rows,qvals):
        r["q_value_fdr"] = q
    pw = pd.DataFrame(rows).sort_values("q_value_fdr")
    return emm, pw

def fit_glm_binom(df: pd.DataFrame, include_covariates: bool):
    if include_covariates:
        formula = "prop ~ C(consensus) * C(phy_group) + log_m + log_L + log_k"
    else:
        formula = "prop ~ C(consensus) * C(phy_group)"
    model = smf.glm(formula, data=df, family=sm.families.Binomial(), freq_weights=df["n"])
    # cluster-robust by inversion
    res = model.fit(cov_type="cluster", cov_kwds={"groups": df["inv_id"]})
    return res

def main():
    print(">>> Loading inputs...")
    df = load_data()
    print(f"Rows (exact matches, n_pairs>0): {len(df)}")

    # ---- Adjusted model (with covariates) ----
    print("\n>>> GLM Binomial (with covariates), cluster-robust by inversion ...")
    res_full = fit_glm_binom(df, include_covariates=True)
    print(res_full.summary())

    emm_full, pw_full = emms_and_pairs(res_full, df, include_covariates=True)
    emm_full = emm_full.sort_values("p_hat", ascending=False).reset_index(drop=True)

    print("\n=== Adjusted model: standardized category means (equal weight per inversion) ===")
    for _, r in emm_full.iterrows():
        print(f"{r['category']:<20} p̂={r['p_hat']:.4f}  (95% CI {r['p_lcl95']:.4f}–{r['p_ucl95']:.4f})")

    print("\n=== Adjusted model: pairwise contrasts (BH-FDR) ===")
    for _, r in pw_full.iterrows():
        print(f"{r['A']} vs {r['B']}: Δlogit={r['diff_logit']:.3f} (SE {r['se_logit']:.3f}), "
              f"p={r['p_value']:.4g}, q(FDR)={r['q_value_fdr']:.4g}; "
              f"Δp={r['diff_prob']:.4f} (95% CI {r['diff_prob_lcl95']:.4f}–{r['diff_prob_ucl95']:.4f})")

    # ---- No-covariates model ----
    print("\n>>> GLM Binomial (NO covariates), cluster-robust by inversion ...")
    res_nocov = fit_glm_binom(df, include_covariates=False)
    print(res_nocov.summary())

    emm_nc, pw_nc = emms_and_pairs(res_nocov, df, include_covariates=False)
    emm_nc = emm_nc.sort_values("p_hat", ascending=False).reset_index(drop=True)

    print("\n=== No-covariates model: standardized category means (equal weight per inversion) ===")
    for _, r in emm_nc.iterrows():
        print(f"{r['category']:<20} p̂={r['p_hat']:.4f}  (95% CI {r['p_lcl95']:.4f}–{r['p_ucl95']:.4f})")

    print("\n=== No-covariates model: pairwise contrasts (BH-FDR) ===")
    for _, r in pw_nc.iterrows():
        print(f"{r['A']} vs {r['B']}: Δlogit={r['diff_logit']:.3f} (SE {r['se_logit']:.3f}), "
              f"p={r['p_value']:.4g}, q(FDR)={r['q_value_fdr']:.4g}; "
              f"Δp={r['diff_prob']:.4f} (95% CI {r['diff_prob_lcl95']:.4f}–{r['diff_prob_ucl95']:.4f})")

    # Save
    emm_full.assign(model="adjusted").to_csv("cds_emm_adjusted.tsv", sep="\t", index=False)
    pw_full.assign(model="adjusted").to_csv("cds_pairwise_adjusted.tsv", sep="\t", index=False)
    emm_nc.assign(model="no_covariates").to_csv("cds_emm_nocov.tsv", sep="\t", index=False)
    pw_nc.assign(model="no_covariates").to_csv("cds_pairwise_nocov.tsv", sep="\t", index=False)

    print("\nWrote: cds_emm_adjusted.tsv, cds_pairwise_adjusted.tsv, cds_emm_nocov.tsv, cds_pairwise_nocov.tsv")
    print("Done.")

if __name__ == "__main__":
    main()
