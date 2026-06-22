#!/usr/bin/env python3
"""
Reviewer 2 #1 -- Calibration of the CDS-conservation GLM.

The reviewer's concern: the CDS-conservation test (proportion of identical
codons, direct vs inverted) may be confounded because inverted haplotypes are
less diverse (more recent, smaller Ne), which alone could inflate apparent
conservation. The reviewer asks us to (a) support the test's POWER under these
conditions and (b) validate that the GLM p-values are RELIABLE / well calibrated.

Rather than building a second, different forward simulator (which the authors
explicitly want to avoid), we calibrate the EXISTING model via a permutation /
label-shuffling null:

  * The fitted dataset (see stats/CDS_identical_model.py) is a *matched* design:
    after the "both orientations present" filter, every
    (inversion, CDS, recurrence) stratum has exactly one Direct row and one
    Inverted row. Permuting the Direct/Inverted label WITHIN each stratum
    therefore (i) breaks any true orientation -> conservation association while
    (ii) preserving every other feature of the data: the per-CDS conservation
    values, the inversion clustering used for cluster-robust SEs, the
    recurrence structure, the covariates (log length / #seqs / #sites move with
    the row, not the label), and the lower diversity of inverted haplotypes is
    *not* relevant because the label is what is shuffled, not the values.

  * Under this null the true orientation effect is zero by construction. We
    refit the SAME GLM (binomial, freq_weights = n_pairs, cluster-robust by
    inversion) on each shuffle and record the orientation contrast statistic.
    This yields:
       - an empirical (permutation) p-value for the observed effect, and
       - a null distribution of the GLM's *parametric* p-values, whose QQ plot
         against Uniform(0,1) directly shows whether the Wald p-values are
         well calibrated, conservative, or anti-conservative.

  * Power (optional): using the SAME refit machinery, we inject a known
    conservation difference into the inverted label and measure the detection
    rate at alpha = 0.05 -- no new generative model, just a shift applied to the
    response of whichever row currently carries the inverted label.

Outputs (written to the current directory, matching the figure pipeline):
  - cds_conservation_calibration.tsv  (parametric vs permutation p-values,
                                       calibration summary, optional power)
  - cds_conservation_calibration.pdf  (QQ of null parametric p-values +
                                       permutation null of the test statistic)

Run from the repository root (the figure pipeline does this for you):
    python stats/cds_conservation_calibration.py
"""

import os
import sys
import math
import warnings

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, kstest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the EXACT model construction / data loading from the primary analysis so
# that calibration is testing the same estimator the paper reports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from stats.CDS_identical_model import load_data, fit_glm_binom  # noqa: E402
except Exception:  # pragma: no cover - fallback when run as a loose script
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    from CDS_identical_model import load_data, fit_glm_binom  # type: ignore

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_SEED = 2027
N_PERM = 2000          # permutations for the calibration / empirical p-value
N_POWER = 500          # permutations per injected effect size for the power curve
POWER_EFFECTS_LOGIT = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)  # logit shift on inverted
ALPHA = 0.05

STRATUM_KEYS = ["inv_id", "cds_id", "consensus"]

OUT_TSV = "cds_conservation_calibration.tsv"
OUT_PDF = "cds_conservation_calibration.pdf"

# The orientation parameter in the treatment-coded formula. This is the
# direct-vs-inverted main effect that the CDS-conservation test turns on.
ORIENT_PARAM = "C(phy_group)[T.1]"
INTERACTION_PARAM = "C(consensus)[T.1]:C(phy_group)[T.1]"


# --------------------------------------------------------------------------- #
# Test statistic from a fitted model
# --------------------------------------------------------------------------- #

def orientation_stat(res) -> dict:
    """
    Extract the orientation test from a fitted GLM result.

    We report the Wald z and parametric (two-sided) p for the Direct->Inverted
    main effect. We also report a *global* orientation Wald chi-square that
    jointly tests the main effect AND its interaction with recurrence, because
    the conservation claim concerns orientation overall (in both recurrence
    classes). The global statistic is the primary calibration target since it is
    invariant to the arbitrary reference recurrence level.
    """
    params = res.params
    cov = res.cov_params()

    out = {}

    # --- main-effect Wald (orientation at the reference recurrence level) ---
    if ORIENT_PARAM in params.index:
        b = float(params[ORIENT_PARAM])
        se = float(np.sqrt(cov.loc[ORIENT_PARAM, ORIENT_PARAM]))
        z = b / se if se > 0 else np.nan
        out["beta_orient"] = b
        out["z_orient"] = z
        out["p_orient"] = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan
    else:
        out["beta_orient"] = np.nan
        out["z_orient"] = np.nan
        out["p_orient"] = np.nan

    # --- global orientation Wald chi-square (main effect + interaction) ---
    orient_terms = [t for t in (ORIENT_PARAM, INTERACTION_PARAM) if t in params.index]
    if orient_terms:
        bvec = params[orient_terms].values.astype(float)
        cmat = cov.loc[orient_terms, orient_terms].values.astype(float)
        try:
            chi2 = float(bvec @ np.linalg.solve(cmat, bvec))
            dfree = len(orient_terms)
            from scipy.stats import chi2 as chi2_dist
            p_global = float(chi2_dist.sf(chi2, dfree))
        except np.linalg.LinAlgError:
            chi2, dfree, p_global = np.nan, len(orient_terms), np.nan
        out["wald_chi2_orient"] = chi2
        out["wald_df_orient"] = dfree
        out["p_orient_global"] = p_global
    else:
        out["wald_chi2_orient"] = np.nan
        out["wald_df_orient"] = 0
        out["p_orient_global"] = np.nan

    return out


def fit_and_stat(df: pd.DataFrame, include_covariates: bool) -> dict:
    """Fit the primary GLM and return the orientation test statistics."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_glm_binom(df, include_covariates=include_covariates)
    return orientation_stat(res)


# --------------------------------------------------------------------------- #
# Permutation machinery (label shuffling within matched strata)
# --------------------------------------------------------------------------- #

def assert_matched_design(df: pd.DataFrame) -> None:
    """Confirm every stratum has exactly one Direct and one Inverted row."""
    g = df.groupby(STRATUM_KEYS)["phy_group"]
    sizes = g.size()
    sums = g.sum()  # phy_group is 0/1, so the sum counts the inverted rows
    bad = sizes[(sizes != 2)]
    if len(bad):
        raise ValueError(
            f"Matched-design assumption violated: {len(bad)} strata do not have "
            f"exactly 2 rows. Permutation calibration assumes the same paired "
            f"design as the GLM's 'both orientations present' filter."
        )
    bad_sum = sums[(sums != 1)]
    if len(bad_sum):
        raise ValueError(
            f"Matched-design assumption violated: {len(bad_sum)} strata do not "
            f"have exactly one Direct and one Inverted row."
        )


def permute_orientation(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Return a copy of df with phy_group shuffled WITHIN each matched stratum.

    Because each stratum has exactly one Direct and one Inverted row, the only
    randomization is whether to keep or swap the two labels. We draw an
    independent fair coin per stratum and, when it lands on "swap", flip the two
    phy_group labels in that stratum. Covariates (log_m, log_L, log_k) travel
    with the row and are NOT shuffled -- only the orientation label moves, so any
    real orientation effect is destroyed while everything else (clustering,
    recurrence, diversity, per-CDS conservation) is preserved exactly.
    """
    out = df.copy()
    # Stable ordering so the two rows of each stratum are adjacent and known.
    out = out.sort_values(STRATUM_KEYS + ["phy_group"], kind="mergesort").reset_index(drop=True)
    n_strata = len(out) // 2
    swap = rng.integers(0, 2, size=n_strata).astype(bool)
    pg = out["phy_group"].to_numpy().copy()
    # Rows come in (Direct=0, Inverted=1) pairs after the sort above.
    for i in np.nonzero(swap)[0]:
        pg[2 * i], pg[2 * i + 1] = pg[2 * i + 1], pg[2 * i]
    out["phy_group"] = pg
    return out


def run_permutation_null(df: pd.DataFrame, include_covariates: bool,
                         n_perm: int, seed: int) -> pd.DataFrame:
    """Refit the GLM on n_perm orientation-shuffled datasets."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_perm):
        dperm = permute_orientation(df, rng)
        try:
            s = fit_and_stat(dperm, include_covariates)
        except Exception as exc:  # a refit may occasionally fail to converge
            s = {"fit_error": str(exc)}
        s["perm"] = k
        rows.append(s)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Optional power check (reuses the same refit; injects a known logit shift)
# --------------------------------------------------------------------------- #

def inject_and_test(df: pd.DataFrame, delta_logit: float, include_covariates: bool,
                   rng: np.random.Generator) -> dict:
    """
    Shuffle orientation, then add a KNOWN logit shift to whichever row now
    carries the inverted label, and rebuild its identical-pair count so the GLM
    sees a genuine orientation signal of known size. Detection at alpha gives
    the empirical power. This reuses the existing estimator -- no new simulator.
    """
    d = permute_orientation(df, rng)
    inv_mask = d["phy_group"] == 1
    # current per-row logit of the identical-pair proportion (regularized)
    p = (d["y"].to_numpy() + 0.5) / (d["n"].to_numpy() + 1.0)
    lp = np.log(p / (1.0 - p))
    lp_new = lp.copy()
    lp_new[inv_mask.to_numpy()] += delta_logit
    p_new = 1.0 / (1.0 + np.exp(-lp_new))
    y_new = np.rint(p_new * d["n"].to_numpy()).astype(int)
    y_new = np.clip(y_new, 0, d["n"].to_numpy())
    d = d.copy()
    d["y"] = y_new
    d["prop"] = d["y"] / d["n"]
    return fit_and_stat(d, include_covariates)


def run_power_curve(df: pd.DataFrame, include_covariates: bool, effects, n_each: int,
                   seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 777)
    rows = []
    for delta in effects:
        det_main = 0
        det_global = 0
        n_ok = 0
        for _ in range(n_each):
            try:
                s = inject_and_test(df, delta, include_covariates, rng)
            except Exception:
                continue
            n_ok += 1
            if np.isfinite(s.get("p_orient", np.nan)) and s["p_orient"] < ALPHA:
                det_main += 1
            if np.isfinite(s.get("p_orient_global", np.nan)) and s["p_orient_global"] < ALPHA:
                det_global += 1
        rows.append({
            "delta_logit": delta,
            "n_reps": n_ok,
            "power_main_effect": det_main / n_ok if n_ok else np.nan,
            "power_global": det_global / n_ok if n_ok else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Calibration summary + plotting
# --------------------------------------------------------------------------- #

def empirical_p(null_stat: np.ndarray, observed: float) -> float:
    """Two-sided-style empirical p using a one-sided upper tail on a chi-square
    style statistic (larger = more extreme). Add-one correction."""
    null_stat = null_stat[np.isfinite(null_stat)]
    if null_stat.size == 0:
        return np.nan
    return (1.0 + np.sum(null_stat >= observed)) / (1.0 + null_stat.size)


def calibration_plot(null_p_global: np.ndarray, null_p_main: np.ndarray,
                    null_chi2: np.ndarray, observed_chi2: float, outfile: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    # ---- Panel A: QQ of null parametric p-values vs Uniform(0,1) ----
    ax = axes[0]
    for pvals, label, color in (
        (null_p_global, "Global orientation Wald", "#8c2d7e"),
        (null_p_main, "Main-effect Wald", "#1f3b78"),
    ):
        p = np.sort(pvals[np.isfinite(pvals)])
        if p.size == 0:
            continue
        expected = (np.arange(1, p.size + 1) - 0.5) / p.size
        ax.plot(expected, p, marker="o", ms=2.5, lw=0.8, alpha=0.8,
                color=color, label=label)
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1.0)
    ax.set_xlabel("Expected p-value quantile (Uniform)")
    ax.set_ylabel("Observed parametric p-value (under null)")
    ax.set_title("A  GLM p-value calibration under the\norientation-label permutation null")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # ---- Panel B: permutation null of the orientation Wald statistic ----
    ax = axes[1]
    finite = null_chi2[np.isfinite(null_chi2)]
    if finite.size:
        ax.hist(finite, bins=40, color="#bdbdd0", edgecolor="white", alpha=0.9)
    if np.isfinite(observed_chi2):
        ax.axvline(observed_chi2, color="#FF6F00", lw=2.0,
                   label=f"observed = {observed_chi2:.2f}")
    ax.set_xlabel("Orientation Wald $\\chi^2$ statistic")
    ax.set_ylabel("Permutations")
    ax.set_title("B  Permutation null of the\norientation test statistic")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    np.random.seed(RANDOM_SEED)
    print(">>> Reviewer 2 #1: CDS-conservation GLM calibration")
    print(">>> Loading data (same loader / filters as CDS_identical_model.py) ...")
    df = load_data()
    print(f"Rows after matched-orientation filter: {len(df)}; "
          f"inversions: {df['inv_id'].nunique()}; "
          f"strata: {df.groupby(STRATUM_KEYS).ngroups}")

    assert_matched_design(df)
    print("Matched-design check passed: every stratum has 1 Direct + 1 Inverted row.")

    summary_rows = []

    for include_cov, model_label in ((True, "adjusted"), (False, "no_covariates")):
        print(f"\n>>> Model: {model_label} "
              f"({'with covariates' if include_cov else 'no covariates'})")

        # --- observed fit ---
        obs = fit_and_stat(df, include_cov)
        print(f"  Observed: beta_orient={obs['beta_orient']:.4f}, "
              f"z={obs['z_orient']:.3f}, parametric p(main)={obs['p_orient']:.4g}, "
              f"Wald chi2(orient)={obs['wald_chi2_orient']:.3f} "
              f"(df={obs['wald_df_orient']}), parametric p(global)={obs['p_orient_global']:.4g}")

        # --- permutation null ---
        print(f"  Running {N_PERM} orientation-label permutations ...")
        null_df = run_permutation_null(df, include_cov, N_PERM, RANDOM_SEED)
        n_fail = int(null_df.get("fit_error", pd.Series(dtype=object)).notna().sum()) \
            if "fit_error" in null_df.columns else 0
        if n_fail:
            print(f"  NOTE: {n_fail}/{N_PERM} permutation refits failed to converge "
                  f"(excluded from the null).")

        null_chi2 = null_df["wald_chi2_orient"].to_numpy(dtype=float)
        null_p_global = null_df["p_orient_global"].to_numpy(dtype=float)
        null_p_main = null_df["p_orient"].to_numpy(dtype=float)
        null_z = null_df["z_orient"].to_numpy(dtype=float)

        # --- empirical p-values for the observed effect ---
        emp_p_global = empirical_p(null_chi2, obs["wald_chi2_orient"])
        # main-effect: two-sided empirical p on |z|
        finite_z = null_z[np.isfinite(null_z)]
        emp_p_main = (1.0 + np.sum(np.abs(finite_z) >= abs(obs["z_orient"]))) / \
            (1.0 + finite_z.size) if finite_z.size and np.isfinite(obs["z_orient"]) else np.nan

        # --- calibration of the parametric p-values under the null ---
        # If the Wald p-values are well calibrated, the null p-values are ~U(0,1):
        #   - KS test against Uniform(0,1)
        #   - realized type-I error at nominal alpha (should be ~alpha)
        def ks_unif(p):
            p = p[np.isfinite(p)]
            if p.size < 5:
                return np.nan, np.nan
            d, pv = kstest(p, "uniform")
            return d, pv

        ks_d_g, ks_p_g = ks_unif(null_p_global)
        ks_d_m, ks_p_m = ks_unif(null_p_main)

        def realized_alpha(p, a=ALPHA):
            p = p[np.isfinite(p)]
            return float(np.mean(p < a)) if p.size else np.nan

        ra_g = realized_alpha(null_p_global)
        ra_m = realized_alpha(null_p_main)

        print(f"  Calibration (global orientation Wald): "
              f"realized type-I at alpha={ALPHA}: {ra_g:.4f} "
              f"(KS vs Uniform: D={ks_d_g:.3f}, p={ks_p_g:.3g})")
        print(f"  Calibration (main-effect Wald):        "
              f"realized type-I at alpha={ALPHA}: {ra_m:.4f} "
              f"(KS vs Uniform: D={ks_d_m:.3f}, p={ks_p_m:.3g})")
        verdict_g = ("well calibrated" if abs(ra_g - ALPHA) <= 0.02 else
                     ("ANTI-conservative" if ra_g > ALPHA else "conservative"))
        print(f"  -> Global Wald p-values appear {verdict_g} "
              f"(realized {ra_g:.3f} vs nominal {ALPHA}).")
        print(f"  Observed global: parametric p={obs['p_orient_global']:.4g} vs "
              f"permutation p={emp_p_global:.4g}")

        # --- power curve (optional; same refit machinery) ---
        print(f"  Power curve: injecting logit shifts {POWER_EFFECTS_LOGIT} "
              f"({N_POWER} reps each) ...")
        power_df = run_power_curve(df, include_cov, POWER_EFFECTS_LOGIT, N_POWER, RANDOM_SEED)
        for _, r in power_df.iterrows():
            print(f"    delta_logit={r['delta_logit']:.2f}: "
                  f"power(main)={r['power_main_effect']:.3f}, "
                  f"power(global)={r['power_global']:.3f} (n={int(r['n_reps'])})")

        # --- plot (only for the adjusted model, the headline analysis) ---
        if include_cov:
            calibration_plot(null_p_global, null_p_main, null_chi2,
                             obs["wald_chi2_orient"], OUT_PDF)
            print(f"  Wrote calibration figure: {OUT_PDF}")

        # --- collect summary rows ---
        summary_rows.append({
            "model": model_label,
            "statistic": "orientation_global_wald",
            "observed_value": obs["wald_chi2_orient"],
            "observed_df": obs["wald_df_orient"],
            "parametric_p": obs["p_orient_global"],
            "permutation_p": emp_p_global,
            "n_perm": int(np.isfinite(null_chi2).sum()),
            "null_realized_typeI_alpha0.05": ra_g,
            "null_ks_uniform_D": ks_d_g,
            "null_ks_uniform_p": ks_p_g,
            "calibration_verdict": verdict_g,
        })
        summary_rows.append({
            "model": model_label,
            "statistic": "orientation_main_effect_wald",
            "observed_value": obs["z_orient"],
            "observed_df": 1,
            "parametric_p": obs["p_orient"],
            "permutation_p": emp_p_main,
            "n_perm": int(np.isfinite(null_z).sum()),
            "null_realized_typeI_alpha0.05": ra_m,
            "null_ks_uniform_D": ks_d_m,
            "null_ks_uniform_p": ks_p_m,
            "calibration_verdict": ("well calibrated" if abs(ra_m - ALPHA) <= 0.02 else
                                    ("anti-conservative" if ra_m > ALPHA else "conservative")),
        })
        # power rows
        for _, r in power_df.iterrows():
            summary_rows.append({
                "model": model_label,
                "statistic": f"power_inject_logit_{r['delta_logit']:.2f}",
                "observed_value": r["delta_logit"],
                "observed_df": np.nan,
                "parametric_p": np.nan,
                "permutation_p": np.nan,
                "n_perm": int(r["n_reps"]),
                "null_realized_typeI_alpha0.05": np.nan,
                "null_ks_uniform_D": np.nan,
                "null_ks_uniform_p": np.nan,
                "calibration_verdict": (
                    f"power_main={r['power_main_effect']:.3f};"
                    f"power_global={r['power_global']:.3f}"
                ),
            })

    out = pd.DataFrame(summary_rows)
    out.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nWrote calibration summary: {OUT_TSV}")
    print("Done.")


if __name__ == "__main__":
    main()
