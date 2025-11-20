import io
import gzip
import math
import urllib.request
import numpy as np
import pandas as pd
from scipy import optimize

# =============================================================================
# PHASE 1: SCRIPT CONFIGURATION
# =============================================================================

# Primary PheCodeX mapping file.
PHECODE_MAP_URL = "https://raw.githubusercontent.com/nhgritctran/PheTK/main/src/PheTK/phecode/phecodeX.csv"

# UK Biobank "phenocode" mappings for enrichment.
UKBB_ICD10_MAP_URL = "https://raw.githubusercontent.com/atgu/ukbb_pan_ancestry/refs/heads/master/data/UKB_PHENOME_ICD10_PHECODE_MAP_20200109.txt"
UKBB_ICD9_MAP_URL  = "https://raw.githubusercontent.com/atgu/ukbb_pan_ancestry/refs/heads/master/data/UKB_PHENOME_ICD9_PHECODE_MAP_20200109.txt"

# UK Biobank heritability manifest for further enrichment (bgzip).
H2_MANIFEST_URL = "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/h2_manifest.tsv.bgz"

# Define the name for the final, comprehensive output file.
OUTPUT_FILENAME = "phecodex_with_phenocode_and_h2_mappings.tsv"

# Diagnostics for mapped-but-missing PheCodes:
MISSING_PHECODES_TSV = "phecodes_without_usable_h2.tsv"

# Define the delimiter for joining lists into a single string.
LIST_DELIMITER = ";"

# --- Statistical knobs ---
H2_THRESHOLD = 0.0   # 0% heritability boundary (used for detection test)
FDR_Q        = 0.05   # BH-FDR across phenocodes
USE_QC       = False  # set True to require QC flags (defined_h2 & in_bounds_h2), if available

# REML parametric bootstrap draws for CI/PI (set None or 0 to disable).
REML_BOOTSTRAP_B = 0

# =============================================================================
# Utility helpers (pandas/numpy only)
# =============================================================================

def read_bgz_tsv(url: str, usecols=None) -> pd.DataFrame:
    """
    Robustly read a (b)gzipped TSV from URL without pyarrow/duckdb/scipy.
    Tries pandas with compression='infer' first; falls back to urllib+gzip.
    """
    try:
        return pd.read_csv(
            url, sep="\t", compression="infer", engine="python",
            usecols=(usecols if usecols is None else (lambda c: c in usecols))
        )
    except Exception:
        with urllib.request.urlopen(url) as resp:
            gzdata = io.BytesIO(resp.read())
        with gzip.GzipFile(fileobj=gzdata, mode="rb") as gz:
            return pd.read_csv(
                gz, sep="\t", engine="python",
                usecols=(usecols if usecols is None else (lambda c: c in usecols))
            )

def list_join_safe(x):
    """Join a list of strings with LIST_DELIMITER or return empty string if list/values are missing."""
    if not isinstance(x, (list, tuple, np.ndarray)):
        return ""
    if len(x) == 0:
        return ""
    return LIST_DELIMITER.join(map(str, x))

def acat(pvals: np.ndarray) -> float:
    """
    ACAT combiner for one-sided p-values.
    Robust to dependence; good power for sparse signals.
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.nan
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    tans = np.tan((0.5 - p) * np.pi)
    T = np.mean(tans)
    p_acat = 0.5 - np.arctan(T) / np.pi
    if not np.isfinite(p_acat):
        return np.nan
    return float(np.clip(p_acat, 0.0, 1.0))

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg q-values (FDR). Returns q-values in the original order.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.empty_like(p_sorted)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = (p_sorted[i] * n) / rank
        if val > prev:
            val = prev
        prev = val
        q_sorted[i] = val
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return q

def combine_codes_unique_sorted(series: pd.Series) -> list:
    """Helper to deduplicate and sort ICD code lists pulled from a series."""
    if series is None or series.empty:
        return []
    vals = series.dropna().astype(str).unique()
    return sorted(vals)

# =============================================================================
# REML HIERARCHICAL META (one-shot across populations & phenocodes)
# =============================================================================

def compute_typical_s2(s2, sigma2_e):
    """
    RE-weighted 'typical' measurement variance for a new phenocode observation:
      w_i = 1 / (s2_i + sigma2_e)
      s2_bar = sum_i w_i * s2_i / sum_i w_i

    Parameters
    ----------
    s2 : array-like of measurement variances (SE^2) on analysis scale
    sigma2_e : float >= 0, idiosyncratic residual variance component

    Returns
    -------
    float or np.nan
    """
    import numpy as np

    s2 = np.asarray(s2, dtype=float)
    mask = np.isfinite(s2)
    s2 = s2[mask]
    if s2.size == 0:
        return np.nan
    w = 1.0 / (s2 + float(max(sigma2_e, 0.0)))
    sw = np.sum(w)
    if sw <= 0:
        return np.nan
    return float(np.sum(w * s2) / sw)

def _prep_transform(y, se, transform):
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    return (
        y,
        se,
        {
            "scale": "liability",
            "back": lambda x: x,
            "delta": lambda mu, se_mu: se_mu,
            "clamp_eps": None,
        },
    )

def fit_overall_h2_reml(
    y, s2, pop_ids, phe_ids, *,
    transform="liability",
    bootstrap_B=None,
    random_state=None,
    pi_target="new_pop_new_phe"
):
    """
    REML for hierarchical model with two random effects:
        y_i = mu + u_{pop(i)} + b_{phe(i)} + eps_i
        u_p   ~ N(0, tau2)
        b_j   ~ N(0, omega2)
        eps_i ~ N(0, s2_i + sigma2_e)   (s2_i = known measurement variance)

    Notes / policy choices implemented:
      - No ad-hoc t-approx fallback for CI/PI. If bootstrap_B is None/0, CI/PI are NaN.
      - Transform == "liability": identity; "logit": delta & back-transform used for mu/se.
    """
    import numpy as np
    from scipy import optimize
    from scipy.stats import norm  # only for constants, not used if bootstrap off

    # ---- basic masking & encoding
    y = np.asarray(y, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    pop_ids = np.asarray(pop_ids)
    phe_ids = np.asarray(phe_ids)

    mask = (np.isfinite(y) & np.isfinite(s2) & (s2 > 0) & (~pd.isna(pop_ids)) & (~pd.isna(phe_ids)) & (pop_ids.astype(str) != "") & (phe_ids.astype(str) != ""))
    y = y[mask]; s2 = s2[mask]; pop_ids = pop_ids[mask]; phe_ids = phe_ids[mask]
    n = y.size

    if n == 0:
        return dict(
            mu=np.nan, se_mu=np.nan, ci95_l=np.nan, ci95_u=np.nan,
            tau2=np.nan, omega2=np.nan, sigma2_e=np.nan,
            I2_between_pops=np.nan, I2_between_phecodes=np.nan, I2_idiosyncratic=np.nan,
            pred_int_l=np.nan, pred_int_u=np.nan,
            n_pops_used=0, n_phecodes_used=0, n_obs_used=0
        )

    # ---- analysis scale transform
    y_t, se_t, meta = _prep_transform(y, np.sqrt(s2), transform)
    s2_t = se_t ** 2

    # ---- encode groups
    pop_idx, phe_idx, uniq_pops, uniq_phe = _encode_groups(pop_ids, phe_ids)
    K = len(uniq_pops)
    J = len(uniq_phe)

    # ---- REML objective
    def neg_reml(theta):
        ltau2, lomega2, lsig2 = theta
        tau2 = np.exp(ltau2)
        omega2 = np.exp(lomega2)
        sigma2_e = np.exp(lsig2)

        d = s2_t + sigma2_e + 1e-12
        sol = _woodbury_multi_solvers(d, pop_idx, phe_idx, tau2, omega2)

        Vinv_y   = sol["apply_Vinv"](y_t)
        XtVinvX  = sol["XtVinvX"]()
        XtVinvY  = sol["XtVinv_vec"](y_t)
        yPy      = float(y_t @ Vinv_y - (XtVinvY ** 2) / XtVinvX)

        logdetV  = sol["logdet_V"]()
        logdet_X = np.log(XtVinvX)

        # REML loglik (drop constants)
        ll = -0.5 * (logdetV + logdet_X + yPy)
        return -ll

    x0 = np.array([-4.0, -4.0, -4.0], dtype=float)
    bounds = [(-20.0, 8.0)] * 3

    res = optimize.minimize(
        neg_reml, x0, method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=300, ftol=1e-9)
    )
    ltau2_opt, lomega2_opt, lsig2_opt = res.x
    tau2_hat    = float(np.exp(ltau2_opt))
    omega2_hat  = float(np.exp(lomega2_opt))
    sigma2_e_hat= float(np.exp(lsig2_opt))

    # ---- GLS summaries at optimum
    d = s2_t + sigma2_e_hat + 1e-12
    sol = _woodbury_multi_solvers(d, pop_idx, phe_idx, tau2_hat, omega2_hat)

    XtVinvX  = sol["XtVinvX"]()
    XtVinvY  = sol["XtVinv_vec"](y_t)
    mu_hat_t = XtVinvY / XtVinvX
    var_mu_t = 1.0 / XtVinvX
    se_mu_t  = float(np.sqrt(var_mu_t))

    # Typical measurement variance for prediction
    s2_bar_t = compute_typical_s2(s2_t, sigma2_e_hat)

    # ---- back-transform point & SE to original scale
    mu_point = meta["back"](mu_hat_t) if meta["scale"] == "logit" else float(mu_hat_t)
    se_point = meta["delta"](mu_hat_t, se_mu_t)

    # ---- intervals: bootstrap only; otherwise NaN
    if bootstrap_B is not None and int(bootstrap_B) > 0:
        boot = parametric_bootstrap_overall_2re(
            y_t, s2_t, pop_idx, phe_idx,
            mu_hat_t, tau2_hat, omega2_hat, sigma2_e_hat, s2_bar_t,
            transform=meta["scale"], B=int(bootstrap_B),
            random_state=random_state, pi_target=pi_target
        )
        ci_l, ci_u = np.percentile(boot["mu_draws"], [2.5, 97.5])
        pi_l, pi_u = np.percentile(boot["pred_draws"], [2.5, 97.5])

        # ensure [0,1] on original h2 scale
        ci_l = float(min(1.0, max(0.0, ci_l)))
        ci_u = float(min(1.0, max(0.0, ci_u)))
        pi_l = float(min(1.0, max(0.0, pi_l)))
        pi_u = float(min(1.0, max(0.0, pi_u)))
    else:
        ci_l = ci_u = pi_l = pi_u = float("nan")

    # ---- I^2 decomposition (analysis scale)
    denom = tau2_hat + omega2_hat + sigma2_e_hat + (s2_bar_t if np.isfinite(s2_bar_t) else 0.0)
    if denom <= 0:
        I2_pop = I2_phe = I2_eps = 0.0
    else:
        I2_pop = float(tau2_hat / denom)
        I2_phe = float(omega2_hat / denom)
        I2_eps = float(sigma2_e_hat / denom)

    return dict(
        mu=float(mu_point), se_mu=float(se_point),
        ci95_l=float(ci_l), ci95_u=float(ci_u),
        tau2=float(tau2_hat), omega2=float(omega2_hat), sigma2_e=float(sigma2_e_hat),
        I2_between_pops=I2_pop, I2_between_phecodes=I2_phe, I2_idiosyncratic=I2_eps,
        pred_int_l=float(pi_l), pred_int_u=float(pi_u),
        n_pops_used=int(K), n_phecodes_used=int(J), n_obs_used=int(n)
    )

def _reml_worker(payload):
    """
    Worker used by build_disease_overall_estimates to fit the 2-RE REML model
    for a single disease. Removes misleading legacy aliases.

    Parameters
    ----------
    payload : tuple
        (y, s2, pop, phe, mix_flag, transform_, B, rs, pi_t)

    Returns
    -------
    dict
        Keys include:
          - h2_overall_REML, se_overall_REML, ci95_l_overall_REML, ci95_u_overall_REML
          - tau2_between_pops, omega2_between_phecodes, sigma2_within_idio
          - I2_between_pops, I2_between_phecodes, I2_idiosyncratic
          - pred_int_l, pred_int_u
          - n_pops_used, n_phecodes_used, n_obs_used
          - any_scale_mix_flag
        (No legacy/misleading aliases are included.)
    """
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    y, s2, pop, phe, mix_flag, transform_, B, rs, pi_t = payload

    fit = fit_overall_h2_reml(
        y, s2, pop, phe,
        transform=transform_,
        bootstrap_B=(int(B) if (B is not None and int(B) > 0) else None),
        random_state=rs,
        pi_target=pi_t
    )

    # Return only clearly labeled components (no legacy aliases)
    return dict(
        h2_overall_REML=fit["mu"],
        se_overall_REML=fit["se_mu"],
        ci95_l_overall_REML=fit["ci95_l"],
        ci95_u_overall_REML=fit["ci95_u"],
        tau2_between_pops=fit["tau2"],
        omega2_between_phecodes=fit["omega2"],
        sigma2_within_idio=fit["sigma2_e"],
        I2_between_pops=fit["I2_between_pops"],
        I2_between_phecodes=fit["I2_between_phecodes"],
        I2_idiosyncratic=fit["I2_idiosyncratic"],
        pred_int_l=fit["pred_int_l"],
        pred_int_u=fit["pred_int_u"],
        n_pops_used=fit["n_pops_used"],
        n_phecodes_used=fit["n_phecodes_used"],
        n_obs_used=fit["n_obs_used"],
        any_scale_mix_flag=mix_flag
    )


def parametric_bootstrap_overall_2re(
    y_t, s2_t, pop_idx, phe_idx,
    mu_hat_t, tau2_hat, omega2_hat, sigma2_e_hat, s2_bar_t,
    *, transform="liability", B=2000, random_state=None, pi_target="new_pop_new_phe"
):
    """
    Parametric bootstrap under the 2-RE model (population + PheCode) on the ANALYSIS scale.
    **Fixes scale mixing**: predictive noise is added on analysis scale; results are
    back-transformed only at the end when needed (logit case).

    Simulate:
      u_p ~ N(0, tau2_hat), b_j ~ N(0, omega2_hat),
      eps_i ~ N(0, s2_t[i] + sigma2_e_hat),
      y*_i = mu_hat_t + u_{pop(i)} + b_{phe(i)} + eps_i.
    Refit REML on y* (bootstrap disabled) to get mu*_b (original scale).
    Predictive draw per b on analysis scale; then back-transform if logit.

    Returns
    -------
    dict: {"mu_draws": (B,), "pred_draws": (B,)} on ORIGINAL scale.
    """
    import numpy as np

    rng = np.random.default_rng(random_state)
    K = int(pop_idx.max()) + 1 if pop_idx.size else 0
    J = int(phe_idx.max()) + 1 if phe_idx.size else 0
    n = y_t.size

    mu_draws = np.empty(B, dtype=float)
    pred_draws = np.empty(B, dtype=float)

    # Predictive variance on analysis scale (depends on target)
    if pi_target == "new_pop_new_phe":
        pred_var_t = tau2_hat + omega2_hat + sigma2_e_hat + (s2_bar_t if np.isfinite(s2_bar_t) else 0.0)
    elif pi_target == "new_pop_existing_phe":
        pred_var_t = tau2_hat + sigma2_e_hat + (s2_bar_t if np.isfinite(s2_bar_t) else 0.0)
    else:  # "existing_pop_new_phe"
        pred_var_t = omega2_hat + sigma2_e_hat + (s2_bar_t if np.isfinite(s2_bar_t) else 0.0)
    pred_sd_t = float(np.sqrt(max(0.0, pred_var_t)))

    for b in range(B):
        u = rng.normal(0.0, np.sqrt(max(tau2_hat, 0.0)), size=K) if K > 0 else np.zeros(0, dtype=float)
        v = rng.normal(0.0, np.sqrt(max(omega2_hat, 0.0)), size=J) if J > 0 else np.zeros(0, dtype=float)
        eps = rng.normal(0.0, np.sqrt(s2_t + max(sigma2_e_hat, 0.0)), size=n)

        y_star = mu_hat_t + (u[pop_idx] if K > 0 else 0.0) + (v[phe_idx] if J > 0 else 0.0) + eps

        # Refit (returns mu on ORIGINAL scale)
        fit_b = fit_overall_h2_reml(
            y_star, s2_t, pop_idx, phe_idx,
            transform="liability",
            bootstrap_B=None,  # no nested bootstrap
            random_state=rng, pi_target=pi_target
        )
        mu_b = fit_b["mu"]               # original scale
        mu_draws[b] = mu_b

        # Predictive draw: add noise on analysis scale, then back-transform
        mu_pred_t = mu_b + rng.normal(0.0, pred_sd_t)
        pred = float(mu_pred_t)

        # Clip to [0,1] on original scale (h2)
        pred_draws[b] = min(1.0, max(0.0, pred))

    return {"mu_draws": mu_draws, "pred_draws": pred_draws}

def _encode_groups(pop_ids, phe_ids):
    """
    Encode string/object labels for populations and PheCodes into 0..K-1 and 0..J-1.

    Parameters
    ----------
    pop_ids : array-like (n,)
    phe_ids : array-like (n,)

    Returns
    -------
    pop_idx : np.ndarray (n,), dtype=int
    phe_idx : np.ndarray (n,), dtype=int
    unique_pops : np.ndarray (K,), original labels in index order
    unique_phe  : np.ndarray (J,), original labels in index order
    """
    import numpy as np

    pop_ids = np.asarray(pop_ids)
    phe_ids = np.asarray(phe_ids)

    unique_pops, pop_idx = np.unique(pop_ids, return_inverse=True)
    unique_phe,  phe_idx = np.unique(phe_ids,  return_inverse=True)

    # Ensure int dtype
    pop_idx = pop_idx.astype(np.int64, copy=False)
    phe_idx = phe_idx.astype(np.int64, copy=False)

    return pop_idx, phe_idx, unique_pops, unique_phe


def _woodbury_multi_solvers(s2_plus_sigma2_e, pop_idx, phe_idx, tau2, omega2):
    """
    Build fast solvers for V = diag(s2 + sigma2_e) + tau2 * Z_pop Z_pop^T + omega2 * Z_phe Z_phe^T.

    Uses the Woodbury identity with U = [sqrt(tau2) * Z_pop, sqrt(omega2) * Z_phe] so that:
      V^{-1} = D^{-1} - D^{-1} U (I + U^T D^{-1} U)^{-1} U^T D^{-1}.

    Parameters
    ----------
    s2_plus_sigma2_e : array-like (n,), strictly positive
    pop_idx : np.ndarray (n,), int in [0..K-1]
    phe_idx : np.ndarray (n,), int in [0..J-1]
    tau2    : float >= 0, between-pop variance component
    omega2  : float >= 0, between-phecode variance component

    Returns
    -------
    dict with callable closures:
      apply_Vinv(v)  -> V^{-1} v
      logdet_V()     -> log|V|
      XtVinvX()      -> 1^T V^{-1} 1
      XtVinv_vec(v)  -> 1^T V^{-1} v
      and metadata keys: n, K, J
    """
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve, LinAlgError

    d = np.asarray(s2_plus_sigma2_e, dtype=float)
    if np.any(~np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("s2_plus_sigma2_e must be finite and > 0.")

    n = d.size
    dinv = 1.0 / d

    pop_idx = np.asarray(pop_idx, dtype=np.int64)
    phe_idx = np.asarray(phe_idx, dtype=np.int64)

    K = int(pop_idx.max()) + 1 if pop_idx.size else 0
    J = int(phe_idx.max()) + 1 if phe_idx.size else 0

    # Precompute group sums with weights dinv
    sum_dinv_by_pop = np.bincount(pop_idx, weights=dinv, minlength=K)
    sum_dinv_by_phe = np.bincount(phe_idx, weights=dinv, minlength=J)

    # Cross (K x J) with weights dinv: Σ_i dinv[i] * 1(pop=k, phe=j)
    pair = pop_idx * J + phe_idx
    cross = np.bincount(pair, weights=dinv, minlength=K * J).reshape(K, J) if (K > 0 and J > 0) else np.zeros((K, J), dtype=float)

    # Build small (K+J)x(K+J) matrix: A = I + U^T D^{-1} U
    # Blocks:
    #   A11 = I_K + tau2 * diag(sum_dinv_by_pop)
    #   A22 = I_J + omega2 * diag(sum_dinv_by_phe)
    #   A12 = sqrt(tau2*omega2) * cross, A21 = A12^T
    I_K = np.eye(K, dtype=float)
    I_J = np.eye(J, dtype=float)

    sqrt_to = np.sqrt(max(tau2, 0.0)) * np.sqrt(max(omega2, 0.0))
    A11 = I_K + max(tau2, 0.0) * np.diag(sum_dinv_by_pop) if K > 0 else np.zeros((0, 0), dtype=float)
    A22 = I_J + max(omega2, 0.0) * np.diag(sum_dinv_by_phe) if J > 0 else np.zeros((0, 0), dtype=float)
    A12 = sqrt_to * cross if (K > 0 and J > 0) else np.zeros((K, J), dtype=float)

    if K == 0 and J == 0:
        # No random effects: V = D
        def apply_Vinv(v):
            v = np.asarray(v, dtype=float)
            return dinv * v
        def logdet_V():
            return float(np.sum(np.log(d)))
        def XtVinvX():
            one = np.ones(n, dtype=float)
            return float(one @ (dinv * one))
        def XtVinv_vec(v):
            return float(np.ones(n, dtype=float) @ (dinv * np.asarray(v, dtype=float)))
        return {
            "apply_Vinv": apply_Vinv,
            "logdet_V": logdet_V,
            "XtVinvX": XtVinvX,
            "XtVinv_vec": XtVinv_vec,
            "n": n, "K": 0, "J": 0
        }

    # Assemble A
    if K > 0 and J > 0:
        top = np.concatenate([A11, A12], axis=1)
        bottom = np.concatenate([A12.T, A22], axis=1)
        A = np.concatenate([top, bottom], axis=0)
    elif K > 0:
        A = A11
    else:
        A = A22

    # Cholesky for A
    try:
        c, lower = cho_factor(A, overwrite_a=False, check_finite=False)
    except LinAlgError:
        # Add a tiny ridge and retry once
        ridge = 1e-10
        A_ridge = A + ridge * np.eye(A.shape[0], dtype=float)
        c, lower = cho_factor(A_ridge, overwrite_a=False, check_finite=False)

    # Helpers to map vectors back/forth through Z and D^{-1}
    def _Zt_Dinv_v(v):
        """Returns concatenated [sqrt(tau2)*Zp^T D^{-1} v ; sqrt(omega2)*Zj^T D^{-1} v]"""
        v = np.asarray(v, dtype=float)
        u = dinv * v
        out_pop = np.bincount(pop_idx, weights=u, minlength=K) if K > 0 else np.zeros(0, dtype=float)
        out_phe = np.bincount(phe_idx, weights=u, minlength=J) if J > 0 else np.zeros(0, dtype=float)
        if K > 0:
            out_pop *= np.sqrt(max(tau2, 0.0))
        if J > 0:
            out_phe *= np.sqrt(max(omega2, 0.0))
        return np.concatenate([out_pop, out_phe], axis=0)

    def _Z_diag_vec(x_pop, x_phe):
        """Compute sqrt(tau2)*Zp x_pop + sqrt(omega2)*Zj x_phe"""
        out = np.zeros(n, dtype=float)
        if K > 0:
            out += np.sqrt(max(tau2, 0.0)) * x_pop[pop_idx]
        if J > 0:
            out += np.sqrt(max(omega2, 0.0)) * x_phe[phe_idx]
        return out

    def apply_Vinv(v):
        v = np.asarray(v, dtype=float)
        u = dinv * v
        a = _Zt_Dinv_v(v)  # length K+J
        # Solve A x = a
        x = cho_solve((c, lower), a, check_finite=False)
        # Split back
        x_pop = x[:K] if K > 0 else np.zeros(0, dtype=float)
        x_phe = x[K:] if J > 0 else np.zeros(0, dtype=float)
        corr = dinv * _Z_diag_vec(x_pop, x_phe)
        return u - corr

    def logdet_V():
        # log|V| = sum(log d) + log|A|
        # log|A| from Cholesky: 2 * sum(log diag(L))
        diagL = np.diag(c) if not lower else np.diag(c)
        return float(np.sum(np.log(d)) + 2.0 * np.sum(np.log(np.abs(diagL))))

    def XtVinvX():
        one = np.ones(n, dtype=float)
        return float(one @ apply_Vinv(one))

    def XtVinv_vec(v):
        return float(np.ones(n, dtype=float) @ apply_Vinv(np.asarray(v, dtype=float)))

    return {
        "apply_Vinv": apply_Vinv,
        "logdet_V": logdet_V,
        "XtVinvX": XtVinvX,
        "XtVinv_vec": XtVinv_vec,
        "n": n, "K": K, "J": J
    }


# =============================================================================
# Aggregation helpers
# =============================================================================

def build_disease_overall_estimates(
    disease_pheno_pop_long,
    grouping_cols,
    *,
    phecode_col="ukbb_phenocode",
    transform="liability",
    bootstrap_B=None,
    random_state=None,
    n_jobs=None,
    show_progress=True,
    pi_target="new_pop_new_phe"
):
    """
    Fit the two-RE REML model per disease (grouping_cols) in parallel.

    Expects columns in `disease_pheno_pop_long`:
      - grouping_cols (keys for disease)
      - 'pop' (population label)
      - phecode_col (PheCode label)
      - 'h2' (float), 'se' (float)
      - 'scale_used' (string), for mix flag only

    Returns a DataFrame with one row per disease including:
      - h2_overall_REML, se_overall_REML, ci95_l_overall_REML, ci95_u_overall_REML
      - tau2_between_pops, omega2_between_phecodes, sigma2_within_idio
      - I2_between_pops, I2_between_phecodes, I2_idiosyncratic
      - pred_int_l, pred_int_u, n_pops_used, n_phecodes_used, n_obs_used, any_scale_mix_flag
      - (plus legacy alias columns for backward compatibility)
    """
    import os
    import numpy as np
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    # Validate columns
    req = set(grouping_cols) | {"pop", phecode_col, "h2", "se", "scale_used"}
    missing = [c for c in req if c not in disease_pheno_pop_long.columns]
    if missing:
        raise ValueError(f"build_disease_overall_estimates: missing columns: {missing}")

    # Drop rows without usable inputs early
    df = disease_pheno_pop_long[list(grouping_cols) + ["pop", phecode_col, "h2", "se", "scale_used"]].copy()
    df = df[df["h2"].notna() & df["se"].notna() & (df["se"] > 0)]
    if df.empty:
        cols = list(grouping_cols) + [
            "h2_overall_REML","se_overall_REML","ci95_l_overall_REML","ci95_u_overall_REML",
            "tau2_between_pops","omega2_between_phecodes","sigma2_within_idio",
            "I2_between_pops","I2_between_phecodes","I2_idiosyncratic",
            "pred_int_l","pred_int_u",
            "n_pops_used","n_phecodes_used","n_obs_used","any_scale_mix_flag",
            # legacy names for compatibility with downstream selection
            "omega2_within_pop","I2_within_pops"
        ]
        return pd.DataFrame(columns=cols)

    # Build tasks per disease
    groups = df.groupby(list(grouping_cols), dropna=False)
    tasks = []
    keys = []

    for key, sub in groups:
        # Minimal payload to worker to reduce pickle size
        y   = sub["h2"].to_numpy(dtype=float)
        s2  = np.square(sub["se"].to_numpy(dtype=float))
        pop = sub["pop"].astype(str).to_numpy()
        phe = sub[phecode_col].astype(str).to_numpy()
        mix = int((sub["scale_used"].astype(str).str.lower() != "liability").any())

        tasks.append((y, s2, pop, phe, mix, transform, bootstrap_B, random_state, pi_target))
        # Store disease key as tuple
        keys.append(key if isinstance(key, tuple) else (key,))

    # Parallel execution
    if n_jobs is None or int(n_jobs) <= 0:
        try:
            import os
            n_jobs = max(1, os.cpu_count() or 1)
        except Exception:
            n_jobs = 1

    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        fut_to_idx = {ex.submit(_reml_worker, payload): i for i, payload in enumerate(tasks)}
        iterator = as_completed(fut_to_idx)
        if show_progress:
            iterator = tqdm(iterator, total=len(fut_to_idx), desc="REML per disease", unit="disease")
        for fut in iterator:
            idx = fut_to_idx[fut]
            results[idx] = fut.result()

    # Build output DF
    rows = []
    for key, res in zip(keys, results):
        row = dict(res)
        # expand grouping key into columns
        for col, val in zip(grouping_cols, key):
            row[col] = val
        rows.append(row)

    out_df = pd.DataFrame(rows)

    # Ensure consistent column order
    ordered_cols = list(grouping_cols) + [
        "h2_overall_REML","se_overall_REML","ci95_l_overall_REML","ci95_u_overall_REML",
        "tau2_between_pops","omega2_between_phecodes","sigma2_within_idio",
        "I2_between_pops","I2_between_phecodes","I2_idiosyncratic",
        "pred_int_l","pred_int_u",
        "n_pops_used","n_phecodes_used","n_obs_used","any_scale_mix_flag",
        # legacy
        "omega2_within_pop","I2_within_pops"
    ]
    for c in ordered_cols:
        if c not in out_df.columns:
            out_df[c] = np.nan
    out_df = out_df[ordered_cols]

    return out_df

# =============================================================================
# Main
# =============================================================================

def main():
    """
    Build the master TSV with:
      (A) ANY-pop detection via ACAT + BH-FDR at H2_THRESHOLD (liability-only),
      (B) REML-based overall heritability per disease (only for diseases passing detection).
    """
    import numpy as np
    import pandas as pd
    import math

    print("Starting combined PheCodeX, UKBB Phenocode, and Heritability mapping process...")

    # -----------------------------
    # PHASE 2: DATA ACQUISITION
    # -----------------------------
    try:
        # --- PheCodeX
        print(f"Downloading PheCodeX data from: {PHECODE_MAP_URL}")
        phecodex_df = pd.read_csv(
            PHECODE_MAP_URL,
            dtype={
                "phecode": "string",
                "ICD": "string",
                "flag": "Int64",
                "phecode_string": "string",
                "category_num": "string",
                "phecode_category": "string",
                "sex": "string",
                "icd10_only": "Int64",
                "code_val": "string",
            }
        )[["phecode", "ICD", "flag", "phecode_string", "phecode_category"]]

        # --- UKBB ICD→PheCode maps
        print("Downloading UKBB ICD-to-Phenocode maps...")
        icd10_map = pd.read_csv(
            UKBB_ICD10_MAP_URL, sep="\t", engine="python",
            dtype={"ICD10": "string", "phecode": "string"}
        )[["ICD10", "phecode"]].rename(columns={"ICD10": "ICD", "phecode": "ukbb_phenocode"})

        icd9_map = pd.read_csv(
            UKBB_ICD9_MAP_URL, sep="\t", engine="python",
            dtype={"ICD9": "string", "phecode": "string"}
        )[["ICD9", "phecode"]].rename(columns={"ICD9": "ICD", "phecode": "ukbb_phenocode"})

        ukbb_lookup_df = pd.concat([icd9_map, icd10_map], ignore_index=True).drop_duplicates()
        print("Created a unified UKBB lookup table.")

        # --- Heritability manifest (liability-only; no observed fallback)
        print(f"Downloading heritability data from: {H2_MANIFEST_URL}")
        usecols = [
            "trait_type", "phenocode", "pop",
            "estimates.final.h2_liability", "estimates.final.h2_liability_se",
            "qcflags.defined_h2", "qcflags.in_bounds_h2", "qcflags.pass_all"
        ]
        h2_raw_df = read_bgz_tsv(H2_MANIFEST_URL, usecols=usecols)
        h2_raw_df = h2_raw_df[h2_raw_df["trait_type"] == "phecode"].copy()

        h2_pre_qc_df = h2_raw_df.copy()
        if USE_QC:
            have_defined = "qcflags.defined_h2" in h2_raw_df.columns
            have_inbounds = "qcflags.in_bounds_h2" in h2_raw_df.columns
            if have_defined and have_inbounds:
                h2_raw_df = h2_raw_df[
                    (h2_raw_df["qcflags.defined_h2"] == True) &
                    (h2_raw_df["qcflags.in_bounds_h2"] == True)
                ].copy()

        # Require liability h2 + SE; drop everything else (no scale mixing)
        liab  = h2_raw_df["estimates.final.h2_liability"]
        liab_se = h2_raw_df["estimates.final.h2_liability_se"]
        keep = liab.notna() & liab_se.notna() & (liab_se > 0)
        h2_rows = h2_raw_df.loc[keep, ["phenocode", "pop"]].copy()
        h2_rows["h2"] = liab[keep].astype(float).to_numpy()
        h2_rows["se"] = liab_se[keep].astype(float).to_numpy()
        h2_rows["scale_used"] = "liability"

        # -----------------------------
        # Detection: ACAT + BH-FDR
        # -----------------------------
        z = (h2_rows["h2"].to_numpy(dtype=float) - H2_THRESHOLD) / h2_rows["se"].to_numpy(dtype=float)
        p_one = np.array([0.5 * math.erfc(val / math.sqrt(2.0)) for val in z], dtype=float)
        p_one = np.clip(p_one, 1e-15, 1.0 - 1e-15)
        h2_rows["p_one"] = p_one

        def acat_group(g: pd.DataFrame) -> pd.Series:
            ps = g["p_one"].to_numpy(dtype=float)
            p_ac = acat(ps)
            h2_max = float(np.nanmax(g["h2"].to_numpy(dtype=float))) if len(g) else np.nan
            pop_min = g.loc[g["p_one"].idxmin(), "pop"] if len(g) else np.nan
            return pd.Series({
                "p_any_acat": p_ac,
                "h2_max_any_pop": h2_max,
                "pop_min_p": pop_min
            })

        per_pheno = (
            h2_rows.groupby("phenocode", dropna=False)
                   .apply(acat_group)
                   .reset_index()
        )

        # Guard for the extreme "no usable h\u00B2 rows" case
        if per_pheno.empty:
            per_pheno = pd.DataFrame(
                columns=["phenocode", "p_any_acat", "h2_max_any_pop", "pop_min_p"]
            )

        # Flag phenocodes that actually have heritability data
        per_pheno["has_h2_data_for_phenocode"] = per_pheno["p_any_acat"].notna().astype("int64")
        per_pheno["q_bh"] = np.nan
        valid_mask = per_pheno["p_any_acat"].notna()
        if valid_mask.any():
            per_pheno.loc[valid_mask, "q_bh"] = bh_fdr(
                per_pheno.loc[valid_mask, "p_any_acat"].to_numpy(dtype=float)
            )

        # Rename detection flag as requested
        per_pheno["phenocode_anypop_gt_threshold_fdr"] = (
            (per_pheno["q_bh"].notna()) & (per_pheno["q_bh"] <= FDR_Q)
        ).astype("int64")

        # EUR descriptive (IVW across multiple EUR rows, if any)
        eur_rows = h2_rows[h2_rows["pop"] == "EUR"].copy()
        if not eur_rows.empty:
            tmp = eur_rows.copy()
            tmp["w"] = 1.0 / (tmp["se"] ** 2)
            tmp["wh2"] = tmp["w"] * tmp["h2"]
            eur_ivw = tmp.groupby("phenocode", as_index=False)[["wh2", "w"]].sum()
            eur_ivw["eur_h2_mean"] = eur_ivw["wh2"] / eur_ivw["w"]
            eur_ivw = eur_ivw[["phenocode", "eur_h2_mean"]]
            per_pheno = per_pheno.merge(eur_ivw, on="phenocode", how="left")
        else:
            per_pheno["eur_h2_mean"] = np.nan

        # Diagnostics: mapped-but-missing phenocodes (optional write)
        mapped_set = set(
            ukbb_lookup_df.loc[
                ukbb_lookup_df["ICD"].isin(phecodex_df["ICD"].dropna().astype(str)),
                "ukbb_phenocode"
            ].dropna().astype(str).unique().tolist()
        )
        considered_set = set(per_pheno.loc[per_pheno["p_any_acat"].notna(), "phenocode"].astype(str).unique().tolist())
        manifest_preqc_set = set(h2_pre_qc_df["phenocode"].dropna().astype(str).unique().tolist())
        missing_set = sorted(mapped_set - considered_set)

        if len(missing_set) > 0:
            def summarize_issues_one(phenocode: str) -> dict:
                pre = h2_pre_qc_df[h2_pre_qc_df["phenocode"].astype(str) == phenocode]
                n_pops_preqc = int(pre.shape[0])
                pops_preqc = ";".join(sorted(pre["pop"].dropna().astype(str).unique().tolist())) if n_pops_preqc else ""
                return dict(
                    phenocode=phenocode,
                    reason=("absent_from_manifest" if phenocode not in manifest_preqc_set else "no_liability_h2_se"),
                    n_pops_manifest_preqc=n_pops_preqc,
                    pops_manifest_preqc=pops_preqc
                )
            try:
                pd.DataFrame([summarize_issues_one(pc) for pc in missing_set]).to_csv(MISSING_PHECODES_TSV, sep="\t", index=False)
                print(f"Wrote detail on missing/unsuitable PheCodes to: {MISSING_PHECODES_TSV}")
            except Exception as _e:
                print(f"Warning: failed to write {MISSING_PHECODES_TSV}: {_e}")

        print("Successfully processed heritability data at the phenocode level (ACAT + BH-FDR).")

    except Exception as e:
        print("------------------------------------------------------------")
        print("FATAL ERROR: Failed to download or parse a required data file.")
        print(f"Error details: {e}")
        return

    # ------------------------------------------
    # PHASE 3: MERGE, DETECTION, & ESTIMATION
    # ------------------------------------------
    print("Enriching data with UKBB PheCodes and Heritability stats...")

    # Map ICD -> PheCodes into PheCodeX rows
    base_df = phecodex_df.merge(ukbb_lookup_df, on="ICD", how="left")

    grouping_cols = ["phecode", "phecode_string", "phecode_category"]

    def agg_disease(g: pd.DataFrame) -> pd.Series:
        icd9 = combine_codes_unique_sorted(g.loc[g["flag"] == 9, "ICD"])
        icd10 = combine_codes_unique_sorted(g.loc[g["flag"] == 10, "ICD"])
        phecodes = combine_codes_unique_sorted(g["ukbb_phenocode"])
        return pd.Series({
            "ICD9_Mappings": icd9,
            "ICD10_Mappings": icd10,
            "ukbb_phenocode": phecodes
        })

    aggregated_df = (
        base_df.groupby(grouping_cols, dropna=False)
               .apply(agg_disease)
               .reset_index()
    )

    # Flag whether each disease maps to at least one UKBB phenocode
    aggregated_df["has_ukbb_phenocode_mapping"] = aggregated_df["ukbb_phenocode"].apply(lambda xs: len(xs) > 0)

    # Detection → disease level
    long_df = aggregated_df.explode("ukbb_phenocode")
    long_df["ukbb_phenocode"] = long_df["ukbb_phenocode"].astype("string")
    per_pheno["phenocode"] = per_pheno["phenocode"].astype("string")

    long_joined = long_df.merge(
        per_pheno[["phenocode", "p_any_acat", "phenocode_anypop_gt_threshold_fdr", "eur_h2_mean", "has_h2_data_for_phenocode"]],
        left_on="ukbb_phenocode",
        right_on="phenocode",
        how="left"
    )


    data_presence = (
        long_joined.groupby(grouping_cols, as_index=False)["has_h2_data_for_phenocode"]
                   .max()
                   .rename(columns={"has_h2_data_for_phenocode": "has_ukbb_heritability_data"})
    )

    disease_signals = (
        long_joined.groupby(grouping_cols, as_index=False)
                   .agg({
                       "phenocode_anypop_gt_threshold_fdr": "max",
                       "eur_h2_mean": "mean",
                       "p_any_acat": "min"   # best (smallest) ACAT p across mapped phenocodes
                   })
                   .rename(columns={
                       "phenocode_anypop_gt_threshold_fdr": "is_h2_significant_in_any_ancestry",
                       "eur_h2_mean": "h2_eur_avg",
                       "p_any_acat": "anypop_acat_p_best_T"
                   })
    )

    final_df = aggregated_df.merge(disease_signals, on=grouping_cols, how="left")
    final_df = final_df.merge(data_presence, on=grouping_cols, how="left")

    final_df["is_h2_significant_in_any_ancestry"] = (
        final_df["is_h2_significant_in_any_ancestry"].astype("Int64")
    )
    final_df["has_ukbb_heritability_data"] = final_df["has_ukbb_heritability_data"].fillna(0).astype("int64")
    final_df["has_ukbb_phenocode_mapping"] = final_df["has_ukbb_phenocode_mapping"].astype("int64")

    # ---------- Estimation (REML) ONLY for diseases passing detection ----------
    sig_keys = final_df.loc[final_df["is_h2_significant_in_any_ancestry"] == 1, grouping_cols].drop_duplicates()
    if not sig_keys.empty:
        # Restrict disease×phenocode list to passing diseases
        long_sig = long_df.merge(sig_keys, on=grouping_cols, how="inner")

        # Attach pop-level liability h2 rows
        disease_pheno_pop_long = long_sig.merge(
            h2_rows.rename(columns={"phenocode": "ukbb_phenocode"}),
            on="ukbb_phenocode",
            how="left"
        )

        B_boot = int(REML_BOOTSTRAP_B) if (REML_BOOTSTRAP_B and int(REML_BOOTSTRAP_B) > 0) else None

        disease_overall_df = build_disease_overall_estimates(
            disease_pheno_pop_long,
            grouping_cols=grouping_cols,
            transform="liability",     # no scale mixing
            bootstrap_B=B_boot,
            random_state=42,
            # other build_disease_overall_estimates args at defaults
        )
        final_df = final_df.merge(disease_overall_df, on=grouping_cols, how="left")
    else:
        # No passing diseases; create empty columns expected downstream
        for c in [
            "h2_overall_REML","se_overall_REML","ci95_l_overall_REML","ci95_u_overall_REML",
            "tau2_between_pops","omega2_between_phecodes","sigma2_within_idio",
            "I2_between_pops","I2_between_phecodes","I2_idiosyncratic",
            "pred_int_l","pred_int_u",
            "n_pops_used","n_obs_used","any_scale_mix_flag"
        ]:
            if c not in final_df.columns:
                final_df[c] = np.nan

    # Back-compat alias if desired (harmless)
    if "h2_overall_REML" in final_df.columns:
        final_df["h2_overall_RE"] = final_df["h2_overall_REML"]

    # -----------------------------
    # PHASE 4: OUTPUT FORMATTING
    # -----------------------------
    print("Formatting final data and preparing for output...")

    final_df["icd9_codes"] = final_df["ICD9_Mappings"].apply(list_join_safe)
    final_df["icd10_codes"] = final_df["ICD10_Mappings"].apply(list_join_safe)
    final_df["ukbb_phenocode"] = final_df["ukbb_phenocode"].apply(list_join_safe)

    if "h2_eur_avg" not in final_df.columns:
        final_df["h2_eur_avg"] = np.nan
    final_df["h2_eur_avg"] = final_df["h2_eur_avg"].round(4)
    final_df["h2_eur_avg"] = final_df["h2_eur_avg"].apply(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    out_cols = [
        "phecode",
        "ukbb_phenocode",
        "phecode_string",
        "phecode_category",
        "is_h2_significant_in_any_ancestry",
        "h2_eur_avg",
        "anypop_acat_p_best_T",   # disease-level ACAT p (best over mapped phenocodes)
        "icd9_codes",
        "icd10_codes",
        "h2_overall_REML",
        "se_overall_REML",
        "ci95_l_overall_REML",
        "ci95_u_overall_REML",
        "tau2_between_pops",
        "omega2_between_phecodes",
        "sigma2_within_idio",
        "I2_between_pops",
        "I2_between_phecodes",
        "I2_idiosyncratic",
        "pred_int_l",
        "pred_int_u",
        "n_pops_used",
        "n_obs_used",
        "any_scale_mix_flag",
        "h2_overall_RE",
        "has_ukbb_phenocode_mapping",
        "has_ukbb_heritability_data",
    ]

    for c in out_cols:
        if c not in final_df.columns:
            final_df[c] = np.nan

    final_df = final_df[out_cols].rename(columns={
        "phecode_string": "disease",
        "phecode_category": "disease_category"
    })

    try:
        final_df.to_csv(OUTPUT_FILENAME, sep="\t", index=False)
        print("-" * 50)
        print("PROCESS COMPLETE!")
        print(f"Successfully created the enriched mapping file: {OUTPUT_FILENAME}")
        print(f"Found and processed {len(final_df)} unique diseases.")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")


if __name__ == "__main__":
    main()
