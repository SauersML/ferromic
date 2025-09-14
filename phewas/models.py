import os
import gc
import hashlib
import warnings
from datetime import datetime, timezone
import traceback
import sys
import atexit

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.special import expit
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

from . import iox as io

CTX = {}  # Worker context with constants from run.py
allowed_fp_by_cat = {}

def safe_basename(name: str) -> str:
    """Allow only [-._a-zA-Z0-9], map others to '_'."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(name))

def _write_meta(meta_path, kind, s_name, category, target, core_cols, core_idx_fp, case_fp, extra=None):
    """Helper to write a standardized metadata JSON file."""
    base = {
        "kind": kind,
        "s_name": s_name,
        "category": category,
        "model_columns": list(core_cols),
        "num_pcs": CTX["NUM_PCS"],
        "min_cases": CTX["MIN_CASES_FILTER"],
        "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
        "target": target,
        "core_index_fp": core_idx_fp,
        "case_idx_fp": case_fp,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    if extra:
        base.update(extra)
    io.atomic_write_json(meta_path, base)

# thresholds (configured via CTX; here are defaults/fallbacks)
DEFAULT_MIN_CASES = 1000
DEFAULT_MIN_CONTROLS = 1000
DEFAULT_MIN_NEFF = 0  # set 0 to disable
DEFAULT_SEX_RESTRICT_PROP = 0.99

def _thresholds(cases_key="MIN_CASES_FILTER", controls_key="MIN_CONTROLS_FILTER", neff_key="MIN_NEFF_FILTER"):
    return (
        int(CTX.get(cases_key, DEFAULT_MIN_CASES)),
        int(CTX.get(controls_key, DEFAULT_MIN_CONTROLS)),
        float(CTX.get(neff_key, DEFAULT_MIN_NEFF)),
    )

def _counts_from_y(y):
    y = np.asarray(y, dtype=np.int8)
    n = y.size
    n_cases = int(np.sum(y))
    n_ctrls = int(n - n_cases)
    pi = (n_cases / n) if n > 0 else 0.0
    n_eff = 4.0 * n * pi * (1.0 - pi) if n > 0 else 0.0
    return n, n_cases, n_ctrls, n_eff

def validate_min_counts_for_fit(y, stage_tag, extra_context=None, cases_key="MIN_CASES_FILTER", controls_key="MIN_CONTROLS_FILTER", neff_key="MIN_NEFF_FILTER"):
    """
    Validate *final* y used for the fit. Returns (ok: bool, reason: str, details: dict)
    stage_tag: 'phewas' | 'lrt_stage1' | 'lrt_followup:<ANC>'
    """
    min_cases, min_ctrls, min_neff = _thresholds(cases_key=cases_key, controls_key=controls_key, neff_key=neff_key)
    n, n_cases, n_ctrls, n_eff = _counts_from_y(y)
    ok = True
    reasons = []
    if n_cases < min_cases:
        ok = False; reasons.append(f"cases<{min_cases}({n_cases})")
    if n_ctrls < min_ctrls:
        ok = False; reasons.append(f"controls<{min_ctrls}({n_ctrls})")
    if min_neff > 0 and n_eff < min_neff:
        ok = False; reasons.append(f"neff<{min_neff:g}({n_eff:.1f})")

    details = {
        "stage": stage_tag,
        "N": n, "N_cases": n_cases, "N_ctrls": n_ctrls, "N_eff": n_eff,
        "min_cases": min_cases, "min_ctrls": min_ctrls, "min_neff": min_neff,
    }
    if extra_context:
        details.update(extra_context)
    reason = "OK" if ok else "insufficient_counts:" + "|".join(reasons)
    return ok, reason, details

def _converged(fit_obj):
    """Checks for convergence in a statsmodels fit object."""
    try:
        if hasattr(fit_obj, "mle_retvals") and isinstance(fit_obj.mle_retvals, dict):
            return bool(fit_obj.mle_retvals.get("converged", False))
        if hasattr(fit_obj, "converged"):
            return bool(fit_obj.converged)
        return False
    except Exception:
        return False

def _logit_fit(model, method, **kw):
    """
    Helper to fit a logit model with per-solver argument routing for stability and correctness.

    For 'newton', only pass 'tol' since 'gtol' is unsupported for that solver.
    For 'bfgs' and 'cg', pass 'gtol' and do not pass 'tol'.
    Falls back gracefully when 'warn_convergence' is unavailable in the installed statsmodels.
    """
    maxiter = kw.get("maxiter", 200)
    start_params = kw.get("start_params", None)

    if method in ("bfgs", "cg"):
        fit_kwargs = {
            "disp": 0,
            "method": method,
            "maxiter": maxiter,
            "start_params": start_params,
        }
        gtol = kw.get("gtol", 1e-8)
        if gtol is not None:
            fit_kwargs["gtol"] = gtol
        try:
            return model.fit(warn_convergence=False, **fit_kwargs)
        except TypeError:
            return model.fit(**fit_kwargs)
    else:
        fit_kwargs = {
            "disp": 0,
            "method": method,
            "maxiter": maxiter,
            "start_params": start_params,
        }
        tol = kw.get("tol", 1e-8)
        if tol is not None:
            fit_kwargs["tol"] = tol
        try:
            return model.fit(warn_convergence=False, **fit_kwargs)
        except TypeError:
            return model.fit(**fit_kwargs)


def _leverages_batched(X_np, XtWX_inv, W, batch=100_000):
    """Compute hat matrix leverages in batches to bound memory usage."""
    n = X_np.shape[0]
    h = np.empty(n, dtype=np.float64)
    for i0 in range(0, n, batch):
        i1 = min(i0 + batch, n)
        Xb = X_np[i0:i1]
        Tb = Xb @ XtWX_inv
        s = np.einsum("ij,ij->i", Tb, Xb)
        h[i0:i1] = np.clip(W[i0:i1] * s, 0.0, 1.0)
    return h

def _fit_logit_ladder(X, y, ridge_ok=True, const_ix=None, prefer_mle_first=False, **kwargs):
    """
    Logistic fit ladder with an option to attempt unpenalized MLE first.
    If numpy arrays are provided, const_ix identifies the intercept column for zero-penalty.
    Returns a tuple (fit_result, reason_tag).
    """
    is_pandas = hasattr(X, "columns")
    if not ridge_ok:
        return None, "ridge_disabled"

    try:
        # If requested, try unpenalized MLE first. This is particularly effective after design restrictions.
        if prefer_mle_first:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=PerfectSeparationWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in exp",
                    category=RuntimeWarning,
                    module=r"statsmodels\.discrete\.discrete_model"
                )
                warnings.filterwarnings(
                    "ignore",
                    message="divide by zero encountered in log",
                    category=RuntimeWarning,
                    module=r"statsmodels\.discrete\.discrete_model"
                )
                try:
                    mle_newton = _logit_fit(
                        sm.Logit(y, X),
                        "newton",
                        maxiter=400,
                        tol=1e-8,
                        start_params=kwargs.get("start_params", None)
                    )
                    if _converged(mle_newton) and hasattr(mle_newton, "bse") and np.all(np.isfinite(mle_newton.bse)) and np.max(mle_newton.bse) <= 100.0:
                        setattr(mle_newton, "_final_is_mle", True)
                        return mle_newton, "mle_first_newton"
                except (Exception, PerfectSeparationWarning):
                    pass
                try:
                    mle_bfgs = _logit_fit(
                        sm.Logit(y, X),
                        "bfgs",
                        maxiter=800,
                        gtol=1e-8,
                        start_params=kwargs.get("start_params", None)
                    )
                    if _converged(mle_bfgs) and hasattr(mle_bfgs, "bse") and np.all(np.isfinite(mle_bfgs.bse)) and np.max(mle_bfgs.bse) <= 100.0:
                        setattr(mle_bfgs, "_final_is_mle", True)
                        return mle_bfgs, "mle_first_bfgs"
                except (Exception, PerfectSeparationWarning):
                    pass

        # Ridge-first pathway with strict MLE gating based on numerical diagnostics.
        p = X.shape[1] - (1 if (is_pandas and "const" in X.columns) or (not is_pandas and const_ix is not None) else 0)
        n = max(1, X.shape[0])
        pi = float(np.mean(y)) if len(y) > 0 else 0.5
        n_eff = max(1.0, 4.0 * float(len(y)) * pi * (1.0 - pi))
        alpha_scalar = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / n_eff), 1e-6)
        # DiscreteModel.fit_regularized expects scalar alpha; per-parameter weights are not reliably supported.
        # Using scalar ridge is OK since we refit MLE (unpenalized) when possible.
        ridge_fit = sm.Logit(y, X).fit_regularized(alpha=float(alpha_scalar), L1_wt=0.0, maxiter=800, disp=0, **kwargs)

        setattr(ridge_fit, "_used_ridge", True)
        setattr(ridge_fit, "_final_is_mle", False)

        try:
            max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, ridge_fit.params)
        except Exception:
            max_abs_linpred, frac_lo, frac_hi = float("inf"), 1.0, 1.0

        neff_gate = float(CTX.get("MLE_REFIT_MIN_NEFF", 0.0))
        blocked_by_gate = ((max_abs_linpred > 15.0) or (frac_lo > 0.02) or (frac_hi > 0.02) or (neff_gate > 0 and n_eff < neff_gate))
        if blocked_by_gate and not prefer_mle_first:
            return ridge_fit, "ridge_only"
        # When prefer_mle_first is True, proceed to attempt an unpenalized refit seeded by ridge even if diagnostics indicate separation-like behavior.

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=PerfectSeparationWarning)
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in exp",
                category=RuntimeWarning,
                module=r"statsmodels\.discrete\.discrete_model"
            )
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in log",
                category=RuntimeWarning,
                module=r"statsmodels\.discrete\.discrete_model"
            )
            try:
                extra_flag = {} if ('_already_failed' in kwargs) else {'_already_failed': True}
                refit_newton = _logit_fit(
                    sm.Logit(y, X),
                    "newton",
                    maxiter=400,
                    tol=1e-8,
                    start_params=ridge_fit.params,
                    **extra_flag,
                    **kwargs
                )
                if _converged(refit_newton) and hasattr(refit_newton, "bse") and np.all(np.isfinite(refit_newton.bse)):
                    if np.max(refit_newton.bse) > 100.0:
                        raise PerfectSeparationWarning("Large SEs detected")
                    setattr(refit_newton, "_used_ridge_seed", True)
                    setattr(refit_newton, "_final_is_mle", True)
                    return refit_newton, "ridge_seeded_refit"
            except (Exception, PerfectSeparationWarning):
                pass

            try:
                extra_flag = {} if ('_already_failed' in kwargs) else {'_already_failed': True}
                refit_bfgs = _logit_fit(
                    sm.Logit(y, X),
                    "bfgs",
                    maxiter=800,
                    gtol=1e-8,
                    start_params=ridge_fit.params,
                    **extra_flag,
                    **kwargs
                )
                if _converged(refit_bfgs) and hasattr(refit_bfgs, "bse") and np.all(np.isfinite(refit_bfgs.bse)):
                    if np.max(refit_bfgs.bse) > 100.0:
                        raise PerfectSeparationWarning("Large SEs detected")
                    setattr(refit_bfgs, "_used_ridge_seed", True)
                    setattr(refit_bfgs, "_final_is_mle", True)
                    return refit_bfgs, "ridge_seeded_refit"
            except (Exception, PerfectSeparationWarning):
                pass

        # Firth fallback for separation-prone designs under prefer_mle_first.
        # This implements bias-reduced logistic regression using the adjusted-score iteration.
        if prefer_mle_first:
            X_np = np.asarray(X, dtype=np.float64)
            y_np = np.asarray(y, dtype=np.float64)
            beta = np.zeros(X_np.shape[1], dtype=np.float64)
            maxiter_firth = 200
            tol_firth = 1e-8
            converged_firth = False
            for _it in range(maxiter_firth):
                eta = X_np @ beta
                p = expit(eta)
                W = p * (1.0 - p)
                if not np.all(np.isfinite(W)) or np.any(W <= 0):
                    break
                XTW = X_np.T * W
                XtWX = XTW @ X_np
                try:
                    XtWX_inv = np.linalg.inv(XtWX)
                except np.linalg.LinAlgError:
                    try:
                        XtWX_inv = np.linalg.pinv(XtWX)
                    except Exception:
                        break
                # Compute leverages without constructing the full N×N hat matrix.
                # h_i = w_i * x_i^T (X' W X)^{-1} x_i
                h = _leverages_batched(X_np, XtWX_inv, W)
                adj = (0.5 - p) * h
                score = X_np.T @ (y_np - p + adj)
                try:
                    delta = XtWX_inv @ score
                except Exception:
                    break
                beta_new = beta + delta
                if not np.all(np.isfinite(beta_new)):
                    break
                if np.max(np.abs(delta)) < tol_firth:
                    beta = beta_new
                    converged_firth = True
                    break
                beta = beta_new
            if converged_firth:
                eta = X_np @ beta
                p = expit(eta)
                W = p * (1.0 - p)
                XTW = X_np.T * W
                XtWX = XTW @ X_np
                try:
                    cov = np.linalg.inv(XtWX)
                except np.linalg.LinAlgError:
                    cov = np.linalg.pinv(XtWX)
                bse = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = beta / bse
                pvals = 2.0 * sp_stats.norm.sf(np.abs(z))
                # Penalized log-likelihood for Firth logistic regression for LRT compatibility.
                # Uses l(β) + 0.5 * log|X' W X|.
                with np.errstate(divide="ignore", invalid="ignore"):
                    loglik = float(np.sum(y_np * np.log(p) + (1.0 - y_np) * np.log(1.0 - p)))
                sign_det, logdet = np.linalg.slogdet(XtWX)
                pll = loglik + 0.5 * logdet if sign_det > 0 else -np.inf
                class _Result:
                    """Lightweight container to mimic statsmodels results where needed."""
                    pass
                firth_res = _Result()
                if is_pandas and hasattr(X, "columns"):
                    firth_res.params = pd.Series(beta, index=X.columns)
                    firth_res.bse = pd.Series(bse, index=X.columns)
                    firth_res.pvalues = pd.Series(pvals, index=X.columns)
                else:
                    firth_res.params = beta
                    firth_res.bse = bse
                    firth_res.pvalues = pvals
                setattr(firth_res, "llf", float(pll))
                setattr(firth_res, "_final_is_mle", True)
                setattr(firth_res, "_used_firth", True)
                return firth_res, "firth_refit"

        return ridge_fit, "ridge_only"
    except Exception as e:
        return None, f"ridge_exception:{type(e).__name__}"

def _drop_zero_variance(X: pd.DataFrame, keep_cols=('const',), always_keep=(), eps=1e-12):
    """Drops columns with no or near-zero variance, keeping specified columns."""
    keep = set(keep_cols) | set(always_keep)
    cols = []
    for c in X.columns:
        if c in keep:
            cols.append(c)
            continue
        s = X[c]
        if pd.isna(s).all():
            continue
        # Treat extremely small variance as zero
        if s.nunique(dropna=True) <= 1 or float(np.nanstd(s)) < eps:
            continue
        cols.append(c)
    return X.loc[:, cols]


def _drop_rank_deficient(X: pd.DataFrame, keep_cols=('const',), always_keep=(), rtol=1e-10):
    """
    Removes columns that render the design matrix rank-deficient by greedily dropping
    non-essential columns based on ascending column standard deviation while preserving
    columns listed in keep_cols and always_keep whenever possible.
    Returns a DataFrame with full column rank or the best achievable subset if no removable columns remain.
    """
    keep = set(keep_cols) | set(always_keep)
    if X.shape[1] == 0:
        return X
    M = X.to_numpy(dtype=np.float64, copy=False)
    rank = np.linalg.matrix_rank(M)
    if rank == X.shape[1]:
        return X
    remaining = list(X.columns)
    removable = [c for c in remaining if c not in keep]
    X_work = X.copy()
    while np.linalg.matrix_rank(X_work.to_numpy(dtype=np.float64, copy=False)) < X_work.shape[1]:
        if not removable:
            break
        stds = np.nanstd(X_work.to_numpy(dtype=np.float64, copy=False), axis=0)
        col_order = np.argsort(stds)
        dropped = False
        for k in col_order:
            colname = X_work.columns[k]
            if colname not in removable:
                continue
            trial = X_work.drop(columns=[colname])
            if np.linalg.matrix_rank(trial.to_numpy(dtype=np.float64, copy=False)) == trial.shape[1]:
                X_work = trial
                remaining = list(X_work.columns)
                removable = [c for c in remaining if c not in keep]
                dropped = True
                break
        if not dropped:
            break
    return X_work


def _fit_diagnostics(X, y, params):
    """
    Computes simple numerical diagnostics for a fitted logistic model:
      - max absolute linear predictor
      - fraction of probabilities effectively at 0 or 1
    """
    X_arr = X if (isinstance(X, np.ndarray) and X.dtype == np.float64) else np.asarray(X, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    linpred = X_arr @ params_arr
    if not np.all(np.isfinite(linpred)):
        max_abs_linpred = float("inf")
        frac_lo = 0.0
        frac_hi = 0.0
    else:
        max_abs_linpred = float(np.max(np.abs(linpred))) if linpred.size else 0.0
        p = expit(linpred)
        frac_lo = float(np.mean(p < 1e-12)) if p.size else 0.0
        frac_hi = float(np.mean(p > 1.0 - 1e-12)) if p.size else 0.0
    return max_abs_linpred, frac_lo, frac_hi


def _print_fit_diag(s_name_safe, stage, model_tag, N_total, N_cases, N_ctrls, solver_tag, X, y, params, notes):
    """
    Emits a single-line diagnostic message for a fit attempt. This is intended for real-time visibility
    into numerical behavior and sample composition while models are running in worker processes.
    """
    max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, params)
    msg = (
        f"[fit] name={s_name_safe} stage={stage} model={model_tag} "
        f"N={int(N_total)}/{int(N_cases)}/{int(N_ctrls)} solver={solver_tag} "
        f"max|Xb|={max_abs_linpred:.6g} p<1e-12:{frac_lo:.2%} p>1-1e-12:{frac_hi:.2%} "
        f"notes={'|'.join(notes) if notes else ''}"
    )
    print(msg, flush=True)

def _suppress_worker_warnings():
    """Configures warning filters for the worker process to ignore specific, benign warnings."""
    # RuntimeWarning: overflow encountered in exp
    warnings.filterwarnings('ignore', message='overflow encountered in exp', category=RuntimeWarning)
    
    # RuntimeWarning: divide by zero encountered in log
    warnings.filterwarnings('ignore', message='divide by zero encountered in log', category=RuntimeWarning)
    
    # ConvergenceWarning: QC check did not pass for X out of Y parameters
    warnings.filterwarnings('ignore', message=r'QC check did not pass', category=ConvergenceWarning)
    
    # ConvergenceWarning: Could not trim params automatically
    warnings.filterwarnings('ignore', message=r'Could not trim params automatically', category=ConvergenceWarning)
    return

REQUIRED_CTX_KEYS = {
 "NUM_PCS", "MIN_CASES_FILTER", "MIN_CONTROLS_FILTER", "CACHE_DIR",
 "RESULTS_CACHE_DIR", "LRT_OVERALL_CACHE_DIR", "LRT_FOLLOWUP_CACHE_DIR", "RIDGE_L2_BASE",
 "PER_ANC_MIN_CASES", "PER_ANC_MIN_CONTROLS"
}

def _validate_ctx(ctx):
    """Raises RuntimeError if required context keys are missing."""
    missing = [k for k in REQUIRED_CTX_KEYS if k not in ctx]
    if missing:
        raise RuntimeError(f"[Worker-{os.getpid()}] Missing CTX keys: {', '.join(missing)}")
def _drop_zero_variance_np(X, keep_ix=(), eps=1e-12):
    """
    Drops columns with no or near-zero variance from a NumPy array.
    `keep_ix` is a set of column indices to always keep.
    Returns the pruned array and the indices of the kept columns from the original array.
    """
    stds = np.nanstd(X, axis=0)
    good_mask = (stds >= eps) & np.isfinite(stds)
    for i in keep_ix:
        if i < len(good_mask):
            good_mask[i] = True
    kept_original_indices = np.flatnonzero(good_mask)
    return X[:, kept_original_indices], kept_original_indices


def _drop_rank_deficiency_np(X, keep_ix=()):
    """
    Greedily removes columns from a NumPy design matrix until it achieves full column rank.
    Columns listed in keep_ix are preserved whenever possible. Removal order is by ascending
    column standard deviation among removable columns. This continues until the matrix is full rank
    or no more columns can be removed.
    Returns the pruned matrix and the kept column indices relative to the original X.
    """
    if X.shape[1] == 0: return X, np.array([], int)
    keep_ix = set(int(i) for i in keep_ix)
    cols = np.arange(X.shape[1])
    while np.linalg.matrix_rank(X) < X.shape[1]:
        removable_positions = [k for k, j in enumerate(cols) if j not in keep_ix]
        if not removable_positions:
            break
        stds = np.nanstd(X, axis=0)
        # Find the position of the column with the minimum standard deviation among those that can be removed.
        # np.argmin operates on the filtered stds, so we need to map the index back to the original position.
        min_std_removable_pos = removable_positions[int(np.argmin(stds[removable_positions]))]
        X = np.delete(X, min_std_removable_pos, axis=1)
        cols = np.delete(cols, min_std_removable_pos, axis=0)
    return X, cols


def _apply_sex_restriction(X: pd.DataFrame, y: pd.Series):
    """
    Returns: (X2, y2, note:str, skip_reason:str|None)
    """
    if 'sex' not in X.columns:
        return X, y, "", None
    tab = pd.crosstab(X['sex'], y).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
    total_cases = int(tab.loc[0.0, 1] + tab.loc[1.0, 1])
    if total_cases <= 0:
        return X, y, "", None
    thr = float(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP))
    cases_by_sex = {0.0: int(tab.loc[0.0, 1]), 1.0: int(tab.loc[1.0, 1])}
    dominant_sex = 0.0 if cases_by_sex[0.0] >= cases_by_sex[1.0] else 1.0
    frac = (cases_by_sex[dominant_sex] / total_cases) if total_cases > 0 else 0.0
    if frac < thr:
        return X, y, "", None
    if int(tab.loc[dominant_sex, 0]) == 0:
        return X, y, "", "sex_no_controls_in_case_sex"
    keep = X['sex'].eq(dominant_sex)
    return X.loc[keep].drop(columns=['sex']), y.loc[keep], f"sex_restricted_to_{int(dominant_sex)}", None


def _apply_sex_restriction_np(X, y, sex_ix):
    """
    NumPy-based sex restriction with strict or majority gating.

    Behavior is controlled via CTX:
      - CTX['SEX_RESTRICT_MODE']: 'strict' or 'majority'.
      - CTX['SEX_RESTRICT_PROP']: float in [0, 1] for majority.
      - CTX['SEX_RESTRICT_MAX_OTHER_CASES']: int cap on stray other-sex cases.
    Returns: (X_restr, y_restr, note:str, skip_reason:str|None, sex_col_dropped:bool)
    """
    if sex_ix is None:
        return X, y, "", None, False

    mode = str(CTX.get("SEX_RESTRICT_MODE", "majority")).lower()
    majority_prop = float(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP))
    max_other = int(CTX.get("SEX_RESTRICT_MAX_OTHER_CASES", 0))

    sex_col = X[:, sex_ix]
    y_bool = y.astype(bool)

    n_f_case = int(np.sum((sex_col == 0.0) & y_bool))
    n_m_case = int(np.sum((sex_col == 1.0) & y_bool))
    n_f_ctrl = int(np.sum((sex_col == 0.0) & (~y_bool)))
    n_m_ctrl = int(np.sum((sex_col == 1.0) & (~y_bool)))

    def _restrict_to(s: float, tag: str):
        if s == 0.0 and n_f_ctrl == 0:
            return X, y, "", "sex_no_controls_in_case_sex", False
        if s == 1.0 and n_m_ctrl == 0:
            return X, y, "", "sex_no_controls_in_case_sex", False
        keep_mask = (sex_col == s)
        cols_to_keep = np.arange(X.shape[1]) != sex_ix
        X_restr = X[keep_mask][:, cols_to_keep]
        y_restr = y[keep_mask]
        return X_restr, y_restr, f"{tag}_{int(s)}", None, True

    case_sexes_present = []
    if n_f_case > 0:
        case_sexes_present.append(0.0)
    if n_m_case > 0:
        case_sexes_present.append(1.0)

    if mode == "strict":
        if len(case_sexes_present) == 1:
            return _restrict_to(case_sexes_present[0], "sex_restricted_to")
        return X, y, "", None, False

    total_cases = n_f_case + n_m_case
    if total_cases > 0:
        if n_f_case >= n_m_case:
            prop = n_f_case / total_cases
            other = n_m_case
            s_dom = 0.0
        else:
            prop = n_m_case / total_cases
            other = n_f_case
            s_dom = 1.0
        if (prop >= majority_prop) or (other <= max_other):
            X2, Y2, note, skip, dropped = _restrict_to(s_dom, "sex_majority_restricted_to")
            if note:
                note = f"{note}:prop={prop:.3f};other_cases={other}"
            return X2, Y2, note, skip, dropped

    return X, y, "", None, False


# --- Bootstrap helpers ---
def _score_test_components(X_red: pd.DataFrame, y: pd.Series, target: str):
    const_ix = X_red.columns.get_loc('const') if 'const' in X_red.columns else None
    fit_red, _ = _fit_logit_ladder(X_red, y, const_ix=const_ix, prefer_mle_first=True)
    if fit_red is None:
        raise ValueError('reduced fit failed')
    eta = X_red.to_numpy(dtype=np.float64, copy=False) @ np.asarray(fit_red.params, dtype=np.float64)
    p_hat = expit(eta)
    W = p_hat * (1.0 - p_hat)
    return fit_red, p_hat, W


def _efficient_score_vector(target_vec: np.ndarray, X_red_mat: np.ndarray, W: np.ndarray):
    XTW = X_red_mat.T * W
    XtWX = XTW @ X_red_mat
    try:
        c = np.linalg.cholesky(XtWX)
        tmp = np.linalg.solve(c, XTW @ target_vec)
        beta_hat = np.linalg.solve(c.T, tmp)
    except np.linalg.LinAlgError:
        beta_hat = np.linalg.pinv(XtWX) @ (XTW @ target_vec)
    proj = X_red_mat @ beta_hat
    h = target_vec - proj
    denom = float(h.T @ (W * h))
    return h, denom

# --- Worker globals ---
# Populated by init_worker and read-only thereafter.
worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, finite_mask_worker = None, None, 0, None, None
# Array-based versions for performance
X_all, col_ix, worker_core_df_cols, worker_core_df_index = None, None, None, None
# Handle to keep shared memory alive in workers
_BASE_SHM_HANDLE = None
# Shared uniform matrix for bootstrap
U_boot, _BOOT_SHM_HANDLE, B_boot = None, None, 0
# Per-category cached indices only (no big matrices)
control_indices_by_cat = {}


def init_worker(base_shm_meta, core_cols, core_index, masks, ctx):
    """Initializer that attaches to the shared core design matrix."""
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
    _suppress_worker_warnings()
    _validate_ctx(ctx)
    global worker_core_df, allowed_mask_by_cat, N_core, CTX, finite_mask_worker
    global X_all, col_ix, worker_core_df_cols, worker_core_df_index, control_indices_by_cat, _BASE_SHM_HANDLE

    worker_core_df = None
    allowed_mask_by_cat, CTX = masks, ctx
    worker_core_df_cols = pd.Index(core_cols)
    worker_core_df_index = pd.Index(core_index)

    X_all, _BASE_SHM_HANDLE = io.attach_shared_ndarray(base_shm_meta)

    def _cleanup():
        try:
            if _BASE_SHM_HANDLE:
                _BASE_SHM_HANDLE.close()
        except Exception:
            pass

    atexit.register(_cleanup)

    N_core = X_all.shape[0]
    col_ix = {name: i for i, name in enumerate(worker_core_df_cols)}

    finite_mask_worker = np.isfinite(X_all).all(axis=1)

    control_indices_by_cat = {}
    for category, allowed_mask in allowed_mask_by_cat.items():
        control_mask = allowed_mask & finite_mask_worker
        control_indices_by_cat[category] = np.flatnonzero(control_mask)

    # Precompute per-category allowed-mask fingerprints once (used in skip checks)
    global allowed_fp_by_cat
    allowed_fp_by_cat = {}
    for category, idx in control_indices_by_cat.items():
        allowed_fp_by_cat[category] = _index_fingerprint(worker_core_df_index[idx] if len(idx) else pd.Index([]))

    bad = ~finite_mask_worker
    if bad.any():
        rows = worker_core_df_index[bad][:5].tolist()
        nonfinite_cols = [c for j, c in enumerate(worker_core_df_cols) if not np.isfinite(X_all[:, j]).all()]
        print(f"[Worker-{os.getpid()}] Non-finite sample rows={rows} cols={nonfinite_cols[:10]}", flush=True)
    print(f"[Worker-{os.getpid()}] Initialized with {N_core} subjects, {len(masks)} masks. Array is shared, no per-worker copy.", flush=True)

def init_lrt_worker(base_shm_meta, core_cols, core_index, masks, anc_series, ctx):
    """Initializer for LRT pools that also provides ancestry labels and context."""
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
    _suppress_worker_warnings()
    _validate_ctx(ctx)
    global worker_core_df, allowed_mask_by_cat, allowed_fp_by_cat, N_core, worker_anc_series, CTX, finite_mask_worker
    global X_all, col_ix, worker_core_df_cols, worker_core_df_index, _BASE_SHM_HANDLE

    worker_core_df = None
    allowed_mask_by_cat, CTX = masks, ctx
    worker_core_df_cols = pd.Index(core_cols)
    worker_core_df_index = pd.Index(core_index)
    worker_anc_series = anc_series.reindex(worker_core_df_index).str.lower()

    X_all, _BASE_SHM_HANDLE = io.attach_shared_ndarray(base_shm_meta)

    def _cleanup():
        try:
            if _BASE_SHM_HANDLE:
                _BASE_SHM_HANDLE.close()
        except Exception:
            pass

    atexit.register(_cleanup)

    N_core = X_all.shape[0]
    col_ix = {name: i for i, name in enumerate(worker_core_df_cols)}

    finite_mask_worker = np.isfinite(X_all).all(axis=1)

    # Precompute per-category allowed-mask fingerprints once (use allowed ∧ finite)
    allowed_fp_by_cat = {}
    for cat, mask in allowed_mask_by_cat.items():
        eff = mask & finite_mask_worker
        idx = np.flatnonzero(eff)
        allowed_fp_by_cat[cat] = _index_fingerprint(
            worker_core_df_index[idx] if idx.size else pd.Index([])
        )

    bad = ~finite_mask_worker
    if bad.any():
        rows = worker_core_df_index[bad][:5].tolist()
        nonfinite_cols = [c for j, c in enumerate(worker_core_df_cols) if not np.isfinite(X_all[:, j]).all()]
        print(f"[Worker-{os.getpid()}] Non-finite sample rows={rows} cols={nonfinite_cols[:10]}", flush=True)
    print(f"[LRT-Worker-{os.getpid()}] Initialized with {N_core} subjects, {len(masks)} masks, {worker_anc_series.nunique()} ancestries.", flush=True)


def init_boot_worker(base_shm_meta, boot_shm_meta, core_cols, core_index, masks, anc_series, ctx):
    init_lrt_worker(base_shm_meta, core_cols, core_index, masks, anc_series, ctx)
    global U_boot, _BOOT_SHM_HANDLE, B_boot
    U_boot, _BOOT_SHM_HANDLE = io.attach_shared_ndarray(boot_shm_meta)
    B_boot = U_boot.shape[1]
    print(f"[Boot-Worker-{os.getpid()}] Attached U matrix shape={U_boot.shape}", flush=True)

def _index_fingerprint(index):
    """Fast, order-insensitive fingerprint of a person_id index using XOR hashing."""
    h = 0
    n = 0
    for pid in map(str, index):
        h ^= int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)
        n += 1
    return f"{h:016x}:{n}"

def _mask_fingerprint(mask: np.ndarray, index: pd.Index) -> str:
    """Fast, order-insensitive fingerprint of a subset of an index using XOR hashing."""
    h = 0
    n = 0
    for pid in map(str, index[mask]):
        h ^= int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)
        n += 1
    return f"{h:016x}:{n}"

def _should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target, allowed_fp):
    """Determines if a model run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta:
        return False
    return (
        meta.get("model_columns") == list(core_df_cols) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp and
        meta.get("allowed_mask_fp") == allowed_fp and
        meta.get("min_cases") == CTX["MIN_CASES_FILTER"] and
        meta.get("min_ctrls") == CTX["MIN_CONTROLS_FILTER"]
    )


def _lrt_meta_should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target, allowed_fp):
    """Determines if an LRT run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta: return False
    return (
        meta.get("model_columns") == list(core_df_cols) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp and
        meta.get("allowed_mask_fp") == allowed_fp
    )


def _pos_in_current(orig_ix, current_ix_array):
    pos = np.flatnonzero(current_ix_array == orig_ix)
    return int(pos[0]) if pos.size else None


def run_single_model_worker(pheno_data, target_inversion, results_cache_dir):
    """CONSUMER: Runs a single model. Executed in a separate process using NumPy arrays."""
    s_name, category = pheno_data["name"], pheno_data["category"]
    case_idx_global = pheno_data.get("case_idx")
    if case_idx_global is None:
        cdr_code = pheno_data.get("cdr_codename", CTX.get("cdr_codename"))
        case_ids = io.load_pheno_cases_from_cache(s_name, CTX["CACHE_DIR"], cdr_code)
        idx = worker_core_df_index.get_indexer(case_ids)
        case_idx_global = idx[idx >= 0].astype(np.int32)
        case_ids_for_fp = case_ids
    else:
        case_idx_global = np.asarray(case_idx_global, dtype=np.int32)
        case_ids_for_fp = worker_core_df_index[case_idx_global] if case_idx_global.size > 0 else pd.Index([])
    s_name_safe = safe_basename(s_name)
    result_path = os.path.join(results_cache_dir, f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"

    try:
        # Prefer precomputed fingerprint if provided
        case_idx_fp = pheno_data.get("case_fp")
        if case_idx_fp is None:
            case_idx_fp = _index_fingerprint(case_ids_for_fp if case_ids_for_fp is not None
                                             else (worker_core_df_index[case_idx_global]
                                                   if case_idx_global.size > 0 else pd.Index([])))

        # Use per-category fingerprint computed in init_worker (fallback to one-off if missing)
        allowed_fp = allowed_fp_by_cat.get(category) if 'allowed_fp_by_cat' in globals() else None
        if allowed_fp is None:
            control_indices = control_indices_by_cat.get(category, np.array([], dtype=int))
            allowed_fp = _index_fingerprint(worker_core_df_index[control_indices])

        core_fp = _index_fingerprint(worker_core_df_index)

        # Repair missing meta for an existing result (enables fast skip)
        if os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False):
            _write_meta(
                meta_path, "phewas_result", s_name, category, target_inversion,
                worker_core_df_cols, core_fp, case_idx_fp,
                extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]}
            )
            print(f"[meta repaired] {s_name_safe}", flush=True)
            return

        if os.path.exists(result_path) and _should_skip(
            meta_path, worker_core_df_cols, core_fp, case_idx_fp, category, target_inversion, allowed_fp
        ):
            print(f"[skip cache-ok] {s_name_safe}", flush=True)
            return

        # --- Array-based data assembly ---
        case_indices_in_X_all = case_idx_global[finite_mask_worker[case_idx_global]]

        ctrl_idx = control_indices_by_cat.get(category, np.array([], dtype=int))
        # Fancy indexing returns a copy (that’s OK; it’s ephemeral and per-fit)
        X_controls = X_all[ctrl_idx].astype(np.float64, copy=False) if ctrl_idx.size else np.empty((0, X_all.shape[1]), dtype=np.float64)
        y_controls = np.zeros(len(X_controls), dtype=np.int8)

        X_cases = X_all[case_indices_in_X_all].astype(np.float64, copy=False)
        y_cases = np.ones(len(X_cases), dtype=np.int8)

        X_clean = np.vstack([X_controls, X_cases])
        y_clean = np.concatenate([y_controls, y_cases])

        # Free big temporaries ASAP after each fit
        del X_controls, y_controls, X_cases

        n_total, n_cases, n_ctrls = len(y_clean), int(y_clean.sum()), len(y_clean) - int(y_clean.sum())

        # --- Pre-fit sanity checks ---
        notes = []
        target_ix = col_ix.get(target_inversion)
        const_ix = col_ix.get('const')
        sex_ix = col_ix.get('sex')

        if target_ix is None:
            skip_reason = "target_inversion_not_in_cols"
        elif X_clean.shape[0] == 0:
            skip_reason = "no_valid_rows"
        elif X_clean.shape[0] > 0 and np.nanstd(X_clean[:, target_ix]) < 1e-12:
            skip_reason = "target_constant"
        else:
            skip_reason = ""

        if skip_reason:
            # ... (identical skip logic as before, just using different N vars)
            result_data = {"Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls, "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "Skip_Reason": skip_reason}
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_idx_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": skip_reason})
            return

        # --- Model fitting pipeline (on arrays) ---
        X_work, y_work = X_clean, y_clean
        # current_col_indices tracks the original column indices from X_all as columns are dropped.
        current_col_indices = np.arange(X_all.shape[1])
        assert current_col_indices.ndim == 1 and current_col_indices.dtype.kind in "iu"

        # Sex restriction behavior is driven by CTX keys: SEX_RESTRICT_MODE, SEX_RESTRICT_PROP, SEX_RESTRICT_MAX_OTHER_CASES.
        X_work, y_work, sex_note, sex_skip, sex_col_dropped = _apply_sex_restriction_np(X_work, y_work, sex_ix)
        if sex_note: notes.append(sex_note)

        if sex_skip:
            # Persist a deterministic skip result when sex restriction leaves no valid control stratum.
            n_total_used = len(y_work)
            n_cases_used = int(y_work.sum())
            n_ctrls_used = n_total_used - n_cases_used
            result_data = {
                "Phenotype": s_name,
                "N_Total": n_total,
                "N_Cases": n_cases,
                "N_Controls": n_ctrls,
                "Skip_Reason": sex_skip,
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
                "Model_Notes": ";".join(notes) if notes else ""
            }
            io.atomic_write_json(result_path, result_data)
            _write_meta(
                meta_path,
                "phewas_result",
                s_name,
                category,
                target_inversion,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_idx_fp,
                extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": sex_skip}
            )
            return

        if sex_col_dropped:
            current_col_indices = np.delete(current_col_indices, sex_ix)
            tgt_orig = col_ix.get(target_inversion) if target_inversion in col_ix else None
            cst_orig = col_ix.get('const') if 'const' in col_ix else None
            target_ix = _pos_in_current(tgt_orig, current_col_indices) if tgt_orig is not None else None
            const_ix = _pos_in_current(cst_orig, current_col_indices) if cst_orig is not None else None

        n_total_used, n_cases_used, n_ctrls_used = len(y_work), int(y_work.sum()), len(y_work) - int(y_work.sum())
        keep_ix_zv = {idx for idx in [target_ix, const_ix] if idx is not None}
        X_work_zv, kept_indices_after_zv = _drop_zero_variance_np(X_work, keep_ix=keep_ix_zv)

        current_col_indices = current_col_indices[kept_indices_after_zv]
        # After dropping zero-variance columns, find the new positions of our key columns.
        tgt_orig = col_ix.get(target_inversion) if target_inversion in col_ix else None
        cst_orig = col_ix.get('const') if 'const' in col_ix else None
        target_ix_mid = _pos_in_current(tgt_orig, current_col_indices) if (tgt_orig is not None) else None
        const_ix_mid = _pos_in_current(cst_orig, current_col_indices) if (cst_orig is not None) else None

        X_work_fd, kept_indices_after_fd = _drop_rank_deficiency_np(X_work_zv, keep_ix={i for i in [target_ix_mid, const_ix_mid] if i is not None})
        current_col_indices = current_col_indices[kept_indices_after_fd]
        # Final pass after rank-deficiency removal to get final positions.
        target_ix_final = _pos_in_current(tgt_orig, current_col_indices) if (tgt_orig is not None) else None
        const_ix_final = _pos_in_current(cst_orig, current_col_indices) if (cst_orig is not None) else None

        ok, reason, det = validate_min_counts_for_fit(y_work, stage_tag="phewas", extra_context={"phenotype": s_name})
        if not ok:
            print(f"[skip] name={s_name_safe} stage=phewas reason={reason} "
                  f"N={det['N']}/{det['N_cases']}/{det['N_ctrls']} "
                  f"min={det['min_cases']}/{det['min_ctrls']} neff={det['N_eff']:.1f}/{det['min_neff']:.1f}", flush=True)
            result_data = {"Phenotype": s_name, "N_Total": det['N'], "N_Cases": det['N_cases'], "N_Controls": det['N_ctrls'], "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "Skip_Reason": reason}
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion,
                        worker_core_df_cols, _index_fingerprint(worker_core_df_index),
                        case_idx_fp, extra={"skip_reason": reason, "counts": det, "allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
            return

        fit, fit_reason = _fit_logit_ladder(X_work_fd, y_work, const_ix=const_ix_final, prefer_mle_first=bool(sex_col_dropped))
        if fit_reason: notes.append(fit_reason)
        if fit is not None:
            _print_fit_diag(
                s_name_safe=s_name_safe,
                stage="phewas",
                model_tag="full",
                N_total=n_total_used,
                N_cases=n_cases_used,
                N_ctrls=n_ctrls_used,
                solver_tag=fit_reason,
                X=X_work_fd,
                y=y_work,
                params=fit.params,
                notes=notes
            )


        if not fit or target_ix_final is None or target_ix_final >= len(fit.params):
            # Persist a deterministic skip result when the fit fails or the target coefficient is unavailable.
            result_data = {
                "Phenotype": s_name,
                "N_Total": n_total,
                "N_Cases": n_cases,
                "N_Controls": n_ctrls,
                "Skip_Reason": "fit_failed_or_target_missing",
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
                "Model_Notes": ";".join(notes) if notes else ""
            }
            io.atomic_write_json(result_path, result_data)
            _write_meta(
                meta_path,
                "phewas_result",
                s_name,
                category,
                target_inversion,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_idx_fp,
                extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": "fit_failed_or_target_missing"}
            )
            return

        beta = float(fit.params[target_ix_final])
        final_is_mle = getattr(fit, "_final_is_mle", False)
        pval = np.nan
        if final_is_mle and hasattr(fit, "pvalues"):
            pval = float(fit.pvalues[target_ix_final])

        se = float(fit.bse[target_ix_final]) if hasattr(fit, "bse") and target_ix_final < len(fit.bse) else np.nan
        or_ci95_str = None
        if final_is_mle and np.isfinite(beta) and np.isfinite(se) and se > 0:
            lo, hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
            or_ci95_str = f"{lo:.3f},{hi:.3f}"

        result = {
            "Phenotype": s_name,
            "N_Total": n_total,
            "N_Cases": n_cases,
            "N_Controls": n_ctrls,
            "Beta": beta,
            "OR": np.exp(beta),
            "P_Value": pval,
            "OR_CI95": or_ci95_str,
            "Used_Ridge": not final_is_mle,
            "Final_Is_MLE": final_is_mle,
        }
        result.update({
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": ";".join(notes) if notes else "",
        })
        io.atomic_write_json(result_path, result)
        _write_meta(meta_path, "phewas_result", s_name, category, target_inversion, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_idx_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
        print(f"[fit OK] name={s_name_safe} OR={np.exp(beta):.3f} p={pval:.3e} notes={'|'.join(notes)}", flush=True)

    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()

def lrt_overall_worker(task):
    """Worker for Stage-1 overall LRT. Uses array-based pipeline."""
    s_name, cat, target = task["name"], task["category"], task["target"]
    s_name_safe = safe_basename(s_name)
    result_path = os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    res_path = os.path.join(CTX["RESULTS_CACHE_DIR"], f"{s_name_safe}.json")
    os.makedirs(CTX["RESULTS_CACHE_DIR"], exist_ok=True)
    try:
        pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")
        if not os.path.exists(pheno_path):
            io.atomic_write_json(result_path, {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_Overall_Reason": "missing_case_cache"})
            return

        # Prefer precomputed case_idx / case_fp; fall back to parquet
        case_idx = None
        if task.get("case_idx") is not None:
            case_idx = np.asarray(task["case_idx"], dtype=np.int32)
        if task.get("case_fp") is not None:
            case_fp = task["case_fp"]
            if case_idx is None:
                case_idx = np.array([], dtype=np.int32)
        else:
            case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
            idx = worker_core_df_index.get_indexer(case_ids)
            case_idx = idx[idx >= 0].astype(np.int32)
            case_fp = _index_fingerprint(worker_core_df_index[case_idx] if case_idx.size > 0 else pd.Index([]))

        # Use per-category allowed fingerprint computed once in worker
        allowed_fp = allowed_fp_by_cat.get(cat) if 'allowed_fp_by_cat' in globals() else _mask_fingerprint(
            allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool)), worker_core_df_index
        )

        core_fp = _index_fingerprint(worker_core_df_index)

        # Optional meta repair to enable skip when result exists but meta is missing
        if os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False):
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp,
                        extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
            print(f"[meta repaired] {s_name_safe} (LRT-Stage1)", flush=True)
            return

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df_cols, core_fp, case_fp, cat, target, allowed_fp
        ):
            if os.path.exists(res_path):
                print(f"[skip cache-ok] {s_name_safe} (LRT-Stage1)", flush=True)
                return
            else:
                print(f"[backfill] {s_name_safe} (LRT-Stage1) missing results JSON; regenerating", flush=True)

        allowed_mask = allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool))
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        anc_cols = [c for c in worker_core_df_cols if c.startswith("ANC_")]
        base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq'] + anc_cols
        base_ix = [col_ix[c] for c in base_cols]

        X_base = pd.DataFrame(
            X_all[np.ix_(valid_mask, base_ix)],
            index=worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index, dtype=np.int8)

        # Pre-sex-restriction counts to mirror main PheWAS semantics
        n_cases_pre = int(y_series.sum())
        n_ctrls_pre = int(len(y_series) - n_cases_pre)
        n_total_pre = int(len(y_series))

        Xb, yb, note, skip = _apply_sex_restriction(X_base, y_series)
        n_total_used, n_cases_used, n_ctrls_used = len(yb), int(yb.sum()), len(yb) - int(yb.sum())

        if skip:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_Overall_Reason": skip, "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used})
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": skip})

            # Write the PheWAS-style result as a skip to mirror main pass outputs
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False,
                "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used,
                "Model_Notes": note or "",
                "Skip_Reason": skip
            })
            io.atomic_write_json(res_path + ".meta.json", {
                "kind": "phewas_result",
                "s_name": s_name,
                "category": cat,
                "model_columns": list(worker_core_df_cols),
                "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"],
                "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target,
                "core_index_fp": _index_fingerprint(worker_core_df_index),
                "case_idx_fp": case_fp,
                "allowed_mask_fp": allowed_fp,
                "ridge_l2_base": CTX["RIDGE_L2_BASE"],
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            return

        ok, reason, det = validate_min_counts_for_fit(yb, stage_tag="lrt_stage1", extra_context={"phenotype": s_name})
        if not ok:
            print(f"[skip] name={s_name_safe} stage=LRT-Stage1 reason={reason} "
                  f"N={det['N']}/{det['N_cases']}/{det['N_ctrls']} "
                  f"min={det['min_cases']}/{det['min_ctrls']} neff={det['N_eff']:.1f}/{det['min_neff']:.1f}", flush=True)
            io.atomic_write_json(result_path, {
                "Phenotype": s_name,
                "P_LRT_Overall": np.nan,
                "LRT_Overall_Reason": reason,
                "N_Total_Used": det['N'],
                "N_Cases_Used": det['N_cases'],
                "N_Controls_Used": det['N_ctrls']
            })
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": reason, "counts": det})

            # Emit a PheWAS-style skip result to keep downstream shape identical
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False,
                "N_Total_Used": det['N'], "N_Cases_Used": det['N_cases'], "N_Controls_Used": det['N_ctrls'],
                "Model_Notes": note or "",
                "Skip_Reason": reason
            })
            io.atomic_write_json(res_path + ".meta.json", {
                "kind": "phewas_result",
                "s_name": s_name,
                "category": cat,
                "model_columns": list(worker_core_df_cols),
                "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"],
                "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target,
                "core_index_fp": _index_fingerprint(worker_core_df_index),
                "case_idx_fp": case_fp,
                "allowed_mask_fp": allowed_fp,
                "ridge_l2_base": CTX["RIDGE_L2_BASE"],
                "created_at": datetime.now(timezone.utc).isoformat()
            })
            return

        X_full_df = Xb

        # Prune the full model first to resolve rank deficiency.
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=(target,))
        X_full_zv = _drop_rank_deficient(X_full_zv, keep_cols=('const',), always_keep=(target,))

        target_ix = X_full_zv.columns.get_loc(target) if target in X_full_zv.columns else None

        if target_ix is None:
            skip_reason = "target_dropped_in_pruning"
            io.atomic_write_json(result_path, {
                "Phenotype": s_name,
                "P_LRT_Overall": np.nan,
                "LRT_Overall_Reason": skip_reason,
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
            })
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan,
                "OR": np.nan,
                "P_Value": np.nan,
                "OR_CI95": None,
                "Used_Ridge": False,
                "Final_Is_MLE": False,
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
                "Model_Notes": note or "",
                "Skip_Reason": skip_reason,
            })
            io.atomic_write_json(res_path + ".meta.json", {
                "kind": "phewas_result",
                "s_name": s_name,
                "category": cat,
                "model_columns": list(worker_core_df_cols),
                "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"],
                "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target,
                "core_index_fp": _index_fingerprint(worker_core_df_index),
                "case_idx_fp": case_fp,
                "allowed_mask_fp": allowed_fp,
                "ridge_l2_base": CTX["RIDGE_L2_BASE"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            _write_meta(
                meta_path,
                "lrt_overall",
                s_name,
                cat,
                target,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_fp,
                extra={
                    "allowed_mask_fp": allowed_fp,
                    "ridge_l2_base": CTX["RIDGE_L2_BASE"],
                    "skip_reason": skip_reason,
                },
            )
            return

        # The reduced model MUST be a subset of the pruned full model for the LRT to be valid.
        # Construct it by dropping the target column from the *already pruned* full model columns.
        if target in X_full_zv.columns:
            red_cols = [c for c in X_full_zv.columns if c != target]
            X_red_zv = X_full_zv[red_cols]
        else:
            # If the target was dropped during pruning, the models are identical.
            X_red_zv = X_full_zv

        const_ix_red = X_red_zv.columns.get_loc('const') if 'const' in X_red_zv.columns else None
        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red, prefer_mle_first=bool(note))
        fit_full, reason_full = _fit_logit_ladder(X_full_zv, yb, const_ix=const_ix_full, prefer_mle_first=bool(note))

        if fit_red is not None:
            _print_fit_diag(
                s_name_safe=s_name_safe,
                stage="LRT-Stage1",
                model_tag="reduced",
                N_total=n_total_used,
                N_cases=n_cases_used,
                N_ctrls=n_ctrls_used,
                solver_tag=reason_red,
                X=X_red_zv,
                y=yb,
                params=fit_red.params,
                notes=[note] if note else []
            )
        if fit_full is not None:
            _print_fit_diag(
                s_name_safe=s_name_safe,
                stage="LRT-Stage1",
                model_tag="full",
                N_total=n_total_used,
                N_cases=n_cases_used,
                N_ctrls=n_ctrls_used,
                solver_tag=reason_full,
                X=X_full_zv,
                y=yb,
                params=fit_full.params,
                notes=[note] if note else []
            )
        red_is_mle = getattr(fit_red, "_final_is_mle", False)
        full_is_mle = getattr(fit_full, "_final_is_mle", False)

        out = {
            "Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_df_Overall": np.nan, "Model_Notes": note,
            "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used,
        }
        if red_is_mle and full_is_mle and fit_full and fit_red and hasattr(fit_full, 'llf') and hasattr(fit_red, 'llf') and fit_full.llf >= fit_red.llf:
            r_full, r_red = np.linalg.matrix_rank(X_full_zv.to_numpy()), np.linalg.matrix_rank(X_red_zv.to_numpy())
            df_lrt = max(0, int(r_full - r_red))
            if df_lrt > 0:
                llr = 2 * (fit_full.llf - fit_red.llf)
                out["P_LRT_Overall"] = sp_stats.chi2.sf(llr, df_lrt)
                out["LRT_df_Overall"] = df_lrt
                print(f"[LRT-Stage1-OK] name={s_name_safe} p={out['P_LRT_Overall']:.3e} df={df_lrt}", flush=True)
            else:
                out["LRT_Overall_Reason"] = "zero_df_lrt"
        else:
            out["LRT_Overall_Reason"] = "penalized_fit_in_path" if (fit_red or fit_full) else "fit_failed"

        # --- Emit PheWAS-style per-phenotype result from the FULL fit ---
        beta_full = np.nan
        or_val = np.nan
        wald_p = np.nan
        ci95 = None
        final_is_mle = getattr(fit_full, "_final_is_mle", False)

        if fit_full is not None and target_ix is not None:
            try:
                params = getattr(fit_full, "params", None)
                if hasattr(params, "index"):
                    beta_full = float(params[target])
                else:
                    beta_full = float(params[target_ix])
                or_val = float(np.exp(beta_full))
            except Exception:
                pass

            if final_is_mle and hasattr(fit_full, "pvalues"):
                try:
                    pvals = fit_full.pvalues
                    try:
                        wald_p = float(pvals[target_ix])
                    except Exception:
                        try:
                            wald_p = float(pvals.loc[target])
                        except Exception:
                            wald_p = np.nan
                    se = np.nan
                    if hasattr(fit_full, "bse"):
                        try:
                            se = float(fit_full.bse[target_ix])
                        except Exception:
                            try:
                                se = float(fit_full.bse.loc[target])
                            except Exception:
                                se = np.nan
                    if np.isfinite(se) and se > 0:
                        lo, hi = np.exp(beta_full - 1.96 * se), np.exp(beta_full + 1.96 * se)
                        ci95 = f"{lo:.3f},{hi:.3f}"
                except Exception:
                    pass

        model_notes = [note] if note else []
        if isinstance(reason_full, str) and reason_full:
            model_notes.append(reason_full)
        if isinstance(reason_red, str) and reason_red:
            model_notes.append(reason_red)

        res_record = {
            "Phenotype": s_name,
            "N_Total": n_total_pre,
            "N_Cases": n_cases_pre,
            "N_Controls": n_ctrls_pre,
            "Beta": beta_full,
            "OR": or_val,
            "P_Value": wald_p,
            "OR_CI95": ci95,
            "Used_Ridge": (not final_is_mle),
            "Final_Is_MLE": bool(final_is_mle),
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": ";".join(model_notes)
        }
        final_cols_names = list(X_full_zv.columns)
        final_cols_pos = [col_ix.get(c, -1) for c in final_cols_names]

        io.atomic_write_json(res_path, res_record)
        io.atomic_write_json(res_path + ".meta.json", {
            "kind": "phewas_result",
            "s_name": s_name,
            "category": cat,
            "model_columns": list(worker_core_df_cols),
            "num_pcs": CTX["NUM_PCS"],
            "min_cases": CTX["MIN_CASES_FILTER"],
            "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
            "target": target,
            "core_index_fp": _index_fingerprint(worker_core_df_index),
            "case_idx_fp": case_fp,
            "allowed_mask_fp": allowed_fp,
            "ridge_l2_base": CTX["RIDGE_L2_BASE"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "final_cols_names": final_cols_names,
            "final_cols_pos": final_cols_pos,
            "full_llf": float(getattr(fit_full, "llf", np.nan)),
            "full_is_mle": bool(final_is_mle),
            "used_firth": bool(getattr(fit_full, "_used_firth", False)),
            "prune_recipe_version": "zv+greedy-rank-v1",
        })

        io.atomic_write_json(result_path, out)
        _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols,
                    _index_fingerprint(worker_core_df_index), case_fp,
                    extra={
                        "allowed_mask_fp": allowed_fp,
                        "ridge_l2_base": CTX["RIDGE_L2_BASE"],
                        "final_cols_names": final_cols_names,
                        "final_cols_pos": final_cols_pos,
                        "full_llf": float(getattr(fit_full, "llf", np.nan)),
                        "full_is_mle": bool(final_is_mle),
                        "used_firth": bool(getattr(fit_full, "_used_firth", False)),
                        "prune_recipe_version": "zv+greedy-rank-v1",
                    })
    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()

def bootstrap_overall_worker(task):
    s_name, cat, target = task["name"], task["category"], task["target"]
    s_name_safe = safe_basename(s_name)
    boot_dir = CTX["BOOT_OVERALL_CACHE_DIR"]
    os.makedirs(boot_dir, exist_ok=True)
    tnull_dir = os.path.join(boot_dir, "t_null")
    os.makedirs(tnull_dir, exist_ok=True)
    result_path = os.path.join(boot_dir, f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    res_path = os.path.join(CTX["RESULTS_CACHE_DIR"], f"{s_name_safe}.json")
    try:
        pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")
        if not os.path.exists(pheno_path):
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": "missing_case_cache"})
            return

        case_idx = None
        if task.get("case_idx") is not None:
            case_idx = np.asarray(task["case_idx"], dtype=np.int32)
        if task.get("case_fp") is not None:
            case_fp = task["case_fp"]
            if case_idx is None:
                case_idx = np.array([], dtype=np.int32)
        else:
            case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
            idx = worker_core_df_index.get_indexer(case_ids)
            case_idx = idx[idx >= 0].astype(np.int32)
            case_fp = _index_fingerprint(worker_core_df_index[case_idx] if case_idx.size > 0 else pd.Index([]))

        allowed_mask = allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool))
        allowed_fp = allowed_fp_by_cat.get(cat) if 'allowed_fp_by_cat' in globals() else None
        if allowed_fp is None:
            allowed_fp = _mask_fingerprint(allowed_mask, worker_core_df_index)
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        anc_cols = [c for c in worker_core_df_cols if c.startswith("ANC_")]
        base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq'] + anc_cols
        base_ix = [col_ix[c] for c in base_cols]
        X_base = pd.DataFrame(
            X_all[np.ix_(valid_mask, base_ix)],
            index=worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index, dtype=np.int8)

        n_cases_pre = int(y_series.sum())
        n_ctrls_pre = int(len(y_series) - n_cases_pre)
        n_total_pre = int(len(y_series))

        Xb, yb, note, skip = _apply_sex_restriction(X_base, y_series)
        n_total_used, n_cases_used = len(yb), int(yb.sum())
        n_ctrls_used = n_total_used - n_cases_used
        if skip:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": skip, "N_Total_Used": n_total_used})
            _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp})
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False,
                "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used,
                "Model_Notes": note or "", "Skip_Reason": skip
            })
            return

        ok, reason, det = validate_min_counts_for_fit(yb, stage_tag="boot_stage1", extra_context={"phenotype": s_name})
        if not ok:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": reason, "N_Total_Used": det['N'], "N_Cases_Used": det['N_cases'], "N_Controls_Used": det['N_ctrls']})
            _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "counts": det})
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False,
                "N_Total_Used": det['N'], "N_Cases_Used": det['N_cases'], "N_Controls_Used": det['N_ctrls'],
                "Model_Notes": reason
            })
            return

        X_full_df = Xb
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=(target,))
        X_full_zv = _drop_rank_deficient(X_full_zv, keep_cols=('const',), always_keep=(target,))
        if target not in X_full_zv.columns:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": "target_dropped_in_pruning"})
            return
        red_cols = [c for c in X_full_zv.columns if c != target]
        X_red_zv = X_full_zv[red_cols]

        fit_red, p_hat, W = _score_test_components(X_red_zv, yb, target)
        t_vec = X_full_zv[target].to_numpy(dtype=np.float64, copy=False)
        Xr = X_red_zv.to_numpy(dtype=np.float64, copy=False)
        h, denom = _efficient_score_vector(t_vec, Xr, W)
        if not np.isfinite(denom) or denom <= 0:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": "nonpos_denom"})
            return
        resid = yb.to_numpy(dtype=np.float64, copy=False) - p_hat
        S_obs = float(h @ resid)
        T_obs = (S_obs * S_obs) / denom

        pos = worker_core_df_index.get_indexer(X_red_zv.index)
        pos = pos[pos >= 0]
        U_slice = U_boot[pos, :]
        B = U_slice.shape[1]
        T_b = np.empty(B, dtype=np.float64)
        for j0 in range(0, B, 64):
            j1 = min(B, j0 + 64)
            Ystar = (U_slice[:, j0:j1] < p_hat[:, None]).astype(np.float64, copy=False)
            R = Ystar - p_hat[:, None]
            S = h @ R
            T_b[j0:j1] = (S * S) / denom
        p_emp = float((1.0 + np.sum(T_b >= T_obs)) / (1.0 + B))

        io.atomic_write_json(result_path, {
            "Phenotype": s_name,
            "T_OBS": T_obs,
            "P_EMP": p_emp,
            "B": int(B),
            "Test_Stat": "score",
            "Boot": "bernoulli",
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": note or ""
        })
        np.save(os.path.join(tnull_dir, f"{s_name_safe}.npy"), T_b)
        _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})

        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None
        fit_full, reason_full = _fit_logit_ladder(X_full_zv, yb, const_ix=const_ix_full, prefer_mle_first=bool(note))
        beta_full, wald_p, ci95, or_val, final_is_mle = np.nan, np.nan, None, np.nan, getattr(fit_full, "_final_is_mle", False)
        if fit_full is not None and target in X_full_zv.columns:
            beta_full = float(getattr(fit_full, "params", pd.Series(np.nan, index=X_full_zv.columns))[target])
            or_val = float(np.exp(beta_full))
            if final_is_mle and hasattr(fit_full, "pvalues"):
                wald_p = float(getattr(fit_full, "pvalues", pd.Series(np.nan, index=X_full_zv.columns))[target])
                se = float(getattr(fit_full, "bse", pd.Series(np.nan, index=X_full_zv.columns))[target]) if hasattr(fit_full, "bse") else np.nan
                if np.isfinite(se) and se > 0:
                    lo, hi = np.exp(beta_full - 1.96*se), np.exp(beta_full + 1.96*se)
                    ci95 = f"{lo:.3f},{hi:.3f}"
        io.atomic_write_json(res_path, {
            "Phenotype": s_name,
            "N_Total": n_total_pre,
            "N_Cases": n_cases_pre,
            "N_Controls": n_ctrls_pre,
            "Beta": beta_full,
            "OR": or_val,
            "P_Value": wald_p,
            "OR_CI95": ci95,
            "Used_Ridge": (not final_is_mle),
            "Final_Is_MLE": bool(final_is_mle),
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": reason_full or note or ""
        })

    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()

def lrt_followup_worker(task):
    """Worker for Stage-2 ancestry×dosage LRT and per-ancestry splits. Uses array-based pipeline."""
    s_name, category, target = task["name"], task["category"], task["target"]
    s_name_safe = safe_basename(s_name)
    result_path = os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    try:
        pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")
        if not os.path.exists(pheno_path):
            io.atomic_write_json(result_path, {'Phenotype': s_name, 'P_LRT_AncestryxDosage': np.nan, 'LRT_df': np.nan, 'LRT_Reason': "missing_case_cache"})
            return

        case_idx = None
        if task.get("case_idx") is not None:
            case_idx = np.asarray(task["case_idx"], dtype=np.int32)
        if task.get("case_fp") is not None:
            case_fp = task["case_fp"]
            if case_idx is None:
                case_idx = np.array([], dtype=np.int32)
        else:
            case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
            idx = worker_core_df_index.get_indexer(case_ids)
            case_idx = idx[idx >= 0].astype(np.int32)
            case_fp = _index_fingerprint(worker_core_df_index[case_idx] if case_idx.size > 0 else pd.Index([]))

        allowed_fp = allowed_fp_by_cat.get(category) if 'allowed_fp_by_cat' in globals() else _mask_fingerprint(
            allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool)), worker_core_df_index
        )

        core_fp = _index_fingerprint(worker_core_df_index)

        if os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False):
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, core_fp, case_fp,
                        extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
            print(f"[meta repaired] {s_name_safe} (LRT-Stage2)", flush=True)
            return

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df_cols, core_fp, case_fp, category, target, allowed_fp
        ):
            print(f"[skip cache-ok] {s_name_safe} (LRT-Stage2)", flush=True)
            return

        allowed_mask = allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool))
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq']
        base_ix = [col_ix[c] for c in base_cols]
        X_base_df = pd.DataFrame(
            X_all[np.ix_(valid_mask, base_ix)],
            index=worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base_df.index, dtype=np.int8)

        Xb, yb, note, skip = _apply_sex_restriction(X_base_df, y_series)
        out = {'Phenotype': s_name, 'P_LRT_AncestryxDosage': np.nan, 'LRT_df': np.nan, 'LRT_Reason': "", "Model_Notes": note}
        if skip:
            out['LRT_Reason'] = skip; io.atomic_write_json(result_path, out)
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": skip})
            return

        anc_vec = worker_anc_series.loc[Xb.index]
        levels = pd.Index(anc_vec.dropna().unique(), dtype=str).tolist()
        levels_sorted = (['eur'] if 'eur' in levels else []) + [x for x in sorted(levels) if x != 'eur']
        out['LRT_Ancestry_Levels'] = ",".join(levels_sorted)

        if len(levels_sorted) < 2:
            out['LRT_Reason'] = "only_one_ancestry_level"; io.atomic_write_json(result_path, out)
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": "only_one_ancestry_level"})
            return

        if 'eur' in levels:
            anc_cat = pd.Categorical(anc_vec, categories=['eur'] + sorted([x for x in levels if x != 'eur']))
        else:
            anc_cat = pd.Categorical(anc_vec)

        A_df = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True).reindex(Xb.index, fill_value=0)
        X_red_df = Xb.join(A_df)

        # Use vectorized broadcasting to create interaction terms
        target_col_np = X_red_df[target].to_numpy(copy=False)[:, None]
        A_np = A_df.to_numpy(copy=False)
        interaction_mat = target_col_np * A_np
        interaction_cols = [f"{target}:{c}" for c in A_df.columns]
        X_full_df = pd.concat([X_red_df, pd.DataFrame(interaction_mat, index=X_red_df.index, columns=interaction_cols)], axis=1)

        # Prune the full model (with interactions) first.
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=[target] + interaction_cols)
        X_full_zv = _drop_rank_deficient(X_full_zv, keep_cols=('const',), always_keep=[target] + interaction_cols)

        # Construct the reduced model by dropping interaction terms from the pruned full model.
        # This ensures the reduced model is properly nested within the full model.
        kept_interaction_cols = [c for c in interaction_cols if c in X_full_zv.columns]
        red_cols = [c for c in X_full_zv.columns if c not in kept_interaction_cols]
        X_red_zv = X_full_zv[red_cols]

        const_ix_red = X_red_zv.columns.get_loc('const') if 'const' in X_red_zv.columns else None
        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red, prefer_mle_first=bool(note))
        fit_full, reason_full = _fit_logit_ladder(X_full_zv, yb, const_ix=const_ix_full, prefer_mle_first=bool(note))

        if fit_red is not None:
            _print_fit_diag(
                s_name_safe=s_name_safe,
                stage="LRT-Stage2",
                model_tag="reduced",
                N_total=len(yb),
                N_cases=int(yb.sum()),
                N_ctrls=int(len(yb) - int(yb.sum())),
                solver_tag=reason_red,
                X=X_red_zv,
                y=yb,
                params=fit_red.params,
                notes=[note] if note else []
            )
        if fit_full is not None:
            _print_fit_diag(
                s_name_safe=s_name_safe,
                stage="LRT-Stage2",
                model_tag="full",
                N_total=len(yb),
                N_cases=int(yb.sum()),
                N_ctrls=int(len(yb) - int(yb.sum())),
                solver_tag=reason_full,
                X=X_full_zv,
                y=yb,
                params=fit_full.params,
                notes=[note] if note else []
            )
        red_is_mle, full_is_mle = getattr(fit_red, "_final_is_mle", False), getattr(fit_full, "_final_is_mle", False)

        if red_is_mle and full_is_mle and fit_full and fit_red and hasattr(fit_full, 'llf') and hasattr(fit_red, 'llf') and fit_full.llf >= fit_red.llf:
            r_full, r_red = np.linalg.matrix_rank(X_full_zv.to_numpy()), np.linalg.matrix_rank(X_red_zv.to_numpy())
            df_lrt = max(0, int(r_full - r_red))
            if df_lrt > 0:
                llr = 2 * (fit_full.llf - fit_red.llf)
                out['P_LRT_AncestryxDosage'] = sp_stats.chi2.sf(llr, df_lrt)
                out['LRT_df'] = df_lrt
            else: out['LRT_Reason'] = "zero_df_lrt"
        else: out['LRT_Reason'] = "penalized_fit_in_path"

        for anc in levels_sorted:
            anc_mask = (anc_vec == anc).to_numpy()
            X_anc, y_anc = Xb[anc_mask], yb[anc_mask]

            _, n_cases_anc, n_ctrls_anc, _ = _counts_from_y(y_anc)
            out[f"{anc.upper()}_N"], out[f"{anc.upper()}_N_Cases"], out[f"{anc.upper()}_N_Controls"] = len(y_anc), n_cases_anc, n_ctrls_anc

            ok, reason, det = validate_min_counts_for_fit(
                y_anc,
                stage_tag=f"lrt_followup:{anc}",
                extra_context={"phenotype": s_name, "ancestry": anc},
                cases_key="PER_ANC_MIN_CASES",
                controls_key="PER_ANC_MIN_CONTROLS"
            )
            if not ok:
                print(f"[skip] name={s_name_safe} stage=LRT-Followup anc={anc} reason={reason} "
                      f"N={det['N']}/{det['N_cases']}/{det['N_ctrls']} "
                      f"min={det['min_cases']}/{det['min_ctrls']} neff={det['N_eff']:.1f}/{det['min_neff']:.1f}", flush=True)
                out[f"{anc.upper()}_REASON"] = reason
                continue

            X_anc_zv = _drop_zero_variance(X_anc, keep_cols=('const',), always_keep=(target,))
            X_anc_zv = _drop_rank_deficient(X_anc_zv, keep_cols=('const',), always_keep=(target,))
            const_ix_anc = X_anc_zv.columns.get_loc('const') if 'const' in X_anc_zv.columns else None
            fit, _ = _fit_logit_ladder(X_anc_zv, y_anc, const_ix=const_ix_anc, prefer_mle_first=bool(note))

            if fit and target in getattr(fit, "params", {}):
                beta = float(fit.params[target])
                out[f"{anc.upper()}_OR"] = float(np.exp(beta))
                final_is_mle = getattr(fit, "_final_is_mle", False)
                if final_is_mle and hasattr(fit, "pvalues"):
                    pval = float(fit.pvalues.get(target, np.nan))
                    out[f"{anc.upper()}_P"] = pval
                    se = float(getattr(fit, "bse", {}).get(target, np.nan)) if hasattr(fit, "bse") else np.nan
                    if np.isfinite(se) and se > 0:
                        lo, hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
                        out[f"{anc.upper()}_CI95"] = f"{lo:.3f},{hi:.3f}"
                else:
                    out[f"{anc.upper()}_REASON"] = "subset_fit_penalized"
            else:
                out[f"{anc.upper()}_REASON"] = "subset_fit_failed"


        io.atomic_write_json(result_path, out)
        _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()
