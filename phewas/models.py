import os
import gc
import hashlib
import warnings
from datetime import datetime, timezone
import traceback
import sys
import atexit
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.special import expit
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

from . import iox as io

CTX = {}  # Worker context with constants from run.py
allowed_fp_by_cat = {}

# --- inference behavior toggles ---
DEFAULT_PREFER_FIRTH_ON_RIDGE = True
DEFAULT_ALLOW_PENALIZED_WALD = False

# Thresholds to detect unusable confidence intervals from penalized fits
PENALIZED_CI_SPAN_RATIO = 1e3
PENALIZED_CI_LO_OR_MAX = 1e-3
PENALIZED_CI_HI_OR_MIN = 1e3

MLE_SE_MAX_ALL = 10.0
MLE_SE_MAX_TARGET = 5.0
MLE_MAX_ABS_XB = 15.0
MLE_FRAC_P_EXTREME = 0.02
EPV_MIN_FOR_MLE = 10.0
TARGET_VAR_MIN_FOR_MLE = 1e-8
PROFILE_MAX_ABS_BETA = 40.0
BOOTSTRAP_DEFAULT_B = 2000
BOOTSTRAP_MAX_B = 131072
BOOTSTRAP_SEQ_ALPHA = 0.01
BOOTSTRAP_CHUNK = 4096
BOOTSTRAP_STREAM_TARGET_BYTES = 32 * 1024 * 1024  # ~32 MiB cap per chunk

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
        "min_neff": CTX.get("MIN_NEFF_FILTER", DEFAULT_MIN_NEFF),
        "target": target,
        "core_index_fp": core_idx_fp,
        "case_idx_fp": case_fp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ctx_tag": CTX.get("CTX_TAG"),
        "cache_version_tag": CTX.get("CACHE_VERSION_TAG"),
        "cdr_codename": CTX.get("cdr_codename"),
        "mode": CTX.get("MODE"),
        "selection": CTX.get("SELECTION"),
    }
    data_keys = CTX.get("DATA_KEYS")
    if data_keys:
        base["data_keys"] = data_keys
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


def _fmt_num(x):
    if not np.isfinite(x):
        if np.isnan(x):
            return "NA"
        return "+inf" if x > 0 else "-inf"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax > 1e3):
        return f"{x:.3e}"
    return f"{x:.3f}"


def _fmt_ci(lo, hi):
    return f"{_fmt_num(lo)},{_fmt_num(hi)}"


def _bootstrap_rng(seed_key):
    seed_base = CTX.get("BOOT_SEED_BASE")
    if not isinstance(seed_key, (tuple, list)):
        seed_key = (seed_key,)
    h = hashlib.blake2b(digest_size=16)
    if seed_base is None:
        h.update(b"default_boot_seed")
    else:
        h.update(str(seed_base).encode("utf-8"))
    for item in seed_key:
        if isinstance(item, (bytes, bytearray)):
            h.update(item)
        elif isinstance(item, (float, np.floating)):
            h.update(np.float64(item).tobytes())
        elif isinstance(item, (int, np.integer)):
            h.update(int(item).to_bytes(8, byteorder="little", signed=True))
        elif item is None:
            h.update(b"None")
        else:
            h.update(str(item).encode("utf-8"))
    seed_bytes = h.digest()[:8]
    seed = int.from_bytes(seed_bytes, "little", signed=False)
    return np.random.default_rng(seed)


def _clopper_pearson_interval(successes, total, alpha=0.01):
    if total <= 0:
        return 0.0, 1.0
    if successes <= 0:
        lower = 0.0
    else:
        lower = float(sp_stats.beta.ppf(alpha / 2.0, successes, total - successes + 1))
    if successes >= total:
        upper = 1.0
    else:
        upper = float(sp_stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return lower, upper


def _ok_mle_fit(fit, X, y, target_ix=None,
                se_max_all=None, se_max_target=None,
                max_abs_xb=None, frac_extreme=None):
    if fit is None or (not hasattr(fit, "bse")):
        return False
    se_max_all = float(CTX.get("MLE_SE_MAX_ALL", MLE_SE_MAX_ALL) if se_max_all is None else se_max_all)
    se_max_target = float(CTX.get("MLE_SE_MAX_TARGET", MLE_SE_MAX_TARGET) if se_max_target is None else se_max_target)
    max_abs_xb = float(CTX.get("MLE_MAX_ABS_XB", MLE_MAX_ABS_XB) if max_abs_xb is None else max_abs_xb)
    frac_extreme = float(CTX.get("MLE_FRAC_P_EXTREME", MLE_FRAC_P_EXTREME) if frac_extreme is None else frac_extreme)
    try:
        bse = np.asarray(fit.bse, dtype=np.float64)
    except Exception:
        return False
    if bse.ndim == 0:
        bse = np.array([float(bse)], dtype=np.float64)
    if not np.all(np.isfinite(bse)):
        return False
    if np.nanmax(bse) > se_max_all:
        return False
    if target_ix is not None and 0 <= int(target_ix) < bse.size:
        if (not np.isfinite(bse[int(target_ix)])) or (bse[int(target_ix)] > se_max_target):
            return False
    try:
        params = getattr(fit, "params", None)
        if params is None:
            return False
        max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, params)
        if (np.isfinite(max_abs_linpred) and max_abs_linpred > max_abs_xb):
            return False
        if (np.isfinite(frac_lo) and frac_lo > frac_extreme) or (np.isfinite(frac_hi) and frac_hi > frac_extreme):
            return False
    except Exception:
        return False
    return True


def _mle_prefit_ok(X, y, target_ix=None, const_ix=None):
    X_np = X.to_numpy(dtype=np.float64, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        return False
    n = float(X_np.shape[0])
    if n <= 0:
        return False
    n_cases = float(np.sum(y_np))
    n_ctrls = n - n_cases
    if n_cases <= 0 or n_ctrls <= 0:
        return False
    p_eff = int(X_np.shape[1])
    if const_ix is not None and 0 <= int(const_ix) < X_np.shape[1]:
        p_eff = max(1, p_eff - 1)
    p_eff = max(1, p_eff)
    epv = min(n_cases, n_ctrls) / float(p_eff)
    epv_min = float(CTX.get("EPV_MIN_FOR_MLE", EPV_MIN_FOR_MLE))
    if epv < epv_min:
        return False
    if target_ix is not None and 0 <= int(target_ix) < X_np.shape[1]:
        tgt_std = float(np.nanstd(X_np[:, int(target_ix)]))
        if tgt_std < float(CTX.get("TARGET_VAR_MIN_FOR_MLE", TARGET_VAR_MIN_FOR_MLE)):
            return False
    return True


def _logit_mle_refit_offset(X, y, offset=None, maxiter=200, tol=1e-8):
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        raise ValueError("design/response mismatch for MLE offset refit")
    n, p = X_np.shape
    if offset is None:
        offset = np.zeros(n, dtype=np.float64)
    else:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (n,):
            raise ValueError("offset shape mismatch")
    beta = np.zeros(p, dtype=np.float64)
    converged = False
    for _ in range(int(maxiter)):
        eta = np.clip(offset + X_np @ beta, -35.0, 35.0)
        p_hat = expit(eta)
        W = p_hat * (1.0 - p_hat)
        z = eta + (y_np - p_hat) / np.clip(W, 1e-12, None)
        XTW = X_np.T * W
        XtWX = XTW @ X_np
        XtWz = XTW @ z
        try:
            delta = np.linalg.solve(XtWX, XtWz - XtWX @ beta)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(XtWX) @ (XtWz - XtWX @ beta)
        beta_new = beta + delta
        if not np.all(np.isfinite(beta_new)):
            break
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    if not converged:
        raise RuntimeError("MLE offset refit failed to converge")
    eta = np.clip(offset + X_np @ beta, -35.0, 35.0)
    p_hat = expit(eta)
    llf = float(np.sum(y_np * np.log(p_hat) + (1.0 - y_np) * np.log(1.0 - p_hat)))
    W = p_hat * (1.0 - p_hat)
    XTW = X_np.T * W
    XtWX = XTW @ X_np
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))

    class _Res:
        pass

    res = _Res()
    res.params = beta
    res.bse = bse
    res.llf = llf
    setattr(res, "_final_is_mle", True)
    setattr(res, "_used_firth", False)
    return res


def _firth_refit_offset(X, y, offset=None, maxiter=200, tol=1e-8):
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        raise ValueError("design/response mismatch for Firth offset refit")
    n, p = X_np.shape
    if offset is None:
        offset = np.zeros(n, dtype=np.float64)
    else:
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (n,):
            raise ValueError("offset shape mismatch")
    beta = np.zeros(p, dtype=np.float64)
    converged = False
    for _ in range(int(maxiter)):
        eta = np.clip(offset + X_np @ beta, -35.0, 35.0)
        p_hat = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
        W = p_hat * (1.0 - p_hat)
        XTW = X_np.T * W
        XtWX = XTW @ X_np
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
        h = _leverages_batched(X_np, XtWX_inv, W)
        score = X_np.T @ (y_np - p_hat + (0.5 - p_hat) * h)
        delta = XtWX_inv @ score
        beta_new = beta + delta
        if not np.all(np.isfinite(beta_new)):
            break
        if np.max(np.abs(delta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    if not converged:
        raise RuntimeError("Firth offset refit failed to converge")
    eta = np.clip(offset + X_np @ beta, -35.0, 35.0)
    p_hat = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
    W = p_hat * (1.0 - p_hat)
    XTW = X_np.T * W
    XtWX = XTW @ X_np
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    loglik = float(np.sum(y_np * np.log(p_hat) + (1.0 - y_np) * np.log(1.0 - p_hat)))
    sign_det, logdet = np.linalg.slogdet(XtWX)
    pll = loglik + 0.5 * logdet if sign_det > 0 else -np.inf

    class _Res:
        pass

    res = _Res()
    res.params = beta
    res.bse = bse
    res.llf = float(pll)
    setattr(res, "_final_is_mle", False)
    setattr(res, "_used_firth", True)
    return res


def _profile_ci_beta(X_full, y, target_ix, fit_full, kind="mle", alpha=0.05, max_abs_beta=None):
    max_abs_beta = float(CTX.get("PROFILE_MAX_ABS_BETA", PROFILE_MAX_ABS_BETA) if max_abs_beta is None else max_abs_beta)
    X_np = X_full.to_numpy(dtype=np.float64, copy=False) if hasattr(X_full, "to_numpy") else np.asarray(X_full, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False, "method": None}
    if target_ix is None or target_ix < 0 or target_ix >= X_np.shape[1]:
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False, "method": None}
    params = getattr(fit_full, "params", None)
    if params is None:
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False, "method": None}
    beta_hat = float(np.asarray(params, dtype=np.float64)[int(target_ix)])
    ll_full = float(getattr(fit_full, "llf", np.nan))
    if not np.isfinite(ll_full):
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False, "method": None}
    X_red = np.delete(X_np, int(target_ix), axis=1)
    x_target = X_np[:, int(target_ix)]
    crit = float(sp_stats.chi2.ppf(1.0 - alpha, df=1))
    refit = _logit_mle_refit_offset if kind == "mle" else _firth_refit_offset

    def dev_at(b0):
        try:
            fit_c = refit(X_red, y_np, offset=b0 * x_target)
        except Exception:
            return np.inf
        ll_con = float(getattr(fit_c, "llf", np.nan))
        if not np.isfinite(ll_con):
            return np.inf
        val = 2.0 * (ll_full - ll_con)
        return float(val)

    base = dev_at(beta_hat)
    if not np.isfinite(base):
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False, "method": None}
    diff0 = base - crit

    def bracket_toward_zero(beta_hat_val, direction):
        a, b = (beta_hat_val, 0.0) if direction < 0 else (0.0, beta_hat_val)
        if a > b:
            a, b = b, a
        fa = dev_at(a) - crit
        fb = dev_at(b) - crit
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None, False
        if fa * fb > 0:
            return None, False
        for _ in range(100):
            m = 0.5 * (a + b)
            fm = dev_at(m) - crit
            if not np.isfinite(fm):
                break
            if abs(fm) < 1e-6 or abs(b - a) < 1e-6:
                return float(m), True
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b), True

    def bracket_far_side(beta_hat_val, direction, max_abs=max_abs_beta, tries=5):
        step = 0.5
        for _ in range(int(tries)):
            cand = beta_hat_val + direction * step
            if abs(cand) > max_abs:
                break
            df = dev_at(cand) - crit
            if np.isfinite(df) and np.isfinite(diff0) and diff0 * df <= 0:
                if direction < 0:
                    a, b = cand, beta_hat_val
                else:
                    a, b = beta_hat_val, cand
                fa = dev_at(a) - crit
                fb = dev_at(b) - crit
                if not (np.isfinite(fa) and np.isfinite(fb)):
                    break
                for _ in range(100):
                    m = 0.5 * (a + b)
                    fm = dev_at(m) - crit
                    if not np.isfinite(fm):
                        break
                    if abs(fm) < 1e-6 or abs(b - a) < 1e-6:
                        return float(m), True
                    if fa * fm <= 0:
                        b, fb = m, fm
                    else:
                        a, fa = m, fm
            step *= 2.0
        return None, False

    dev_zero = dev_at(0.0)
    if not np.isfinite(dev_zero):
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False,
                "method": "profile" if kind == "mle" else "profile_penalized"}

    blo = bhi = None
    ok_lo = ok_hi = False
    if dev_zero > crit:
        if beta_hat > 0:
            blo, ok_lo = bracket_toward_zero(beta_hat, direction=-1)
            bhi, ok_hi = bracket_far_side(beta_hat, direction=+1)
        elif beta_hat < 0:
            bhi, ok_hi = bracket_toward_zero(beta_hat, direction=+1)
            blo, ok_lo = bracket_far_side(beta_hat, direction=-1)
        else:
            return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False,
                    "method": "profile" if kind == "mle" else "profile_penalized"}
    else:
        blo, ok_lo = bracket_far_side(beta_hat, direction=-1)
        bhi, ok_hi = bracket_far_side(beta_hat, direction=+1)

    if not ok_lo and not ok_hi:
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False,
                "method": "profile" if kind == "mle" else "profile_penalized"}
    if dev_zero > crit and (not ok_lo or not ok_hi):
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False,
                "method": "profile" if kind == "mle" else "profile_penalized"}

    sided = "two"
    if not ok_lo:
        blo = -np.inf
        sided = "one"
    if not ok_hi:
        bhi = np.inf
        sided = "one"
    return {
        "lo": float(blo) if blo is not None else np.nan,
        "hi": float(bhi) if bhi is not None else np.nan,
        "sided": sided,
        "valid": True,
        "method": "profile" if kind == "mle" else "profile_penalized",
    }


def _score_stat_at_beta(X_red, y, x_target, beta0, kind="mle"):
    Xr = X_red.to_numpy(dtype=np.float64, copy=False) if hasattr(X_red, "to_numpy") else np.asarray(X_red, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    xt = np.asarray(x_target, dtype=np.float64)
    if Xr.ndim != 2 or yv.ndim != 1 or xt.ndim != 1 or Xr.shape[0] != yv.shape[0] or xt.shape[0] != yv.shape[0]:
        return np.nan
    offset = beta0 * xt
    try:
        if kind == "mle":
            fit_red = _logit_mle_refit_offset(Xr, yv, offset=offset)
        else:
            fit_red = _firth_refit_offset(Xr, yv, offset=offset)
    except Exception:
        return np.nan
    params = getattr(fit_red, "params", None)
    if params is None:
        return np.nan
    coef_red = np.asarray(params, dtype=np.float64)
    if coef_red.ndim != 1 or coef_red.shape[0] != Xr.shape[1]:
        return np.nan
    eta = np.clip(offset + Xr @ coef_red, -35.0, 35.0)
    p_hat = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
    W = p_hat * (1.0 - p_hat)
    h, denom = _efficient_score_vector(xt, Xr, W)
    if not (np.isfinite(denom) and denom > 0):
        return np.nan
    resid = yv - p_hat
    S = float(h @ resid)
    stat = (S * S) / denom
    return float(stat) if np.isfinite(stat) else np.nan


def _score_ci_beta(X_red, y, x_target, beta_hat, alpha=0.05, kind="mle", max_abs_beta=None):
    max_abs_beta = float(CTX.get("PROFILE_MAX_ABS_BETA", PROFILE_MAX_ABS_BETA) if max_abs_beta is None else max_abs_beta)
    Xr = X_red.to_numpy(dtype=np.float64, copy=False) if hasattr(X_red, "to_numpy") else np.asarray(X_red, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    xt = np.asarray(x_target, dtype=np.float64)
    if Xr.ndim != 2 or yv.ndim != 1 or xt.ndim != 1 or Xr.shape[0] != yv.shape[0] or xt.shape[0] != yv.shape[0]:
        return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_inversion", "sided": "two"}
    if not np.isfinite(beta_hat):
        return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_inversion", "sided": "two"}
    crit = float(sp_stats.chi2.ppf(1.0 - alpha, 1))
    cache = {}

    def stat_minus_crit(beta0):
        key = float(beta0)
        if key not in cache:
            cache[key] = _score_stat_at_beta(Xr, yv, xt, key, kind=kind)
        val = cache[key]
        if not np.isfinite(val):
            return np.nan
        return val - crit

    T0 = _score_stat_at_beta(Xr, yv, xt, 0.0, kind=kind)
    if not np.isfinite(T0):
        return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_inversion", "sided": "two"}
    diff_hat = stat_minus_crit(beta_hat)

    def root_bracket(a, b):
        fa = stat_minus_crit(a)
        fb = stat_minus_crit(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None, False
        if fa * fb > 0:
            return None, False
        for _ in range(70):
            mid = 0.5 * (a + b)
            fm = stat_minus_crit(mid)
            if not np.isfinite(fm):
                break
            if abs(fm) < 1e-6 or abs(b - a) < 1e-6:
                return float(mid), True
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        return 0.5 * (a + b), True

    blo = bhi = None
    ok_lo = ok_hi = False
    step = 0.5

    if T0 > crit:
        if beta_hat > 0:
            blo, ok_lo = root_bracket(0.0, beta_hat)
            if np.isfinite(diff_hat):
                b = beta_hat
                prev = diff_hat
                for _ in range(10):
                    cand = b + step
                    if abs(cand) > max_abs_beta:
                        break
                    diff_c = stat_minus_crit(cand)
                    if np.isfinite(diff_c) and prev * diff_c <= 0:
                        bhi, ok_hi = root_bracket(b, cand)
                        break
                    b = cand
                    prev = diff_c
                    step *= 2.0
        elif beta_hat < 0:
            bhi, ok_hi = root_bracket(beta_hat, 0.0)
            if np.isfinite(diff_hat):
                a = beta_hat
                prev = diff_hat
                step = 0.5
                for _ in range(10):
                    cand = a - step
                    if abs(cand) > max_abs_beta:
                        break
                    diff_c = stat_minus_crit(cand)
                    if np.isfinite(diff_c) and prev * diff_c <= 0:
                        blo, ok_lo = root_bracket(cand, a)
                        break
                    a = cand
                    prev = diff_c
                    step *= 2.0
        else:
            return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_inversion", "sided": "two"}
    else:
        if np.isfinite(diff_hat):
            left = beta_hat
            right = beta_hat
            fa = diff_hat
            fb = diff_hat
            for _ in range(10):
                left_candidate = left - step
                right_candidate = right + step
                if abs(left_candidate) <= max_abs_beta:
                    fa2 = stat_minus_crit(left_candidate)
                    if np.isfinite(fa2) and fa * fa2 <= 0:
                        blo, ok_lo = root_bracket(left_candidate, left)
                    left = left_candidate
                    fa = fa2 if np.isfinite(fa2) else fa
                if abs(right_candidate) <= max_abs_beta:
                    fb2 = stat_minus_crit(right_candidate)
                    if np.isfinite(fb2) and fb * fb2 <= 0:
                        bhi, ok_hi = root_bracket(right, right_candidate)
                    right = right_candidate
                    fb = fb2 if np.isfinite(fb2) else fb
                if ok_lo and ok_hi:
                    break
                step *= 2.0

    if ok_lo and ok_hi:
        return {
            "lo": float(blo),
            "hi": float(bhi),
            "valid": True,
            "method": "score_inversion",
            "sided": "two",
        }
    return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_inversion", "sided": "two"}


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

def _fit_logit_ladder(X, y, ridge_ok=True, const_ix=None, target_ix=None, prefer_mle_first=False, **kwargs):
    """
    Logistic fit ladder with an option to attempt unpenalized MLE first.
    If numpy arrays are provided, const_ix identifies the intercept column for zero-penalty.
    Returns a tuple (fit_result, reason_tag).
    """
    # avoid accidental duplication/override of start_params
    kwargs = dict(kwargs)
    user_start = kwargs.pop("start_params", None)

    is_pandas = hasattr(X, "columns")
    prefer_firth_on_ridge = bool(CTX.get("PREFER_FIRTH_ON_RIDGE", DEFAULT_PREFER_FIRTH_ON_RIDGE))
    allow_mle = _mle_prefit_ok(X, y, target_ix=target_ix, const_ix=const_ix)
    prefit_gate_tags = [] if allow_mle else ["gate:mle_prefit_blocked"]

    def _maybe_firth(path_tags):
        if not prefer_firth_on_ridge:
            return None
        firth_res = _firth_refit(X, y)
        if firth_res is None:
            return None
        tags = list(path_tags)
        tags.append("firth_refit")
        # Firth refits triggered from the ridge pathway are still penalized fits.
        # Mark the result accordingly so downstream consumers suppress inference.
        setattr(firth_res, "_used_ridge", True)
        setattr(firth_res, "_path_reasons", tags)
        return firth_res, "firth_refit"

    if not ridge_ok:
        return None, "ridge_disabled"

    try:
        # If requested, try unpenalized MLE first. This is particularly effective after design restrictions.
        if prefer_mle_first and allow_mle:
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
                        start_params=user_start
                    )
                    if _converged(mle_newton) and _ok_mle_fit(mle_newton, X, y, target_ix=target_ix):
                        setattr(mle_newton, "_final_is_mle", True)
                        setattr(mle_newton, "_path_reasons", ["mle_first_newton"] + prefit_gate_tags)
                        return mle_newton, "mle_first_newton"
                except (Exception, PerfectSeparationWarning):
                    pass
                try:
                    mle_bfgs = _logit_fit(
                        sm.Logit(y, X),
                        "bfgs",
                        maxiter=800,
                        gtol=1e-8,
                        start_params=user_start
                    )
                    if _converged(mle_bfgs) and _ok_mle_fit(mle_bfgs, X, y, target_ix=target_ix):
                        setattr(mle_bfgs, "_final_is_mle", True)
                        setattr(mle_bfgs, "_path_reasons", ["mle_first_bfgs"] + prefit_gate_tags)
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
        ridge_fit = sm.Logit(y, X).fit_regularized(
            alpha=float(alpha_scalar),
            L1_wt=0.0,
            maxiter=800,
            disp=0,
            start_params=user_start,
            **kwargs,
        )

        setattr(ridge_fit, "_ridge_alpha", float(alpha_scalar))
        setattr(ridge_fit, "_ridge_const_ix", None if const_ix is None else int(const_ix))
        setattr(ridge_fit, "_used_ridge", True)
        setattr(ridge_fit, "_final_is_mle", False)

        try:
            max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, ridge_fit.params)
        except Exception:
            max_abs_linpred, frac_lo, frac_hi = float("inf"), 1.0, 1.0

        neff_gate = float(CTX.get("MLE_REFIT_MIN_NEFF", 0.0))
        gate_tags = _ridge_gate_reasons(max_abs_linpred, frac_lo, frac_hi, n_eff, neff_gate)
        blocked_by_gate = ((max_abs_linpred > 15.0) or (frac_lo > 0.02) or (frac_hi > 0.02) or (neff_gate > 0 and n_eff < neff_gate))
        path_prefix = ["ridge_reached"] + gate_tags + prefit_gate_tags
        if blocked_by_gate or (not allow_mle):
            firth_attempt = _maybe_firth(path_prefix)
            if firth_attempt is not None:
                return firth_attempt
            tags = ["ridge_only"] + gate_tags
            if prefer_firth_on_ridge:
                tags.append("firth_failed")
            setattr(ridge_fit, "_path_reasons", tags)
            return ridge_fit, "ridge_only"
        # Proceed to attempt an unpenalized refit seeded by ridge if allowed.
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
                if _converged(refit_newton) and _ok_mle_fit(refit_newton, X, y, target_ix=target_ix):
                    setattr(refit_newton, "_used_ridge_seed", True)
                    setattr(refit_newton, "_final_is_mle", True)
                    tags = ["ridge_seeded_refit"] + gate_tags + prefit_gate_tags
                    setattr(refit_newton, "_path_reasons", tags)
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
                if _converged(refit_bfgs) and _ok_mle_fit(refit_bfgs, X, y, target_ix=target_ix):
                    setattr(refit_bfgs, "_used_ridge_seed", True)
                    setattr(refit_bfgs, "_final_is_mle", True)
                    tags = ["ridge_seeded_refit"] + gate_tags + prefit_gate_tags
                    setattr(refit_bfgs, "_path_reasons", tags)
                    return refit_bfgs, "ridge_seeded_refit"
            except (Exception, PerfectSeparationWarning):
                pass

        firth_path = list(path_prefix) + ["seeded_refit_failed"]
        firth_attempt = _maybe_firth(firth_path)
        if firth_attempt is not None:
            return firth_attempt

        tags = ["ridge_only"] + gate_tags
        if prefer_firth_on_ridge:
            tags.append("firth_failed")
        setattr(ridge_fit, "_path_reasons", tags)
        return ridge_fit, "ridge_only"
    except Exception as e:
        return None, f"ridge_exception:{type(e).__name__}"


def _is_ridge_fit(fit):
    """Returns True when the provided fit corresponds to a ridge solution."""
    if fit is None:
        return False
    used_ridge = bool(getattr(fit, "_used_ridge", False))
    if not used_ridge:
        return False
    if bool(getattr(fit, "_used_firth", False)):
        return False
    return True

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


def _wald_ci_or_from_fit(fit, target_ix, alpha=0.05, *, penalized=False):
    """
    Return a dict with a Wald CI on the OR scale computed from a fitted model:
      {"valid": bool, "lo_or": float, "hi_or": float, "method": str}
    If penalized=True, apply sanity checks (span and extreme endpoints).
    """
    if fit is None or (not hasattr(fit, "params")) or (not hasattr(fit, "bse")):
        return {"valid": False}

    try:
        params = np.asarray(fit.params, dtype=np.float64).ravel()
        bse = np.asarray(fit.bse, dtype=np.float64).ravel()
        beta = float(params[int(target_ix)])
        se = float(bse[int(target_ix)])
    except Exception:
        return {"valid": False}

    if not (np.isfinite(beta) and np.isfinite(se)) or se <= 0.0:
        return {"valid": False}

    z = float(sp_stats.norm.ppf(1.0 - 0.5 * alpha))
    lo_beta = beta - z * se
    hi_beta = beta + z * se
    lo_or = float(np.exp(lo_beta))
    hi_or = float(np.exp(hi_beta))
    ok = np.isfinite(lo_or) and np.isfinite(hi_or) and (lo_or > 0.0) and (hi_or > 0.0)

    if ok and penalized:
        span = hi_or / max(lo_or, 1e-300)
        if span > float(CTX.get("PENALIZED_CI_SPAN_RATIO", PENALIZED_CI_SPAN_RATIO)):
            ok = False
        if lo_or < float(CTX.get("PENALIZED_CI_LO_OR_MAX", PENALIZED_CI_LO_OR_MAX)):
            ok = False
        if hi_or > float(CTX.get("PENALIZED_CI_HI_OR_MIN", PENALIZED_CI_HI_OR_MIN)):
            ok = False

    method = (
        "wald_firth"
        if bool(getattr(fit, "_used_firth", False))
        else ("wald_penalized" if bool(getattr(fit, "_used_ridge", False)) else "wald_mle")
    )
    return {
        "valid": ok,
        "lo_or": lo_or if ok else np.nan,
        "hi_or": hi_or if ok else np.nan,
        "method": method,
    }


def _ridge_gate_reasons(max_abs_linpred, frac_lo, frac_hi, n_eff, neff_gate):
    reasons = []
    if np.isfinite(max_abs_linpred) and max_abs_linpred > 15.0:
        reasons.append("gate:max|Xb|>15")
    if np.isfinite(frac_lo) and frac_lo > 0.02:
        reasons.append("gate:p<1e-12>2%")
    if np.isfinite(frac_hi) and frac_hi > 0.02:
        reasons.append("gate:p>1-1e-12>2%")
    if (neff_gate is not None) and (neff_gate > 0) and np.isfinite(n_eff) and (n_eff < neff_gate):
        reasons.append(f"gate:neff<{neff_gate:g}")
    return reasons


def _firth_refit(X, y):
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        return None

    beta = np.zeros(X_np.shape[1], dtype=np.float64)
    maxiter_firth = 200
    tol_firth = 1e-8
    converged_firth = False

    for _it in range(maxiter_firth):
        eta = np.clip(X_np @ beta, -35.0, 35.0)
        p = expit(eta)
        p = np.clip(p, 1e-12, 1.0 - 1e-12)
        W = p * (1.0 - p)
        if not np.all(np.isfinite(W)):
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

    if not converged_firth:
        return None

    eta = np.clip(X_np @ beta, -35.0, 35.0)
    p = expit(eta)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    W = p * (1.0 - p)
    XTW = X_np.T * W
    XtWX = XTW @ X_np
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(XtWX)
    bse = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.divide(beta, bse, out=np.zeros_like(beta), where=bse > 0)
    pvals = 2.0 * sp_stats.norm.sf(np.abs(z))
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = float(np.sum(y_np * np.log(p) + (1.0 - y_np) * np.log(1.0 - p)))
    sign_det, logdet = np.linalg.slogdet(XtWX)
    pll = loglik + 0.5 * logdet if sign_det > 0 else -np.inf

    class _Result:
        """Lightweight container to mimic statsmodels results where needed."""

        pass

    firth_res = _Result()
    if hasattr(X, "columns"):
        firth_res.params = pd.Series(beta, index=X.columns)
        firth_res.bse = pd.Series(bse, index=X.columns)
        firth_res.pvalues = pd.Series(pvals, index=X.columns)
    else:
        firth_res.params = beta
        firth_res.bse = bse
        firth_res.pvalues = pvals
    setattr(firth_res, "llf", float(pll))
    setattr(firth_res, "_final_is_mle", False)
    setattr(firth_res, "_used_firth", True)
    return firth_res


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
 "RESULTS_CACHE_DIR", "LRT_OVERALL_CACHE_DIR", "LRT_FOLLOWUP_CACHE_DIR",
 "RIDGE_L2_BASE", "PER_ANC_MIN_CASES", "PER_ANC_MIN_CONTROLS",
 "BOOT_OVERALL_CACHE_DIR"
}

def _validate_ctx(ctx):
    """Raises RuntimeError if required context keys are missing."""
    missing = [k for k in REQUIRED_CTX_KEYS if k not in ctx]
    if missing:
        raise RuntimeError(f"[Worker-{os.getpid()}] Missing CTX keys: {', '.join(missing)}")
    ctx.setdefault("BOOTSTRAP_B", BOOTSTRAP_DEFAULT_B)
    ctx.setdefault("BOOTSTRAP_B_MAX", BOOTSTRAP_MAX_B)
    ctx.setdefault("BOOTSTRAP_CHUNK", BOOTSTRAP_CHUNK)
    ctx.setdefault("BOOTSTRAP_SEQ_ALPHA", BOOTSTRAP_SEQ_ALPHA)
    ctx.setdefault("FDR_ALPHA", 0.05)
    ctx.setdefault("BOOT_SEED_BASE", None)
    ctx.setdefault("ALLOW_PENALIZED_WALD", DEFAULT_ALLOW_PENALIZED_WALD)




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


def _score_test_from_reduced(X_red, y, x_target, const_ix=None):
    """Analytic 1-df Rao score test computed from the reduced (null) model."""
    fit_red, _ = _fit_logit_ladder(X_red, y, const_ix=const_ix, prefer_mle_first=True)
    if fit_red is None:
        return np.nan, np.nan
    if not bool(getattr(fit_red, "_final_is_mle", False)) or bool(getattr(fit_red, "_used_firth", False)):
        return np.nan, np.nan
    Xr = X_red.to_numpy(dtype=np.float64, copy=False) if hasattr(X_red, "to_numpy") else np.asarray(X_red, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    beta = np.asarray(getattr(fit_red, "params", np.zeros(Xr.shape[1])), dtype=np.float64)
    eta = np.clip(Xr @ beta, -35.0, 35.0)
    p_hat = expit(eta)
    W = p_hat * (1.0 - p_hat)
    x_tgt = np.asarray(x_target, dtype=np.float64)
    h, denom = _efficient_score_vector(x_tgt, Xr, W)
    S = float(h @ (yv - p_hat))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.nan, np.nan
    T_obs = (S * S) / denom
    if not np.isfinite(T_obs):
        return np.nan, np.nan
    p = float(sp_stats.chi2.sf(T_obs, 1))
    return p, T_obs


def _score_bootstrap_bits(Xr, yv, xt, beta0, kind="mle"):
    if Xr.ndim != 2 or yv.ndim != 1 or xt.ndim != 1 or Xr.shape[0] != yv.shape[0] or xt.shape[0] != yv.shape[0]:
        return None
    offset = beta0 * xt
    try:
        if kind == "mle":
            fit = _logit_mle_refit_offset(Xr, yv, offset=offset)
        else:
            fit = _firth_refit_offset(Xr, yv, offset=offset)
    except Exception:
        if kind == "mle":
            try:
                fit = _firth_refit_offset(Xr, yv, offset=offset)
                kind = "firth"
            except Exception:
                return None
        else:
            return None
    params = getattr(fit, "params", None)
    if params is None:
        return None
    coef = np.asarray(params, dtype=np.float64)
    if coef.ndim != 1 or coef.shape[0] != Xr.shape[1]:
        return None
    eta = np.clip(offset + Xr @ coef, -35.0, 35.0)
    mu = np.clip(expit(eta), 1e-12, 1.0 - 1e-12)
    W = mu * (1.0 - mu)
    h_vec, denom = _efficient_score_vector(xt, Xr, W)
    if not (np.isfinite(denom) and denom > 0.0):
        return None
    resid = yv - mu
    S_obs = float(h_vec @ resid)
    T_obs = (S_obs * S_obs) / denom
    if not np.isfinite(T_obs):
        return None
    return {
        "h_resid": np.asarray(h_vec * resid, dtype=np.float64),
        "den": float(denom),
        "T_obs": float(T_obs),
        "fit_kind": kind,
    }


def _bootstrap_chunk_exceed(h_resid, threshold_val, rng, reps, *, target_bytes=BOOTSTRAP_STREAM_TARGET_BYTES):
    reps = int(reps)
    if reps <= 0:
        return 0
    n = int(h_resid.shape[0])
    if n <= 0:
        return 0
    bytes_per_entry = 8.0  # float64
    block_cols = max(1, int(target_bytes // (bytes_per_entry * max(1, reps))))
    exceed = 0
    sr = np.zeros(reps, dtype=np.float64)
    for start in range(0, n, block_cols):
        stop = min(n, start + block_cols)
        width = stop - start
        if width <= 0:
            continue
        g_block = rng.standard_normal(size=(reps, width))
        sr += g_block @ h_resid[start:stop]
    exceed = int(np.sum((sr * sr) >= threshold_val))
    return exceed


def _score_bootstrap_p_from_bits(
    bits,
    B=None,
    B_max=None,
    alpha=None,
    rng=None,
    *,
    min_total=None,
    return_detail=False,
):
    if bits is None:
        if return_detail:
            return {"p": np.nan, "draws": 0, "exceed": 0}
        return np.nan
    den = float(bits.get("den", np.nan))
    T_obs = float(bits.get("T_obs", np.nan))
    if not (np.isfinite(den) and den > 0.0 and np.isfinite(T_obs)):
        if return_detail:
            return {"p": np.nan, "draws": 0, "exceed": 0}
        return np.nan
    h_resid = np.asarray(bits.get("h_resid"), dtype=np.float64)
    if h_resid.ndim != 1:
        if return_detail:
            return {"p": np.nan, "draws": 0, "exceed": 0}
        return np.nan
    rng = np.random.default_rng() if rng is None else rng
    base_B = int(B if B is not None else CTX.get("BOOTSTRAP_B", BOOTSTRAP_DEFAULT_B))
    if base_B <= 0:
        base_B = BOOTSTRAP_DEFAULT_B
    base_B = max(32, base_B)
    max_B = int(B_max if B_max is not None else CTX.get("BOOTSTRAP_B_MAX", BOOTSTRAP_MAX_B))
    if max_B < base_B:
        max_B = base_B
    chunk_limit = int(CTX.get("BOOTSTRAP_CHUNK", BOOTSTRAP_CHUNK))
    if chunk_limit <= 0:
        chunk_limit = BOOTSTRAP_CHUNK
    cp_alpha = float(CTX.get("BOOTSTRAP_SEQ_ALPHA", BOOTSTRAP_SEQ_ALPHA))
    alpha_target = float(alpha) if alpha is not None else None
    min_total = int(min_total) if min_total is not None else None
    if min_total is not None:
        if min_total <= 0:
            min_total = None
        else:
            min_total = min(min_total, max_B)
    total = 0
    exceed = 0
    target = base_B if min_total is None else max(base_B, min_total)
    threshold_val = T_obs * den
    while True:
        while total < target and total < max_B:
            draw = min(chunk_limit, target - total, max_B - total)
            if draw <= 0:
                break
            exceed += _bootstrap_chunk_exceed(h_resid, threshold_val, rng, draw)
            total += draw
        if total >= target:
            if alpha_target is not None:
                lower, upper = _clopper_pearson_interval(exceed, total, alpha=cp_alpha)
                if upper < alpha_target or lower > alpha_target:
                    break
            if target >= max_B:
                break
            next_target = min(target * 2, max_B)
            if min_total is not None:
                next_target = max(next_target, min_total)
            if next_target <= target:
                break
            target = next_target
        else:
            break
    if total <= 0:
        result = np.nan
    else:
        result = float((1.0 + exceed) / (1.0 + total))
    if return_detail:
        return {"p": result, "draws": int(total), "exceed": int(exceed)}
    return result


def _score_bootstrap_from_reduced(
    X_red,
    y,
    x_target,
    B=None,
    rng=None,
    alpha=None,
    seed_key=None,
    kind="mle",
    B_max=None,
    min_total=None,
):
    """Multiplier (wild) bootstrap of the Rao score statistic under the reduced model."""
    Xr = X_red.to_numpy(dtype=np.float64, copy=False) if hasattr(X_red, "to_numpy") else np.asarray(X_red, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    xt = np.asarray(x_target, dtype=np.float64)
    if Xr.ndim != 2 or yv.ndim != 1 or xt.ndim != 1 or Xr.shape[0] != yv.shape[0] or xt.shape[0] != yv.shape[0]:
        return np.nan, np.nan
    bits = _score_bootstrap_bits(Xr, yv, xt, 0.0, kind=kind)
    if bits is None and kind == "mle":
        bits = _score_bootstrap_bits(Xr, yv, xt, 0.0, kind="firth")
    if bits is None:
        return np.nan, np.nan
    alpha_target = float(alpha) if alpha is not None else float(CTX.get("FDR_ALPHA", 0.05))
    base_key = seed_key if seed_key is not None else ("score_boot", Xr.shape[0], Xr.shape[1], float(np.sum(np.abs(xt))))
    rng_local = rng if rng is not None else _bootstrap_rng((base_key, 0.0))
    detail = _score_bootstrap_p_from_bits(
        bits,
        B=B,
        B_max=B_max,
        alpha=alpha_target,
        rng=rng_local,
        min_total=min_total,
        return_detail=True,
    )
    return detail.get("p", np.nan), bits["T_obs"], detail.get("draws", 0), detail.get("exceed", 0)


def _score_boot_ci_beta(
    X_red,
    y,
    x_target,
    beta_hat,
    alpha=0.05,
    kind="mle",
    B=None,
    B_max=None,
    seed_key=None,
    p_at_zero=None,
    max_abs_beta=None,
):
    max_abs_beta = float(CTX.get("PROFILE_MAX_ABS_BETA", PROFILE_MAX_ABS_BETA) if max_abs_beta is None else max_abs_beta)
    Xr = X_red.to_numpy(dtype=np.float64, copy=False) if hasattr(X_red, "to_numpy") else np.asarray(X_red, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    xt = np.asarray(x_target, dtype=np.float64)
    if Xr.ndim != 2 or yv.ndim != 1 or xt.ndim != 1 or Xr.shape[0] != yv.shape[0] or xt.shape[0] != yv.shape[0]:
        return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_boot_multiplier", "sided": "two"}
    if not np.isfinite(beta_hat):
        return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_boot_multiplier", "sided": "two"}

    base_key = seed_key if seed_key is not None else ("score_boot_ci", Xr.shape[0], Xr.shape[1], float(np.sum(np.abs(xt))))
    base_B_local = int(B if B is not None else CTX.get("BOOTSTRAP_B", BOOTSTRAP_DEFAULT_B))
    if base_B_local <= 0:
        base_B_local = BOOTSTRAP_DEFAULT_B
    base_B_local = max(32, base_B_local)
    max_B_local = int(B_max if B_max is not None else CTX.get("BOOTSTRAP_B_MAX", BOOTSTRAP_MAX_B))
    if max_B_local < base_B_local:
        max_B_local = base_B_local

    cache = {}
    if p_at_zero is not None and np.isfinite(p_at_zero):
        cache[0.0] = {"p": float(p_at_zero), "draws": base_B_local}

    def _cache_draws(beta0):
        entry = cache.get(float(beta0))
        if entry is None:
            return 0
        if isinstance(entry, dict):
            return int(entry.get("draws", 0))
        return 0

    def p_eval(beta0, *, min_total=None):
        key = float(beta0)
        min_req = int(min_total) if min_total is not None else None
        if min_req is not None:
            if min_req <= 0:
                min_req = None
            else:
                min_req = max(base_B_local, min_req)
                min_req = min(min_req, max_B_local)
        entry = cache.get(key)
        if isinstance(entry, dict):
            if min_req is None or entry.get("draws", 0) >= min_req:
                return float(entry.get("p", np.nan))
        draw_key = min_req if min_req is not None else base_B_local
        rng_local = _bootstrap_rng((base_key, draw_key))
        bits = _score_bootstrap_bits(Xr, yv, xt, key, kind=kind)
        if bits is None and kind == "mle":
            bits = _score_bootstrap_bits(Xr, yv, xt, key, kind="firth")
        if bits is None:
            cache[key] = {"p": np.nan, "draws": 0}
        else:
            detail = _score_bootstrap_p_from_bits(
                bits,
                B=base_B_local,
                B_max=max_B_local,
                alpha=alpha,
                rng=rng_local,
                min_total=min_req,
                return_detail=True,
            )
            cache[key] = {
                "p": float(detail.get("p", np.nan)),
                "draws": int(detail.get("draws", 0)),
            }
        return float(cache[key]["p"])

    def diff(beta0, *, min_total=None):
        val = p_eval(beta0, min_total=min_total)
        if not np.isfinite(val):
            return np.nan
        return val - alpha

    p0 = p_eval(0.0)
    if not np.isfinite(p0):
        return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_boot_multiplier", "sided": "two"}

    diff_hat = diff(beta_hat)

    def root_bracket(a, b):
        a0 = float(a)
        b0 = float(b)
        if a0 == b0:
            return None, False
        for attempt in range(2):
            fa = diff(a0)
            fb = diff(b0)
            if not (np.isfinite(fa) and np.isfinite(fb)):
                return None, False
            if fa * fb > 0:
                return None, False
            left, right = a0, b0
            f_left, f_right = fa, fb
            for _ in range(70):
                mid = 0.5 * (left + right)
                fm = diff(mid)
                if not np.isfinite(fm):
                    break
                if abs(fm) < 1e-3 or abs(right - left) < 1e-3:
                    return float(mid), True
                if f_left * fm <= 0:
                    right, f_right = mid, fm
                else:
                    left, f_left = mid, fm
            if attempt == 0:
                draw_a = _cache_draws(a0)
                draw_b = _cache_draws(b0)
                draw_mid = _cache_draws(0.5 * (a0 + b0))
                best_draws = max(draw_a, draw_b, draw_mid)
                if best_draws < max_B_local:
                    min_req = max(best_draws * 2 if best_draws else base_B_local * 4, base_B_local * 4)
                    min_req = min(min_req, max_B_local)
                    diff(a0, min_total=min_req)
                    diff(b0, min_total=min_req)
                    diff(0.5 * (a0 + b0), min_total=min_req)
                    continue
            return 0.5 * (left + right), True
        return 0.5 * (a0 + b0), True

    blo = bhi = None
    ok_lo = ok_hi = False

    if p0 < alpha:
        if beta_hat > 0:
            blo, ok_lo = root_bracket(0.0, beta_hat)
            step = 0.5
            prev = diff_hat if np.isfinite(diff_hat) else diff(beta_hat)
            b = beta_hat
            for _ in range(12):
                cand = b + step
                if abs(cand) > max_abs_beta:
                    break
                diff_c = diff(cand)
                if np.isfinite(prev) and np.isfinite(diff_c) and prev * diff_c <= 0:
                    bhi, ok_hi = root_bracket(b, cand)
                    break
                b = cand
                prev = diff_c
                step *= 2.0
        elif beta_hat < 0:
            bhi, ok_hi = root_bracket(beta_hat, 0.0)
            step = 0.5
            prev = diff_hat if np.isfinite(diff_hat) else diff(beta_hat)
            a = beta_hat
            for _ in range(12):
                cand = a - step
                if abs(cand) > max_abs_beta:
                    break
                diff_c = diff(cand)
                if np.isfinite(prev) and np.isfinite(diff_c) and prev * diff_c <= 0:
                    blo, ok_lo = root_bracket(cand, a)
                    break
                a = cand
                prev = diff_c
                step *= 2.0
        else:
            return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_boot_multiplier", "sided": "two"}
    else:
        step = 0.5
        left = beta_hat
        right = beta_hat
        fa = diff_hat if np.isfinite(diff_hat) else diff(left)
        fb = fa
        for _ in range(12):
            did_work = False
            left_candidate = left - step
            right_candidate = right + step
            if abs(left_candidate) <= max_abs_beta:
                fa2 = diff(left_candidate)
                if np.isfinite(fa2) and np.isfinite(fa) and fa * fa2 <= 0:
                    blo, ok_lo = root_bracket(left_candidate, left)
                left = left_candidate
                fa = fa2 if np.isfinite(fa2) else fa
                did_work = True
            if abs(right_candidate) <= max_abs_beta:
                fb2 = diff(right_candidate)
                if np.isfinite(fb2) and np.isfinite(fb) and fb * fb2 <= 0:
                    bhi, ok_hi = root_bracket(right, right_candidate)
                right = right_candidate
                fb = fb2 if np.isfinite(fb2) else fb
                did_work = True
            if ok_lo and ok_hi:
                break
            if not did_work:
                break
            step *= 2.0

    if ok_lo and ok_hi:
        return {
            "lo": float(blo),
            "hi": float(bhi),
            "valid": True,
            "method": "score_boot_multiplier",
            "sided": "two",
        }
    return {"lo": np.nan, "hi": np.nan, "valid": False, "method": "score_boot_multiplier", "sided": "two"}


def plan_score_bootstrap_refinement(results_dir, ctx, *, safety_factor=8.0):
    """Identify score-bootstrap results that need additional draws for BH stability."""
    if not results_dir or not os.path.isdir(results_dir):
        return []
    try:
        files = [
            f
            for f in os.listdir(results_dir)
            if f.endswith(".json") and not f.endswith(".meta.json")
        ]
    except FileNotFoundError:
        return []

    alpha_global = float(ctx.get("FDR_ALPHA", 0.05))
    if not np.isfinite(alpha_global) or alpha_global <= 0.0:
        return []

    all_pvals = []
    boot_records = []
    for fn in files:
        path = os.path.join(results_dir, fn)
        try:
            rec = pd.read_json(path, typ="series")
        except Exception:
            continue
        try:
            p_val = float(rec.get("P_Value"))
        except (TypeError, ValueError):
            p_val = np.nan
        if np.isfinite(p_val):
            all_pvals.append(p_val)
        inf_type = str(rec.get("Inference_Type", "")).lower()
        if inf_type != "score_boot":
            continue
        try:
            draws = float(rec.get("Boot_Total", np.nan))
            exceed = float(rec.get("Boot_Exceed", np.nan))
        except (TypeError, ValueError):
            draws = np.nan
            exceed = np.nan
        if not np.isfinite(draws) or draws <= 0:
            continue
        if not np.isfinite(exceed) or exceed < 0:
            continue
        name = rec.get("Phenotype")
        if not isinstance(name, str) or not name:
            name = os.path.splitext(fn)[0]
        boot_records.append({
            "name": name,
            "draws": int(draws),
            "exceed": int(exceed),
        })

    m = len(all_pvals)
    if m == 0:
        return []

    sorted_p = np.sort(np.asarray(all_pvals, dtype=float))
    thresholds = alpha_global * (np.arange(1, m + 1, dtype=float) / m)
    hits = sorted_p <= thresholds
    if np.any(hits):
        idx = int(np.max(np.nonzero(hits)[0]))
        t_star = float(thresholds[idx])
    else:
        t_star = float(thresholds[0])

    if not np.isfinite(t_star) or t_star <= 0.0:
        return []

    cp_alpha = float(ctx.get("BOOTSTRAP_SEQ_ALPHA", BOOTSTRAP_SEQ_ALPHA))
    max_B = int(ctx.get("BOOTSTRAP_B_MAX", BOOTSTRAP_MAX_B))
    plan = []
    for rec in boot_records:
        draws = rec["draws"]
        exceed = rec["exceed"]
        if draws <= 0 or draws >= max_B:
            continue
        lower, upper = _clopper_pearson_interval(exceed, draws, alpha=cp_alpha)
        if lower <= t_star <= upper:
            target = math.ceil(safety_factor / max(t_star, 1e-12)) - 1
            target = max(target, draws + 1)
            target = min(target, max_B)
            if target > draws:
                plan.append({
                    "name": rec["name"],
                    "min_total": int(target),
                    "alpha_target": float(t_star),
                })
    return plan

# --- Worker globals ---
# Populated by init_lrt_worker / init_boot_worker and read-only thereafter.
worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, finite_mask_worker = None, None, 0, None, None
# Array-based versions for performance
X_all, col_ix, worker_core_df_cols, worker_core_df_index = None, None, None, None
# Handle to keep shared memory alive in workers
_BASE_SHM_HANDLE = None
# Shared uniform matrix for bootstrap
U_boot, _BOOT_SHM_HANDLE, B_boot = None, None, 0


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





def _should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target, allowed_fp, *,
                 used_index_fp=None, sex_cfg=None, thresholds=None):
    """Determines if a model run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta:
        return False
    if CTX.get("CTX_TAG") and meta.get("ctx_tag") != CTX.get("CTX_TAG"):
        return False
    if CTX.get("cdr_codename") and meta.get("cdr_codename") != CTX.get("cdr_codename"):
        return False
    if CTX.get("CACHE_VERSION_TAG") and meta.get("cache_version_tag") != CTX.get("CACHE_VERSION_TAG"):
        return False
    base_ok = (
        meta.get("model_columns") == list(core_df_cols) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp and
        meta.get("allowed_mask_fp") == allowed_fp and
        meta.get("target") == target
    )
    if not base_ok:
        return False
    data_keys = CTX.get("DATA_KEYS")
    if data_keys and meta.get("data_keys") != data_keys:
        return False
    if used_index_fp is not None and meta.get("used_index_fp") != used_index_fp:
        return False
    if sex_cfg:
        for k, v in sex_cfg.items():
            if meta.get(k) != v:
                return False
    if thresholds:
        for k, v in thresholds.items():
            if meta.get(k) != v:
                return False
    return True


def _lrt_meta_should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target, allowed_fp, *,
                          used_index_fp=None, sex_cfg=None, thresholds=None):
    """Determines if an LRT run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta: return False
    if CTX.get("CTX_TAG") and meta.get("ctx_tag") != CTX.get("CTX_TAG"):
        return False
    if CTX.get("cdr_codename") and meta.get("cdr_codename") != CTX.get("cdr_codename"):
        return False
    if CTX.get("CACHE_VERSION_TAG") and meta.get("cache_version_tag") != CTX.get("CACHE_VERSION_TAG"):
        return False
    base_ok = (
        meta.get("model_columns") == list(core_df_cols) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp and
        meta.get("allowed_mask_fp") == allowed_fp and
        meta.get("target") == target
    )
    if not base_ok:
        return False
    data_keys = CTX.get("DATA_KEYS")
    if data_keys and meta.get("data_keys") != data_keys:
        return False
    if used_index_fp is not None and meta.get("used_index_fp") != used_index_fp:
        return False
    if sex_cfg:
        for k, v in sex_cfg.items():
            if meta.get(k) != v:
                return False
    if thresholds:
        for k, v in thresholds.items():
            if meta.get(k) != v:
                return False
    return True


def _pos_in_current(orig_ix, current_ix_array):
    pos = np.flatnonzero(current_ix_array == orig_ix)
    return int(pos[0]) if pos.size else None




def lrt_overall_worker(task):
    """Worker for Stage-1 overall LRT. Uses array-based pipeline."""
    s_name, cat, target = task["name"], task["category"], task["target"]
    s_name_safe = safe_basename(s_name)
    result_path = os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name_safe}.json")
    meta_path = os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name_safe}.meta.json")
    res_path = os.path.join(CTX["RESULTS_CACHE_DIR"], f"{s_name_safe}.json")
    res_meta_path = os.path.join(CTX["RESULTS_CACHE_DIR"], f"{s_name_safe}.meta.json")
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

        repair_meta = os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False)

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

        used_index_fp = _index_fingerprint(Xb.index)
        sex_cfg = {
            "sex_restrict_mode": str(CTX.get("SEX_RESTRICT_MODE", "majority")).lower(),
            "sex_restrict_prop": float(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP)),
            "sex_restrict_max_other": int(CTX.get("SEX_RESTRICT_MAX_OTHER_CASES", 0)),
        }
        thresholds = {
            "min_cases": int(CTX.get("MIN_CASES_FILTER", DEFAULT_MIN_CASES)),
            "min_ctrls": int(CTX.get("MIN_CONTROLS_FILTER", DEFAULT_MIN_CONTROLS)),
            "min_neff": float(CTX.get("MIN_NEFF_FILTER", DEFAULT_MIN_NEFF)),
        }
        meta_extra_common = {
            "allowed_mask_fp": allowed_fp,
            "ridge_l2_base": CTX.get("RIDGE_L2_BASE", 1.0),
            "used_index_fp": used_index_fp,
        }
        meta_extra_common.update(sex_cfg)
        meta_extra_common.update(thresholds)

        if repair_meta:
            extra_meta = dict(meta_extra_common)
            if skip:
                extra_meta["skip_reason"] = skip
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp, extra=extra_meta)
            print(f"[meta repaired] {s_name_safe} (LRT-Stage1)", flush=True)

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df_cols, core_fp, case_fp, cat, target, allowed_fp,
            used_index_fp=used_index_fp, sex_cfg=sex_cfg, thresholds=thresholds
        ):
            if os.path.exists(res_path):
                print(f"[skip cache-ok] {s_name_safe} (LRT-Stage1)", flush=True)
                return
            else:
                print(f"[backfill] {s_name_safe} (LRT-Stage1) missing results JSON; regenerating", flush=True)

        if skip:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_Overall_Reason": skip, "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used})
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = skip
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=meta_extra)

            # Write the PheWAS-style result as a skip to mirror main pass outputs
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False, "Used_Firth": False,
                "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used,
                "Model_Notes": note or "",
                "Skip_Reason": skip
            })
            meta_extra_result = dict(meta_extra_common)
            meta_extra_result["skip_reason"] = skip
            meta_extra_result.update({
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
            })
            _write_meta(
                res_meta_path,
                "phewas_result",
                s_name,
                cat,
                target,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_fp,
                extra=meta_extra_result,
            )
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
            meta_extra = dict(meta_extra_common)
            meta_extra.update({"skip_reason": reason, "counts": det})
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=meta_extra)

            # Emit a PheWAS-style skip result to keep downstream shape identical
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False, "Used_Firth": False,
                "N_Total_Used": det['N'], "N_Cases_Used": det['N_cases'], "N_Controls_Used": det['N_ctrls'],
                "Model_Notes": note or "",
                "Skip_Reason": reason
            })
            meta_extra_result = dict(meta_extra_common)
            meta_extra_result.update({
                "skip_reason": reason,
                "counts": det,
                "N_Total_Used": det['N'],
                "N_Cases_Used": det['N_cases'],
                "N_Controls_Used": det['N_ctrls'],
            })
            _write_meta(
                res_meta_path,
                "phewas_result",
                s_name,
                cat,
                target,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_fp,
                extra=meta_extra_result,
            )
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
                "Used_Firth": False,
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
                "Model_Notes": note or "",
                "Skip_Reason": skip_reason,
            })
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = skip_reason
            _write_meta(
                meta_path,
                "lrt_overall",
                s_name,
                cat,
                target,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_fp,
                extra=meta_extra,
            )
            meta_extra_result = dict(meta_extra_common)
            meta_extra_result.update({
                "skip_reason": skip_reason,
                "N_Total_Used": n_total_used,
                "N_Cases_Used": n_cases_used,
                "N_Controls_Used": n_ctrls_used,
            })
            _write_meta(
                res_meta_path,
                "phewas_result",
                s_name,
                cat,
                target,
                worker_core_df_cols,
                _index_fingerprint(worker_core_df_index),
                case_fp,
                extra=meta_extra_result,
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

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red)
        fit_full, reason_full = _fit_logit_ladder(
            X_full_zv,
            yb,
            const_ix=const_ix_full,
            target_ix=target_ix,
        )

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
        full_is_mle = bool(getattr(fit_full, "_final_is_mle", False)) and not bool(getattr(fit_full, "_used_firth", False))
        red_is_mle = bool(getattr(fit_red, "_final_is_mle", False)) and not bool(getattr(fit_red, "_used_firth", False))
        inference_family = None
        fit_full_use = None
        fit_red_use = None
        if (
            fit_full is not None
            and fit_red is not None
            and full_is_mle
            and red_is_mle
            and _ok_mle_fit(fit_full, X_full_zv, yb, target_ix=target_ix)
            and _ok_mle_fit(fit_red, X_red_zv, yb)
        ):
            inference_family = "mle"
            fit_full_use = fit_full
            fit_red_use = fit_red
        else:
            fit_full_firth = fit_full if bool(getattr(fit_full, "_used_firth", False)) else _firth_refit(X_full_zv, yb)
            fit_red_firth = fit_red if bool(getattr(fit_red, "_used_firth", False)) else _firth_refit(X_red_zv, yb)
            if (fit_full_firth is not None) and (fit_red_firth is not None):
                inference_family = "firth"
                fit_full_use = fit_full_firth
                fit_red_use = fit_red_firth

        p_value = np.nan
        p_source = None
        ci_method = None
        ci_sided = "two"
        ci_label = ""
        ci_valid = False
        ci_lo_or = np.nan
        ci_hi_or = np.nan
        or_ci95 = None
        beta_full = np.nan
        or_val = np.nan

        if inference_family is not None:
            ll_full = float(getattr(fit_full_use, "llf", np.nan))
            ll_red = float(getattr(fit_red_use, "llf", np.nan))
            if np.isfinite(ll_full) and np.isfinite(ll_red):
                stat = max(0.0, 2.0 * (ll_full - ll_red))
                p_value = float(sp_stats.chi2.sf(stat, 1))
                p_source = "lrt_mle" if inference_family == "mle" else "lrt_firth"
                ci_info = _profile_ci_beta(X_full_zv, yb, target_ix, fit_full_use, kind=inference_family)
                ci_method = ci_info.get("method")
                ci_sided = ci_info.get("sided", "two")
                ci_valid = bool(ci_info.get("valid", False))
                if ci_valid:
                    lo_beta = ci_info.get("lo")
                    hi_beta = ci_info.get("hi")
                    if lo_beta == -np.inf:
                        ci_lo_or = 0.0
                    elif np.isfinite(lo_beta):
                        ci_lo_or = float(np.exp(lo_beta))
                    else:
                        ci_lo_or = np.nan
                    if hi_beta == np.inf:
                        ci_hi_or = np.inf
                    elif np.isfinite(hi_beta):
                        ci_hi_or = float(np.exp(hi_beta))
                    else:
                        ci_hi_or = np.nan
                    or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                    if ci_sided == "one":
                        ci_label = "one-sided (boundary)"
                params = getattr(fit_full_use, "params", None)
                if params is not None:
                    try:
                        if hasattr(params, "__getitem__"):
                            beta_full = float(params[target]) if hasattr(params, "index") else float(params[target_ix])
                        else:
                            beta_full = float(np.asarray(params)[target_ix])
                        or_val = float(np.exp(beta_full))
                    except Exception:
                        beta_full = np.nan
                        or_val = np.nan
            else:
                inference_family = None
                p_value = np.nan
                p_source = None

        if (
            inference_family is None
            and target_ix is not None
            and target in X_full_zv.columns
        ):
            x_target_vec = X_full_zv.iloc[:, int(target_ix)].to_numpy(dtype=np.float64, copy=False)
            p_sc, _ = _score_test_from_reduced(
                X_red_zv,
                yb,
                x_target_vec,
                const_ix=const_ix_red,
            )
            if np.isfinite(p_sc):
                p_value = p_sc
                p_source = "score_chi2"
                inference_family = "score"
            else:
                boot_res = _score_bootstrap_from_reduced(
                    X_red_zv,
                    yb,
                    x_target_vec,
                    seed_key=("lrt_overall", s_name_safe, target, "pval"),
                )
                if isinstance(boot_res, tuple):
                    p_emp = boot_res[0]
                else:
                    p_emp = np.nan
                if np.isfinite(p_emp):
                    p_value = p_emp
                    p_source = "score_boot"
                    inference_family = "score_boot"

        if (
            (not np.isfinite(beta_full))
            and fit_full is not None
            and target_ix is not None
            and target in X_full_zv.columns
        ):
            params_full = getattr(fit_full, "params", None)
            if params_full is not None:
                try:
                    if hasattr(params_full, "__getitem__"):
                        if hasattr(params_full, "index"):
                            beta_full = float(params_full[target])
                        else:
                            beta_full = float(params_full[target_ix])
                    else:
                        beta_full = float(np.asarray(params_full)[target_ix])
                    or_val = float(np.exp(beta_full))
                except Exception:
                    beta_full = np.nan
                    or_val = np.nan

        inference_type = inference_family if inference_family is not None else "none"
        p_valid = bool(np.isfinite(p_value))

        if inference_type == "score":
            if (
                target_ix is not None
                and target in X_full_zv.columns
                and np.isfinite(beta_full)
            ):
                x_target_vec_ci = X_full_zv.iloc[:, int(target_ix)].to_numpy(dtype=np.float64, copy=False)
                ci_info = _score_ci_beta(
                    X_red_zv,
                    yb,
                    x_target_vec_ci,
                    beta_full,
                    kind="mle",
                )
                ci_method = ci_info.get("method")
                ci_sided = ci_info.get("sided", "two")
                ci_valid = bool(ci_info.get("valid", False))
                if ci_valid:
                    lo_beta = ci_info.get("lo")
                    hi_beta = ci_info.get("hi")
                    if lo_beta == -np.inf:
                        ci_lo_or = 0.0
                    elif np.isfinite(lo_beta):
                        ci_lo_or = float(np.exp(lo_beta))
                    else:
                        ci_lo_or = np.nan
                    if hi_beta == np.inf:
                        ci_hi_or = np.inf
                    elif np.isfinite(hi_beta):
                        ci_hi_or = float(np.exp(hi_beta))
                    else:
                        ci_hi_or = np.nan
                    or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                else:
                    ci_lo_or = np.nan
                    ci_hi_or = np.nan
                    or_ci95 = None
            else:
                ci_valid = False
                ci_lo_or = np.nan
                ci_hi_or = np.nan
                or_ci95 = None
                ci_method = None
        elif inference_type == "score_boot":
            if (
                target_ix is not None
                and target in X_full_zv.columns
                and np.isfinite(beta_full)
            ):
                x_target_vec_ci = X_full_zv.iloc[:, int(target_ix)].to_numpy(dtype=np.float64, copy=False)
                ci_info = _score_boot_ci_beta(
                    X_red_zv,
                    yb,
                    x_target_vec_ci,
                    beta_full,
                    kind="mle",
                    seed_key=("lrt_overall", s_name_safe, target, "ci"),
                    p_at_zero=p_value if p_valid else None,
                )
                ci_method = ci_info.get("method")
                ci_sided = ci_info.get("sided", "two")
                ci_valid = bool(ci_info.get("valid", False))
                if ci_valid:
                    lo_beta = ci_info.get("lo")
                    hi_beta = ci_info.get("hi")
                    if lo_beta == -np.inf:
                        ci_lo_or = 0.0
                    elif np.isfinite(lo_beta):
                        ci_lo_or = float(np.exp(lo_beta))
                    else:
                        ci_lo_or = np.nan
                    if hi_beta == np.inf:
                        ci_hi_or = np.inf
                    elif np.isfinite(hi_beta):
                        ci_hi_or = float(np.exp(hi_beta))
                    else:
                        ci_hi_or = np.nan
                    or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                    ci_label = "score bootstrap (inverted)"
                else:
                    ci_lo_or = np.nan
                    ci_hi_or = np.nan
                    or_ci95 = None
            else:
                ci_valid = False
                ci_lo_or = np.nan
                ci_hi_or = np.nan
                or_ci95 = None
                ci_method = None

        if (not ci_valid) and (target_ix is not None):
            cand_fit_for_wald = fit_full_use if (fit_full_use is not None) else fit_full
            if cand_fit_for_wald is not None:
                wald = _wald_ci_or_from_fit(
                    cand_fit_for_wald,
                    target_ix,
                    alpha=0.05,
                    penalized=bool(getattr(cand_fit_for_wald, "_used_ridge", False)),
                )
            else:
                wald = {"valid": False}
            if wald.get("valid", False):
                ci_valid = True
                ci_method = wald["method"]
                ci_sided = "two"
                ci_lo_or = float(wald["lo_or"])
                ci_hi_or = float(wald["hi_or"])
                or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)

        ridge_in_path_full = bool(getattr(fit_full, "_used_ridge", False))
        used_firth_full = (
            bool(getattr(fit_full, "_used_firth", False))
            or bool(getattr(fit_full_use, "_used_firth", False))
            or (inference_type == "firth")
        )

        out = {
            "Phenotype": s_name,
            "P_LRT_Overall": float(p_value) if np.isfinite(p_value) else np.nan,
            "P_Overall_Valid": p_valid,
            "P_Source": p_source,
            "P_Method": p_source,
            "LRT_df_Overall": 1 if p_valid else np.nan,
            "Inference_Type": inference_type,
            "CI_Method": ci_method,
            "CI_Sided": ci_sided,
            "CI_Label": ci_label,
            "CI_Valid": bool(ci_valid),
            "CI_LO_OR": ci_lo_or,
            "CI_HI_OR": ci_hi_or,
            "Model_Notes": note,
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
        }
        if not p_valid:
            out["LRT_Overall_Reason"] = "fit_failed"

        model_notes = [note] if note else []
        if isinstance(reason_full, str) and reason_full:
            model_notes.append(reason_full)
        if isinstance(reason_red, str) and reason_red:
            model_notes.append(reason_red)
        model_notes.append(f"inference={inference_type}")
        if ci_method:
            model_notes.append(f"ci={ci_method}")

        final_cols_names = list(X_full_zv.columns)
        final_cols_pos = [col_ix.get(c, -1) for c in final_cols_names]

        res_record = {
            "Phenotype": s_name,
            "N_Total": n_total_pre,
            "N_Cases": n_cases_pre,
            "N_Controls": n_ctrls_pre,
            "Beta": beta_full,
            "OR": or_val,
            "P_Value": float(p_value) if np.isfinite(p_value) else p_value,
            "P_Valid": p_valid,
            "P_Source": p_source,
            "OR_CI95": or_ci95,
            "CI_Method": ci_method,
            "CI_Sided": ci_sided,
            "CI_Label": ci_label,
            "CI_Valid": bool(ci_valid),
            "CI_LO_OR": ci_lo_or,
            "CI_HI_OR": ci_hi_or,
            "Used_Ridge": ridge_in_path_full,
            "Final_Is_MLE": inference_type == "mle",
            "Used_Firth": used_firth_full,
            "Inference_Type": inference_type,
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": ";".join(model_notes),
        }

        if inference_type == "mle":
            penalized = any(_is_ridge_fit(candidate) for candidate in (fit_full_use, fit_red_use))
        elif inference_type == "firth":
            penalized = False
        elif inference_type in {"score", "score_boot"}:
            penalized = False
        else:
            penalized = _is_ridge_fit(fit_full)
        if penalized:
            out.update(
                {
                    "P_LRT_Overall": np.nan,
                    "P_Overall_Valid": False,
                    "P_Source": None,
                    "P_Method": None,
                    "LRT_df_Overall": np.nan,
                    "LRT_Overall_Reason": "penalized_fit_in_path",
                }
            )
            reason_tag = "penalized_fit_in_path"
            res_record.update(
                {
                    "P_Value": np.nan,
                    "P_Valid": False,
                    "P_Source": None,
                }
            )
            out_notes = out.get("Model_Notes")
            out["Model_Notes"] = f"{out_notes};{reason_tag}" if out_notes else reason_tag
            rec_notes = res_record.get("Model_Notes")
            res_record["Model_Notes"] = f"{rec_notes};{reason_tag}" if rec_notes else reason_tag

            if bool(CTX.get("ALLOW_PENALIZED_WALD", DEFAULT_ALLOW_PENALIZED_WALD)) and (target_ix is not None):
                cand_fit = fit_full if bool(getattr(fit_full, "_used_firth", False)) else _firth_refit(X_full_zv, yb)
                firth_for_ci = cand_fit
                if firth_for_ci is not None:
                    wald = _wald_ci_or_from_fit(firth_for_ci, target_ix, alpha=0.05, penalized=True)
                    if wald.get("valid", False):
                        ci_valid = True
                        ci_method = "wald_firth_fallback"
                        ci_sided = "two"
                        ci_lo_or = float(wald["lo_or"])
                        ci_hi_or = float(wald["hi_or"])
                        ci_label = "fallback (no p-value)"
                        or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                        out.update(
                            {
                                "CI_Method": ci_method,
                                "CI_Sided": ci_sided,
                                "CI_Label": "fallback (no p-value)",
                                "CI_Valid": True,
                                "CI_LO_OR": ci_lo_or,
                                "CI_HI_OR": ci_hi_or,
                            }
                        )
                        res_record.update(
                            {
                                "OR_CI95": or_ci95,
                                "CI_Method": ci_method,
                                "CI_Sided": ci_sided,
                                "CI_Label": "fallback (no p-value)",
                                "CI_Valid": True,
                                "CI_LO_OR": ci_lo_or,
                                "CI_HI_OR": ci_hi_or,
                            }
                        )
            if not (out.get("CI_Valid", False)):
                out.update(
                    {
                        "CI_Method": None,
                        "CI_Sided": None,
                        "CI_Label": "",
                        "CI_Valid": False,
                        "CI_LO_OR": np.nan,
                        "CI_HI_OR": np.nan,
                    }
                )
                res_record.update(
                    {
                        "OR_CI95": None,
                        "CI_Method": None,
                        "CI_Sided": None,
                        "CI_Label": "",
                        "CI_Valid": False,
                        "CI_LO_OR": np.nan,
                        "CI_HI_OR": np.nan,
                    }
                )

        io.atomic_write_json(res_path, res_record)
        meta_extra_result = dict(meta_extra_common)
        meta_extra_result.update({
            "final_cols_names": final_cols_names,
            "final_cols_pos": final_cols_pos,
            "full_llf": float(getattr(fit_full, "llf", np.nan)),
            "full_is_mle": bool(res_record.get("Final_Is_MLE", False)),
            "used_firth": used_firth_full,
            "used_ridge": ridge_in_path_full,
            "prune_recipe_version": "zv+greedy-rank-v1",
        })
        _write_meta(
            res_meta_path,
            "phewas_result",
            s_name,
            cat,
            target,
            worker_core_df_cols,
            _index_fingerprint(worker_core_df_index),
            case_fp,
            extra=meta_extra_result,
        )

        io.atomic_write_json(result_path, out)
        meta_extra = dict(meta_extra_common)
        meta_extra.update({
            "final_cols_names": final_cols_names,
            "final_cols_pos": final_cols_pos,
            "full_llf": float(getattr(fit_full, "llf", np.nan)),
            "full_is_mle": bool(res_record.get("Final_Is_MLE", False)),
            "used_firth": used_firth_full,
            "used_ridge": ridge_in_path_full,
            "prune_recipe_version": "zv+greedy-rank-v1",
        })
        _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols,
                    _index_fingerprint(worker_core_df_index), case_fp,
                    extra=meta_extra)
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
    meta_path = os.path.join(boot_dir, f"{s_name_safe}.meta.json")
    res_dir = CTX["RESULTS_CACHE_DIR"]
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, f"{s_name_safe}.json")
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

        used_index_fp = _index_fingerprint(Xb.index)
        sex_cfg = {
            "sex_restrict_mode": str(CTX.get("SEX_RESTRICT_MODE", "majority")).lower(),
            "sex_restrict_prop": float(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP)),
            "sex_restrict_max_other": int(CTX.get("SEX_RESTRICT_MAX_OTHER_CASES", 0)),
        }
        thresholds = {
            "min_cases": int(CTX.get("MIN_CASES_FILTER", DEFAULT_MIN_CASES)),
            "min_ctrls": int(CTX.get("MIN_CONTROLS_FILTER", DEFAULT_MIN_CONTROLS)),
            "min_neff": float(CTX.get("MIN_NEFF_FILTER", DEFAULT_MIN_NEFF)),
        }
        meta_extra_common = {
            "allowed_mask_fp": allowed_fp,
            "ridge_l2_base": CTX.get("RIDGE_L2_BASE", 1.0),
            "used_index_fp": used_index_fp,
        }
        meta_extra_common.update(sex_cfg)
        meta_extra_common.update(thresholds)
        if skip:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": skip, "N_Total_Used": n_total_used})
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = skip
            _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=meta_extra)
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False, "Used_Firth": False,
                "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used,
                "Model_Notes": note or "", "Skip_Reason": skip
            })
            return

        ok, reason, det = validate_min_counts_for_fit(yb, stage_tag="boot_stage1", extra_context={"phenotype": s_name})
        if not ok:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": reason, "N_Total_Used": det['N'], "N_Cases_Used": det['N_cases'], "N_Controls_Used": det['N_ctrls']})
            meta_extra = dict(meta_extra_common)
            meta_extra.update({"counts": det, "skip_reason": reason})
            _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=meta_extra)
            io.atomic_write_json(res_path, {
                "Phenotype": s_name,
                "N_Total": n_total_pre,
                "N_Cases": n_cases_pre,
                "N_Controls": n_ctrls_pre,
                "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "OR_CI95": None,
                "Used_Ridge": False, "Final_Is_MLE": False, "Used_Firth": False,
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
        if not np.isfinite(denom) or denom <= 1e-14:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": "nonpos_denom"})
            return
        resid = yb.to_numpy(dtype=np.float64, copy=False) - p_hat
        S_obs = float(h @ resid)
        T_obs = (S_obs * S_obs) / denom

        pos = worker_core_df_index.get_indexer(X_red_zv.index)
        pos = pos[pos >= 0]
        B = U_boot.shape[1]
        T_b = np.empty(B, dtype=np.float64)
        for j0 in range(0, B, 64):
            j1 = min(B, j0 + 64)
            U_blk = U_boot[np.ix_(pos, np.arange(j0, j1))]
            Ystar = (U_blk < p_hat[:, None]).astype(np.float64, copy=False)
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
            "P_Source": "score_boot",
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": note or ""
        })
        np.save(os.path.join(tnull_dir, f"{s_name_safe}.npy"), T_b.astype(np.float32, copy=False))
        _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=dict(meta_extra_common))

        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None
        target_ix_full = X_full_zv.columns.get_loc(target) if target in X_full_zv.columns else None
        fit_full, reason_full = _fit_logit_ladder(
            X_full_zv,
            yb,
            const_ix=const_ix_full,
            target_ix=target_ix_full,
        )
        beta_full, or_val = np.nan, np.nan
        final_is_mle = bool(getattr(fit_full, "_final_is_mle", False))
        used_firth_full = bool(getattr(fit_full, "_used_firth", False))
        used_ridge_full = bool(getattr(fit_full, "_used_ridge", False))
        if fit_full is not None and target in X_full_zv.columns:
            beta_full = float(getattr(fit_full, "params", pd.Series(np.nan, index=X_full_zv.columns))[target])
            or_val = float(np.exp(beta_full))
        ci_lo_or = np.nan
        ci_hi_or = np.nan
        or_ci95 = None
        ci_method = None
        ci_valid = False
        if fit_full is not None and target in X_full_zv.columns:
            wald = _wald_ci_or_from_fit(
                fit_full,
                target_ix_full,
                alpha=0.05,
                penalized=bool(getattr(fit_full, "_used_ridge", False)),
            )
            if (
                not wald.get("valid", False)
                and bool(getattr(fit_full, "_used_ridge", False))
                and bool(CTX.get("ALLOW_PENALIZED_WALD", DEFAULT_ALLOW_PENALIZED_WALD))
            ):
                firth_for_ci = _firth_refit(X_full_zv, yb)
                if firth_for_ci is not None:
                    wald = _wald_ci_or_from_fit(firth_for_ci, target_ix_full, alpha=0.05, penalized=True)
            if wald.get("valid", False):
                ci_lo_or = float(wald["lo_or"])
                ci_hi_or = float(wald["hi_or"])
                or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                ci_method = wald["method"]
                ci_valid = True
        io.atomic_write_json(res_path, {
            "Phenotype": s_name,
            "N_Total": n_total_pre,
            "N_Cases": n_cases_pre,
            "N_Controls": n_ctrls_pre,
            "Beta": beta_full,
            "OR": or_val,
            "P_Value": p_emp,
            "P_Source": "score_boot",
            "OR_CI95": or_ci95,
            "CI_Method": ci_method,
            "CI_Valid": bool(ci_valid),
            "CI_LO_OR": ci_lo_or,
            "CI_HI_OR": ci_hi_or,
            "Used_Ridge": used_ridge_full,
            "Final_Is_MLE": bool(final_is_mle),
            "Used_Firth": used_firth_full,
            "Inference_Type": "score_boot",
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
    meta_path = os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name_safe}.meta.json")
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

        repair_meta = os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False)

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
        out = {
            'Phenotype': s_name,
            'P_LRT_AncestryxDosage': np.nan,
            'P_Stage2_Valid': False,
            'P_Method': None,
            'P_Source': None,
            'Inference_Type': 'none',
            'LRT_df': np.nan,
            'LRT_Reason': "",
            'Model_Notes': note,
        }
        used_index_fp = _index_fingerprint(Xb.index)
        sex_cfg = {
            "sex_restrict_mode": str(CTX.get("SEX_RESTRICT_MODE", "majority")).lower(),
            "sex_restrict_prop": float(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP)),
            "sex_restrict_max_other": int(CTX.get("SEX_RESTRICT_MAX_OTHER_CASES", 0)),
        }
        thresholds = {
            "min_cases": int(CTX.get("MIN_CASES_FILTER", DEFAULT_MIN_CASES)),
            "min_ctrls": int(CTX.get("MIN_CONTROLS_FILTER", DEFAULT_MIN_CONTROLS)),
            "min_neff": float(CTX.get("MIN_NEFF_FILTER", DEFAULT_MIN_NEFF)),
        }
        meta_extra_common = {
            "allowed_mask_fp": allowed_fp,
            "ridge_l2_base": CTX.get("RIDGE_L2_BASE", 1.0),
            "used_index_fp": used_index_fp,
        }
        meta_extra_common.update(sex_cfg)
        meta_extra_common.update(thresholds)
        if repair_meta:
            extra_meta = dict(meta_extra_common)
            if skip:
                extra_meta["skip_reason"] = skip
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, core_fp, case_fp, extra=extra_meta)
            print(f"[meta repaired] {s_name_safe} (LRT-Stage2)", flush=True)
        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df_cols, core_fp, case_fp, category, target, allowed_fp,
            used_index_fp=used_index_fp, sex_cfg=sex_cfg, thresholds=thresholds
        ):
            print(f"[skip cache-ok] {s_name_safe} (LRT-Stage2)", flush=True)
            return
        if skip:
            out['LRT_Reason'] = skip; io.atomic_write_json(result_path, out)
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = skip
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=meta_extra)
            return

        anc_vec = worker_anc_series.loc[Xb.index]
        levels = pd.Index(anc_vec.dropna().unique(), dtype=str).tolist()
        levels_sorted = (['eur'] if 'eur' in levels else []) + [x for x in sorted(levels) if x != 'eur']
        out['LRT_Ancestry_Levels'] = ",".join(levels_sorted)

        if len(levels_sorted) < 2:
            out['LRT_Reason'] = "only_one_ancestry_level"; io.atomic_write_json(result_path, out)
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = "only_one_ancestry_level"
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=meta_extra)
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

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red)
        target_ix_full = X_full_zv.columns.get_loc(target) if target in X_full_zv.columns else None
        fit_full, reason_full = _fit_logit_ladder(
            X_full_zv,
            yb,
            const_ix=const_ix_full,
            target_ix=target_ix_full,
        )

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
        r_full = np.linalg.matrix_rank(X_full_zv.to_numpy(dtype=np.float64, copy=False))
        r_red = np.linalg.matrix_rank(X_red_zv.to_numpy(dtype=np.float64, copy=False))
        df_lrt = max(0, int(r_full - r_red))
        inference_family = None
        fit_full_use = None
        fit_red_use = None
        if df_lrt > 0:
            full_is_mle = bool(getattr(fit_full, "_final_is_mle", False)) and not bool(getattr(fit_full, "_used_firth", False))
            red_is_mle = bool(getattr(fit_red, "_final_is_mle", False)) and not bool(getattr(fit_red, "_used_firth", False))
            if (
                fit_full is not None
                and fit_red is not None
                and full_is_mle
                and red_is_mle
                and _ok_mle_fit(fit_full, X_full_zv, yb)
                and _ok_mle_fit(fit_red, X_red_zv, yb)
            ):
                inference_family = "mle"
                fit_full_use = fit_full
                fit_red_use = fit_red
            else:
                fit_full_firth = fit_full if bool(getattr(fit_full, "_used_firth", False)) else _firth_refit(X_full_zv, yb)
                fit_red_firth = fit_red if bool(getattr(fit_red, "_used_firth", False)) else _firth_refit(X_red_zv, yb)
                if (fit_full_firth is not None) and (fit_red_firth is not None):
                    inference_family = "firth"
                    fit_full_use = fit_full_firth
                    fit_red_use = fit_red_firth

        if inference_family is not None:
            ll_full = float(getattr(fit_full_use, "llf", np.nan))
            ll_red = float(getattr(fit_red_use, "llf", np.nan))
            if np.isfinite(ll_full) and np.isfinite(ll_red):
                stat = max(0.0, 2.0 * (ll_full - ll_red))
                p_stage2 = float(sp_stats.chi2.sf(stat, df_lrt))
                out['P_LRT_AncestryxDosage'] = p_stage2
                out['P_Stage2_Valid'] = np.isfinite(p_stage2)
                out['P_Method'] = "lrt_mle" if inference_family == "mle" else "lrt_firth"
                out['P_Source'] = out['P_Method']
                out['Inference_Type'] = inference_family
                out['LRT_df'] = df_lrt
            else:
                out['LRT_Reason'] = "fit_failed"
        else:
            out['LRT_Reason'] = "zero_df_lrt" if df_lrt == 0 else "fit_failed"

        for anc in levels_sorted:
            anc_mask = (anc_vec == anc).to_numpy()
            X_anc, y_anc = Xb[anc_mask], yb[anc_mask]

            anc_upper = anc.upper()
            n_total_anc = len(y_anc)
            n_cases_anc = int(y_anc.sum())
            n_ctrls_anc = n_total_anc - n_cases_anc

            out[f"{anc_upper}_N"] = n_total_anc
            out[f"{anc_upper}_N_Cases"] = n_cases_anc
            out[f"{anc_upper}_N_Controls"] = n_ctrls_anc
            out[f"{anc_upper}_OR"] = np.nan
            out[f"{anc_upper}_P"] = np.nan
            out[f"{anc_upper}_P_Valid"] = False
            out[f"{anc_upper}_P_Source"] = None
            out[f"{anc_upper}_Inference_Type"] = "none"
            out[f"{anc_upper}_CI_Method"] = None
            out[f"{anc_upper}_CI_Sided"] = "two"
            out[f"{anc_upper}_CI_Label"] = ""
            out[f"{anc_upper}_CI_Valid"] = False
            out[f"{anc_upper}_CI_LO_OR"] = np.nan
            out[f"{anc_upper}_CI_HI_OR"] = np.nan
            out[f"{anc_upper}_CI95"] = None
            out[f"{anc_upper}_REASON"] = ""

            ok, reason, det = validate_min_counts_for_fit(
                y_anc,
                stage_tag=f"lrt_followup:{anc}",
                extra_context={"phenotype": s_name, "ancestry": anc},
                cases_key="PER_ANC_MIN_CASES",
                controls_key="PER_ANC_MIN_CONTROLS",
            )
            if not ok:
                print(
                    f"[skip] name={s_name_safe} stage=LRT-Followup anc={anc} reason={reason} "
                    f"N={det['N']}/{det['N_cases']}/{det['N_ctrls']} "
                    f"min={det['min_cases']}/{det['min_ctrls']} neff={det['N_eff']:.1f}/{det['min_neff']:.1f}",
                    flush=True,
                )
                out[f"{anc_upper}_REASON"] = reason
                continue

            X_anc_zv = _drop_zero_variance(X_anc, keep_cols=("const",), always_keep=(target,))
            X_anc_zv = _drop_rank_deficient(X_anc_zv, keep_cols=("const",), always_keep=(target,))

            if target not in X_anc_zv.columns:
                out[f"{anc_upper}_REASON"] = "target_pruned"
                continue

            const_ix_anc = X_anc_zv.columns.get_loc('const') if 'const' in X_anc_zv.columns else None
            target_ix_anc = X_anc_zv.columns.get_loc(target)

            red_cols = [c for c in X_anc_zv.columns if c != target]
            X_anc_red = X_anc_zv[red_cols]
            const_ix_red = X_anc_red.columns.get_loc('const') if 'const' in X_anc_red.columns else None

            fit_full, reason_full = _fit_logit_ladder(
                X_anc_zv,
                y_anc,
                const_ix=const_ix_anc,
                target_ix=target_ix_anc,
            )
            fit_red, reason_red = _fit_logit_ladder(
                X_anc_red,
                y_anc,
                const_ix=const_ix_red,
            )

            if fit_full is not None:
                _print_fit_diag(
                    s_name_safe=s_name_safe,
                    stage="LRT-Followup",
                    model_tag=f"{anc}_full",
                    N_total=n_total_anc,
                    N_cases=n_cases_anc,
                    N_ctrls=n_ctrls_anc,
                    solver_tag=reason_full,
                    X=X_anc_zv,
                    y=y_anc,
                    params=fit_full.params,
                    notes=[note, f"anc={anc}"] if note else [f"anc={anc}"],
                )
            if fit_red is not None:
                _print_fit_diag(
                    s_name_safe=s_name_safe,
                    stage="LRT-Followup",
                    model_tag=f"{anc}_reduced",
                    N_total=n_total_anc,
                    N_cases=n_cases_anc,
                    N_ctrls=n_ctrls_anc,
                    solver_tag=reason_red,
                    X=X_anc_red,
                    y=y_anc,
                    params=fit_red.params,
                    notes=[note, f"anc={anc}"] if note else [f"anc={anc}"],
                )

            inference_family = None
            fit_full_use = None
            fit_red_use = None

            if (
                fit_full is not None
                and fit_red is not None
                and bool(getattr(fit_full, "_final_is_mle", False))
                and not bool(getattr(fit_full, "_used_firth", False))
                and bool(getattr(fit_red, "_final_is_mle", False))
                and not bool(getattr(fit_red, "_used_firth", False))
                and _ok_mle_fit(fit_full, X_anc_zv, y_anc, target_ix=target_ix_anc)
                and _ok_mle_fit(fit_red, X_anc_red, y_anc)
            ):
                inference_family = "mle"
                fit_full_use = fit_full
                fit_red_use = fit_red
            else:
                fit_full_firth = fit_full if bool(getattr(fit_full, "_used_firth", False)) else _firth_refit(X_anc_zv, y_anc)
                fit_red_firth = fit_red if bool(getattr(fit_red, "_used_firth", False)) else _firth_refit(X_anc_red, y_anc)
                if (fit_full_firth is not None) and (fit_red_firth is not None):
                    inference_family = "firth"
                    fit_full_use = fit_full_firth
                    fit_red_use = fit_red_firth

            p_val = np.nan
            p_source = None
            inference_type = "none"
            ci_method = None
            ci_sided = "two"
            ci_label = ""
            ci_valid = False
            ci_lo_or = np.nan
            ci_hi_or = np.nan
            ci_str = None
            beta_val = np.nan
            or_val = np.nan

            if inference_family is not None:
                ll_full = float(getattr(fit_full_use, "llf", np.nan))
                ll_red = float(getattr(fit_red_use, "llf", np.nan))
                if np.isfinite(ll_full) and np.isfinite(ll_red):
                    stat = max(0.0, 2.0 * (ll_full - ll_red))
                    p_val = float(sp_stats.chi2.sf(stat, 1))
                    p_source = "lrt_mle" if inference_family == "mle" else "lrt_firth"
                    inference_type = inference_family
                    ci_info = _profile_ci_beta(X_anc_zv, y_anc, target_ix_anc, fit_full_use, kind=inference_family)
                    ci_method = ci_info.get("method")
                    ci_sided = ci_info.get("sided", "two")
                    ci_valid = bool(ci_info.get("valid", False))
                    if ci_valid:
                        lo_beta = ci_info.get("lo")
                        hi_beta = ci_info.get("hi")
                        if lo_beta == -np.inf:
                            ci_lo_or = 0.0
                        elif np.isfinite(lo_beta):
                            ci_lo_or = float(np.exp(lo_beta))
                        if hi_beta == np.inf:
                            ci_hi_or = np.inf
                        elif np.isfinite(hi_beta):
                            ci_hi_or = float(np.exp(hi_beta))
                        ci_str = _fmt_ci(ci_lo_or, ci_hi_or)
                        if ci_sided == "one":
                            ci_label = "one-sided (boundary)"
                    params_full = getattr(fit_full_use, "params", None)
                    if params_full is not None:
                        try:
                            beta_val = float(np.asarray(params_full, dtype=np.float64)[target_ix_anc])
                            or_val = float(np.exp(beta_val))
                        except Exception:
                            beta_val = np.nan
                            or_val = np.nan
                else:
                    inference_family = None

            if inference_family is None:
                x_target_vec = X_anc_zv.iloc[:, int(target_ix_anc)].to_numpy(dtype=np.float64, copy=False)
                p_sc, _ = _score_test_from_reduced(
                    X_anc_red,
                    y_anc,
                    x_target_vec,
                    const_ix=const_ix_red,
                )
                if np.isfinite(p_sc):
                    p_val = p_sc
                    p_source = "score_chi2"
                    inference_type = "score"
                else:
                    boot_res = _score_bootstrap_from_reduced(
                        X_anc_red,
                        y_anc,
                        x_target_vec,
                        seed_key=("lrt_followup", s_name_safe, anc, target, "pval"),
                    )
                    if isinstance(boot_res, tuple):
                        p_emp = boot_res[0]
                    else:
                        p_emp = np.nan
                    if np.isfinite(p_emp):
                        p_val = p_emp
                        p_source = "score_boot"
                        inference_type = "score_boot"

            if (
                (not np.isfinite(beta_val))
                and fit_full is not None
                and target_ix_anc is not None
                and target in X_anc_zv.columns
            ):
                params_full = getattr(fit_full, "params", None)
                if params_full is not None:
                    try:
                        beta_val = float(np.asarray(params_full, dtype=np.float64)[int(target_ix_anc)])
                        or_val = float(np.exp(beta_val))
                    except Exception:
                        beta_val = np.nan
                        or_val = np.nan

            if inference_type == "score":
                if (
                    target_ix_anc is not None
                    and target in X_anc_zv.columns
                    and np.isfinite(beta_val)
                ):
                    x_target_vec_ci = X_anc_zv.iloc[:, int(target_ix_anc)].to_numpy(dtype=np.float64, copy=False)
                    ci_info = _score_ci_beta(
                        X_anc_red,
                        y_anc,
                        x_target_vec_ci,
                        beta_val,
                        kind="mle",
                    )
                    ci_method = ci_info.get("method")
                    ci_sided = ci_info.get("sided", "two")
                    ci_valid = bool(ci_info.get("valid", False))
                    if ci_valid:
                        lo_beta = ci_info.get("lo")
                        hi_beta = ci_info.get("hi")
                        if lo_beta == -np.inf:
                            ci_lo_or = 0.0
                        elif np.isfinite(lo_beta):
                            ci_lo_or = float(np.exp(lo_beta))
                        else:
                            ci_lo_or = np.nan
                        if hi_beta == np.inf:
                            ci_hi_or = np.inf
                        elif np.isfinite(hi_beta):
                            ci_hi_or = float(np.exp(hi_beta))
                        else:
                            ci_hi_or = np.nan
                        ci_str = _fmt_ci(ci_lo_or, ci_hi_or)
                    else:
                        ci_lo_or = np.nan
                        ci_hi_or = np.nan
                        ci_str = None
                else:
                    ci_valid = False
                    ci_lo_or = np.nan
                    ci_hi_or = np.nan
                    ci_str = None
                    ci_method = None
            elif inference_type == "score_boot":
                if (
                    target_ix_anc is not None
                    and target in X_anc_zv.columns
                    and np.isfinite(beta_val)
                ):
                    x_target_vec_ci = X_anc_zv.iloc[:, int(target_ix_anc)].to_numpy(dtype=np.float64, copy=False)
                    ci_info = _score_boot_ci_beta(
                        X_anc_red,
                        y_anc,
                        x_target_vec_ci,
                        beta_val,
                        kind="mle",
                        seed_key=("lrt_followup", s_name_safe, anc, target, "ci"),
                        p_at_zero=p_val if np.isfinite(p_val) else None,
                    )
                    ci_method = ci_info.get("method")
                    ci_sided = ci_info.get("sided", "two")
                    ci_valid = bool(ci_info.get("valid", False))
                    if ci_valid:
                        lo_beta = ci_info.get("lo")
                        hi_beta = ci_info.get("hi")
                        if lo_beta == -np.inf:
                            ci_lo_or = 0.0
                        elif np.isfinite(lo_beta):
                            ci_lo_or = float(np.exp(lo_beta))
                        else:
                            ci_lo_or = np.nan
                        if hi_beta == np.inf:
                            ci_hi_or = np.inf
                        elif np.isfinite(hi_beta):
                            ci_hi_or = float(np.exp(hi_beta))
                        else:
                            ci_hi_or = np.nan
                        ci_str = _fmt_ci(ci_lo_or, ci_hi_or)
                        ci_label = "score bootstrap (inverted)"
                    else:
                        ci_lo_or = np.nan
                        ci_hi_or = np.nan
                        ci_str = None
                else:
                    ci_valid = False
                    ci_lo_or = np.nan
                    ci_hi_or = np.nan
                    ci_str = None
                    ci_method = None

            if inference_type == "mle":
                ridge_inference = any(
                    _is_ridge_fit(candidate) for candidate in (fit_full_use, fit_red_use)
                )
            elif inference_type == "firth":
                ridge_inference = False
            elif inference_type in {"score", "score_boot"}:
                ridge_inference = False
            else:
                ridge_inference = any(
                    _is_ridge_fit(candidate) for candidate in (fit_full, fit_red)
                )
            if ridge_inference:
                p_val = np.nan
                p_source = None
                ci_method = None
                ci_sided = None
                ci_label = ""
                ci_valid = False
                ci_lo_or = np.nan
                ci_hi_or = np.nan
                ci_str = None
                if bool(CTX.get("ALLOW_PENALIZED_WALD", DEFAULT_ALLOW_PENALIZED_WALD)) and (target_ix_anc is not None):
                    firth_for_ci = _firth_refit(X_anc_zv, y_anc)
                    if firth_for_ci is not None:
                        wald = _wald_ci_or_from_fit(
                            firth_for_ci,
                            target_ix_anc,
                            alpha=0.05,
                            penalized=True,
                        )
                        if wald.get("valid", False):
                            ci_valid = True
                            ci_method = "wald_firth_fallback"
                            ci_sided = "two"
                            ci_lo_or = float(wald["lo_or"])
                            ci_hi_or = float(wald["hi_or"])
                            ci_str = _fmt_ci(ci_lo_or, ci_hi_or)
                            ci_label = "fallback (no p-value)"
                if ci_valid and (not np.isfinite(p_val)):
                    out[f"{anc_upper}_REASON"] = ""
                if not out[f"{anc_upper}_REASON"]:
                    out[f"{anc_upper}_REASON"] = "penalized_fit"

            p_valid = bool(np.isfinite(p_val))
            if (not p_valid) and (not ci_valid):
                if not out[f"{anc_upper}_REASON"]:
                    out[f"{anc_upper}_REASON"] = "subset_fit_failed"
                continue

            out[f"{anc_upper}_OR"] = or_val
            out[f"{anc_upper}_P"] = float(p_val) if np.isfinite(p_val) else np.nan
            out[f"{anc_upper}_P_Valid"] = bool(p_valid)
            out[f"{anc_upper}_P_Source"] = p_source
            out[f"{anc_upper}_Inference_Type"] = inference_type
            out[f"{anc_upper}_CI_Method"] = ci_method
            out[f"{anc_upper}_CI_Sided"] = ci_sided
            out[f"{anc_upper}_CI_Label"] = ci_label
            out[f"{anc_upper}_CI_Valid"] = bool(ci_valid)
            out[f"{anc_upper}_CI_LO_OR"] = ci_lo_or
            out[f"{anc_upper}_CI_HI_OR"] = ci_hi_or
            out[f"{anc_upper}_CI95"] = ci_str
            if p_valid:
                out[f"{anc_upper}_REASON"] = ""

            if not out.get(f"{anc_upper}_REASON"):
                out.pop(f"{anc_upper}_REASON", None)



        io.atomic_write_json(result_path, out)
        _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra=dict(meta_extra_common))
    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()
