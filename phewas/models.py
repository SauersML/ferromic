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
from scipy import linalg as sp_linalg
from scipy import stats as sp_stats
from scipy.special import expit
from scipy.optimize import brentq
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

from . import iox as io
from . import logging_utils

CTX = {}  # Worker context with constants from run.py
allowed_fp_by_cat = {}
worker_core_index_fp = None


class CaseCacheReadError(RuntimeError):
    """Raised when a phenotype case cache cannot be read."""

    def __init__(self, phenotype: str, path: str, stage: str, original: Exception):
        self.phenotype = phenotype
        self.path = path
        self.stage = stage
        self.original = original
        self.detail = f"{type(original).__name__}: {original}"
        super().__init__(f"{phenotype} [{stage}] case cache read failed: {self.detail}")

# --- inference behavior toggles ---
DEFAULT_PREFER_FIRTH_ON_RIDGE = True
DEFAULT_ALLOW_POST_FIRTH_MLE_REFIT = True
ENABLE_SCORE_BOOT_MLE = False  # Disable score bootstrap (too slow); set invalid p-value instead
ENABLE_SCORE_BOOT_PER_ANCESTRY = True  # Enable score bootstrap ONLY for per-ancestry tests

ALLOWED_P_SOURCES = {"lrt_mle", "score_chi2", "score_boot_mle", "score_boot_firth", "rao_score"}
ALLOWED_CI_METHODS = {
    "profile",
    "profile_penalized",
    "score_inversion",
    "score_boot_multiplier",
    "wald_mle",
}

MLE_SE_MAX_ALL = 100.0 # It looks at the Standard Error for every variable in the model. Looks for confidence interval massive across any covar
MLE_SE_MAX_TARGET = 3.0 # Standard Error for the Inversion/SNP of interest
MLE_MAX_ABS_XB = 25.0
MLE_FRAC_P_EXTREME = 0.05
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

def _canon_list(seq):
    if seq is None:
        return None
    return sorted(list(seq))


def _same_members_ignore_order(left, right):
    if left is None or right is None:
        return left is None and right is None
    if len(left) != len(right):
        return False
    return sorted(left) == sorted(right)


def _write_meta(meta_path, kind, s_name, category, target, core_cols, core_idx_fp, case_fp, extra=None):
    """Helper to write a standardized metadata JSON file."""
    base = {
        "kind": kind,
        "s_name": s_name,
        "category": category,
        "model_columns": _canon_list(core_cols),
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
        "phenotype_filter": CTX.get("PHENOTYPE_FILTER"),
    }
    data_keys = CTX.get("DATA_KEYS")
    if data_keys:
        base["data_keys"] = _canon_list(data_keys)
    if extra:
        base.update(extra)
    io.atomic_write_json(meta_path, base)


def _read_case_cache(path: str, *, phenotype: str, stage: str, columns=None) -> pd.DataFrame:
    """Read a phenotype case cache, converting I/O errors into diagnostics."""
    cols = ['is_case'] if columns is None else columns
    try:
        return pd.read_parquet(path, columns=cols)
    except Exception as exc:  # pragma: no cover - exercised via custom tests
        err = CaseCacheReadError(phenotype, path, stage, exc)
        print(
            f"[cache ERROR] name={safe_basename(phenotype)} stage={stage} "
            f"path={path} error={err.detail}",
            flush=True,
        )
        raise err from exc

# thresholds (configured via CTX; here are defaults/fallbacks)
DEFAULT_MIN_CASES = 100
DEFAULT_MIN_CONTROLS = 100
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


def _check_separation_in_strata(X, y, target_col, pheno_name="unknown"):
    """Check if target variable has variance (not constant) in overall and within strata."""
    if target_col not in X.columns:
        return

    target = X[target_col].to_numpy()
    cases = y.to_numpy()

    # Dosage descriptive statistics
    dosage_quantiles = np.quantile(target, [0.01, 0.05, 0.50, 0.95, 0.99])
    print(
        f"[TARGET-DOSAGE] name={pheno_name} "
        f"p01={dosage_quantiles[0]:.4g} p05={dosage_quantiles[1]:.4g} "
        f"p50={dosage_quantiles[2]:.4g} p95={dosage_quantiles[3]:.4g} p99={dosage_quantiles[4]:.4g}",
        flush=True
    )

    # Dosage statistics by case/control status
    case_mask = (cases == 1)
    ctrl_mask = (cases == 0)

    dosage_case_mean = np.mean(target[case_mask]) if case_mask.sum() > 0 else 0.0
    dosage_case_std = np.std(target[case_mask]) if case_mask.sum() > 0 else 0.0
    dosage_ctrl_mean = np.mean(target[ctrl_mask]) if ctrl_mask.sum() > 0 else 0.0
    dosage_ctrl_std = np.std(target[ctrl_mask]) if ctrl_mask.sum() > 0 else 0.0

    print(
        f"[DOSAGE-STATS] name={pheno_name} stratum=overall "
        f"cases_n={int(case_mask.sum())} cases_mean={dosage_case_mean:.4g} cases_std={dosage_case_std:.4g} "
        f"ctrls_n={int(ctrl_mask.sum())} ctrls_mean={dosage_ctrl_mean:.4g} ctrls_std={dosage_ctrl_std:.4g}",
        flush=True
    )

    # Check overall variance
    if np.var(target) == 0:
        unique_val = target[0]
        print(
            f"[SEPARATION-STRATUM] name={pheno_name} site=design_check "
            f"stratum=overall target_constant={unique_val:.3g} "
            f"driver=target action=gate_to_penalized",
            flush=True
        )

    # Check within sex strata if available
    if 'sex' in X.columns:
        for sex_val in [0.0, 1.0]:
            mask = X['sex'] == sex_val
            if mask.sum() == 0:
                continue
            target_stratum = target[mask]
            cases_stratum = cases[mask]

            # Dosage statistics in stratum
            case_mask_s = (cases_stratum == 1)
            ctrl_mask_s = (cases_stratum == 0)

            dosage_case_mean_s = np.mean(target_stratum[case_mask_s]) if case_mask_s.sum() > 0 else 0.0
            dosage_case_std_s = np.std(target_stratum[case_mask_s]) if case_mask_s.sum() > 0 else 0.0
            dosage_ctrl_mean_s = np.mean(target_stratum[ctrl_mask_s]) if ctrl_mask_s.sum() > 0 else 0.0
            dosage_ctrl_std_s = np.std(target_stratum[ctrl_mask_s]) if ctrl_mask_s.sum() > 0 else 0.0

            print(
                f"[DOSAGE-STATS] name={pheno_name} stratum=sex={int(sex_val)} "
                f"cases_n={int(case_mask_s.sum())} cases_mean={dosage_case_mean_s:.4g} cases_std={dosage_case_std_s:.4g} "
                f"ctrls_n={int(ctrl_mask_s.sum())} ctrls_mean={dosage_ctrl_mean_s:.4g} ctrls_std={dosage_ctrl_std_s:.4g}",
                flush=True
            )

            if np.var(target_stratum) == 0:
                unique_val = target_stratum[0]
                print(
                    f"[SEPARATION-STRATUM] name={pheno_name} site=design_check "
                    f"stratum=sex={int(sex_val)} target_constant={unique_val:.3g} "
                    f"driver=target action=gate_to_penalized",
                    flush=True
                )


def _check_collinearity(X, pheno_name="unknown"):
    """Check for multicollinearity issues in design matrix."""
    try:
        X_arr = X.to_numpy(dtype=np.float64) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=np.float64)
        n, p = X_arr.shape
        
        if p == 0 or n < p:
            return
        
        # Compute SVD
        try:
            U, s, Vt = np.linalg.svd(X_arr, full_matrices=False)
            smin = s[-1] if len(s) > 0 else 0.0
            smax = s[0] if len(s) > 0 else 1.0
            kappa = smax / smin if smin > 1e-15 else np.inf
            rank = np.sum(s > 1e-10)
        except:
            return
        
        # Compute VIF for each column (excluding constant)
        vifs = []
        col_names = X.columns.tolist() if hasattr(X, 'columns') else [f"X{i}" for i in range(p)]
        
        for i, col_name in enumerate(col_names):
            if col_name == 'const':
                continue
            try:
                X_i = X_arr[:, i]
                X_other = np.delete(X_arr, i, axis=1)
                if X_other.shape[1] == 0:
                    continue
                # R² from regressing X_i on other columns
                X_other_pinv = np.linalg.pinv(X_other)
                pred = X_other @ (X_other_pinv @ X_i)
                ss_res = np.sum((X_i - pred) ** 2)
                ss_tot = np.sum((X_i - np.mean(X_i)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
                vif = 1.0 / (1.0 - r2) if r2 < 0.9999 else np.inf
                vifs.append((col_name, vif))
            except:
                continue
        
        # Check thresholds
        if smin < 1e-8 or kappa > 1e8 or any(vif > 30 for _, vif in vifs):
            top_vif = max(vifs, key=lambda x: x[1]) if vifs else ("none", 0.0)
            suspects = [name for name, vif in vifs if vif > 10]
            
            print(
                f"[COLLINEAR-CLUMP] name={pheno_name} site=design_check "
                f"rank={rank}/{p} smin={smin:.2e} kappa={kappa:.2e} "
                f"top_vif={top_vif[1]:.2f}:{top_vif[0]} "
                f"suspects={','.join(suspects[:5]) if suspects else 'none'} "
                f"action=ridge_gate",
                flush=True
            )
    except Exception:
        pass


def _check_leverage_influence(X, y, fit, pheno_name="unknown"):
    """Check for high leverage points and influential observations."""
    try:
        X_arr = X.to_numpy(dtype=np.float64) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=np.float64)
        n, p = X_arr.shape
        
        if n < p or fit is None:
            return
        
        # Get fitted probabilities
        params = getattr(fit, 'params', None)
        if params is None:
            return
        
        eta = X_arr @ np.asarray(params, dtype=np.float64)
        eta = np.clip(eta, -35, 35)
        p_hat = expit(eta)
        W = p_hat * (1 - p_hat)
        
        # Compute hat matrix diagonal (leverage)
        W_sqrt = np.sqrt(W + 1e-10)
        X_weighted = X_arr * W_sqrt[:, None]
        try:
            XtWX_inv = np.linalg.pinv(X_weighted.T @ X_weighted)
            h = np.sum((X_weighted @ XtWX_inv) * X_weighted, axis=1)
        except:
            return
        
        # Check leverage
        mean_h = p / n
        max_h = np.max(h)
        h_threshold = max(4 * p / n, 10 * mean_h)
        
        if max_h > h_threshold:
            n_flagged = int(np.sum(h > h_threshold))
            print(
                f"[LEVERAGE-SPIKE] name={pheno_name} site=fit_diagnostics "
                f"n_flagged={n_flagged} max_h={max_h:.4f} max_cookd=0.0000 "
                f"top_obs= effect_shift_if_dropped=Δβ≈0.000",
                flush=True
            )
    except Exception:
        pass


def _check_bootstrap_instability(invalid_count, total_draws, mc_se_p, reasons, pheno_name="unknown"):
    """Check for bootstrap instability issues."""
    if total_draws == 0:
        return
    
    invalid_pct = 100.0 * invalid_count / total_draws
    
    if invalid_pct > 10.0 or (mc_se_p is not None and mc_se_p > 0.02):
        # Get top reasons
        reason_counts = {}
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        reasons_str = '|'.join([f"{r}:{c}" for r, c in top_reasons])
        
        mc_se_str = f"{mc_se_p:.4f}" if mc_se_p is not None else "NA"
        
        print(
            f"[BOOTSTRAP-UNSTABLE] name={pheno_name} site=bootstrap_worker "
            f"draws={total_draws} invalid={invalid_pct:.1f}% mc_se_p={mc_se_str} "
            f"reasons_top={reasons_str} action=suppress_or_flag_p",
            flush=True
        )


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


def _extract_fit_provenance(fit, X, y):
    """Extract core diagnostics from a fit object for logging."""
    info = {}
    try:
        info["converged"] = bool(getattr(fit, "converged", False))
        info["niter"] = int(getattr(fit, "niter", -1)) if hasattr(fit, "niter") else -1
        info["llf"] = float(getattr(fit, "llf", float("nan")))

        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_arr = np.asarray(y)
        info["n"] = int(X_arr.shape[0])
        info["n_cases"] = int(np.sum(y_arr))
        info["n_ctrls"] = info["n"] - info["n_cases"]
        info["p_events"] = float(info["n_cases"]) / max(1, info["n"])
        info["p_covars"] = int(X_arr.shape[1])
    except Exception:
        pass
    return info


def _ok_mle_fit(fit, X, y, target_ix=None,
                se_max_all=None, se_max_target=None,
                max_abs_xb=None, frac_extreme=None):
    """
    Check if MLE fit meets quality gates.
    Returns: (ok: bool, fail_reason: str, fail_details: dict)
    """
    # Caps
    se_max_all = float(CTX.get("MLE_SE_MAX_ALL", MLE_SE_MAX_ALL) if se_max_all is None else se_max_all)
    se_max_target = float(CTX.get("MLE_SE_MAX_TARGET", MLE_SE_MAX_TARGET) if se_max_target is None else se_max_target)
    max_abs_xb = float(CTX.get("MLE_MAX_ABS_XB", MLE_MAX_ABS_XB) if max_abs_xb is None else max_abs_xb)
    frac_extreme = float(CTX.get("MLE_FRAC_P_EXTREME", MLE_FRAC_P_EXTREME) if frac_extreme is None else frac_extreme)

    fail_reason = None
    fail_details = {}

    # Basic checks
    if fit is None or (not hasattr(fit, "bse")):
        fail_reason = "no_fit_or_no_bse"
        print(f"[OK-MLE-FIT] ok=False FAIL_REASON={fail_reason}", flush=True)
        return False, fail_reason, fail_details

    try:
        bse = np.asarray(fit.bse, dtype=np.float64)
    except Exception as e:
        fail_reason = "bse_extraction_failed"
        fail_details["error"] = str(e)
        print(f"[OK-MLE-FIT] ok=False FAIL_REASON={fail_reason}", flush=True)
        return False, fail_reason, fail_details

    if bse.ndim == 0:
        bse = np.array([float(bse)], dtype=np.float64)

    n_nonfinite_bse = int(np.sum(~np.isfinite(bse)))
    if n_nonfinite_bse > 0:
        fail_reason = "bse_not_finite"
        fail_details["n_nonfinite_bse"] = n_nonfinite_bse
        finite_bse = bse[np.isfinite(bse)]
        fail_details["bse_max"] = float(np.max(finite_bse)) if finite_bse.size else float("nan")
        print(f"[OK-MLE-FIT] ok=False FAIL_REASON={fail_reason} n_nonfinite_bse={n_nonfinite_bse}", flush=True)
        return False, fail_reason, fail_details

    # Extract core metrics
    bse_max_val = float(np.nanmax(bse))
    bse_tgt_val = None
    if target_ix is not None and 0 <= int(target_ix) < bse.size:
        bse_tgt_val = float(bse[int(target_ix)])

    try:
        params = getattr(fit, "params", None)
        if params is None:
            fail_reason = "no_params"
            print(f"[OK-MLE-FIT] ok=False FAIL_REASON={fail_reason}", flush=True)
            return False, fail_reason, fail_details
        max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, params)
        frac_total = frac_lo + frac_hi

        # Extract p_hat for additional diagnostics
        X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        params_arr = np.asarray(params, dtype=np.float64)
        linpred = X_arr @ params_arr
        p_hat = expit(linpred)
        p_min = float(np.min(p_hat))
        p_max = float(np.max(p_hat))
    except Exception as e:
        fail_reason = "diagnostics_failed"
        fail_details["error"] = str(e)
        print(f"[OK-MLE-FIT] ok=False FAIL_REASON={fail_reason} error={str(e)}", flush=True)
        return False, fail_reason, fail_details

    # Check design matrix properties
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float64)
    X_rank = int(np.linalg.matrix_rank(X_arr))
    X_ncols = int(X_arr.shape[1])
    X_full_rank = X_rank == X_ncols

    # vcov checks
    vcov_ok = True
    vcov_min_eig = float("nan")
    vcov_cond = float("nan")
    try:
        if hasattr(fit, "cov_params"):
            vcov = np.asarray(fit.cov_params(), dtype=np.float64)
            eigvals = np.linalg.eigvalsh(vcov)
            vcov_min_eig = float(np.min(eigvals))
            if vcov_min_eig > 0:
                vcov_cond = float(np.max(eigvals) / vcov_min_eig)
            else:
                vcov_ok = False
                vcov_cond = float("inf")
    except Exception:
        vcov_ok = False

    # Evaluate all guards with per-guard logging
    g_finite_bse = n_nonfinite_bse == 0
    g_se_all = bse_max_val <= se_max_all
    g_se_tgt = True
    if bse_tgt_val is not None:
        g_se_tgt = np.isfinite(bse_tgt_val) and (bse_tgt_val <= se_max_target)
    g_linpred = not (np.isfinite(max_abs_linpred) and max_abs_linpred > max_abs_xb)
    g_extreme = not ((np.isfinite(frac_lo) and frac_lo > frac_extreme) or
                     (np.isfinite(frac_hi) and frac_hi > frac_extreme))
    g_vcov = vcov_ok
    g_rank = X_full_rank

    # Print per-guard PASS/FAIL
    print(f"QC_FINITE_BSE: nonfinite={n_nonfinite_bse} -> {'PASS' if g_finite_bse else 'FAIL'}", flush=True)
    print(f"QC_SE_ALL: value={bse_max_val:.4g} cap={se_max_all} -> {'PASS' if g_se_all else 'FAIL'}", flush=True)
    if bse_tgt_val is not None:
        print(f"QC_SE_TARGET: value={bse_tgt_val:.4g} cap={se_max_target} -> {'PASS' if g_se_tgt else 'FAIL'}", flush=True)
    print(f"QC_MAX_ABS_XB: value={max_abs_linpred:.4g} cap={max_abs_xb} -> {'PASS' if g_linpred else 'FAIL'}", flush=True)
    print(f"QC_EXTREME_P_FRAC: lo={frac_lo:.4g} hi={frac_hi:.4g} total={frac_total:.4g} cap={frac_extreme} -> {'PASS' if g_extreme else 'FAIL'}", flush=True)
    print(f"QC_VCOV_POSDEF: min_eig={vcov_min_eig:.4g} cond={vcov_cond:.4g} -> {'PASS' if g_vcov else 'FAIL'}", flush=True)
    print(f"QC_X_RANK: rank={X_rank} ncols={X_ncols} full_rank={X_full_rank} -> {'PASS' if g_rank else 'FAIL'}", flush=True)

    # Determine first failed guard
    ok = g_finite_bse and g_se_all and g_se_tgt and g_linpred and g_extreme and g_vcov and g_rank

    if not ok:
        if not g_finite_bse:
            fail_reason = "bse_not_finite"
        elif not g_se_all:
            fail_reason = "bse_all_exceeds_cap"
            fail_details["bse_max"] = bse_max_val
            fail_details["cap"] = se_max_all
        elif not g_se_tgt:
            fail_reason = "bse_target_exceeds_cap"
            fail_details["bse_target"] = bse_tgt_val
            fail_details["cap"] = se_max_target
        elif not g_linpred:
            fail_reason = "max_abs_xb_exceeds_cap"
            fail_details["max_abs_xb"] = max_abs_linpred
            fail_details["cap"] = max_abs_xb
        elif not g_extreme:
            fail_reason = "extreme_probs_exceed_cap"
            fail_details["frac_lo"] = frac_lo
            fail_details["frac_hi"] = frac_hi
            fail_details["cap"] = frac_extreme
        elif not g_vcov:
            fail_reason = "vcov_not_posdef"
            fail_details["min_eig"] = vcov_min_eig
        elif not g_rank:
            fail_reason = "rank_deficiency"
            fail_details["rank"] = X_rank
            fail_details["ncols"] = X_ncols

    # Canonical summary line
    fail_details_str = " ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                                 for k, v in fail_details.items())

    print(
        f"[OK-MLE-FIT] ok={ok} FAIL_REASON={fail_reason if not ok else 'none'} | "
        f"bse_max={bse_max_val:.4g} bse_tgt={bse_tgt_val if bse_tgt_val is not None else 'NA'} "
        f"max_abs_xb={max_abs_linpred:.4g} frac_extreme={frac_total:.4%} "
        f"pmin={p_min:.4g} pmax={p_max:.4g} vcov_cond={vcov_cond:.4g} X_rank={X_rank}/{X_ncols} | "
        f"{fail_details_str}",
        flush=True
    )

    return ok, fail_reason, fail_details


def _mle_prefit_ok(X, y, target_ix=None, const_ix=None):
    X_np = X.to_numpy(dtype=np.float64, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        print("[MLE-PREFIT-OK] ok=False reason=shape_mismatch", flush=True)
        return False
    n = float(X_np.shape[0])
    if n <= 0:
        print("[MLE-PREFIT-OK] ok=False reason=empty_data", flush=True)
        return False
    n_cases = float(np.sum(y_np))
    n_ctrls = n - n_cases
    if n_cases <= 0 or n_ctrls <= 0:
        print(f"[MLE-PREFIT-OK] ok=False reason=no_variation n_cases={n_cases:.0f} n_ctrls={n_ctrls:.0f}", flush=True)
        return False
    p_eff = int(X_np.shape[1])
    if const_ix is not None and 0 <= int(const_ix) < X_np.shape[1]:
        p_eff = max(1, p_eff - 1)
    p_eff = max(1, p_eff)
    epv = min(n_cases, n_ctrls) / float(p_eff)
    epv_min = float(CTX.get("EPV_MIN_FOR_MLE", EPV_MIN_FOR_MLE))

    tgt_std = None
    tgt_var_min = None
    if target_ix is not None and 0 <= int(target_ix) < X_np.shape[1]:
        tgt_std = float(np.nanstd(X_np[:, int(target_ix)]))
        tgt_var_min = float(CTX.get("TARGET_VAR_MIN_FOR_MLE", TARGET_VAR_MIN_FOR_MLE))

    g_epv = epv >= epv_min
    g_tgt_var = True
    if tgt_std is not None and tgt_var_min is not None:
        g_tgt_var = tgt_std >= tgt_var_min

    ok = g_epv and g_tgt_var

    print(
        f"[MLE-PREFIT-OK] ok={ok} | "
        f"n={n:.0f} n_cases={n_cases:.0f} n_ctrls={n_ctrls:.0f} p_eff={p_eff} | "
        f"epv={epv:.2f}>=cap{epv_min}:{g_epv} | "
        f"tgt_std={tgt_std if tgt_std is not None else 'NA'}>=cap{tgt_var_min if tgt_var_min is not None else 'NA'}:{g_tgt_var}",
        flush=True
    )

    return ok


def _logit_mle_refit_offset(X, y, offset=None, maxiter=200, tol=1e-8, start_beta=None):
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
    if start_beta is not None:
        beta = np.asarray(start_beta, dtype=np.float64).copy()
        if beta.shape != (p,):
            raise ValueError("start_beta shape mismatch")
    else:
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
    eta = offset + X_np @ beta
    eta_work = np.clip(eta, -35.0, 35.0)
    p_hat = expit(eta_work)
    llf = float(np.sum(y_np * eta - np.logaddexp(0.0, eta)))
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


def _firth_refit_offset(X, y, offset=None, maxiter=200, tol=1e-8, start_beta=None):
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
    if start_beta is not None:
        beta = np.asarray(start_beta, dtype=np.float64).copy()
        if beta.shape != (p,):
            raise ValueError("start_beta shape mismatch")
    else:
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
    eta = offset + X_np @ beta
    eta_work = np.clip(eta, -35.0, 35.0)
    p_hat = np.clip(expit(eta_work), 1e-12, 1.0 - 1e-12)
    W = p_hat * (1.0 - p_hat)
    XTW = X_np.T * W
    XtWX = XTW @ X_np
    loglik = float(np.sum(y_np * eta - np.logaddexp(0.0, eta)))
    sign_det, logdet = np.linalg.slogdet(XtWX)
    pll = loglik + 0.5 * logdet if sign_det > 0 else -np.inf

    class _Res:
        pass

    res = _Res()
    res.params = beta
    p_dim = X_np.shape[1]
    res.bse = np.full(p_dim, np.nan)
    res.llf = float(pll)
    setattr(res, "_final_is_mle", False)
    setattr(res, "_used_firth", True)
    return res


def _firth_refit_with_fixed_coef(X, y, target_ix, fixed_value, maxiter=200, tol=1e-8, start_beta=None):
    """
    Fit Firth logistic regression with one coefficient fixed.
    Keeps full design matrix and constrains beta[target_ix] = fixed_value.
    This maintains the same penalty dimension as the unconstrained fit.
    """
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
        raise ValueError("design/response mismatch for Firth fixed coef refit")
    n, p = X_np.shape
    target_ix = int(target_ix)
    if target_ix < 0 or target_ix >= p:
        raise ValueError(f"target_ix {target_ix} out of range [0, {p})")
    
    # Initialize all coefficients
    if start_beta is not None:
        beta = np.asarray(start_beta, dtype=np.float64).copy()
        if beta.shape != (p,):
            raise ValueError("start_beta shape mismatch")
    else:
        beta = np.zeros(p, dtype=np.float64)
    beta[target_ix] = float(fixed_value)
    
    # Create mask for free parameters
    free_mask = np.ones(p, dtype=bool)
    free_mask[target_ix] = False
    free_indices = np.where(free_mask)[0]
    
    converged = False
    for _ in range(int(maxiter)):
        eta = np.clip(X_np @ beta, -35.0, 35.0)
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
        
        # Only update free parameters
        delta_full = XtWX_inv @ score
        delta_free = delta_full[free_indices]
        
        beta_new = beta.copy()
        beta_new[free_indices] += delta_free
        
        if not np.all(np.isfinite(beta_new)):
            break
        if np.max(np.abs(delta_free)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new
    
    if not converged:
        raise RuntimeError("Firth fixed coef refit failed to converge")
    
    eta = X_np @ beta
    eta_work = np.clip(eta, -35.0, 35.0)
    p_hat = np.clip(expit(eta_work), 1e-12, 1.0 - 1e-12)
    W = p_hat * (1.0 - p_hat)
    XTW = X_np.T * W
    XtWX = XTW @ X_np
    loglik = float(np.sum(y_np * eta - np.logaddexp(0.0, eta)))
    sign_det, logdet = np.linalg.slogdet(XtWX)
    pll = loglik + 0.5 * logdet if sign_det > 0 else -np.inf
    
    class _Res:
        pass
    
    res = _Res()
    res.params = beta
    res.bse = np.full(p, np.nan)
    res.llf = float(pll)
    setattr(res, "_final_is_mle", False)
    setattr(res, "_used_firth", True)
    return res


def _profile_ci_beta(X_full, y, target_ix, fit_full, kind="mle", alpha=0.05, max_abs_beta=None):
    """
    Compute profile likelihood CI for a single coefficient.
    
    For Firth regression, uses _firth_refit_with_fixed_coef to maintain
    the same penalty dimension between full and constrained models.
    """
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
    
    crit = float(sp_stats.chi2.ppf(1.0 - alpha, df=1))
    
    # Continuation state: track last successful coefficient vector per side
    # This enables warm-start homotopy for path-following
    warm_state = {"last_beta": None}
    
    # Memoization cache to avoid redundant fits during Brent interpolation
    memo_cache = {}
    
    if kind == "firth":
        # Use fixed coefficient approach to maintain penalty dimension
        def dev_at(b0, use_warm=True):
            key = float(b0)
            if key in memo_cache:
                return memo_cache[key]
            try:
                start = warm_state["last_beta"] if (use_warm and warm_state["last_beta"] is not None) else None
                fit_c = _firth_refit_with_fixed_coef(X_np, y_np, target_ix, b0, start_beta=start)
                warm_state["last_beta"] = fit_c.params.copy()
            except Exception:
                memo_cache[key] = np.inf
                return np.inf
            ll_con = float(getattr(fit_c, "llf", np.nan))
            if not np.isfinite(ll_con):
                memo_cache[key] = np.inf
                return np.inf
            val = 2.0 * (ll_full - ll_con)
            memo_cache[key] = float(val)
            return float(val)
    else:
        # MLE: use offset approach (dropping column is fine for MLE)
        X_red = np.delete(X_np, int(target_ix), axis=1)
        x_target = X_np[:, int(target_ix)]
        
        def dev_at(b0, use_warm=True):
            key = float(b0)
            if key in memo_cache:
                return memo_cache[key]
            try:
                start = warm_state["last_beta"] if (use_warm and warm_state["last_beta"] is not None) else None
                fit_c = _logit_mle_refit_offset(X_red, y_np, offset=b0 * x_target, start_beta=start)
                warm_state["last_beta"] = fit_c.params.copy()
            except Exception:
                memo_cache[key] = np.inf
                return np.inf
            ll_con = float(getattr(fit_c, "llf", np.nan))
            if not np.isfinite(ll_con):
                memo_cache[key] = np.inf
                return np.inf
            val = 2.0 * (ll_full - ll_con)
            memo_cache[key] = float(val)
            return float(val)

    base = dev_at(beta_hat, use_warm=False)
    if not np.isfinite(base):
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False, "method": None}
    diff0 = base - crit

    def solve_root_brent(a, b, reset_warm=True):
        """Use Brent-Dekker method for fast root finding with safeguarding."""
        if reset_warm:
            warm_state["last_beta"] = None
        
        def objective(x):
            return dev_at(x) - crit
        
        # Check bracket validity
        fa = objective(a)
        fb = objective(b)
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None, False
        if fa * fb > 0:
            return None, False
        
        try:
            root = brentq(objective, a, b, xtol=1e-6, rtol=1e-6, maxiter=50)
            return float(root), True
        except (ValueError, RuntimeError):
            # Fallback to bisection if Brent fails
            for _ in range(50):
                m = 0.5 * (a + b)
                fm = objective(m)
                if not np.isfinite(fm):
                    break
                if abs(fm) < 1e-6 or abs(b - a) < 1e-6:
                    return float(m), True
                if fa * fm <= 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return 0.5 * (a + b), True

    def bracket_toward_zero(beta_hat_val, direction):
        # Reset warm state for this side
        a, b = (beta_hat_val, 0.0) if direction < 0 else (0.0, beta_hat_val)
        if a > b:
            a, b = b, a
        return solve_root_brent(a, b, reset_warm=True)

    def bracket_far_side(beta_hat_val, direction, max_abs=max_abs_beta, tries=5):
        # Reset warm state for this side
        warm_state["last_beta"] = None
        step = 0.5
        for _ in range(int(tries)):
            cand = beta_hat_val + direction * step
            if abs(cand) > max_abs:
                break
            df = dev_at(cand, use_warm=False) - crit
            if np.isfinite(df) and np.isfinite(diff0) and diff0 * df <= 0:
                if direction < 0:
                    a, b = cand, beta_hat_val
                else:
                    a, b = beta_hat_val, cand
                return solve_root_brent(a, b, reset_warm=False)
            step *= 2.0
        return None, False

    dev_zero = dev_at(0.0, use_warm=False)
    if not np.isfinite(dev_zero):
        return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False,
                "method": "profile" if kind == "mle" else "profile_penalized"}

    blo = bhi = None
    ok_lo = ok_hi = False
    if dev_zero > crit:
        if beta_hat > 0:
            # Lower bound: toward zero (warm-start from beta_hat)
            blo, ok_lo = bracket_toward_zero(beta_hat, direction=-1)
            # Upper bound: far side (reset and warm-start from beta_hat)
            bhi, ok_hi = bracket_far_side(beta_hat, direction=+1)
        elif beta_hat < 0:
            # Upper bound: toward zero (warm-start from beta_hat)
            bhi, ok_hi = bracket_toward_zero(beta_hat, direction=+1)
            # Lower bound: far side (reset and warm-start from beta_hat)
            blo, ok_lo = bracket_far_side(beta_hat, direction=-1)
        else:
            return {"lo": np.nan, "hi": np.nan, "sided": "two", "valid": False,
                    "method": "profile" if kind == "mle" else "profile_penalized"}
    else:
        # Both bounds on far side
        blo, ok_lo = bracket_far_side(beta_hat, direction=-1)
        # Reset for opposite side
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


def _ridge_column_scales(X, const_ix=None, *, floor=1e-12):
    """Compute per-column scale factors used to standardize the ridge design."""
    X_np = X.to_numpy(dtype=np.float64, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float64)
    if X_np.ndim != 2:
        return None
    p = X_np.shape[1]
    scales = np.ones(p, dtype=np.float64)
    const_ix_eff = None if const_ix is None else int(const_ix)
    for j in range(p):
        if const_ix_eff is not None and j == const_ix_eff:
            scales[j] = 1.0
            continue
        col = X_np[:, j]
        try:
            scale = float(np.nanstd(col, dtype=np.float64))
        except TypeError:
            col = np.asarray(col, dtype=np.float64)
            scale = float(np.nanstd(col))
        if not np.isfinite(scale) or scale <= floor:
            scales[j] = 1.0
        else:
            scales[j] = float(scale)
    return scales


def _fit_logit_ladder(
    X,
    y,
    ridge_ok=True,
    const_ix=None,
    target_ix=None,
    prefer_mle_first=False,
    ridge_zero_penalty_ixs=None,
    **kwargs,
):
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
    allow_post_firth_mle = bool(
        CTX.get("ALLOW_POST_FIRTH_MLE_REFIT", DEFAULT_ALLOW_POST_FIRTH_MLE_REFIT)
    )
    if const_ix is None and is_pandas and "const" in getattr(X, "columns", []):
        try:
            const_ix = int(list(X.columns).index("const"))
        except ValueError:
            const_ix = None
    elif const_ix is not None:
        try:
            const_ix = int(const_ix)
        except (TypeError, ValueError):
            const_ix = None

    zero_penalty_ixs = set()
    if const_ix is not None:
        try:
            zero_penalty_ixs.add(int(const_ix))
        except (TypeError, ValueError):
            pass
    if ridge_zero_penalty_ixs is not None:
        if np.isscalar(ridge_zero_penalty_ixs):
            ridge_zero_penalty_ixs = [ridge_zero_penalty_ixs]
        for ix in ridge_zero_penalty_ixs:
            if isinstance(ix, str) and is_pandas:
                try:
                    ix = X.columns.get_loc(ix)
                except KeyError:
                    continue
            try:
                zero_penalty_ixs.add(int(ix))
            except (TypeError, ValueError):
                continue

    allow_mle = _mle_prefit_ok(X, y, target_ix=target_ix, const_ix=const_ix)
    prefit_gate_tags = [] if allow_mle else ["gate:mle_prefit_blocked"]

    def _maybe_firth(path_tags):
        if not prefer_firth_on_ridge:
            return None
        
        # Print Firth entry
        pheno_name = kwargs.get("pheno_name", CTX.get("current_phenotype", "unknown"))
        print(
            f"[FIRTH-ENTER] name={pheno_name} site=_fit_logit_ladder "
            f"path={'|'.join(path_tags)} seeded=false reason=after_ridge_or_gate",
            flush=True
        )
        
        firth_res = _firth_refit(X, y)
        if firth_res is None:
            return None
        tags = list(path_tags)
        tags.append("firth_refit")
        
        # Print Firth success
        pll = getattr(firth_res, "llf", float("nan"))
        print(
            f"[FIRTH-OK] name={pheno_name} site=_fit_logit_ladder "
            f"used_firth=true pll={pll:.4f} path={'|'.join(tags)}",
            flush=True
        )
        
        # Firth refits triggered from the ridge pathway should allow inference.
        # Mark that ridge was in the path, but don't suppress Firth inference.
        setattr(firth_res, "_ridge_in_path", True)
        setattr(firth_res, "_path_reasons", tags)
        if allow_post_firth_mle:
            params = getattr(firth_res, "params", None)
            if params is not None:
                try:
                    start = np.asarray(params, dtype=np.float64)
                    if start.shape[0] != int(X.shape[1]):
                        start = None
                except Exception:
                    start = None
            else:
                start = None
            if start is not None and np.all(np.isfinite(start)):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=PerfectSeparationWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="overflow encountered in exp",
                        category=RuntimeWarning,
                        module=r"statsmodels\.discrete\.discrete_model",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="divide by zero encountered in log",
                        category=RuntimeWarning,
                        module=r"statsmodels\.discrete\.discrete_model",
                    )
                    solver_kwargs = dict(kwargs)
                    extra_flag = (
                        {}
                        if ("_already_failed" in solver_kwargs)
                        else {"_already_failed": True}
                    )
                    solver_kwargs.update(extra_flag)
                    for method, fit_kwargs in (
                        ("newton", {"maxiter": 200, "tol": 1e-8}),
                        ("bfgs", {"maxiter": 400, "gtol": 1e-8}),
                    ):
                        print(
                            f"[MLE-REFIT-FROM-FIRTH] name={pheno_name} site=_fit_logit_ladder "
                            f"method={method} start=from_firth",
                            flush=True
                        )
                        try:
                            refit = _logit_fit(
                                sm.Logit(y, X),
                                method,
                                start_params=start,
                                **fit_kwargs,
                                **solver_kwargs,
                            )
                        except (Exception, PerfectSeparationWarning) as e:
                            error_msg = str(e)[:400]
                            print(
                                f"[MLE-REFIT-FAIL] name={pheno_name} site=_fit_logit_ladder "
                                f"method={method} reason={type(e).__name__} message={error_msg}",
                                flush=True
                            )
                            continue
                        refit_ok, fail_reason, fail_details = _ok_mle_fit(refit, X, y, target_ix=target_ix)
                        if _converged(refit) and refit_ok:
                            print(
                                f"[MLE-REFIT-SUCCESS] name={pheno_name} site=_fit_logit_ladder "
                                f"method={method} ok_mle_fit=true",
                                flush=True
                            )
                            setattr(refit, "_final_is_mle", True)
                            setattr(refit, "_ridge_in_path", True)
                            setattr(refit, "_firth_in_path", True)
                            setattr(refit, "_path_reasons", tags + ["firth_seeded_refit"])
                            setattr(refit, "_firth_seeded_refit", True)
                            return refit, "firth_seeded_refit"
                        else:
                            fail_details_str = " ".join(f"{k}={v}" for k, v in fail_details.items())
                            print(
                                f"[MLE-REFIT-FAIL] name={pheno_name} site=_fit_logit_ladder "
                                f"method={method} FAIL_REASON={fail_reason} {fail_details_str}",
                                flush=True
                            )
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
                    mle_newton_ok, _, _ = _ok_mle_fit(mle_newton, X, y, target_ix=target_ix)
                    if _converged(mle_newton) and mle_newton_ok:
                        setattr(mle_newton, "_final_is_mle", True)
                        setattr(mle_newton, "_path_reasons", ["mle_first_newton"] + prefit_gate_tags)
                        return mle_newton, "mle_first_newton"
                except (Exception, PerfectSeparationWarning) as e:
                    error_msg = str(e)[:200]
                    print(
                        f"[MLE-FIRST-FAIL] name={pheno_name or 'unknown'} site=_fit_logit_ladder "
                        f"method=newton reason={type(e).__name__} message={error_msg}",
                        flush=True
                    )
                try:
                    mle_bfgs = _logit_fit(
                        sm.Logit(y, X),
                        "bfgs",
                        maxiter=800,
                        gtol=1e-8,
                        start_params=user_start
                    )
                    mle_bfgs_ok, _, _ = _ok_mle_fit(mle_bfgs, X, y, target_ix=target_ix)
                    if _converged(mle_bfgs) and mle_bfgs_ok:
                        setattr(mle_bfgs, "_final_is_mle", True)
                        setattr(mle_bfgs, "_path_reasons", ["mle_first_bfgs"] + prefit_gate_tags)
                        return mle_bfgs, "mle_first_bfgs"
                except (Exception, PerfectSeparationWarning) as e:
                    error_msg = str(e)[:200]
                    print(
                        f"[MLE-FIRST-FAIL] name={pheno_name or 'unknown'} site=_fit_logit_ladder "
                        f"method=bfgs reason={type(e).__name__} message={error_msg}",
                        flush=True
                    )

        # Ridge-first pathway with strict MLE gating
        n_params = int(X.shape[1])
        valid_zero_ixs = sorted(ix for ix in zero_penalty_ixs if isinstance(ix, int) and 0 <= ix < n_params)
        penalized_param_count = n_params - len(valid_zero_ixs)
        if penalized_param_count <= 0:
            penalized_param_count = n_params or 1
        n = max(1, X.shape[0])
        pi = float(np.mean(y)) if len(y) > 0 else 0.5
        n_eff = max(1.0, 4.0 * float(len(y)) * pi * (1.0 - pi))
        alpha_scalar = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(penalized_param_count) / n_eff), 1e-6)
        pen_weight = np.ones(n_params, dtype=np.float64)
        if valid_zero_ixs:
            pen_weight[valid_zero_ixs] = 0.0

        if const_ix is not None:
            const_ix_eff = int(const_ix)
        elif is_pandas and "const" in X.columns:
            const_ix_eff = int(X.columns.get_loc("const"))
        else:
            const_ix_eff = None
        scales = _ridge_column_scales(X, const_ix=const_ix_eff)

        if scales is None:
            X_ridge = X
            start_scaled = user_start
        else:
            X_np = X.to_numpy(dtype=np.float64, copy=False) if is_pandas else np.asarray(X, dtype=np.float64)
            X_scaled_np = np.array(X_np, dtype=np.float64, copy=True)
            for j, scale in enumerate(scales):
                if scale != 1.0:
                    X_scaled_np[:, j] = X_scaled_np[:, j] / scale
            if is_pandas:
                X_ridge = pd.DataFrame(X_scaled_np, index=X.index, columns=X.columns)
            else:
                X_ridge = X_scaled_np
            if user_start is None:
                start_scaled = None
            else:
                start_arr = np.asarray(user_start, dtype=np.float64)
                if start_arr.shape[-1] != len(scales):
                    start_scaled = start_arr
                else:
                    start_scaled = start_arr * scales

        logit_model = sm.Logit(y, X_ridge)
        fit_regularized_kwargs = dict(kwargs)
        fit_regularized_kwargs.update({
            "alpha": float(alpha_scalar),
            "L1_wt": 0.0,
            "maxiter": 800,
            "disp": 0,
            "start_params": start_scaled,
            "pen_weight": pen_weight,
        })
        try:
            ridge_fit = logit_model.fit_regularized(**fit_regularized_kwargs)
        except TypeError:
            fit_regularized_kwargs.pop("pen_weight", None)
            fit_regularized_kwargs["alpha"] = np.asarray(pen_weight * float(alpha_scalar), dtype=np.float64)
            ridge_fit = logit_model.fit_regularized(**fit_regularized_kwargs)


        if scales is not None:
            params_scaled = np.asarray(ridge_fit.params, dtype=np.float64)
            if params_scaled.shape[-1] == len(scales):
                params_unscaled = params_scaled / np.where(scales == 0.0, 1.0, scales)
                if hasattr(ridge_fit, "params"):
                    if isinstance(ridge_fit.params, pd.Series):
                        ridge_fit.params = pd.Series(params_unscaled, index=ridge_fit.params.index)
                    else:
                        ridge_fit.params[:] = params_unscaled
                if hasattr(ridge_fit, "_results") and hasattr(ridge_fit._results, "params"):
                    ridge_fit._results.params[:] = params_unscaled
                setattr(ridge_fit, "_ridge_scales", scales)

        setattr(ridge_fit, "_ridge_alpha", float(alpha_scalar))
        setattr(ridge_fit, "_ridge_const_ix", None if const_ix is None else int(const_ix))
        setattr(ridge_fit, "_ridge_zero_penalty_ixs", valid_zero_ixs)
        setattr(ridge_fit, "_ridge_penalty_weights", pen_weight)
        setattr(ridge_fit, "_used_ridge", True)
        setattr(ridge_fit, "_final_is_mle", False)

        try:
            max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, ridge_fit.params)
        except Exception:
            max_abs_linpred, frac_lo, frac_hi = float("inf"), 1.0, 1.0

        neff_gate = float(CTX.get("MLE_REFIT_MIN_NEFF", 0.0))
        gate_tags = _ridge_gate_reasons(max_abs_linpred, frac_lo, frac_hi, n_eff, neff_gate)
        blocked_by_gate = ((max_abs_linpred > 15.0) or (frac_lo > 0.02) or (frac_hi > 0.02) or (neff_gate > 0 and n_eff < neff_gate))
        
        # Print ridge gate details if gated or not allowing MLE
        if gate_tags or not allow_mle:
            pheno_name = kwargs.get("pheno_name", CTX.get("current_phenotype", "unknown"))
            zero_penalty_info = f"{len(ridge_zero_penalty_ixs)}" if ridge_zero_penalty_ixs else "0"
            print(
                f"[RIDGE-GATE] name={pheno_name} site=_fit_logit_ladder "
                f"reasons={'|'.join(gate_tags) if gate_tags else 'none'} "
                f"max|Xb|={max_abs_linpred:.4f} frac_p_lo={frac_lo:.4f} frac_p_hi={frac_hi:.4f} "
                f"neff={n_eff:.2f} alpha={alpha_scalar:.6f} "
                f"penalized_params={X.shape[1]} zero_penalty_ixs={zero_penalty_info}",
                flush=True
            )
        ridge_unpenalized_tag = "ridge_unpenalized_terms" if valid_zero_ixs else None
        path_prefix = ["ridge_reached"]
        if ridge_unpenalized_tag is not None:
            path_prefix.append(ridge_unpenalized_tag)
        path_prefix.extend(gate_tags)
        path_prefix.extend(prefit_gate_tags)
        if blocked_by_gate or (not allow_mle):
            firth_attempt = _maybe_firth(path_prefix)
            if firth_attempt is not None:
                return firth_attempt
            tags = ["ridge_only"]
            if ridge_unpenalized_tag is not None:
                tags.append(ridge_unpenalized_tag)
            tags += gate_tags
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
                refit_newton_ok, _, _ = _ok_mle_fit(refit_newton, X, y, target_ix=target_ix)
                if _converged(refit_newton) and refit_newton_ok:
                    setattr(refit_newton, "_used_ridge_seed", True)
                    setattr(refit_newton, "_final_is_mle", True)
                    tags = ["ridge_seeded_refit"] + gate_tags + prefit_gate_tags
                    setattr(refit_newton, "_path_reasons", tags)
                    return refit_newton, "ridge_seeded_refit"
            except (Exception, PerfectSeparationWarning) as e:
                error_msg = str(e)[:400]
                print(
                    f"[MLE-REFIT-FAIL] name={pheno_name} site=_fit_logit_ladder "
                    f"method=newton reason={type(e).__name__} message={error_msg}",
                    flush=True
                )

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
                refit_bfgs_ok, _, _ = _ok_mle_fit(refit_bfgs, X, y, target_ix=target_ix)
                if _converged(refit_bfgs) and refit_bfgs_ok:
                    setattr(refit_bfgs, "_used_ridge_seed", True)
                    setattr(refit_bfgs, "_final_is_mle", True)
                    tags = ["ridge_seeded_refit"] + gate_tags + prefit_gate_tags
                    setattr(refit_bfgs, "_path_reasons", tags)
                    return refit_bfgs, "ridge_seeded_refit"
            except (Exception, PerfectSeparationWarning) as e:
                error_msg = str(e)[:400]
                print(
                    f"[MLE-REFIT-FAIL] name={pheno_name} site=_fit_logit_ladder "
                    f"method=bfgs reason={type(e).__name__} message={error_msg}",
                    flush=True
                )

        firth_path = list(path_prefix) + ["seeded_refit_failed"]
        firth_attempt = _maybe_firth(firth_path)
        if firth_attempt is not None:
            return firth_attempt

        tags = ["ridge_only"]
        if ridge_unpenalized_tag is not None:
            tags.append(ridge_unpenalized_tag)
        tags += gate_tags
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


def _final_stage_penalized(fit, reason_tag=None):
    """Return True if the terminal fit in a ladder relied on penalization."""
    if fit is None:
        return True
    if bool(getattr(fit, "_final_is_mle", False)):
        return False
    if bool(getattr(fit, "_used_firth", False)):
        return True
    if bool(getattr(fit, "_used_ridge", False)):
        return True
    path = getattr(fit, "_path_reasons", None)
    last_tag = None
    if isinstance(path, (list, tuple)) and path:
        last_tag = path[-1]
    elif isinstance(reason_tag, str) and reason_tag:
        last_tag = reason_tag
    if isinstance(last_tag, str):
        if last_tag.startswith("ridge"):
            return True
        if last_tag in {"firth_refit"}:
            return True
    return False

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


def _drop_rank_deficient(
    X: pd.DataFrame, keep_cols=("const",), always_keep=(), rtol=1e-2
):
    """Drop linearly dependent columns using prioritized QR checks.

    Columns listed in ``keep_cols`` and ``always_keep`` are evaluated first so
    they are retained whenever they add independent information. Lower-priority
    columns are dropped when they do not increase the rank of the design matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Design matrix to evaluate.
    keep_cols : tuple or list, default ("const",)
        Columns that should be considered first when building the independent
        basis (for example, the intercept).
    always_keep : tuple or list, default ()
        Additional high-priority columns (for example, the target variable).
    rtol : float, default 1e-2
        Relative tolerance used to decide when a column materially increases the
        rank. Higher values drop near-dependent columns more aggressively.

    Returns
    -------
    (pd.DataFrame, list[str])
        Tuple of the pruned design matrix and the list of columns that were
        removed due to rank deficiency.
    """
    if X.shape[1] == 0:
        return X, []

    # Establish priority order: keep_cols, always_keep, then everything else in
    # their existing order.
    priority = []
    seen = set()
    for name in list(keep_cols) + list(always_keep):
        if name in X.columns and name not in seen:
            priority.append(name)
            seen.add(name)
    for name in X.columns:
        if name not in seen:
            priority.append(name)
            seen.add(name)

    X_ordered = X.loc[:, priority]
    X_arr = np.asarray(X_ordered, dtype=np.float64)

    # Scale columns to unit norm before QR so the tolerance is based on linear
    # independence rather than raw magnitude (which can vary wildly across
    # covariates like an intercept vs. age-squared).
    col_norms = sp_linalg.norm(X_arr, axis=0, ord=2)
    safe_norms = np.where(col_norms == 0.0, 1.0, col_norms)
    X_scaled = X_arr / safe_norms

    # Use pivoted QR to obtain a stable rank estimate and tolerance threshold on
    # the scaled design.
    try:
        R, _ = sp_linalg.qr(X_scaled, mode="r", pivoting=True)
    except Exception:
        # If QR fails for any reason, return the original matrix to avoid
        # blocking the pipeline; downstream steps will handle the failure.
        return X, []

    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return X, []
    tol = diag.max() * float(rtol)

    kept_cols = []
    kept_mat = np.empty((X_scaled.shape[0], 0), dtype=np.float64)
    for idx, name in enumerate(priority):
        col = X_scaled[:, [idx]]
        if kept_mat.shape[1] == 0:
            residual_norm = float(sp_linalg.norm(col, ord=2))
        else:
            coeffs, _, _, _ = sp_linalg.lstsq(kept_mat, col)
            residual = col - kept_mat @ coeffs
            residual_norm = float(sp_linalg.norm(residual, ord=2))

        if residual_norm > tol:
            kept_cols.append(name)
            kept_mat = np.hstack([kept_mat, col]) if kept_mat.size else col

    dropped = [name for name in X.columns if name not in kept_cols]
    return X.loc[:, kept_cols], dropped


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
    
    Only supports standard MLE fits. Returns invalid for penalized fits (Firth/ridge).
    The penalized parameter is ignored (kept for API compatibility).
    """
    if fit is None or (not hasattr(fit, "params")) or (not hasattr(fit, "bse")):
        return {"valid": False}

    if bool(getattr(fit, "_used_firth", False)) or bool(getattr(fit, "_used_ridge", False)):
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

    method = "wald_mle"
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
    maxiter_firth = 400  # Increased from 200 to handle slow convergence in perfect separation scenarios
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

    eta = X_np @ beta
    eta_work = np.clip(eta, -35.0, 35.0)
    p = np.clip(expit(eta_work), 1e-12, 1.0 - 1e-12)
    W = p * (1.0 - p)
    XTW = X_np.T * W
    XtWX = XTW @ X_np
    loglik = float(np.sum(y_np * eta - np.logaddexp(0.0, eta)))
    sign_det, logdet = np.linalg.slogdet(XtWX)
    pll = loglik + 0.5 * logdet if sign_det > 0 else -np.inf

    class _Result:
        """Lightweight container to mimic statsmodels results where needed."""

        pass

    firth_res = _Result()
    if hasattr(X, "columns"):
        idx = X.columns
        firth_res.params = pd.Series(beta, index=idx)
        firth_res.bse = pd.Series(np.full(beta.shape, np.nan), index=idx)
        firth_res.pvalues = pd.Series(np.full(beta.shape, np.nan), index=idx)
    else:
        firth_res.params = beta
        firth_res.bse = np.full(beta.shape, np.nan)
        firth_res.pvalues = np.full(beta.shape, np.nan)
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




# Phenotypes that always require sex restriction to the majority sex
SEX_SPECIFIC_PHENOTYPES = {
    "Amenorrhea",
    "Antepartum_hemorrhage",
    "Bleeding_in_pregnancy",
    "Complications_of_labor_and_delivery_NEC",
    "Decreased_fetal_movements_affecting_management_of_mother",
    "Disorders_of_the_breast_associated_with_childbirth_and_disorders_of_lactation",
    "Drugs_and_alcohol_during_pregnancy",
    "Ectopic_pregnancy",
    "Dysmenorrhea",
    "Estrogen_receptor_positive_status_ER+",
    "Estrogen_receptor_status",
    "Excessive_vomiting_in_pregnancy",
    "Excessive_fetal_growth",
    "High_risk_human_papillomavirus_HPV_DNA_test_positive",
    "Malignant_neoplasm_of_male_genitalia",
    "Other_drug_use_during_pregnancy",
    "Recurrent_pregnancy_loss",
}

def _apply_sex_restriction(X: pd.DataFrame, y: pd.Series, pheno_name: str = None):
    """
    Returns: (X2, y2, note:str, skip_reason:str|None)
    
    Applies sex restriction based on case distribution. For certain sex-specific
    phenotypes, restriction is always applied to the majority sex regardless of
    the proportion threshold.
    """
    if 'sex' not in X.columns:
        return X, y, "", None
    tab = pd.crosstab(X['sex'], y).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
    total_cases = int(tab.loc[0.0, 1] + tab.loc[1.0, 1])
    if total_cases <= 0:
        return X, y, "", None
    
    cases_by_sex = {0.0: int(tab.loc[0.0, 1]), 1.0: int(tab.loc[1.0, 1])}
    dominant_sex = 0.0 if cases_by_sex[0.0] >= cases_by_sex[1.0] else 1.0
    
    # Check if this is a sex-specific phenotype that always requires restriction
    force_restriction = pheno_name in SEX_SPECIFIC_PHENOTYPES if pheno_name else False
    
    if force_restriction:
        # Always restrict to majority sex for sex-specific phenotypes
        should_restrict = True
    else:
        # Normal threshold-based logic
        thr = float(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP))
        frac = (cases_by_sex[dominant_sex] / total_cases) if total_cases > 0 else 0.0
        should_restrict = frac >= thr
    
    if not should_restrict:
        return X, y, "", None
    
    if int(tab.loc[dominant_sex, 0]) == 0:
        return X, y, "", "sex_no_controls_in_case_sex"

    keep = X['sex'].eq(dominant_sex)
    note_parts = [f"sex_restricted_to_{int(dominant_sex)}"]
    
    if force_restriction:
        note_parts.append(f"sex_forced_restriction_to_{int(dominant_sex)}")
    else:
        mode = str(CTX.get("SEX_RESTRICT_MODE", "majority")).lower()
        if mode == "majority":
            note_parts.append(f"sex_majority_restricted_to_{int(dominant_sex)}")

    note = ";".join(note_parts)
    
    # Calculate pre/post restriction counts
    N_pre = len(X)
    C_pre = int((y == 1).sum())
    K_pre = int((y == 0).sum())
    X_post = X.loc[keep].drop(columns=['sex'])
    y_post = y.loc[keep]
    N_post = len(X_post)
    C_post = int((y_post == 1).sum())
    K_post = int((y_post == 0).sum())
    
    # Print sex restriction details
    mode_str = str(CTX.get("SEX_RESTRICT_MODE", "majority"))
    thr_str = str(CTX.get("SEX_RESTRICT_PROP", DEFAULT_SEX_RESTRICT_PROP))
    print(
        f"[SEX-RESTRICT] name={pheno_name or 'unknown'} site=_apply_sex_restriction "
        f"dominant_sex={int(dominant_sex)} forced={str(force_restriction).lower()} "
        f"mode={mode_str} thr={thr_str} "
        f"pre_N/Cases/Ctrls={N_pre}/{C_pre}/{K_pre} "
        f"post_N/Cases/Ctrls={N_post}/{C_post}/{K_post} "
        f"note={note}",
        flush=True
    )
    
    return X_post, y_post, note, None




# --- Bootstrap helpers ---
def _score_test_components(X_red: pd.DataFrame, y: pd.Series, target: str):
    const_ix = X_red.columns.get_loc('const') if 'const' in X_red.columns else None
    fit_red, _ = _fit_logit_ladder(X_red, y, const_ix=const_ix, prefer_mle_first=True)
    if fit_red is None:
        raise ValueError('reduced fit failed')

    used_ridge = bool(getattr(fit_red, "_used_ridge", False))
    used_firth = bool(getattr(fit_red, "_used_firth", False))
    final_is_mle = bool(getattr(fit_red, "_final_is_mle", False))
    if used_ridge:
        fit_type = "ridge"
    elif used_firth:
        fit_type = "firth"
    elif final_is_mle:
        fit_type = "mle"
    else:
        fit_type = "unknown"

    if fit_type != "mle":
        return fit_red, fit_type, None, None

    beta = np.asarray(getattr(fit_red, "params", np.zeros(X_red.shape[1])), dtype=np.float64)
    eta = X_red.to_numpy(dtype=np.float64, copy=False) @ beta
    eta = np.clip(eta, -35.0, 35.0)
    p_hat = expit(eta)
    W = p_hat * (1.0 - p_hat)
    return fit_red, fit_type, p_hat, W


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


def _rao_score_block(y, X0, X1, clip_w=1e-12, rcond=1e-12, fit_red=None):
    """
    Multi-df Rao score test for adding block X1 to reduced model X0 in logistic regression.
    Uses statsmodels for the reduced MLE fit and SVD-based pseudoinverses for stability.

    Args:
        y: outcome vector
        X0: reduced model design matrix
        X1: interaction block to test
        clip_w: minimum weight for numerical stability
        rcond: condition number tolerance for rank determination
        fit_red: optional pre-fitted reduced model (if None, will fit fresh)

    Returns (statistic, df, pval, details_dict).
    """
    # Ensure arrays
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X0 = np.asarray(X0, dtype=np.float64)
    X1 = np.asarray(X1, dtype=np.float64)

    # 1) Use pre-fitted reduced model if provided, otherwise fit fresh
    if fit_red is not None:
        res0 = fit_red
        # Check that params are available
        if not hasattr(res0, "params") or getattr(res0, "params", None) is None:
            return np.nan, 0, np.nan, {"error": "reduced_no_params"}
        
        # CRITICAL: Rao score test's χ² calibration requires true MLE, not penalized fit
        # Check if this is a genuine MLE (not pure Firth)
        is_mle = bool(getattr(res0, "_final_is_mle", False))
        if not is_mle:
            # This is a penalized fit (Firth without MLE refit)
            # χ² calibration is invalid - caller should use bootstrap instead
            return np.nan, 0, np.nan, {"error": "reduced_not_mle"}
    else:
        # Fit reduced model by mainstream library (statsmodels)
        #    Newton with generous iterations; this is usually stable for the reduced model.
        try:
            res0 = sm.Logit(y, X0).fit(disp=0, method="newton", maxiter=200)
        except Exception:
            return np.nan, 0, np.nan, {"error": "reduced_fit_failed"}

        if not bool(getattr(res0, "converged", False)):
            return np.nan, 0, np.nan, {"error": "reduced_not_converged"}
        
        # Fresh fit from statsmodels is always MLE
        setattr(res0, "_final_is_mle", True)

    # Get fitted probabilities (handle both statsmodels and Firth _Result objects)
    if hasattr(res0, "predict"):
        p = res0.predict()
    else:
        # Manual prediction for Firth _Result objects
        params = np.asarray(getattr(res0, "params", None), dtype=np.float64)
        if params is None or params.shape[0] != X0.shape[1]:
            return np.nan, 0, np.nan, {"error": "reduced_params_mismatch"}
        from scipy.special import expit
        eta = X0 @ params
        p = expit(eta)

    w = p * (1.0 - p)                                  # logistic weights p(1-p)
    w = np.clip(w, clip_w, 0.25)                       # guard against vanishing weights

    # 2) Efficient information for X1 given X0:
    #    I_eff = X1' W X1  -  X1' W X0 (X0' W X0)^(-1) X0' W X1
    WX0 = X0 * w[:, None]
    WX1 = X1 * w[:, None]
    XtWX0 = X0.T @ WX0
    X1tWX1 = X1.T @ WX1
    X1tWX0 = X1.T @ WX0

    # Stable pseudo-inverse for symmetric PSD matrices via SVD
    def _sym_pinv(A, tol=rcond):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        if s.size == 0:
            return A * 0.0
        s_inv = np.where(s > tol * s.max(), 1.0 / s, 0.0)
        return (Vt.T * s_inv) @ U.T

    XtWX0_inv = _sym_pinv(XtWX0)
    I_eff = X1tWX1 - X1tWX0 @ XtWX0_inv @ X1tWX0.T

    # 3) Score vector for X1 at reduced fit: U = X1' (y - p)
    U = X1.T @ (y - p)

    # 4) Test statistic: U' I_eff^(-1) U  ~  χ²_df
    I_eff_inv = _sym_pinv(I_eff)
    stat = float(U.T @ I_eff_inv @ U)

    # Rank for df (robust to dropped/aliased columns)
    df = int(np.linalg.matrix_rank(I_eff, tol=rcond))

    # p-value
    pval = float(sp_stats.chi2.sf(stat, df)) if df > 0 else np.nan

    details = {
        "W_mean": float(w.mean()),
        "I_eff_cond": float(np.linalg.cond(I_eff)) if df > 0 else np.inf,
        "rank": df,
        "reduced_converged": bool(getattr(res0, "converged", True)),
        "llf_reduced": float(getattr(res0, "llf", np.nan)),
    }
    return stat, df, pval, details


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

    def _invalid_result():
        return {"p": np.nan, "T_obs": np.nan, "draws": 0, "exceed": 0, "fit_kind": None, "den": np.nan}

    Xr = X_red.to_numpy(dtype=np.float64, copy=False) if hasattr(X_red, "to_numpy") else np.asarray(X_red, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    xt = np.asarray(x_target, dtype=np.float64)
    if Xr.ndim != 2 or yv.ndim != 1 or xt.ndim != 1 or Xr.shape[0] != yv.shape[0] or xt.shape[0] != yv.shape[0]:
        return _invalid_result()
    bits = _score_bootstrap_bits(Xr, yv, xt, 0.0, kind=kind)
    if bits is None and kind == "mle":
        bits = _score_bootstrap_bits(Xr, yv, xt, 0.0, kind="firth")
    if bits is None:
        return _invalid_result()
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
    return {
        "p": detail.get("p", np.nan),
        "T_obs": float(bits.get("T_obs", np.nan)),
        "draws": int(detail.get("draws", 0)),
        "exceed": int(detail.get("exceed", 0)),
        "fit_kind": bits.get("fit_kind", kind),
        "den": float(bits.get("den", np.nan)),
    }


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
    global worker_core_index_fp
    worker_core_index_fp = _index_fingerprint(worker_core_df_index)
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

def _index_fingerprint(index: pd.Index) -> str:
    """Order-invariant fingerprint of an index.

    The hash is computed on the sorted stringified identifiers so that pure row
    re-orderings (including core index reorders) do not invalidate the cache,
    while changes in membership or count still alter the fingerprint.
    """
    h = hashlib.blake2b(digest_size=16)
    if len(index) == 0:
        return f"{h.hexdigest()}:0"

    ids = np.sort(index.astype(str, copy=False))
    for pid in ids:
        h.update(pid.encode("utf-8"))
    return f"{h.hexdigest()}:{ids.size}"

def _mask_fingerprint(mask: np.ndarray, index: pd.Index) -> str:
    """Order-invariant fingerprint of a masked subset of an index."""
    if mask.size != len(index):
        raise ValueError("mask length must match index length for fingerprinting")

    subset = index[mask]
    h = hashlib.blake2b(digest_size=16)
    if subset.empty:
        return f"{h.hexdigest()}:0"

    ids = np.sort(subset.astype(str, copy=False))
    for pid in ids:
        h.update(pid.encode("utf-8"))
    return f"{h.hexdigest()}:{ids.size}"


def _core_index_fp() -> str:
    """Return the precomputed core index fingerprint (order-invariant)."""
    return worker_core_index_fp if worker_core_index_fp is not None else _index_fingerprint(worker_core_df_index)





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
        _same_members_ignore_order(meta.get("model_columns"), core_df_cols) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp and
        meta.get("allowed_mask_fp") == allowed_fp and
        meta.get("target") == target
    )
    if not base_ok:
        return False
    data_keys = CTX.get("DATA_KEYS")
    if data_keys and not _same_members_ignore_order(meta.get("data_keys"), data_keys):
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
        _same_members_ignore_order(meta.get("model_columns"), core_df_cols) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp and
        meta.get("allowed_mask_fp") == allowed_fp and
        meta.get("target") == target
    )
    if not base_ok:
        return False
    data_keys = CTX.get("DATA_KEYS")
    if data_keys and not _same_members_ignore_order(meta.get("data_keys"), data_keys):
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
    s_name = task["name"]
    previous_pheno = CTX.get("current_phenotype")
    with logging_utils.phenotype_logging(s_name):
        try:
            CTX["current_phenotype"] = s_name
            return _lrt_overall_worker_impl(task)
        finally:
            if previous_pheno is None:
                CTX.pop("current_phenotype", None)
            else:
                CTX["current_phenotype"] = previous_pheno


def _lrt_overall_worker_impl(task):
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
            try:
                case_df = _read_case_cache(
                    pheno_path,
                    phenotype=s_name,
                    stage="LRT-Stage1",
                    columns=['is_case'],
                )
            except CaseCacheReadError as err:
                message = err.detail
                io.atomic_write_json(result_path, {
                    "Phenotype": s_name,
                    "P_LRT_Overall": np.nan,
                    "LRT_Overall_Reason": "case_cache_error",
                    "LRT_Overall_Message": message,
                })
                io.atomic_write_json(res_path, {
                    "Phenotype": s_name,
                    "N_Total": np.nan,
                    "N_Cases": np.nan,
                    "N_Controls": np.nan,
                    "Beta": np.nan,
                    "OR": np.nan,
                    "P_Value": np.nan,
                    "OR_CI95": None,
                    "Used_Ridge": False,
                    "Final_Is_MLE": False,
                    "Used_Firth": False,
                    "N_Total_Used": np.nan,
                    "N_Cases_Used": np.nan,
                    "N_Controls_Used": np.nan,
                    "Model_Notes": "",
                    "Skip_Reason": "case_cache_error",
                    "Skip_Message": message,
                })
                return
            case_ids = case_df.query("is_case == 1").index
            idx = worker_core_df_index.get_indexer(case_ids)
            case_idx = idx[idx >= 0].astype(np.int32)
            case_fp = _index_fingerprint(worker_core_df_index[case_idx] if case_idx.size > 0 else pd.Index([]))

        # Use per-category allowed fingerprint computed once in worker
        allowed_fp = allowed_fp_by_cat.get(cat) if 'allowed_fp_by_cat' in globals() else _mask_fingerprint(
            allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool)), worker_core_df_index
        )

        core_fp = _core_index_fp()

        repair_meta = os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False)

        allowed_mask = allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool))
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        anc_cols = [c for c in worker_core_df_cols if c.startswith("ANC_")]
        required_cols = ['const', target]
        missing_required = [c for c in required_cols if c not in col_ix]
        if missing_required:
            raise KeyError(f"missing required design columns: {missing_required}")

        def _existing(names):
            return [name for name in names if name in col_ix]

        base_cols = list(required_cols)
        base_cols += _existing(['sex'])
        base_cols += _existing(pc_cols)
        base_cols += _existing(['AGE_c', 'AGE_c_sq'])
        base_cols += _existing(anc_cols)
        base_ix = [col_ix[c] for c in base_cols]

        X_base = pd.DataFrame(
            X_all[valid_mask][:, base_ix],
            index=worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index, dtype=np.int8)

        # Pre-sex-restriction counts to mirror main PheWAS semantics
        n_cases_pre = int(y_series.sum())
        n_ctrls_pre = int(len(y_series) - n_cases_pre)
        n_total_pre = int(len(y_series))

        Xb, yb, note, skip = _apply_sex_restriction(X_base, y_series, pheno_name=s_name)
        n_total_used, n_cases_used, n_ctrls_used = len(yb), int(yb.sum()), len(yb) - int(yb.sum())
        
        # Initialize notes list for collecting additional diagnostic messages
        notes = []

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
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp, extra=meta_extra)

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
                core_fp,
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
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp, extra=meta_extra)

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
                core_fp,
                case_fp,
                extra=meta_extra_result,
            )
            return

        X_full_df = Xb

        # Prune the full model first to resolve rank deficiency.
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=(target,))
        X_full_zv, dropped_rank_cols = _drop_rank_deficient(
            X_full_zv, keep_cols=("const",), always_keep=(target,)
        )

        if dropped_rank_cols:
            drop_note = f"dropped_rank_def={','.join(dropped_rank_cols)}"
            note = f"{note};{drop_note}" if note else drop_note

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
                core_fp,
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
                core_fp,
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

        # Run diagnostic checks before fitting
        _check_separation_in_strata(X_full_zv, yb, target, pheno_name=s_name)
        _check_collinearity(X_full_zv, pheno_name=s_name)

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red, pheno_name=s_name)
        fit_full, reason_full = _fit_logit_ladder(
            X_full_zv,
            yb,
            const_ix=const_ix_full,
            target_ix=target_ix,
            pheno_name=s_name,
        )
        
        # Check leverage/influence after initial fit
        if fit_full is not None:
            _check_leverage_influence(X_full_zv, yb, fit_full, pheno_name=s_name)

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

        full_ok, _, _ = _ok_mle_fit(fit_full, X_full_zv, yb, target_ix=target_ix) if fit_full is not None else (False, None, {})
        red_ok, _, _ = _ok_mle_fit(fit_red, X_red_zv, yb) if fit_red is not None else (False, None, {})

        inference_family = None
        fit_full_use = None
        fit_red_use = None
        if (
            fit_full is not None
            and fit_red is not None
            and full_is_mle
            and red_is_mle
            and full_ok
            and red_ok
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
                
                # Print inference choice
                full_path = getattr(fit_full, "_path_reasons", ["unknown"])
                red_path = getattr(fit_red, "_path_reasons", ["unknown"])
                print(
                    f"[INFERENCE-CHOICE] name={s_name} site=lrt_overall_worker "
                    f"choice=firth full_fit={'|'.join(full_path)} red_fit={'|'.join(red_path)}",
                    flush=True
                )

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
        inference_type = inference_family if inference_family is not None else None

        if inference_family is not None:
            if target_ix is not None and fit_full_use is not None:
                ci_info = _profile_ci_beta(X_full_zv, yb, target_ix, fit_full_use, kind=inference_family)
                ci_method = ci_info.get("method")
                ci_sided = ci_info.get("sided", "two")
                ci_valid = bool(ci_info.get("valid", False))
                
                # Print CI profile penalized details if using profile_penalized method
                if ci_method == "profile_penalized":
                    beta_hat = getattr(fit_full_use, "params", [np.nan])[target_ix] if hasattr(fit_full_use, "params") else np.nan
                    lo_or_val = np.exp(ci_info.get("lo", np.nan)) if np.isfinite(ci_info.get("lo", np.nan)) else np.nan
                    hi_or_val = np.exp(ci_info.get("hi", np.nan)) if np.isfinite(ci_info.get("hi", np.nan)) else np.nan
                    print(
                        f"[CI-PROFILE-PENALIZED] name={s_name} site=lrt_overall_worker "
                        f"sided={ci_sided} lo_or={lo_or_val:.4f} hi_or={hi_or_val:.4f} "
                        f"valid={str(ci_valid).lower()} beta_hat={beta_hat:.4f} note={ci_info.get('note', '')}",
                        flush=True
                    )
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
            if params is not None and target_ix is not None:
                try:
                    if hasattr(params, "__getitem__"):
                        beta_full = float(params[target]) if hasattr(params, "index") else float(params[target_ix])
                    else:
                        beta_full = float(np.asarray(params)[target_ix])
                    or_val = float(np.exp(beta_full))
                except Exception:
                    beta_full = np.nan
                    or_val = np.nan
            if inference_family == "mle":
                ll_full = float(getattr(fit_full_use, "llf", np.nan))
                ll_red = float(getattr(fit_red_use, "llf", np.nan))
                if np.isfinite(ll_full) and np.isfinite(ll_red):
                    stat = max(0.0, 2.0 * (ll_full - ll_red))
                    p_value = float(sp_stats.chi2.sf(stat, 1))
                    p_source = "lrt_mle"
                    inference_type = "mle"
                else:
                    inference_type = None
            elif inference_family == "firth":
                # Skip the nominal LRT for penalized fits; a score-based fallback
                # is attempted below. Keep inference_type = "firth" to indicate
                # that Firth regression was used for coefficient estimates and CIs.
                ll_full = float(getattr(fit_full_use, "llf", np.nan))
                ll_red = float(getattr(fit_red_use, "llf", np.nan))
                # Don't set inference_type = None here; preserve "firth"

        if (
            (not np.isfinite(p_value))
            and target_ix is not None
            and target in X_full_zv.columns
            and red_is_mle
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
                # Preserve inference_type if already set (e.g., "firth")
                if inference_type is None:
                    inference_type = "score"
            else:
                if ENABLE_SCORE_BOOT_MLE:
                    boot_res = _score_bootstrap_from_reduced(
                        X_red_zv,
                        yb,
                        x_target_vec,
                        seed_key=("lrt_overall", s_name_safe, target, "pval"),
                    )
                    p_emp = float(boot_res.get("p", np.nan))
                    if np.isfinite(p_emp):
                        p_value = p_emp
                        p_source = "score_boot_firth" if boot_res.get("fit_kind") == "firth" else "score_boot_mle"
                        # Preserve inference_type if already set (e.g., "firth")
                        if inference_type is None:
                            inference_type = "score_boot"
                else:
                    # Bootstrap disabled - mark p-value as invalid
                    p_value = np.nan
                    notes.append("score_boot_disabled:score_chi2_failed")


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

        if inference_type is None:
            inference_type = inference_family if inference_family is not None else "none"

        p_finite = bool(np.isfinite(p_value))
        p_valid = bool(p_finite and (p_source in ALLOWED_P_SOURCES))

        used_firth_for_ci = False

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
            wald = {"valid": False}
            wald_fit = None
            if fit_full is not None and not bool(getattr(fit_full, "_used_ridge", False)) and not bool(
                getattr(fit_full, "_used_firth", False)
            ):
                fit_full_wald_ok, _, _ = _ok_mle_fit(fit_full, X_full_zv, yb, target_ix=target_ix)
                if fit_full_wald_ok:
                    wald_fit = fit_full
            if (
                wald_fit is None
                and fit_full_use is not None
                and not bool(getattr(fit_full_use, "_used_ridge", False))
                and not bool(getattr(fit_full_use, "_used_firth", False))
            ):
                fit_full_use_ok, _, _ = _ok_mle_fit(fit_full_use, X_full_zv, yb, target_ix=target_ix)
                if fit_full_use_ok:
                    wald_fit = fit_full_use
            if wald_fit is not None:
                wald = _wald_ci_or_from_fit(wald_fit, target_ix, alpha=0.05, penalized=False)
            if wald.get("valid", False):
                ci_valid = True
                ci_method = wald["method"]
                ci_sided = "two"
                ci_lo_or = float(wald["lo_or"])
                ci_hi_or = float(wald["hi_or"])
                or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)

        # Note: Wald CI from Firth fits is not supported (_wald_ci_or_from_fit rejects penalized fits)
        # Profile CIs are used for Firth instead

        if ci_valid:
            method_allowed = ci_method in ALLOWED_CI_METHODS
            if not method_allowed:
                ci_valid = False
                ci_method = None
                ci_label = ""
                ci_sided = None
                ci_lo_or = np.nan
                ci_hi_or = np.nan
                or_ci95 = None

        ridge_in_path_full = bool(getattr(fit_full, "_used_ridge", False)) or bool(getattr(fit_full, "_ridge_in_path", False))
        used_firth_full = (
            bool(getattr(fit_full, "_used_firth", False))
            or bool(getattr(fit_full_use, "_used_firth", False))
            or (inference_family == "firth")
            or used_firth_for_ci
        )

        p_lrt_overall = float(p_value) if (p_valid and p_source == "lrt_mle") else np.nan
        p_value_for_output = float(p_value) if p_valid else np.nan
        # Report inference_type even if p-value is invalid, as long as CI is valid
        inference_type_out = inference_type if (p_valid or ci_valid) else "none"
        inference_type = inference_type_out
        
        # Track coefficient source separately from p-value source
        # Coefficients come from fit_full_use if available, otherwise fit_full
        coef_source_fit = fit_full_use if fit_full_use is not None else fit_full
        coef_is_mle = (
            inference_family == "mle" or 
            (inference_family is None and coef_source_fit is not None and 
             bool(getattr(coef_source_fit, "_final_is_mle", False)) and 
             not bool(getattr(coef_source_fit, "_used_firth", False)))
        )

        out = {
            "Phenotype": s_name,
            "P_LRT_Overall": p_lrt_overall,
            "P_Value": p_value_for_output,
            "P_Overall_Valid": bool(p_valid),
            "P_Source": p_source,
            "P_Method": p_source if p_source is not None else None,
            "LRT_df_Overall": 1 if (p_valid and p_source == "lrt_mle") else np.nan,
            "Inference_Type": inference_type_out,
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
        # Merge any additional notes collected during processing
        model_notes.extend(notes)
        model_notes.append(f"inference={inference_type_out}")
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
            "P_Value": p_value_for_output,
            "P_Valid": bool(p_valid),
            "P_Source": p_source,
            "OR_CI95": or_ci95,
            "CI_Method": ci_method,
            "CI_Sided": ci_sided,
            "CI_Label": ci_label,
            "CI_Valid": bool(ci_valid),
            "CI_LO_OR": ci_lo_or,
            "CI_HI_OR": ci_hi_or,
            "Used_Ridge": ridge_in_path_full,
            "Final_Is_MLE": coef_is_mle,
            "Used_Firth": used_firth_full,
            "Inference_Type": inference_type_out,
            "Coef_Source": "mle" if coef_is_mle else ("firth" if inference_family == "firth" else "unknown"),
            "P_Value_Method": p_source,
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": ";".join(model_notes),
        }

        if inference_type == "mle":
            penalized = any(
                _final_stage_penalized(candidate, reason)
                for candidate, reason in (
                    (
                        fit_full_use if fit_full_use is not None else fit_full,
                        reason_full,
                    ),
                    (
                        fit_red_use if fit_red_use is not None else fit_red,
                        reason_red,
                    ),
                )
            )
        elif inference_type == "firth":
            penalized = True
        elif inference_type in {"score", "score_boot"}:
            penalized = False
        else:
            penalized = _final_stage_penalized(
                fit_full_use if fit_full_use is not None else fit_full,
                reason_full,
            )
        if penalized and not p_valid:
            # Print p-value blocked due to penalized fit
            full_path = getattr(fit_full_use if fit_full_use is not None else fit_full, "_path_reasons", ["unknown"])
            red_path = getattr(fit_red_use if fit_red_use is not None else fit_red, "_path_reasons", ["unknown"])
            attempted_sources = []
            if p_source:
                attempted_sources.append(f"{p_source}:{p_value if np.isfinite(p_value) else 'nan'}")
            print(
                f"[P-BLOCKED-PENALIZED] name={s_name} site=lrt_overall_worker "
                f"reason=penalized_fit_in_path full_path={'|'.join(full_path)} red_path={'|'.join(red_path)} "
                f"attempted_p_sources={';'.join(attempted_sources) if attempted_sources else 'none'}",
                flush=True
            )
            
            out.update(
                {
                    "P_LRT_Overall": np.nan,
                    "P_Value": np.nan,
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
                    "Beta": np.nan,
                    "OR": np.nan,
                    "OR_CI95": None,
                    "CI_Method": None,
                    "CI_Sided": None,
                    "CI_Label": "",
                    "CI_Valid": False,
                    "CI_LO_OR": np.nan,
                    "CI_HI_OR": np.nan,
                }
            )
            out_notes = out.get("Model_Notes")
            out["Model_Notes"] = f"{out_notes};{reason_tag}" if out_notes else reason_tag
            rec_notes = res_record.get("Model_Notes")
            res_record["Model_Notes"] = f"{rec_notes};{reason_tag}" if rec_notes else reason_tag

            if target_ix is not None:
                # Note: Wald CI from Firth fits is not supported (_wald_ci_or_from_fit rejects penalized fits)
                # Profile CIs are used for Firth instead
                pass
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
            core_fp,
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
                    core_fp, case_fp,
                    extra=meta_extra)
    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()

def bootstrap_overall_worker(task):
    """Stage-1 parametric bootstrap worker with phenotype-specific logging."""
    s_name = task["name"]
    previous_pheno = CTX.get("current_phenotype")
    with logging_utils.phenotype_logging(s_name):
        try:
            CTX["current_phenotype"] = s_name
            return _bootstrap_overall_worker_impl(task)
        finally:
            if previous_pheno is None:
                CTX.pop("current_phenotype", None)
            else:
                CTX["current_phenotype"] = previous_pheno


def _bootstrap_overall_worker_impl(task):
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
    core_fp = _core_index_fp()
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
            try:
                case_df = _read_case_cache(
                    pheno_path,
                    phenotype=s_name,
                    stage="Bootstrap-Stage1",
                    columns=['is_case'],
                )
            except CaseCacheReadError as err:
                message = err.detail
                io.atomic_write_json(result_path, {
                    "Phenotype": s_name,
                    "Reason": "case_cache_error",
                    "Message": message,
                })
                if not os.path.exists(res_path):
                    io.atomic_write_json(res_path, {
                        "Phenotype": s_name,
                        "N_Total": np.nan,
                        "N_Cases": np.nan,
                        "N_Controls": np.nan,
                        "Beta": np.nan,
                        "OR": np.nan,
                        "P_Value": np.nan,
                        "OR_CI95": None,
                        "Used_Ridge": False,
                        "Final_Is_MLE": False,
                        "Used_Firth": False,
                        "N_Total_Used": np.nan,
                        "N_Cases_Used": np.nan,
                        "N_Controls_Used": np.nan,
                        "Model_Notes": "",
                        "Skip_Reason": "case_cache_error",
                        "Skip_Message": message,
                    })
                return
            case_ids = case_df.query("is_case == 1").index
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
        required_cols = ['const', target]
        missing_required = [c for c in required_cols if c not in col_ix]
        if missing_required:
            raise KeyError(f"missing required design columns: {missing_required}")

        def _existing(names):
            return [name for name in names if name in col_ix]

        base_cols = list(required_cols)
        base_cols += _existing(['sex'])
        base_cols += _existing(pc_cols)
        base_cols += _existing(['AGE_c', 'AGE_c_sq'])
        base_cols += _existing(anc_cols)
        base_ix = [col_ix[c] for c in base_cols]
        X_base = pd.DataFrame(
            X_all[valid_mask][:, base_ix],
            index=worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index, dtype=np.int8)

        n_cases_pre = int(y_series.sum())
        n_ctrls_pre = int(len(y_series) - n_cases_pre)
        n_total_pre = int(len(y_series))

        Xb, yb, note, skip = _apply_sex_restriction(X_base, y_series, pheno_name=s_name)
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
            _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp, extra=meta_extra)
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
            _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp, extra=meta_extra)
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
        X_full_zv, dropped_rank_cols = _drop_rank_deficient(
            X_full_zv, keep_cols=("const",), always_keep=(target,)
        )
        if dropped_rank_cols:
            drop_note = f"dropped_rank_def={','.join(dropped_rank_cols)}"
            note = f"{note};{drop_note}" if note else drop_note
        if target not in X_full_zv.columns:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": "target_dropped_in_pruning"})
            return
        red_cols = [c for c in X_full_zv.columns if c != target]
        X_red_zv = X_full_zv[red_cols]

        fit_red, reduced_fit_type, p_hat, W = _score_test_components(X_red_zv, yb, target)
        red_final_is_mle = bool(getattr(fit_red, "_final_is_mle", False))
        red_used_firth = bool(getattr(fit_red, "_used_firth", False))
        red_used_ridge = bool(getattr(fit_red, "_used_ridge", False))
        if reduced_fit_type == "unknown":
            if red_used_ridge:
                reduced_fit_type = "ridge"
            elif red_used_firth:
                reduced_fit_type = "firth"
            elif red_final_is_mle:
                reduced_fit_type = "mle"

        meta_extra_common.update({
            "reduced_fit_type": reduced_fit_type,
            "reduced_final_is_mle": red_final_is_mle,
            "reduced_used_firth": red_used_firth,
            "reduced_used_ridge": red_used_ridge,
        })

        t_vec = X_full_zv[target].to_numpy(dtype=np.float64, copy=False)
        Xr = X_red_zv.to_numpy(dtype=np.float64, copy=False)

        p_emp = np.nan
        T_obs = np.nan
        boot_engine = None
        boot_fit_kind = None
        boot_draws = 0
        boot_exceed = 0
        p_valid = False
        p_source = None
        p_reason = None

        if red_used_ridge or reduced_fit_type == "ridge":
            p_reason = "penalized_reduced_null"
            # Print bootstrap null penalized
            print(
                f"[BOOT-NULL-PENALIZED] name={s_name} site=bootstrap_overall_worker "
                f"reduced_fit_type=ridge action=skip_empirical_p",
                flush=True
            )
        elif reduced_fit_type == "mle" and red_final_is_mle and not red_used_firth:
            if not ENABLE_SCORE_BOOT_MLE:
                p_reason = "score_boot_disabled:bernoulli_bootstrap"
            elif p_hat is None or W is None:
                p_reason = "missing_mle_probabilities"
            else:
                h, denom = _efficient_score_vector(t_vec, Xr, W)
                if not np.isfinite(denom) or denom <= 1e-14:
                    p_reason = "nonpos_denom"
                else:
                    resid = yb.to_numpy(dtype=np.float64, copy=False) - p_hat
                    S_obs = float(h @ resid)
                    T_obs = (S_obs * S_obs) / denom
                    pos = worker_core_df_index.get_indexer(X_red_zv.index)
                    pos = pos[pos >= 0]
                    B = int(U_boot.shape[1]) if U_boot is not None else 0
                    if B <= 0:
                        p_reason = "invalid_bootstrap_matrix"
                    else:
                        T_b = np.empty(B, dtype=np.float64)
                        for j0 in range(0, B, 64):
                            j1 = min(B, j0 + 64)
                            U_blk = U_boot[np.ix_(pos, np.arange(j0, j1))]
                            Ystar = (U_blk < p_hat[:, None]).astype(np.float64, copy=False)
                            R = Ystar - p_hat[:, None]
                            S = h @ R
                            T_b[j0:j1] = (S * S) / denom
                        exceed = int(np.sum(T_b >= T_obs))
                        p_emp = float((1.0 + exceed) / (1.0 + B))
                        if np.isfinite(p_emp):
                            p_valid = True
                            p_source = "score_boot_mle"
                            boot_engine = "bernoulli"
                            boot_draws = B
                            boot_exceed = exceed
                            np.save(os.path.join(tnull_dir, f"{s_name_safe}.npy"), T_b.astype(np.float32, copy=False))
                            
                            # Check bootstrap stability (simplified - assumes all draws valid for Bernoulli)
                            mc_se = np.sqrt(p_emp * (1 - p_emp) / B) if B > 0 else None
                            _check_bootstrap_instability(0, B, mc_se, [], pheno_name=s_name)
                        else:
                            p_reason = "bootstrap_failed"
        elif reduced_fit_type == "firth" or red_used_firth:
            if not ENABLE_SCORE_BOOT_MLE:
                p_reason = "score_boot_disabled:wild_refit_bootstrap"
            else:
                boot_engine = "wild_refit"
                boot_res = _score_bootstrap_from_reduced(
                    X_red_zv,
                    yb,
                    t_vec,
                    seed_key=("boot_overall", s_name_safe, target, "score"),
                    kind="mle",
                )
                boot_fit_kind = boot_res.get("fit_kind")
                p_emp = float(boot_res.get("p", np.nan))
                T_obs = float(boot_res.get("T_obs", np.nan))
                boot_draws = int(boot_res.get("draws", 0))
                boot_exceed = int(boot_res.get("exceed", 0))
                if np.isfinite(p_emp):
                    p_valid = True
                    p_source = "score_boot_firth" if boot_fit_kind == "firth" else "score_boot_mle"
                else:
                    p_reason = "bootstrap_failed"
        else:
            p_reason = "reduced_fit_unknown"

        p_finite = np.isfinite(p_emp)
        p_valid = bool(p_finite and (p_source in ALLOWED_P_SOURCES))
        if (not p_valid) and p_reason == "penalized_reduced_null":
            p_emp = np.nan
            p_finite = False
        if not p_valid:
            p_source = None

        result_payload = {
            "Phenotype": s_name,
            "T_OBS": float(T_obs) if np.isfinite(T_obs) else np.nan,
            "P_EMP": float(p_emp) if (p_valid and np.isfinite(p_emp)) else np.nan,
            "P_Value": float(p_emp) if (p_valid and np.isfinite(p_emp)) else np.nan,
            "P_Valid": bool(p_valid),
            "P_Source": p_source,
            "Boot": boot_engine,
            "Boot_Engine": boot_engine,
            "Boot_Fit_Kind": boot_fit_kind,
            "Boot_Draws": int(boot_draws),
            "Boot_Exceed": int(boot_exceed),
            "Reduced_Fit_Type": reduced_fit_type,
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": note or "",
        }
        result_payload["Test_Stat"] = "score"
        if boot_engine == "bernoulli":
            result_payload["B"] = int(boot_draws)
        if p_reason:
            result_payload["Reason"] = p_reason
        io.atomic_write_json(result_path, result_payload)
        _write_meta(meta_path, "boot_overall", s_name, cat, target, worker_core_df_cols, core_fp, case_fp, extra=dict(meta_extra_common))

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
        ci_label = None
        ci_valid = False
        if p_valid and target in X_full_zv.columns and np.isfinite(beta_full):
            try:
                x_target_vec_ci = X_full_zv[target].to_numpy(dtype=np.float64, copy=False)
                ci_info = _score_boot_ci_beta(
                    X_red_zv,
                    yb,
                    x_target_vec_ci,
                    beta_full,
                    kind="mle",
                    seed_key=("boot_overall", s_name_safe, target, "ci"),
                    p_at_zero=p_emp,
                )
            except Exception:
                ci_info = {"valid": False}
            if ci_info.get("valid", False):
                lo_beta = float(ci_info.get("lo", np.nan))
                hi_beta = float(ci_info.get("hi", np.nan))
                ci_lo_or = 0.0 if lo_beta == -np.inf else (
                    float(np.exp(lo_beta)) if np.isfinite(lo_beta) else np.nan
                )
                ci_hi_or = np.inf if hi_beta == np.inf else (
                    float(np.exp(hi_beta)) if np.isfinite(hi_beta) else np.nan
                )
                or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                ci_method = ci_info.get("method", "score_boot_multiplier")
                ci_label = "score bootstrap (inverted)"
                ci_valid = True
        if (not ci_valid) and p_valid and fit_full is not None and target in X_full_zv.columns:
            wald = {"valid": False}
            fit_full_wald_ok2, _, _ = _ok_mle_fit(fit_full, X_full_zv, yb, target_ix=target_ix_full)
            if (
                not bool(getattr(fit_full, "_used_ridge", False))
                and not bool(getattr(fit_full, "_used_firth", False))
                and fit_full_wald_ok2
            ):
                wald = _wald_ci_or_from_fit(fit_full, target_ix_full, alpha=0.05, penalized=False)
            if wald.get("valid", False):
                ci_valid = True
                ci_method = wald["method"]
                ci_sided = "two"
                ci_lo_or = float(wald["lo_or"])
                ci_hi_or = float(wald["hi_or"])
                or_ci95 = _fmt_ci(ci_lo_or, ci_hi_or)
                ci_label = None

        # Note: Wald CI from Firth fits is not supported (_wald_ci_or_from_fit rejects penalized fits)
        # Profile CIs are used for Firth instead

        if ci_valid:
            method_allowed = ci_method in ALLOWED_CI_METHODS
            if not method_allowed:
                ci_valid = False
                ci_method = None
                ci_label = None
                ci_sided = None
                ci_lo_or = np.nan
                ci_hi_or = np.nan
                or_ci95 = None

        model_notes_parts = []
        if reason_full:
            model_notes_parts.append(reason_full)
        if note:
            model_notes_parts.append(note)
        if (not p_valid) and p_reason:
            model_notes_parts.append(p_reason)
        model_notes = ";".join([part for part in model_notes_parts if part])

        io.atomic_write_json(res_path, {
            "Phenotype": s_name,
            "N_Total": n_total_pre,
            "N_Cases": n_cases_pre,
            "N_Controls": n_ctrls_pre,
            "Beta": beta_full,
            "OR": or_val,
            "P_Value": float(p_emp) if (p_valid and np.isfinite(p_emp)) else np.nan,
            "P_Source": p_source,
            "P_Valid": bool(p_valid),
            "P_Reason": p_reason if (not p_valid and p_reason) else None,
            "Reduced_Fit_Type": reduced_fit_type,
            "Boot_Engine": boot_engine,
            "Boot_Draws": int(boot_draws),
            "Boot_Exceed": int(boot_exceed),
            "Boot_Fit_Kind": boot_fit_kind,
            "OR_CI95": or_ci95,
            "CI_Method": ci_method,
            "CI_Label": ci_label,
            "CI_Valid": bool(ci_valid),
            "CI_LO_OR": ci_lo_or,
            "CI_HI_OR": ci_hi_or,
            "Used_Ridge": used_ridge_full,
            "Final_Is_MLE": bool(final_is_mle),
            "Used_Firth": used_firth_full,
            "Inference_Type": "score_boot" if p_valid else "none",
            "N_Total_Used": n_total_used,
            "N_Cases_Used": n_cases_used,
            "N_Controls_Used": n_ctrls_used,
            "Model_Notes": model_notes,
        })

    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()

def lrt_followup_worker(task):
    """Stage-2 LRT follow-up worker with phenotype-specific logging."""
    s_name = task["name"]
    previous_pheno = CTX.get("current_phenotype")
    with logging_utils.phenotype_logging(s_name):
        try:
            CTX["current_phenotype"] = s_name
            return _lrt_followup_worker_impl(task)
        finally:
            if previous_pheno is None:
                CTX.pop("current_phenotype", None)
            else:
                CTX["current_phenotype"] = previous_pheno


def _lrt_followup_worker_impl(task):
    """Worker for Stage-2 ancestry×dosage LRT and per-ancestry splits. Uses array-based pipeline."""
    s_name, category, target = task["name"], task["category"], task["target"]
    s_name_safe = safe_basename(s_name)
    result_path = os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name_safe}.json")
    meta_path = os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name_safe}.meta.json")
    
    print(f"\n{'='*80}", flush=True)
    print(f"[Stage2] Starting: {s_name_safe}", flush=True)
    print(f"[Stage2]   Target: {target}", flush=True)
    print(f"[Stage2]   Category: {category}", flush=True)
    print(f"{'='*80}", flush=True)
    
    try:
        pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")
        if not os.path.exists(pheno_path):
            print(f"[Stage2] SKIP: Missing case cache at {pheno_path}", flush=True)
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

        core_fp = _core_index_fp()

        repair_meta = os.path.exists(result_path) and (not os.path.exists(meta_path)) and CTX.get("REPAIR_META_IF_MISSING", False)

        allowed_mask = allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool))
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        required_cols = ['const', target]
        missing_required = [c for c in required_cols if c not in col_ix]
        if missing_required:
            raise KeyError(f"missing required design columns: {missing_required}")

        def _existing(names):
            return [name for name in names if name in col_ix]

        base_cols = list(required_cols)
        base_cols += _existing(['sex'])
        base_cols += _existing(pc_cols)
        base_cols += _existing(['AGE_c', 'AGE_c_sq'])
        base_ix = [col_ix[c] for c in base_cols]
        X_base_df = pd.DataFrame(
            X_all[valid_mask][:, base_ix],
            index=worker_core_df_index[valid_mask],
            columns=base_cols,
        ).astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base_df.index, dtype=np.int8)

        Xb, yb, note, skip = _apply_sex_restriction(X_base_df, y_series, pheno_name=s_name)
        anc_vec = worker_anc_series.reindex(Xb.index)
        if anc_vec.isna().all():
            skip = skip or "no_ancestry_labels"
        else:
            valid_anc_mask = anc_vec.notna()
            if not valid_anc_mask.all():
                Xb = Xb.loc[valid_anc_mask]
                yb = yb.loc[valid_anc_mask]
                anc_vec = anc_vec.loc[valid_anc_mask]
                note = f"{note};dropped_missing_ancestry" if note else "dropped_missing_ancestry"

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
            'Boot_Engine': None,
            'Boot_Draws': 0,
            'Boot_Exceed': 0,
            'Boot_Fit_Kind': None,
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
            print(f"[Stage2] Meta repaired for {s_name_safe}", flush=True)
        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df_cols, core_fp, case_fp, category, target, allowed_fp,
            used_index_fp=used_index_fp, sex_cfg=sex_cfg, thresholds=thresholds
        ):
            print(f"[Stage2] Cache valid, skipping {s_name_safe}", flush=True)
            return
        if skip:
            print(f"[Stage2] SKIP: {skip} (N_total={len(yb)}, N_cases={int(yb.sum())}, N_ctrls={len(yb)-int(yb.sum())})", flush=True)
            out['LRT_Reason'] = skip; io.atomic_write_json(result_path, out)
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = skip
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, core_fp, case_fp, extra=meta_extra)
            return

        levels = pd.Index(anc_vec.dropna().unique(), dtype=str).tolist()
        levels_sorted = (['eur'] if 'eur' in levels else []) + [x for x in sorted(levels) if x != 'eur']
        out['LRT_Ancestry_Levels'] = ",".join(levels_sorted)
        
        print(f"[Stage2] Sample composition: N_total={len(yb)}, N_cases={int(yb.sum())}, N_ctrls={len(yb)-int(yb.sum())}", flush=True)
        print(f"[Stage2] Ancestry levels detected: {', '.join(levels_sorted)} (n={len(levels_sorted)})", flush=True)
        
        # Print per-ancestry counts
        for anc in levels_sorted:
            anc_mask = (anc_vec == anc).to_numpy()
            n_anc = anc_mask.sum()
            n_cases_anc = int(yb[anc_mask].sum())
            n_ctrls_anc = n_anc - n_cases_anc
            print(f"[Stage2]   - {anc.upper()}: N={n_anc} (cases={n_cases_anc}, ctrls={n_ctrls_anc})", flush=True)

        if len(levels_sorted) < 2:
            print(f"[Stage2] SKIP: Only one ancestry level, cannot test heterogeneity", flush=True)
            out['LRT_Reason'] = "only_one_ancestry_level"; io.atomic_write_json(result_path, out)
            meta_extra = dict(meta_extra_common)
            meta_extra["skip_reason"] = "only_one_ancestry_level"
            _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, core_fp, case_fp, extra=meta_extra)
            return

        if 'eur' in levels:
            anc_cat = pd.Categorical(anc_vec, categories=['eur'] + sorted([x for x in levels if x != 'eur']))
        else:
            anc_cat = pd.Categorical(anc_vec)

        # Create Series with index before get_dummies to preserve person IDs
        A_df = pd.get_dummies(pd.Series(anc_cat, index=anc_vec.index), prefix='ANC', drop_first=True)
        X_red_df = Xb.join(A_df)

        # Use vectorized broadcasting to create interaction terms
        target_col_np = X_red_df[target].to_numpy(copy=False)[:, None]
        A_np = A_df.to_numpy(copy=False)
        interaction_mat = target_col_np * A_np
        interaction_cols = [f"{target}:{c}" for c in A_df.columns]
        X_full_df = pd.concat([X_red_df, pd.DataFrame(interaction_mat, index=X_red_df.index, columns=interaction_cols)], axis=1)

        # Prune the full model (with interactions) first.
        print(f"[Stage2] Building models...", flush=True)
        print(f"[Stage2]   Full model: {len(X_full_df.columns)} covariates (before pruning)", flush=True)
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=[target] + interaction_cols)
        X_full_zv, dropped_rank_cols = _drop_rank_deficient(
            X_full_zv, keep_cols=("const",), always_keep=[target] + interaction_cols
        )
        if dropped_rank_cols:
            drop_note = f"dropped_rank_def={','.join(dropped_rank_cols)}"
            note = f"{note};{drop_note}" if note else drop_note
        print(f"[Stage2]   Full model: {len(X_full_zv.columns)} covariates (after pruning)", flush=True)

        # Construct the reduced model by dropping interaction terms from the pruned full model.
        # This ensures the reduced model is properly nested within the full model.
        kept_interaction_cols = [c for c in interaction_cols if c in X_full_zv.columns]
        red_cols = [c for c in X_full_zv.columns if c not in kept_interaction_cols]
        X_red_zv = X_full_zv[red_cols]
        print(f"[Stage2]   Reduced model: {len(X_red_zv.columns)} covariates", flush=True)
        print(f"[Stage2]   Interaction terms: {len(kept_interaction_cols)} ({', '.join(kept_interaction_cols[:3])}{'...' if len(kept_interaction_cols) > 3 else ''})", flush=True)

        const_ix_red = X_red_zv.columns.get_loc('const') if 'const' in X_red_zv.columns else None
        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None

        print(f"[Stage2] Fitting reduced model...", flush=True)
        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red)
        print(f"[Stage2]   Reduced: {reason_red if fit_red else 'FAILED'}", flush=True)
        
        print(f"[Stage2] Fitting full model (with interactions)...", flush=True)
        target_ix_full = X_full_zv.columns.get_loc(target) if target in X_full_zv.columns else None
        fit_full, reason_full = _fit_logit_ladder(
            X_full_zv,
            yb,
            const_ix=const_ix_full,
            target_ix=target_ix_full,
        )
        print(f"[Stage2]   Full: {reason_full if fit_full else 'FAILED'}", flush=True)

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
        
        print(f"[Stage2] Model ranks: full={r_full}, reduced={r_red}, df_LRT={df_lrt}", flush=True)
        
        inference_family = None
        fit_full_use = None
        fit_red_use = None
        boot_engine = None
        boot_draws = 0
        boot_exceed = 0
        boot_fit_kind = None
        p_val = np.nan
        p_source = None
        inference_type = "none"
        if df_lrt > 0:
            full_is_mle = bool(getattr(fit_full, "_final_is_mle", False)) and not bool(getattr(fit_full, "_used_firth", False))
            red_is_mle = bool(getattr(fit_red, "_final_is_mle", False)) and not bool(getattr(fit_red, "_used_firth", False))

            full_ok2, _, _ = _ok_mle_fit(fit_full, X_full_zv, yb) if fit_full is not None else (False, None, {})
            red_ok2, _, _ = _ok_mle_fit(fit_red, X_red_zv, yb) if fit_red is not None else (False, None, {})

            print(f"[Stage2] Inference eligibility:", flush=True)
            print(f"[Stage2]   Full: is_MLE={full_is_mle}, ok={full_ok2}", flush=True)
            print(f"[Stage2]   Reduced: is_MLE={red_is_mle}, ok={red_ok2}", flush=True)

            if (
                fit_full is not None
                and fit_red is not None
                and full_is_mle
                and red_is_mle
                and full_ok2
                and red_ok2
            ):
                inference_family = "mle"
                fit_full_use = fit_full
                fit_red_use = fit_red
                print(f"[Stage2] Using MLE-based LRT", flush=True)

        if inference_family == "mle":
            ll_full = float(getattr(fit_full_use, "llf", np.nan))
            ll_red = float(getattr(fit_red_use, "llf", np.nan))
            if np.isfinite(ll_full) and np.isfinite(ll_red):
                stat = max(0.0, 2.0 * (ll_full - ll_red))
                p_val = float(sp_stats.chi2.sf(stat, df_lrt))
                p_source = "lrt_mle"
                inference_type = "mle"
                out['LRT_df'] = df_lrt
                print(f"[Stage2] LRT statistic: chi2={stat:.4f}, df={df_lrt}, p={p_val:.4e}", flush=True)
            else:
                out['LRT_Reason'] = "fit_failed"
                print(f"[Stage2] LRT failed: non-finite log-likelihoods (ll_full={ll_full}, ll_red={ll_red})", flush=True)
        elif df_lrt == 0:
            out['LRT_Reason'] = "zero_df_lrt"
            print(f"[Stage2] Zero degrees of freedom for LRT", flush=True)
        elif df_lrt == 1:
            print(f"[Stage2] Falling back to score test (df=1)...", flush=True)
            x_target_vec = None
            if kept_interaction_cols:
                int_col = kept_interaction_cols[0]
                if int_col in X_full_zv.columns:
                    x_target_vec = X_full_zv[int_col].to_numpy(dtype=np.float64, copy=False)
            if x_target_vec is None:
                out['LRT_Reason'] = "score_target_missing"
                print(f"[Stage2] Score test failed: target vector missing", flush=True)
            else:
                p_sc, _ = _score_test_from_reduced(
                    X_red_zv,
                    yb,
                    x_target_vec,
                    const_ix=const_ix_red,
                )
                if np.isfinite(p_sc):
                    p_val = p_sc
                    p_source = "score_chi2"
                    inference_type = "score"
                    out['LRT_Reason'] = ""
                    print(f"[Stage2] Score test: p={p_val:.4e}", flush=True)
                else:
                    if ENABLE_SCORE_BOOT_MLE:
                        print(f"[Stage2] Score test returned non-finite p-value, trying bootstrap...", flush=True)
                        boot_res = _score_bootstrap_from_reduced(
                            X_red_zv,
                            yb,
                            x_target_vec,
                            seed_key=("lrt_followup", s_name_safe, "stage2", target, "pval"),
                        )
                        p_emp = float(boot_res.get("p", np.nan))
                        boot_draws = int(boot_res.get("draws", 0))
                        boot_exceed = int(boot_res.get("exceed", 0))
                        boot_fit_kind = boot_res.get("fit_kind")
                        if np.isfinite(p_emp):
                            p_val = p_emp
                            p_source = "score_boot_firth" if boot_fit_kind == "firth" else "score_boot_mle"
                            inference_type = "score_boot"
                            out['LRT_Reason'] = ""
                            boot_engine = "score_bootstrap"
                            print(f"[Stage2] Bootstrap score test: p={p_val:.4e} (draws={boot_draws}, exceed={boot_exceed}, fit={boot_fit_kind})", flush=True)
                        else:
                            out['LRT_Reason'] = "score_boot_failed"
                            print(f"[Stage2] Bootstrap score test failed", flush=True)
                    else:
                        out['LRT_Reason'] = "score_boot_disabled"
                        print(f"[Stage2] Score test failed, bootstrap disabled", flush=True)
        else:
            # Multi-df case (df_lrt > 1): Use robust Rao score test computed at reduced model
            # This avoids fitting the unstable full interaction model
            print(f"[Stage2] Using Rao score test (df={df_lrt} > 1)...", flush=True)
            if kept_interaction_cols and len(kept_interaction_cols) > 0:
                try:
                    # Extract interaction block from full design matrix
                    X_int_cols = [c for c in kept_interaction_cols if c in X_full_zv.columns]
                    if X_int_cols:
                        X_int = X_full_zv[X_int_cols].to_numpy(dtype=np.float64, copy=False)
                        X_red_arr = X_red_zv.to_numpy(dtype=np.float64, copy=False)

                        stat, df_eff, pval, det = _rao_score_block(y=yb, X0=X_red_arr, X1=X_int, fit_red=fit_red)

                        # Check if Rao score test succeeded
                        if "error" in det:
                            # Handle specific error cases
                            if det["error"] == "reduced_not_mle":
                                # Reduced model required penalization - chi2 calibration invalid
                                # TODO: Could fall back to score-bootstrap here in future
                                out['LRT_Reason'] = "rao_score_reduced_not_mle"
                                out['Stage2_Model_Notes'] = "rao_score_multi_failed;reduced_penalized"
                                print(f"[Stage2] Rao score failed: reduced model not MLE (penalized)", flush=True)
                            else:
                                out['LRT_Reason'] = f"rao_score_{det['error']}"
                                out['Stage2_Model_Notes'] = f"rao_score_multi_failed;{det['error']}"
                                print(f"[Stage2] Rao score failed: {det['error']}", flush=True)
                        elif df_eff > 0 and np.isfinite(pval):
                            p_val = pval
                            p_source = "rao_score"
                            inference_type = "rao_score"  # Distinct from full MLE LRT
                            out['LRT_df'] = df_eff
                            out['LRT_Reason'] = ""
                            # Include diagnostics for debugging
                            rank = det.get('rank', df_eff)
                            cond = det.get('I_eff_cond', np.nan)
                            out['Stage2_Model_Notes'] = f"rao_score_multi;rank={rank};cond={cond:.2e}"
                            print(f"[Stage2] Rao score test: chi2={stat:.4f}, df_eff={df_eff}, p={pval:.4e}, rank={rank}, cond={cond:.2e}", flush=True)
                        else:
                            out['LRT_Reason'] = "score_info_singular" if df_eff == 0 else "score_p_nan"
                            out['Stage2_Model_Notes'] = "rao_score_multi_failed"
                            print(f"[Stage2] Rao score failed: df_eff={df_eff}, pval={pval}", flush=True)
                    else:
                        out['LRT_Reason'] = "score_unavailable_multi_df"
                except (np.linalg.LinAlgError, ValueError, TypeError) as e:
                    # Expected numerical issues - log and continue
                    err_msg = str(e)[:100]  # Truncate to avoid massive error strings
                    out['LRT_Reason'] = f"score_linalg_error"
                    out['Stage2_Model_Notes'] = f"rao_score_multi_exception:{type(e).__name__}:{err_msg}"
                    print(
                        f"[WARN] Rao score numerical error for {s_name_safe}: {type(e).__name__}: {err_msg}",
                        flush=True,
                    )
                except Exception as e:
                    # Unexpected error - log with more detail but don't crash the worker
                    err_msg = str(e)[:100]
                    out['LRT_Reason'] = "score_unexpected_error"
                    out['Stage2_Model_Notes'] = f"rao_score_multi_unexpected:{type(e).__name__}:{err_msg}"
                    print(
                        f"[ERROR] Unexpected Rao score error for {s_name_safe}: {type(e).__name__}: {e}",
                        flush=True,
                    )
            else:
                out['LRT_Reason'] = "score_unavailable_multi_df"

        out['Boot_Engine'] = boot_engine
        out['Boot_Draws'] = int(boot_draws)
        out['Boot_Exceed'] = int(boot_exceed)
        out['Boot_Fit_Kind'] = boot_fit_kind

        p_finite = bool(np.isfinite(p_val))
        p_valid = bool(p_finite and (p_source in ALLOWED_P_SOURCES))
        if not p_valid:
            p_source = None
            out['LRT_df'] = np.nan
        out['P_LRT_AncestryxDosage'] = float(p_val) if p_valid else np.nan
        out['P_Stage2_Valid'] = bool(p_valid)
        out['P_Method'] = p_source
        out['P_Source'] = p_source
        out['Inference_Type'] = inference_type if p_valid else "none"
        if not out['P_Stage2_Valid'] and not out['LRT_Reason']:
            out['LRT_Reason'] = "fit_failed"
        
        print(f"[Stage2] Interaction test summary:", flush=True)
        print(f"[Stage2]   P_valid: {p_valid}, P_value: {p_val if p_finite else 'NaN'}, Method: {p_source or 'none'}", flush=True)
        print(f"[Stage2]   Inference: {inference_type}, Reason: {out['LRT_Reason'] or 'success'}", flush=True)

        lrt_df_val = out.get('LRT_df')
        if isinstance(lrt_df_val, (np.integer, int)):
            stage2_df = int(lrt_df_val)
        elif isinstance(lrt_df_val, (float, np.floating)) and np.isfinite(lrt_df_val):
            stage2_df = int(lrt_df_val)
        else:
            stage2_df = None
        meta_extra_common.update({
            "stage2_df": stage2_df,
            "stage2_inference_type": out.get('Inference_Type'),
            "stage2_p_method": out.get('P_Method'),
            "stage2_reason": out.get('LRT_Reason') or None,
            "stage2_boot_engine": out.get('Boot_Engine'),
        })

        print(f"\n[Stage2] Per-ancestry stratified analysis:", flush=True)
        for anc in levels_sorted:
            anc_mask = (anc_vec == anc).to_numpy()
            X_anc, y_anc = Xb[anc_mask], yb[anc_mask]

            anc_upper = anc.upper()
            n_total_anc = len(y_anc)
            print(f"[Stage2]   Analyzing {anc_upper}: N={n_total_anc}...", flush=True)
            print(
                f"[DEBUG-START] name={s_name} anc={anc_upper} "
                f"N_total={n_total_anc} N_cases={int(y_anc.sum())} N_controls={n_total_anc - int(y_anc.sum())}",
                flush=True
            )
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
            X_anc_zv, dropped_rank_cols = _drop_rank_deficient(
                X_anc_zv, keep_cols=("const",), always_keep=(target,)
            )

            if dropped_rank_cols:
                drop_note = f"dropped_rank_def={','.join(dropped_rank_cols)}"
                note = f"{note};{drop_note}" if note else drop_note

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

            print(
                f"[DEBUG-FIT] name={s_name} anc={anc_upper} "
                f"fit_full={'present' if fit_full is not None else 'None'} "
                f"fit_red={'present' if fit_red is not None else 'None'}",
                flush=True
            )
            full_ok_anc, _, _ = _ok_mle_fit(fit_full, X_anc_zv, y_anc, target_ix=target_ix_anc) if fit_full is not None else (False, None, {})
            red_ok_anc, _, _ = _ok_mle_fit(fit_red, X_anc_red, y_anc) if fit_red is not None else (False, None, {})

            if (
                fit_full is not None
                and fit_red is not None
                and bool(getattr(fit_full, "_final_is_mle", False))
                and not bool(getattr(fit_full, "_used_firth", False))
                and bool(getattr(fit_red, "_final_is_mle", False))
                and not bool(getattr(fit_red, "_used_firth", False))
                and full_ok_anc
                and red_ok_anc
            ):
                inference_family = "mle"
                fit_full_use = fit_full
                fit_red_use = fit_red
            else:
                print(
                    f"[DEBUG-FIT] name={s_name} anc={anc_upper} "
                    f"action=mle_not_suitable attempting_firth",
                    flush=True
                )
                fit_full_firth = fit_full if bool(getattr(fit_full, "_used_firth", False)) else _firth_refit(X_anc_zv, y_anc)
                fit_red_firth = fit_red if bool(getattr(fit_red, "_used_firth", False)) else _firth_refit(X_anc_red, y_anc)
                print(
                    f"[DEBUG-FIT] name={s_name} anc={anc_upper} "
                    f"fit_full_firth={'present' if fit_full_firth is not None else 'None'} "
                    f"fit_red_firth={'present' if fit_red_firth is not None else 'None'}",
                    flush=True
                )
                if (fit_full_firth is not None) and (fit_red_firth is not None):
                    inference_family = "firth"
                    fit_full_use = fit_full_firth
                    fit_red_use = fit_red_firth
                    print(
                        f"[DEBUG-FIT] name={s_name} anc={anc_upper} "
                        f"inference_family=firth",
                        flush=True
                    )

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
                print(
                    f"[DEBUG-LL] name={s_name} anc={anc_upper} "
                    f"inference_family={inference_family} ll_full={ll_full} ll_red={ll_red} "
                    f"ll_full_finite={np.isfinite(ll_full)} ll_red_finite={np.isfinite(ll_red)}",
                    flush=True
                )
                if np.isfinite(ll_full) and np.isfinite(ll_red):
                    if inference_family == "mle":
                        stat = max(0.0, 2.0 * (ll_full - ll_red))
                        p_val = float(sp_stats.chi2.sf(stat, 1))
                        p_source = "lrt_mle"
                        inference_type = "mle"
                    else:
                        # For Firth, set inference_type but don't compute LRT p-value
                        # (will fall through to score tests below)
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

            # Attempt score tests if no p-value yet (including for Firth)
            print(
                f"[DEBUG-SCORE] name={s_name} anc={anc_upper} "
                f"p_val={p_val} finite={np.isfinite(p_val)} "
                f"inference_family={inference_family} inference_type={inference_type}",
                flush=True
            )
            if not np.isfinite(p_val):
                print(
                    f"[DEBUG-SCORE] name={s_name} anc={anc_upper} "
                    f"action=entering_score_test_section",
                    flush=True
                )
                x_target_vec = X_anc_zv.iloc[:, int(target_ix_anc)].to_numpy(dtype=np.float64, copy=False)
                p_sc, _ = _score_test_from_reduced(
                    X_anc_red,
                    y_anc,
                    x_target_vec,
                    const_ix=const_ix_red,
                )
                print(
                    f"[DEBUG-SCORE] name={s_name} anc={anc_upper} "
                    f"p_sc={p_sc} finite={np.isfinite(p_sc)}",
                    flush=True
                )
                if np.isfinite(p_sc):
                    p_val = p_sc
                    p_source = "score_chi2"
                    print(
                        f"[DEBUG-SCORE] name={s_name} anc={anc_upper} "
                        f"action=score_chi2_succeeded p_val={p_val}",
                        flush=True
                    )
                    # Create composite label when score test provides p-value after Firth/MLE fit
                    if inference_type == "none":
                        inference_type = "score"
                    elif inference_type in {"firth", "mle"}:
                        # Composite: coefficients from firth/mle, p-value from score test
                        inference_type = f"{inference_type}+score"
                else:
                    if ENABLE_SCORE_BOOT_PER_ANCESTRY:
                        print(
                            f"[BOOT-PER-ANCESTRY] name={s_name} anc={anc_upper} "
                            f"reason=score_chi2_failed action=attempting_score_bootstrap",
                            flush=True
                        )
                        boot_res = _score_bootstrap_from_reduced(
                            X_anc_red,
                            y_anc,
                            x_target_vec,
                            seed_key=("lrt_followup", s_name_safe, anc, target, "pval"),
                        )
                        p_emp = float(boot_res.get("p", np.nan))
                        if np.isfinite(p_emp):
                            p_val = p_emp
                            p_source = "score_boot_firth" if boot_res.get("fit_kind") == "firth" else "score_boot_mle"
                            print(
                                f"[BOOT-PER-ANCESTRY] name={s_name} anc={anc_upper} "
                                f"p_emp={p_emp:.4e} source={p_source} action=bootstrap_succeeded",
                                flush=True
                            )
                            # Create composite label when bootstrap provides p-value after Firth/MLE fit
                            if inference_type == "none":
                                inference_type = "score_boot"
                            elif inference_type in {"firth", "mle"}:
                                # Composite: coefficients from firth/mle, p-value from bootstrap
                                inference_type = f"{inference_type}+score_boot"
                        else:
                            print(
                                f"[BOOT-PER-ANCESTRY] name={s_name} anc={anc_upper} "
                                f"action=bootstrap_failed",
                                flush=True
                            )
                    # If bootstrap disabled, p_val remains non-finite and will be handled downstream

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

            if ci_valid:
                method_allowed = ci_method in ALLOWED_CI_METHODS
                if not method_allowed:
                    ci_valid = False
                    ci_method = None
                    ci_label = ""
                    ci_sided = None
                    ci_lo_or = np.nan
                    ci_hi_or = np.nan
                    ci_str = None

            if inference_type == "mle":
                penalized_inference = any(
                    _final_stage_penalized(candidate, reason)
                    for candidate, reason in (
                        (
                            fit_full_use if fit_full_use is not None else fit_full,
                            reason_full,
                        ),
                        (
                            fit_red_use if fit_red_use is not None else fit_red,
                            reason_red,
                        ),
                    )
                )
            elif inference_type == "firth":
                penalized_inference = True
            elif inference_type in {"score", "score_boot", "firth+score", "firth+score_boot", "mle+score", "mle+score_boot"}:
                penalized_inference = False
            else:
                penalized_inference = _final_stage_penalized(
                    fit_full_use if fit_full_use is not None else fit_full,
                    reason_full,
                )
            candidate_p_valid = bool(np.isfinite(p_val) and (p_source in ALLOWED_P_SOURCES))
            print(
                f"[DEBUG-PENALIZED] name={s_name} anc={anc_upper} "
                f"inference_type={inference_type} penalized_inference={penalized_inference} "
                f"candidate_p_valid={candidate_p_valid} p_val={p_val} p_source={p_source}",
                flush=True
            )
            if penalized_inference and not candidate_p_valid:
                p_val = np.nan
                p_source = None
                ci_method = None
                ci_sided = None
                ci_label = ""
                ci_valid = False
                ci_lo_or = np.nan
                ci_hi_or = np.nan
                ci_str = None
                # Note: Wald CI from Firth fits is not supported (_wald_ci_or_from_fit rejects penalized fits)
                # Profile CIs are used for Firth instead
                pass
                if ci_valid and (not np.isfinite(p_val)):
                    out[f"{anc_upper}_REASON"] = ""
                if not out[f"{anc_upper}_REASON"]:
                    out[f"{anc_upper}_REASON"] = "penalized_fit"

            p_finite = bool(np.isfinite(p_val))
            p_valid = bool(p_finite and (p_source in ALLOWED_P_SOURCES))
            if not p_valid:
                p_source = None
            if (not p_valid) and (not ci_valid):
                if not out[f"{anc_upper}_REASON"]:
                    out[f"{anc_upper}_REASON"] = "subset_fit_failed"
                continue

            out[f"{anc_upper}_OR"] = or_val
            out[f"{anc_upper}_P"] = float(p_val) if p_valid else np.nan
            out[f"{anc_upper}_P_Valid"] = bool(p_valid)
            out[f"{anc_upper}_P_Source"] = p_source
            # Report inference_type even if p-value is invalid, as long as CI is valid
            out[f"{anc_upper}_Inference_Type"] = inference_type if (p_valid or ci_valid) else "none"
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
            
            # Print per-ancestry result
            p_str = f"{p_val:.4e}" if p_valid else "NaN"
            or_str = f"{or_val:.3f}" if np.isfinite(or_val) else "NaN"
            ci_str_display = ci_str if ci_valid else "N/A"
            print(f"[Stage2]     {anc_upper}: OR={or_str}, P={p_str}, CI95={ci_str_display}, method={p_source or 'none'}", flush=True)

        io.atomic_write_json(result_path, out)
        _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, core_fp, case_fp, extra=dict(meta_extra_common))
        
        print(f"\n[Stage2] COMPLETE: {s_name_safe}", flush=True)
        p_display = f"{out['P_LRT_AncestryxDosage']:.4e}" if out['P_Stage2_Valid'] else 'NaN'
        print(f"[Stage2]   Interaction P: {p_display}", flush=True)
        print(f"[Stage2]   Results saved to: {os.path.basename(result_path)}", flush=True)
        print(f"{'='*80}\n", flush=True)
        
    except Exception as e:
        print(f"\n[Stage2] EXCEPTION in {s_name_safe}: {type(e).__name__}: {e}", flush=True)
        print(f"{'='*80}\n", flush=True)
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()
