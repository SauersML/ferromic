import os
import gc
import hashlib
import warnings
from datetime import datetime, timezone
import traceback
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from scipy.special import expit
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


import iox as io

CTX = {}  # Worker context with constants from run.py

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

def _fit_logit_ladder(X, y, ridge_ok=True, const_ix=None):
    """
    Ridge-first logistic fit with strict MLE gating based on numerical diagnostics.
    If numpy arrays are provided, const_ix identifies the intercept column for zero-penalty.
    Returns a tuple (fit_result, reason_tag).
    """
    is_pandas = hasattr(X, "columns")
    if not ridge_ok:
        return None, "ridge_disabled"

    try:
        p = X.shape[1] - (1 if (is_pandas and "const" in X.columns) or (not is_pandas and const_ix is not None) else 0)
        n = max(1, X.shape[0])
        pi = float(np.mean(y)) if len(y) > 0 else 0.5
        n_eff = max(1.0, 4.0 * float(len(y)) * pi * (1.0 - pi))
        alpha_scalar = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / n_eff), 1e-6)
        alphas = np.full(X.shape[1], alpha_scalar, dtype=float)
        if is_pandas and "const" in X.columns:
            alphas[X.columns.get_loc("const")] = 0.0
        elif not is_pandas and const_ix is not None and const_ix < X.shape[1]:
            alphas[const_ix] = 0.0

        ridge_fit = sm.Logit(y, X).fit_regularized(alpha=alphas, L1_wt=0.0, maxiter=800)
        setattr(ridge_fit, "_used_ridge", True)
        setattr(ridge_fit, "_final_is_mle", False)

        try:
            max_abs_linpred, frac_lo, frac_hi = _fit_diagnostics(X, y, ridge_fit.params)
        except Exception:
            max_abs_linpred, frac_lo, frac_hi = float("inf"), 1.0, 1.0

        if (max_abs_linpred > 50.0) or (frac_lo > 0.20) or (frac_hi > 0.20):
            return ridge_fit, "ridge_only"

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=PerfectSeparationWarning)
            try:
                refit_newton = _logit_fit(
                    sm.Logit(y, X),
                    "newton",
                    maxiter=400,
                    tol=1e-8,
                    start_params=ridge_fit.params
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
                refit_bfgs = _logit_fit(
                    sm.Logit(y, X),
                    "bfgs",
                    maxiter=800,
                    gtol=1e-8,
                    start_params=ridge_fit.params
                )
                if _converged(refit_bfgs) and hasattr(refit_bfgs, "bse") and np.all(np.isfinite(refit_bfgs.bse)):
                    if np.max(refit_bfgs.bse) > 100.0:
                        raise PerfectSeparationWarning("Large SEs detected")
                    setattr(refit_bfgs, "_used_ridge_seed", True)
                    setattr(refit_bfgs, "_final_is_mle", True)
                    return refit_bfgs, "ridge_seeded_refit"
            except (Exception, PerfectSeparationWarning):
                pass

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
    X_arr = np.asarray(X, dtype=np.float64)
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
    """ALL WARNINGS ARE ENABLED. This function now ensures all warnings are always shown."""
    warnings.simplefilter("always")
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


def _drop_rank_deficiency_np(X, keep_ix=(), max_iter=100):
    """
    Greedily removes columns from a NumPy design matrix until it achieves full column rank.
    Columns listed in keep_ix are preserved whenever possible. Removal order is by ascending
    column standard deviation among removable columns, recomputed after each drop.
    Returns the pruned matrix and the kept column indices relative to the original X.
    """
    if X.shape[1] == 0:
        return X, np.array([], dtype=int)
    original_cols = np.arange(X.shape[1])
    keep_ix = set(int(i) for i in keep_ix)
    X_work = X
    cols_work = original_cols.copy()
    iter_ct = 0
    while np.linalg.matrix_rank(X_work) < X_work.shape[1] and iter_ct < max_iter:
        removable_positions = [k for k, j in enumerate(cols_work) if j not in keep_ix]
        if not removable_positions:
            break
        stds = np.nanstd(X_work, axis=0)
        stds_removable = stds[removable_positions]
        kmin_local = int(np.argmin(stds_removable))
        pos_to_drop = removable_positions[kmin_local]
        cols_work = np.delete(cols_work, pos_to_drop, axis=0)
        X_work = np.delete(X_work, pos_to_drop, axis=1)
        iter_ct += 1
    return X_work, cols_work


def _apply_sex_restriction(X: pd.DataFrame, y: pd.Series):
    """
    Enforce: if all cases are one sex, only use that sex's rows (and drop 'sex').
    If that sex has zero controls, signal skip.
    Returns: (X2, y2, note:str, skip_reason:str|None)
    """
    if 'sex' not in X.columns: return X, y, "", None
    tab = pd.crosstab(X['sex'], y).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
    case_sexes = [s for s in [0.0, 1.0] if s in tab.index and tab.loc[s, 1] > 0]
    if len(case_sexes) != 1: return X, y, "", None
    s = float(case_sexes[0])
    if tab.loc[s, 0] == 0: return X, y, "", "sex_no_controls_in_case_sex"
    keep = X['sex'].eq(s)
    return X.loc[keep].drop(columns=['sex']), y.loc[keep], f"sex_restricted_to_{int(s)}", None

def _apply_sex_restriction_np(X, y, sex_ix):
    """
    NumPy-based sex restriction. If all cases are one sex, filters to that sex.
    Returns: (X_restr, y_restr, note:str, skip_reason:str|None, sex_col_dropped:bool)
    """
    if sex_ix is None:
        return X, y, "", None, False

    sex_col = X[:, sex_ix]
    y_bool = y.astype(bool)

    n_xx_case = np.sum((sex_col == 0.0) & y_bool)
    n_xy_case = np.sum((sex_col == 1.0) & y_bool)

    case_sexes_present = []
    if n_xx_case > 0: case_sexes_present.append(0.0)
    if n_xy_case > 0: case_sexes_present.append(1.0)

    if len(case_sexes_present) != 1:
        return X, y, "", None, False

    s = case_sexes_present[0]

    n_s_ctrl = np.sum((sex_col == s) & (~y_bool))

    if n_s_ctrl == 0:
        return X, y, "", "sex_no_controls_in_case_sex", False

    keep_mask = (sex_col == s)
    cols_to_keep = np.arange(X.shape[1]) != sex_ix

    X_restr = X[keep_mask][:, cols_to_keep]
    y_restr = y[keep_mask]

    return X_restr, y_restr, f"sex_restricted_to_{int(s)}", None, True

# --- Worker globals ---
# Populated by init_worker and read-only thereafter.
worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, finite_mask_worker = None, None, 0, None, None
# Array-based versions for performance
X_all, col_ix, worker_core_df_cols, worker_core_df_index = None, None, None, None
# Per-category cached data for Stage 1
control_indices_by_cat, X_controls_by_cat, y_controls_by_cat = {}, {}, {}


def init_worker(df_to_share, masks, ctx):
    """
    Sends the large core_df, precomputed masks, and context to each worker process.
    Converts pandas DataFrame to numpy array and pre-caches control matrices for performance.
    """
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
    _suppress_worker_warnings()
    _validate_ctx(ctx)
    global worker_core_df, allowed_mask_by_cat, N_core, CTX, finite_mask_worker
    global X_all, col_ix, worker_core_df_cols, worker_core_df_index
    global control_indices_by_cat, X_controls_by_cat, y_controls_by_cat

    worker_core_df = df_to_share  # Keep for fingerprints, metadata, and index
    allowed_mask_by_cat, N_core, CTX = masks, len(df_to_share), ctx
    worker_core_df_cols = df_to_share.columns
    worker_core_df_index = df_to_share.index

    # --- Array conversion ---
    X_all = df_to_share.to_numpy(dtype=np.float64, copy=False)
    col_ix = {name: i for i, name in enumerate(worker_core_df_cols)}

    finite_mask_worker = np.isfinite(X_all).all(axis=1)

    # --- Pre-cache control matrices per category ---
    for category, allowed_mask in allowed_mask_by_cat.items():
        control_mask = allowed_mask & finite_mask_worker
        control_indices = np.flatnonzero(control_mask)
        control_indices_by_cat[category] = control_indices
        X_controls_by_cat[category] = X_all[control_indices]
        y_controls_by_cat[category] = np.zeros(len(control_indices), dtype=np.int8)

    bad = ~finite_mask_worker
    if bad.any():
        rows = worker_core_df_index[bad][:5].tolist()
        bad_cols = [c for c in worker_core_df_cols if not np.isfinite(worker_core_df[c]).all()]
        print(f"[Worker-{os.getpid()}] Non-finite sample rows={rows} cols={bad_cols[:10]}", flush=True)
    print(f"[Worker-{os.getpid()}] Initialized with {N_core} subjects, {len(masks)} masks. Array conversion complete.", flush=True)


def init_lrt_worker(df_to_share, masks, anc_series, ctx):
    """Initializer for LRT pools that also provides ancestry labels and context."""
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
    _suppress_worker_warnings()
    _validate_ctx(ctx)
    global worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, CTX, finite_mask_worker
    global X_all, col_ix, worker_core_df_cols, worker_core_df_index

    worker_core_df, allowed_mask_by_cat, N_core, CTX = df_to_share, masks, len(df_to_share), ctx
    worker_anc_series = anc_series.reindex(df_to_share.index).str.lower()
    worker_core_df_cols = df_to_share.columns
    worker_core_df_index = df_to_share.index

    # --- Array conversion ---
    X_all = df_to_share.to_numpy(dtype=np.float64, copy=False)
    col_ix = {name: i for i, name in enumerate(worker_core_df_cols)}

    finite_mask_worker = np.isfinite(X_all).all(axis=1)
    bad = ~finite_mask_worker
    if bad.any():
        rows = worker_core_df_index[bad][:5].tolist()
        bad_cols = [c for c in worker_core_df_cols if not np.isfinite(worker_core_df[c]).all()]
        print(f"[Worker-{os.getpid()}] Non-finite sample rows={rows} cols={bad_cols[:10]}", flush=True)
    print(f"[LRT-Worker-{os.getpid()}] Initialized with {N_core} subjects, {len(masks)} masks, {worker_anc_series.nunique()} ancestries.", flush=True)

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

def _should_skip(meta_path, core_df, case_idx_fp, category, target, allowed_fp):
    """Determines if a model run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta: return False
    # Check that the metadata matches the current context
    return (
        meta.get("model_columns") == list(core_df.columns) and
        meta.get("ridge_l2_base") == CTX["RIDGE_L2_BASE"] and
        meta.get("core_index_fp") == _index_fingerprint(core_df.index) and
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


def run_single_model_worker(pheno_data, target_inversion, results_cache_dir):
    """CONSUMER: Runs a single model. Executed in a separate process using NumPy arrays."""
    s_name, category, case_idx_global = pheno_data["name"], pheno_data["category"], pheno_data["case_idx"]
    s_name_safe = safe_basename(s_name)
    result_path = os.path.join(results_cache_dir, f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"

    try:
        case_ids_for_fp = worker_core_df_index[case_idx_global] if case_idx_global.size > 0 else pd.Index([])
        case_idx_fp = _index_fingerprint(case_ids_for_fp)

        # Fingerprint the controls for this category for skipping
        control_indices = control_indices_by_cat.get(category, np.array([], dtype=int))
        allowed_fp = _index_fingerprint(worker_core_df_index[control_indices])

        if os.path.exists(result_path) and _should_skip(meta_path, worker_core_df, case_idx_fp, category, target_inversion, allowed_fp):
            return

        # --- Array-based data assembly ---
        case_indices_in_X_all = case_idx_global[finite_mask_worker[case_idx_global]]

        X_controls = X_controls_by_cat.get(category, np.empty((0, X_all.shape[1])))
        y_controls = y_controls_by_cat.get(category, np.empty((0,)))

        X_cases = X_all[case_indices_in_X_all]
        y_cases = np.ones(len(X_cases), dtype=np.int8)

        X_clean = np.vstack([X_controls, X_cases])
        y_clean = np.concatenate([y_controls, y_cases])

        n_total, n_cases, n_ctrls = len(y_clean), int(y_clean.sum()), len(y_clean) - int(y_clean.sum())

        # --- Pre-fit sanity checks ---
        skip_reason, notes = "", []
        target_ix = col_ix.get(target_inversion)
        const_ix = col_ix.get('const')
        sex_ix = col_ix.get('sex')

        if n_cases < CTX["MIN_CASES_FILTER"] or n_ctrls < CTX["MIN_CONTROLS_FILTER"]:
            skip_reason = "insufficient_cases_or_controls"
        elif target_ix is None:
            skip_reason = "target_inversion_not_in_cols"
        elif X_clean.shape[0] == 0:
            skip_reason = "no_valid_rows"
        elif np.nanstd(X_clean[:, target_ix]) < 1e-12:
            skip_reason = "target_constant"

        if skip_reason:
            # ... (identical skip logic as before, just using different N vars)
            result_data = {"Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls, "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "Skip_Reason": skip_reason}
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_idx_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": skip_reason})
            return

        # --- Model fitting pipeline (on arrays) ---
        X_work, y_work = X_clean, y_clean
        current_col_indices = np.arange(X_all.shape[1])

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
            # Update indices of other key columns if they were after the sex column
            target_ix = np.where(current_col_indices == target_ix)[0][0] if target_ix is not None else None
            const_ix = np.where(current_col_indices == const_ix)[0][0] if const_ix is not None else None

        n_total_used, n_cases_used, n_ctrls_used = len(y_work), int(y_work.sum()), len(y_work) - int(y_work.sum())
        keep_ix_zv = {idx for idx in [target_ix, const_ix] if idx is not None}
        X_work_zv, kept_indices_after_zv = _drop_zero_variance_np(X_work, keep_ix=keep_ix_zv)

        current_col_indices = current_col_indices[kept_indices_after_zv]
        target_ix_mid = np.where(current_col_indices == col_ix.get(target_inversion))[0][0] if target_ix is not None else None
        const_ix_mid = np.where(current_col_indices == col_ix.get('const'))[0][0] if const_ix is not None else None

        X_work_fd, kept_indices_after_fd = _drop_rank_deficiency_np(X_work_zv, keep_ix={i for i in [target_ix_mid, const_ix_mid] if i is not None})
        current_col_indices = current_col_indices[kept_indices_after_fd]
        target_ix_final = np.where(current_col_indices == col_ix.get(target_inversion))[0][0] if target_ix is not None else None
        const_ix_final = np.where(current_col_indices == col_ix.get('const'))[0][0] if const_ix is not None else None

        fit, fit_reason = _fit_logit_ladder(X_work_fd, y_work, const_ix=const_ix_final)
        if fit_reason in ("ridge_seeded_refit", "ridge_only"): notes.append(fit_reason)
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

        result = {"Phenotype": s_name, "N_Cases": n_cases, "N_Controls": n_ctrls, "Beta": beta, "OR": np.exp(beta), "P_Value": pval, "OR_CI95": or_ci95_str, "Used_Ridge": not final_is_mle, "Final_Is_MLE": final_is_mle}
        result.update({"N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used, "Model_Notes": ";".join(notes) if notes else ""})
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
    try:
        pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")
        if not os.path.exists(pheno_path):
            io.atomic_write_json(result_path, {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_Overall_Reason": "missing_case_cache"})
            return

        case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
        case_idx = worker_core_df_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0]
        case_fp = _index_fingerprint(worker_core_df_index[case_idx] if case_idx.size > 0 else pd.Index([]))

        allowed_mask = allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool))
        allowed_fp = _mask_fingerprint(allowed_mask, worker_core_df_index)

        if os.path.exists(result_path) and _lrt_meta_should_skip(meta_path, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, cat, target, allowed_fp): return

        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0: case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        anc_cols = [c for c in worker_core_df_cols if c.startswith("ANC_")]
        base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq'] + anc_cols

        X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64, copy=False)
        y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index, dtype=np.int8)

        Xb, yb, note, skip = _apply_sex_restriction(X_base, y_series)
        n_total_used, n_cases_used, n_ctrls_used = len(yb), int(yb.sum()), len(yb) - int(yb.sum())

        if skip:
            io.atomic_write_json(result_path, {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_Overall_Reason": skip, "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used})
            _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": skip})
            return

        X_full_df, X_red_df = Xb, Xb.drop(columns=[target])

        # Ranks for LRT df are computed on matrices after dropping degenerate and rank-deficient columns.
        X_red_zv = _drop_zero_variance(X_red_df, keep_cols=('const',))
        X_red_zv = _drop_rank_deficient(X_red_zv, keep_cols=('const',))
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=(target,))
        X_full_zv = _drop_rank_deficient(X_full_zv, keep_cols=('const',), always_keep=(target,))

        const_ix_red = X_red_zv.columns.get_loc('const') if 'const' in X_red_zv.columns else None
        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red)
        fit_full, reason_full = _fit_logit_ladder(X_full_zv, yb, const_ix=const_ix_full)
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

        io.atomic_write_json(result_path, out)
        _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df.columns,
                    _index_fingerprint(worker_core_df.index), case_fp,
                    extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
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

        case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
        case_idx = worker_core_df_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0]
        case_fp = _index_fingerprint(worker_core_df_index[case_idx] if case_idx.size > 0 else pd.Index([]))

        allowed_mask = allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool))
        allowed_fp = _mask_fingerprint(allowed_mask, worker_core_df_index)

        if os.path.exists(result_path) and _lrt_meta_should_skip(meta_path, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, category, target, allowed_fp):
            return

        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0: case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq']
        X_base_df = worker_core_df.loc[valid_mask, base_cols].astype(np.float64, copy=False)
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

        X_red_zv = _drop_zero_variance(X_red_df, keep_cols=('const',), always_keep=(target,))
        X_red_zv = _drop_rank_deficient(X_red_zv, keep_cols=('const',), always_keep=(target,))
        X_full_zv = _drop_zero_variance(X_full_df, keep_cols=('const',), always_keep=[target] + interaction_cols)
        X_full_zv = _drop_rank_deficient(X_full_zv, keep_cols=('const',), always_keep=[target] + interaction_cols)

        const_ix_red = X_red_zv.columns.get_loc('const') if 'const' in X_red_zv.columns else None
        const_ix_full = X_full_zv.columns.get_loc('const') if 'const' in X_full_zv.columns else None

        fit_red, reason_red = _fit_logit_ladder(X_red_zv, yb, const_ix=const_ix_red)
        fit_full, reason_full = _fit_logit_ladder(X_full_zv, yb, const_ix=const_ix_full)
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
            out[f"{anc.upper()}_N"], out[f"{anc.upper()}_N_Cases"], out[f"{anc.upper()}_N_Controls"] = len(y_anc), int(y_anc.sum()), len(y_anc) - int(y_anc.sum())
            if out[f"{anc.upper()}_N_Cases"] < CTX["PER_ANC_MIN_CASES"] or out[f"{anc.upper()}_N_Controls"] < CTX["PER_ANC_MIN_CONTROLS"]:
                out[f"{anc.upper()}_REASON"] = "insufficient_stratum_counts"; continue

            X_anc_zv = _drop_zero_variance(X_anc, keep_cols=('const',), always_keep=(target,))
            const_ix_anc = X_anc_zv.columns.get_loc('const') if 'const' in X_anc_zv.columns else None
            fit, _ = _fit_logit_ladder(X_anc_zv, y_anc, const_ix=const_ix_anc)

            if fit and target in fit.params:
                beta, pval = float(fit.params[target]), float(fit.pvalues.get(target, np.nan))
                out[f"{anc.upper()}_OR"], out[f"{anc.upper()}_P"] = float(np.exp(beta)), pval
                if getattr(fit, "_final_is_mle", False) and not getattr(fit, '_used_ridge_seed', False):
                    se = float(fit.bse.get(target, np.nan))
                    if np.isfinite(se) and se > 0:
                        lo, hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
                        out[f"{anc.upper()}_CI95"] = f"{lo:.3f},{hi:.3f}"
            else: out[f"{anc.upper()}_REASON"] = "subset_fit_failed"

        io.atomic_write_json(result_path, out)
        _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df_cols, _index_fingerprint(worker_core_df_index), case_fp, extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
    except Exception as e:
        io.atomic_write_json(result_path, {"Phenotype": s_name, "Skip_Reason": f"exception:{type(e).__name__}"})
        traceback.print_exc()
    finally:
        gc.collect()
