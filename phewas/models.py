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
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

import iox as io

CTX = {}  # Worker context with constants from run.py

def _safe_basename(name: str) -> str:
    """Allow only [-._a-zA-Z0-9], map others to '_'."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(name))

def _write_meta(meta_path, kind, s_name, category, target, core_cols, core_idx_fp, case_fp, extra=None):
    """Helper to write a standardized metadata JSON file."""
    base = {
        "kind": kind, "s_name": s_name, "category": category, "model_columns": list(core_cols),
        "num_pcs": CTX["NUM_PCS"], "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
        "target": target, "core_index_fp": core_idx_fp, "case_idx_fp": case_fp,
        "created_at": datetime.now(timezone.utc).isoformat(),
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

def _fit_logit_ladder(X, y, ridge_ok=True):
    """
    Tries fitting a logistic regression model with a ladder of increasingly robust methods.
    Includes a ridge-seeded refit attempt.
    Returns (fit, reason_str)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=PerfectSeparationWarning)
        try:
            fit_try = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=False)
            if _converged(fit_try):
                setattr(fit_try, "_used_ridge", False); setattr(fit_try, "_final_is_mle", True)
                return fit_try, "newton"
        except (Exception, PerfectSeparationWarning): pass
        try:
            fit_try = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=800, gtol=1e-8, warn_convergence=False)
            if _converged(fit_try):
                setattr(fit_try, "_used_ridge", False); setattr(fit_try, "_final_is_mle", True)
                return fit_try, "bfgs"
        except (Exception, PerfectSeparationWarning): pass
    if ridge_ok:
        try:
            p = X.shape[1] - (1 if 'const' in X.columns else 0)
            n = max(1, X.shape[0])
            alpha_scalar = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / float(n)), 1e-6)
            alphas = np.full(X.shape[1], alpha_scalar, dtype=float)
            if 'const' in X.columns: alphas[X.columns.get_loc('const')] = 0.0
            ridge_fit = sm.Logit(y, X).fit_regularized(alpha=alphas, L1_wt=0.0, maxiter=800)
            try:
                refit = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=400, tol=1e-8, start_params=ridge_fit.params, warn_convergence=False)
                if _converged(refit):
                    setattr(refit, "_used_ridge_seed", True); setattr(refit, "_final_is_mle", True)
                    return refit, "ridge_seeded_refit"
            except Exception: pass
            setattr(ridge_fit, "_used_ridge", True); setattr(ridge_fit, "_final_is_mle", False)
            return ridge_fit, "ridge_only"
        except Exception as e: return None, f"ridge_exception:{type(e).__name__}"
    return None, "all_methods_failed"

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

worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, finite_mask_worker = None, None, 0, None, None

def init_worker(df_to_share, masks, ctx):
    """Sends the large core_df, precomputed masks, and context to each worker process."""
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
    global worker_core_df, allowed_mask_by_cat, N_core, CTX, finite_mask_worker
    worker_core_df, allowed_mask_by_cat, N_core, CTX = df_to_share, masks, len(df_to_share), ctx
    finite_mask_worker = np.isfinite(worker_core_df.to_numpy()).all(axis=1)

def init_lrt_worker(df_to_share, masks, anc_series, ctx):
    """Initializer for LRT pools that also provides ancestry labels and context."""
    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[v] = "1"
    global worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, CTX, finite_mask_worker
    worker_core_df, allowed_mask_by_cat, N_core, CTX = df_to_share, masks, len(df_to_share), ctx
    worker_anc_series = anc_series.reindex(df_to_share.index).str.lower()
    finite_mask_worker = np.isfinite(worker_core_df.to_numpy()).all(axis=1)

def _index_fingerprint(index):
    """Order-insensitive fingerprint of a person_id index."""
    s = '\n'.join(sorted(map(str, index)))
    return hashlib.sha256(s.encode()).hexdigest()[:16] + f":{len(index)}"

def _mask_fingerprint(mask: np.ndarray, index: pd.Index) -> str:
    """Order-insensitive fingerprint of a subset of an index."""
    s = '\n'.join(sorted(map(str, index[mask])))
    return hashlib.sha256(s.encode()).hexdigest()[:16] + f":{int(mask.sum())}"

def _should_skip(meta_path, core_df, case_idx_fp, category, target, allowed_fp):
    """Determines if a model run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta: return False
    return (meta.get("model_columns")==list(core_df.columns) and meta.get("ridge_l2_base")==CTX["RIDGE_L2_BASE"] and
            meta.get("core_index_fp")==_index_fingerprint(core_df.index) and meta.get("case_idx_fp")==case_idx_fp and
            meta.get("allowed_mask_fp")==allowed_fp)

def _lrt_meta_should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target, allowed_fp):
    """Determines if an LRT run can be skipped based on metadata."""
    meta = io.read_meta_json(meta_path)
    if not meta: return False
    return (meta.get("model_columns")==list(core_df_cols) and meta.get("ridge_l2_base")==CTX["RIDGE_L2_BASE"] and
            meta.get("core_index_fp")==core_index_fp and meta.get("case_idx_fp")==case_idx_fp and
            meta.get("allowed_mask_fp")==allowed_fp)

def run_single_model_worker(pheno_data, target_inversion, results_cache_dir):
    """CONSUMER: Runs a single model. Executed in a separate process."""
    s_name, category, case_idx = pheno_data["name"], pheno_data["category"], pheno_data["case_idx"]
    s_name_safe = _safe_basename(s_name)
    result_path = os.path.join(results_cache_dir, f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    case_ids_for_fp = worker_core_df.index[case_idx] if case_idx.size > 0 else pd.Index([], name=worker_core_df.index.name)
    case_idx_fp = _index_fingerprint(case_ids_for_fp)
    allowed_mask = allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool))
    allowed_fp = _mask_fingerprint(allowed_mask, worker_core_df.index)
    if os.path.exists(result_path) and _should_skip(meta_path, worker_core_df, case_idx_fp, category, target_inversion, allowed_fp): return

    case_mask = np.zeros(N_core, dtype=bool)
    if case_idx.size > 0: case_mask[case_idx] = True
    valid_mask = (allowed_mask | case_mask) & finite_mask_worker

    n_total = int(valid_mask.sum())
    y = np.zeros(n_total, dtype=np.int8)
    case_positions = np.nonzero(case_mask[valid_mask])[0]
    if case_positions.size > 0: y[case_positions] = 1

    X_clean = worker_core_df[valid_mask].astype(np.float64, copy=False)
    y_clean = pd.Series(y, index=X_clean.index, name='is_case')

    n_cases = int(y_clean.sum())
    n_ctrls = int(n_total - n_cases)

    def write_skip(reason):
        result_data = {"Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                       "Beta": np.nan, "OR": np.nan, "P_Value": np.nan, "Skip_Reason": reason}
        io.atomic_write_json(result_path, result_data)
        _write_meta(meta_path, "phewas_result", s_name, category, target_inversion, worker_core_df.columns,
                    _index_fingerprint(worker_core_df.index), case_idx_fp,
                    extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"], "skip_reason": reason})

    if n_cases < CTX["MIN_CASES_FILTER"] or n_ctrls < CTX["MIN_CONTROLS_FILTER"]:
        write_skip("insufficient_cases_or_controls")
        return

    if X_clean[target_inversion].nunique(dropna=False) <= 1:
        write_skip("target_constant")
        return

    X_work, y_work, sex_note, sex_skip = _apply_sex_restriction(X_clean, y_clean)
    if sex_skip:
        write_skip(sex_skip)
        return

    fit, fit_reason = _fit_logit_ladder(X_work, y_work)

    if not fit or target_inversion not in fit.params:
        write_skip(f"fit_failed:{fit_reason}")
        return

    beta = float(fit.params[target_inversion])
    pval = float(fit.pvalues.get(target_inversion, np.nan))
    se = float(fit.bse.get(target_inversion, np.nan)) if hasattr(fit, "bse") else np.nan

    final_is_mle = getattr(fit, "_final_is_mle", False)
    or_ci95_str = None
    if final_is_mle and np.isfinite(se) and se > 0:
        lo, hi = np.exp(beta - 1.96 * se), np.exp(beta + 1.96 * se)
        or_ci95_str = f"{lo:.3f},{hi:.3f}"

    result = {"Phenotype": s_name, "N_Cases": n_cases, "N_Controls": n_ctrls, "Beta": beta, "OR": np.exp(beta),
              "P_Value": pval, "OR_CI95": or_ci95_str, "Used_Ridge": not final_is_mle, "Final_Is_MLE": final_is_mle}
    io.atomic_write_json(result_path, result)
    _write_meta(meta_path, "phewas_result", s_name, category, target_inversion, worker_core_df.columns,
                _index_fingerprint(worker_core_df.index), case_idx_fp,
                extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})

def lrt_overall_worker(task):
    """Worker for Stage-1 overall LRT."""
    s_name, cat, target = task["name"], task["category"], task["target"]
    s_name_safe = _safe_basename(s_name)
    result_path = os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")
    if not os.path.exists(pheno_path):
        io.atomic_write_json(result_path, {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_Overall_Reason": "missing_case_cache"})
        return
    case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
    case_idx = worker_core_df.index.get_indexer(case_ids)
    case_idx = case_idx[case_idx >= 0]
    case_fp = _index_fingerprint(worker_core_df.index[case_idx] if case_idx.size > 0 else pd.Index([]))
    allowed_mask = allowed_mask_by_cat.get(cat, np.ones(N_core, dtype=bool))
    allowed_fp = _mask_fingerprint(allowed_mask, worker_core_df.index)
    if os.path.exists(result_path) and _lrt_meta_should_skip(result_path + ".meta.json", worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_fp, cat, target, allowed_fp): return

    case_mask = np.zeros(N_core, dtype=bool)
    if case_idx.size > 0: case_mask[case_idx] = True
    valid_mask = (allowed_mask | case_mask) & finite_mask_worker
    pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
    anc_cols = [c for c in worker_core_df.columns if c.startswith("ANC_")]
    base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq'] + anc_cols
    X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64)
    y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index)

    X_full, X_red = X_base, X_base.drop(columns=[target])

    def _fit_logit_hardened(X, y):
        try:
            fit = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=200)
            return fit, "", True
        except Exception: pass
        try:
            fit = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=800)
            return fit, "", True
        except Exception: pass
        p, n = X.shape[1], X.shape[0]
        alphas = np.full(p, max(CTX.get("RIDGE_L2_BASE", 1.0) * (p / max(1, n)), 1e-6))
        if 'const' in X.columns: alphas[X.columns.get_loc('const')] = 0.0
        try:
            ridge_fit = sm.Logit(y, X).fit_regularized(alpha=alphas, L1_wt=0.0)
            return ridge_fit, "", False
        except Exception as e: return None, f"fit_exception:{type(e).__name__}", False

    fit_red, _, red_is_mle = _fit_logit_hardened(X_red, y_series)
    fit_full, _, full_is_mle = _fit_logit_hardened(X_full, y_series)

    out = {"Phenotype": s_name, "P_LRT_Overall": np.nan, "LRT_df_Overall": np.nan}
    if red_is_mle and full_is_mle and fit_full and fit_red and fit_full.llf >= fit_red.llf:
        r_full, r_red = np.linalg.matrix_rank(X_full), np.linalg.matrix_rank(X_red)
        df_lrt = max(0, int(r_full - r_red))
        if df_lrt > 0:
            llr = 2 * (fit_full.llf - fit_red.llf)
            out["P_LRT_Overall"] = sp_stats.chi2.sf(llr, df_lrt)
            out["LRT_df_Overall"] = df_lrt
    else:
        out["LRT_Overall_Reason"] = "penalized_fit_in_path" if (fit_red or fit_full) else "fit_failed"
    io.atomic_write_json(result_path, out)
    _write_meta(meta_path, "lrt_overall", s_name, cat, target, worker_core_df.columns,
                _index_fingerprint(worker_core_df.index), case_fp,
                extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})

def lrt_followup_worker(task):
    """Worker for Stage-2 ancestry×dosage LRT and per-ancestry splits."""
    s_name, category, target = task["name"], task["category"], task["target"]
    s_name_safe = _safe_basename(s_name)
    result_path = os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    pheno_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{task['cdr_codename']}.parquet")

    if not os.path.exists(pheno_path):
        io.atomic_write_json(result_path, {'Phenotype': s_name, 'P_LRT_AncestryxDosage': np.nan, 'LRT_df': np.nan, 'LRT_Reason': "missing_case_cache"})
        return

    case_ids = pd.read_parquet(pheno_path, columns=['is_case']).query("is_case == 1").index
    case_idx = worker_core_df.index.get_indexer(case_ids)
    case_idx = case_idx[case_idx >= 0]
    case_fp = _index_fingerprint(worker_core_df.index[case_idx] if case_idx.size > 0 else pd.Index([]))
    allowed_mask = allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool))
    allowed_fp = _mask_fingerprint(allowed_mask, worker_core_df.index)

    if os.path.exists(result_path) and _lrt_meta_should_skip(meta_path, worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_fp, category, target, allowed_fp):
        return

    case_mask = np.zeros(N_core, dtype=bool)
    if case_idx.size > 0: case_mask[case_idx] = True
    valid_mask = (allowed_mask | case_mask) & finite_mask_worker

    pc_cols = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
    base_cols = ['const', target, 'sex'] + pc_cols + ['AGE_c', 'AGE_c_sq']
    X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64)
    y_series = pd.Series(np.where(case_mask[valid_mask], 1, 0), index=X_base.index)
    anc_vec = worker_anc_series.loc[X_base.index]

    anc_levels = anc_vec.dropna().unique().tolist()
    out = {'Phenotype': s_name, 'P_LRT_AncestryxDosage': np.nan, 'LRT_df': np.nan, 'LRT_Ancestry_Levels': ",".join(anc_levels), 'LRT_Reason': ""}

    if len(anc_levels) < 2:
        out['LRT_Reason'] = "only_one_ancestry_level"
        io.atomic_write_json(result_path, out)
        return

    A = pd.get_dummies(anc_vec, prefix='ANC', drop_first=True)
    X_red = X_base.join(A)
    X_full = X_red.copy()
    for c in A.columns: X_full[f"{target}:{c}"] = X_red[target] * X_red[c]

    fit_red, _ = _fit_logit_ladder(X_red, y_series)
    fit_full, _ = _fit_logit_ladder(X_full, y_series)
    red_is_mle = getattr(fit_red, "_final_is_mle", False)
    full_is_mle = getattr(fit_full, "_final_is_mle", False)

    if red_is_mle and full_is_mle and fit_full and fit_red and fit_full.llf >= fit_red.llf:
        r_full, r_red = np.linalg.matrix_rank(X_full), np.linalg.matrix_rank(X_red)
        df_lrt = max(0, int(r_full - r_red))
        if df_lrt > 0:
            llr = 2 * (fit_full.llf - fit_red.llf)
            out['P_LRT_AncestryxDosage'] = sp_stats.chi2.sf(llr, df_lrt)
            out['LRT_df'] = df_lrt
    else:
        out['LRT_Reason'] = "penalized_fit_in_path"

    for anc in anc_levels:
        anc_mask = anc_vec == anc
        X_anc = X_base[anc_mask]
        y_anc = y_series[anc_mask]
        n_cases, n_ctrls = y_anc.sum(), len(y_anc) - y_anc.sum()
        out[f"{anc.upper()}_N_Cases"], out[f"{anc.upper()}_N_Controls"] = n_cases, n_ctrls
        if n_cases < CTX["PER_ANC_MIN_CASES"] or n_ctrls < CTX["PER_ANC_MIN_CONTROLS"]:
            out[f"{anc.upper()}_REASON"] = "insufficient_stratum_counts"
            continue
        fit, _ = _fit_logit_ladder(X_anc, y_anc)
        if fit and target in fit.params:
            beta = fit.params[target]
            out[f"{anc.upper()}_OR"] = np.exp(beta)
            out[f"{anc.upper()}_P"] = fit.pvalues[target]
        else:
            out[f"{anc.upper()}_REASON"] = "subset_fit_failed"

    io.atomic_write_json(result_path, out)
    _write_meta(meta_path, "lrt_followup", s_name, category, target, worker_core_df.columns,
                _index_fingerprint(worker_core_df.index), case_fp,
                extra={"allowed_mask_fp": allowed_fp, "ridge_l2_base": CTX["RIDGE_L2_BASE"]})
