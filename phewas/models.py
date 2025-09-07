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

def _safe_basename(name: str) -> str:
    """Allow only [-._a-zA-Z0-9], map others to '_'."""
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in os.path.basename(str(name)))

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
        # 1. Newton-Raphson
        try:
            fit_try = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=False)
            if _converged(fit_try):
                setattr(fit_try, "_used_ridge", False)
                return fit_try, "newton"
        except (Exception, PerfectSeparationWarning):
            pass

        # 2. BFGS
        try:
            fit_try = sm.Logit(y, X).fit(disp=0, method='bfgs', maxiter=800, gtol=1e-8, warn_convergence=False)
            if _converged(fit_try):
                setattr(fit_try, "_used_ridge", False)
                return fit_try, "bfgs"
        except (Exception, PerfectSeparationWarning):
            pass

        # 3. Ridge-seeded refit
        if ridge_ok:
            try:
                p = X.shape[1] - (1 if 'const' in X.columns else 0)
                n = max(1, X.shape[0])
                alpha_scalar = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / float(n)), 1e-6)
                alphas = np.full(X.shape[1], alpha_scalar, dtype=float)
                if 'const' in X.columns:
                    alphas[X.columns.get_loc('const')] = 0.0
                ridge_fit = sm.Logit(y, X).fit_regularized(alpha=alphas, L1_wt=0.0, maxiter=800)

                try:
                    refit = sm.Logit(y, X).fit(disp=0, method='newton', maxiter=400, tol=1e-8, start_params=ridge_fit.params, warn_convergence=False)
                    if _converged(refit):
                        setattr(refit, "_used_ridge", True)
                        return refit, "ridge_seeded_refit"
                except Exception:
                    pass

                setattr(ridge_fit, "_used_ridge", True)
                return ridge_fit, "ridge_only"
            except Exception as e:
                return None, f"ridge_exception:{type(e).__name__}"

    return None, "all_methods_failed"

def _apply_sex_restriction(X: pd.DataFrame, y: pd.Series):
    """
    Enforce: if all cases are one sex, only use that sex's rows (and drop 'sex').
    If that sex has zero controls, signal skip.
    Returns: (X2, y2, note:str, skip_reason:str|None)
    """
    if 'sex' not in X.columns:
        return X, y, "", None

    tab = pd.crosstab(X['sex'], y).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
    case_sexes = [s for s in [0.0, 1.0] if s in tab.index and tab.loc[s, 1] > 0]

    if len(case_sexes) != 1:
        return X, y, "", None

    s = case_sexes[0]
    if tab.loc[s, 0] == 0:
        return X, y, "", "sex_no_controls_in_case_sex"

    keep = X['sex'].eq(s)
    X2 = X.loc[keep].drop(columns=['sex'])
    y2 = y.loc[keep]
    return X2, y2, "sex_restricted", None

# --- Module-level globals for worker processes ---
# These are populated by initializer functions.
worker_core_df = None
allowed_mask_by_cat = None
N_core = 0
worker_anc_series = None
finite_mask_worker = None
CTX = {}  # Worker context with constants from run.py


def init_worker(df_to_share, masks, ctx):
    """Sends the large core_df, precomputed masks, and context to each worker process."""
    warnings.filterwarnings("ignore", message=r"^overflow encountered in exp", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r"^divide by zero encountered in log", category=RuntimeWarning)

    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[v] = "1"

    global worker_core_df, allowed_mask_by_cat, N_core, CTX, finite_mask_worker
    worker_core_df = df_to_share
    allowed_mask_by_cat = masks
    N_core = len(df_to_share)
    CTX = ctx
    finite_mask_worker = np.isfinite(worker_core_df.to_numpy()).all(axis=1)

    required_keys = ["NUM_PCS", "MIN_CASES_FILTER", "MIN_CONTROLS_FILTER", "CACHE_DIR", "RESULTS_CACHE_DIR", "RIDGE_L2_BASE"]
    for key in required_keys:
        if key not in CTX:
            raise ValueError(f"Required key '{key}' not found in worker context for init_worker.")

    print(f"[Worker-{os.getpid()}] Initialized and received shared core dataframe, masks, and context.", flush=True)


def init_lrt_worker(df_to_share, masks, anc_series, ctx):
    """Initializer for LRT pools that also provides ancestry labels and context."""
    warnings.filterwarnings("ignore", message=r"^overflow encountered in exp", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r"^divide by zero encountered in log", category=RuntimeWarning)

    for v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[v] = "1"

    global worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, CTX, finite_mask_worker
    worker_core_df = df_to_share
    allowed_mask_by_cat = masks
    N_core = len(df_to_share)
    worker_anc_series = anc_series.reindex(df_to_share.index).str.lower()
    CTX = ctx
    finite_mask_worker = np.isfinite(worker_core_df.to_numpy()).all(axis=1)

    required_keys = ["NUM_PCS", "MIN_CASES_FILTER", "MIN_CONTROLS_FILTER", "CACHE_DIR", "LRT_FOLLOWUP_CACHE_DIR", "PER_ANC_MIN_CASES", "PER_ANC_MIN_CONTROLS", "RIDGE_L2_BASE"]
    for key in required_keys:
        if key not in CTX:
            raise ValueError(f"Required key '{key}' not found in worker context for init_lrt_worker.")

    print(f"[Worker-{os.getpid()}] Initialized and received shared core dataframe, masks, ancestry, and context.", flush=True)


def _index_fingerprint(index) -> str:
    """Order-insensitive fingerprint of a person_id index."""
    s = '\n'.join(sorted(map(str, index)))
    return hashlib.sha256(s.encode()).hexdigest()[:16] + f":{len(index)}"


def _should_skip(meta_path, core_df, case_idx_fp, category, target):
    meta = io.read_meta_json(meta_path)
    if not meta:
        return False
    same_cols = meta.get("model_columns") == list(core_df.columns)
    same_params = (
        meta.get("num_pcs") == CTX["NUM_PCS"] and
        meta.get("min_cases") == CTX["MIN_CASES_FILTER"] and
        meta.get("min_ctrls") == CTX["MIN_CONTROLS_FILTER"] and
        meta.get("target") == target and
        meta.get("category") == category
    )
    same_core = meta.get("core_index_fp") == _index_fingerprint(core_df.index)
    same_case = meta.get("case_idx_fp") == case_idx_fp
    return all([same_cols, same_params, same_core, same_case])


def _lrt_meta_should_skip(meta_path, core_df_cols, core_index_fp, case_idx_fp, category, target):
    meta = io.read_meta_json(meta_path)
    if not meta:
        return False
    same = (
        meta.get("model_columns") == list(core_df_cols) and
        meta.get("num_pcs") == CTX["NUM_PCS"] and
        meta.get("min_cases") == CTX["MIN_CASES_FILTER"] and
        meta.get("min_ctrls") == CTX["MIN_CONTROLS_FILTER"] and
        meta.get("target") == target and
        meta.get("category") == category and
        meta.get("core_index_fp") == core_index_fp and
        meta.get("case_idx_fp") == case_idx_fp
    )
    return bool(same)


def run_single_model_worker(pheno_data, target_inversion, results_cache_dir):
    """CONSUMER: Runs a single model. Executed in a separate process using integer indices and precomputed masks."""
    global worker_core_df, allowed_mask_by_cat, N_core
    s_name = pheno_data["name"]
    s_name_safe = _safe_basename(s_name)
    category = pheno_data["category"]
    case_idx = pheno_data["case_idx"]
    result_path = os.path.join(results_cache_dir, f"{s_name_safe}.json")
    meta_path = result_path + ".meta.json"
    # Order-insensitive fingerprint of the case set based on person_id values to ensure stable caching across runs.
    case_ids_for_fp = worker_core_df.index[case_idx] if case_idx.size > 0 else pd.Index([], name=worker_core_df.index.name)
    case_idx_fp = _index_fingerprint(case_ids_for_fp)
    if os.path.exists(result_path) and _should_skip(meta_path, worker_core_df, case_idx_fp, category, target_inversion):
        return

    try:
        case_mask = np.zeros(N_core, dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True

        # Use the pre-computed finite mask.
        allowed_mask = allowed_mask_by_cat.get(category, np.ones(N_core, dtype=bool))
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker

        n_total = int(valid_mask.sum())
        if n_total == 0:
            print(f"[Worker-{os.getpid()}] - [SKIP] {s_name:<40s} | Reason=no_valid_rows_after_mask", flush=True)
            return

        # Construct response aligned to valid rows.
        y = np.zeros(n_total, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        # Harden design matrix for numeric stability.
        X_clean = worker_core_df[valid_mask].astype(np.float64, copy=False)
        if not np.isfinite(X_clean.to_numpy()).all():
            bad_cols = [c for c in X_clean.columns if not np.isfinite(X_clean[c].to_numpy()).all()]
            bad_rows_mask = ~np.isfinite(X_clean.to_numpy()).all(axis=1)
            bad_idx_sample = X_clean.index[bad_rows_mask][:10].tolist()
            print(f"[fit FAIL] name={s_name} err=non_finite_in_design columns={','.join(bad_cols)} sample_rows={bad_idx_sample}", flush=True)
            traceback.print_stack(file=sys.stderr)
            sys.stderr.flush()
        y_clean = pd.Series(y, index=X_clean.index, name='is_case')

        n_cases = int(y_clean.sum())
        n_ctrls = int(n_total - n_cases)
        if n_cases < CTX["MIN_CASES_FILTER"] or n_ctrls < CTX["MIN_CONTROLS_FILTER"]:
            result_data = {
                "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                "Beta": float('nan'), "OR": float('nan'), "P_Value": float('nan'),
                "Skip_Reason": "insufficient_cases_or_controls"
            }
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion,
                        worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_idx_fp,
                        extra={"skip_reason": "insufficient_cases_or_controls"})
            print(f"[fit SKIP] name={s_name} N={n_total} cases={n_cases} ctrls={n_ctrls} reason=insufficient_cases_or_controls", flush=True)
            return

        drop_candidates = [c for c in X_clean.columns if c not in ('const', target_inversion)]
        zero_var_cols = [c for c in drop_candidates if X_clean[c].nunique(dropna=False) <= 1]
        if zero_var_cols:
            X_clean = X_clean.drop(columns=zero_var_cols)

        if X_clean[target_inversion].nunique(dropna=False) <= 1:
            result_data = {
                "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                "Beta": float('nan'), "OR": float('nan'), "P_Value": float('nan')
            }
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion,
                        worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_idx_fp)
            print(f"[fit SKIP] name={s_name} N={n_total} cases={n_cases} ctrls={n_ctrls} reason=target_constant", flush=True)
            return

        X_work = X_clean
        y_work = y_clean
        model_notes_worker = []

        X_work, y_work, note, skip_reason = _apply_sex_restriction(X_work, y_work)
        if skip_reason:
            result_data = {
                "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                "Beta": float('nan'), "OR": float('nan'), "P_Value": float('nan'), "Skip_Reason": skip_reason
            }
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion,
                        worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_idx_fp,
                        extra={"skip_reason": skip_reason})
            print(f"[fit SKIP] name={s_name} N={n_total} cases={n_cases} ctrls={n_ctrls} reason={skip_reason}", flush=True)
            return

        if note:
            model_notes_worker.append(note)

        zvars = [c for c in X_work.columns if c not in ['const', target_inversion] and X_work[c].nunique(dropna=False) <= 1]
        if zvars:
            X_work = X_work.drop(columns=zvars)

        fit, fit_reason = _fit_logit_ladder(X_work, y_work, ridge_ok=True)
        if fit:
            model_notes_worker.append(fit_reason)
            setattr(fit, "_model_note", ";".join(model_notes_worker))

        if fit is None or target_inversion not in fit.params:
            result_data = {
                "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                "Beta": float('nan'), "OR": float('nan'), "P_Value": float('nan')
            }
            io.atomic_write_json(result_path, result_data)
            _write_meta(meta_path, "phewas_result", s_name, category, target_inversion,
                        worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_idx_fp)
            what = f"fit_failed:{fit_reason}" if fit is None else "coefficient_missing"
            print(f"[fit FAIL] name={s_name} N={n_total} cases={n_cases} ctrls={n_ctrls} err={what}", flush=True)
            return

        beta = float(fit.params[target_inversion])
        try:
            pval = float(fit.pvalues[target_inversion])
            pval_reason = ""
        except Exception as e:
            pval = float('nan')
            pval_reason = f"pvalue_unavailable({type(e).__name__})"

        se = None
        if hasattr(fit, "bse"):
            try:
                se = float(fit.bse[target_inversion])
            except Exception:
                se = None

        used_ridge = bool(getattr(fit, "_used_ridge", False))
        or_ci95_str = None
        if se is not None and np.isfinite(se) and se > 0.0 and not used_ridge:
            lo = float(np.exp(beta - 1.96 * se))
            hi = float(np.exp(beta + 1.96 * se))
            or_ci95_str = f"{lo:.3f},{hi:.3f}"

        notes = []
        if hasattr(fit, "_model_note"): notes.append(fit._model_note)
        if used_ridge: notes.append("used_ridge")
        if pval_reason: notes.append(pval_reason)
        notes_str = ";".join(filter(None, notes))

        n_total_used = int(len(y_work))
        n_cases_used = int(y_work.sum())
        n_ctrls_used = n_total_used - n_cases_used

        print(f"[fit OK] name={s_name} N={n_total_used} cases={n_cases_used} ctrls={n_ctrls_used} beta={beta:+.4f} OR={np.exp(beta):.4f} p={pval:.3e} notes={notes_str}", flush=True)

        result_data = {
            "Phenotype": s_name,
            "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
            "N_Total_Used": n_total_used, "N_Cases_Used": n_cases_used, "N_Controls_Used": n_ctrls_used,
            "Beta": beta, "OR": float(np.exp(beta)), "P_Value": pval, "OR_CI95": or_ci95_str,
            "Model_Notes": notes_str, "Used_Ridge": used_ridge
        }
        io.atomic_write_json(result_path, result_data)
        _write_meta(meta_path, "phewas_result", s_name, category, target_inversion,
                    worker_core_df.columns, _index_fingerprint(worker_core_df.index), case_idx_fp)

    except Exception as e:
        print(f"[Worker-{os.getpid()}] - [FAIL] {s_name:<40s} | Error occurred. Full traceback follows:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()

    finally:
        if 'pheno_data' in locals(): del pheno_data
        if 'y' in locals(): del y
        if 'X_clean' in locals(): del X_clean
        if 'y_clean' in locals(): del y_clean
        gc.collect()


def lrt_overall_worker(task):
    """
    Worker for Stage-1 overall LRT. Reads case set from per-phenotype cache, builds reduced and full models
    on the same rows, computes df via rank difference, and writes result + meta atomically.
    """
    try:
        s_name = task["name"]
        s_name_safe = _safe_basename(s_name)
        category = task["category"]
        cdr_codename = task["cdr_codename"]
        target_inversion = task["target"]
        result_path = os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name_safe}.json")
        meta_path = result_path + ".meta.json"

        pheno_cache_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{cdr_codename}.parquet")
        if not os.path.exists(pheno_cache_path):
            io.atomic_write_json(result_path, {
                "Phenotype": s_name, "P_LRT_Overall": float('nan'), "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "missing_case_cache"
            })
            _write_meta(meta_path, "lrt_overall", s_name, category, target_inversion,
                        worker_core_df.columns, _index_fingerprint(worker_core_df.index), "")
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=missing_case_cache", flush=True)
            return

        ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
        case_ids = ph.index[ph['is_case'] == 1].astype(str)
        core_index = worker_core_df.index
        case_idx = core_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0].astype(np.int32)
        case_ids_for_fp = core_index[case_idx] if case_idx.size > 0 else pd.Index([], name=core_index.name)
        case_fp = _index_fingerprint(case_ids_for_fp)

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df.columns, _index_fingerprint(core_index),
            case_fp, category, target_inversion
        ):
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} CACHE_HIT", flush=True)
            return

        allowed_mask = allowed_mask_by_cat.get(category, np.ones(len(core_index), dtype=bool))
        case_mask = np.zeros(len(core_index), dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker
        n_valid = int(valid_mask.sum())
        y = np.zeros(n_valid, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        pc_cols_local = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        anc_cols = [c for c in worker_core_df.columns if c.startswith("ANC_")]
        base_cols = ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE_c', 'AGE_c_sq'] + anc_cols
        X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64, copy=False)
        y_series = pd.Series(y, index=X_base.index, name='is_case')

        if 'sex' in X_base.columns:
            try:
                tab = pd.crosstab(X_base['sex'], y_series).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
                valid_sexes = []
                for s in [0.0, 1.0]:
                    if s in tab.index:
                        if bool(tab.loc[s, 0] > 0) and bool(tab.loc[s, 1] > 0):
                            valid_sexes.append(s)
                if len(valid_sexes) == 1:
                    keep = X_base['sex'].isin(valid_sexes)
                    X_base = X_base.loc[keep]
                    y_series = y_series.loc[X_base.index]
                elif len(valid_sexes) == 0:
                    X_base = X_base.drop(columns=['sex'])
            except Exception:
                pass

        zvars = [c for c in X_base.columns if c not in ['const', target_inversion] and pd.Series(X_base[c]).nunique(dropna=False) <= 1]
        if len(zvars) > 0:
            X_base = X_base.drop(columns=zvars)

        n_cases = int(y_series.sum())
        n_ctrls = int(len(y_series) - n_cases)
        if n_cases < CTX["MIN_CASES_FILTER"] or n_ctrls < CTX["MIN_CONTROLS_FILTER"]:
            io.atomic_write_json(result_path, {
                "Phenotype": s_name, "P_LRT_Overall": float('nan'), "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "insufficient_counts"
            })
            io.atomic_write_json(meta_path, {
                "kind": "lrt_overall", "s_name": s_name, "category": category,
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp, "created_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=insufficient_counts", flush=True)
            return

        if target_inversion not in X_base.columns or pd.Series(X_base[target_inversion]).nunique(dropna=False) <= 1:
            io.atomic_write_json(result_path, {
                "Phenotype": s_name, "P_LRT_Overall": float('nan'), "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "target_constant"
            })
            io.atomic_write_json(meta_path, {
                "kind": "lrt_overall", "s_name": s_name, "category": category,
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp, "created_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=target_constant", flush=True)
            return

        def _fit_logit_hardened(X, y_in, require_target):
            if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
            X = X.astype(np.float64, copy=False)
            y_arr = np.asarray(y_in, dtype=np.float64).reshape(-1)
            if require_target:
                if target_inversion not in X.columns or pd.Series(X[target_inversion]).nunique(dropna=False) <= 1:
                    return None, "target_constant"
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception: pass
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception: pass
            try:
                p = X.shape[1] - (1 if 'const' in X.columns else 0)
                n = max(1, X.shape[0])
                alpha = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / float(n)), 1e-6)
                ridge_fit = sm.Logit(y_arr, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=800)
                return ridge_fit, ""
            except Exception as e: return None, f"fit_exception:{type(e).__name__}"

        X_full = X_base.copy()
        X_red = X_base.drop(columns=[target_inversion])
        fit_red, r_reason = _fit_logit_hardened(X_red, y_series, require_target=False)
        fit_full, f_reason = _fit_logit_hardened(X_full, y_series, require_target=True)

        out = {
            "Phenotype": s_name, "P_LRT_Overall": float('nan'),
            "LRT_df_Overall": float('nan'), "LRT_Overall_Reason": ""
        }

        if (fit_red is not None) and (fit_full is not None) and hasattr(fit_full, "llf") and hasattr(fit_red, "llf") and (fit_full.llf >= fit_red.llf):
            df_lrt = int(max(0, X_full.shape[1] - X_red.shape[1]))
            if df_lrt > 0:
                llr = 2.0 * (fit_full.llf - fit_red.llf)
                out["P_LRT_Overall"] = float(sp_stats.chi2.sf(llr, df_lrt))
                out["LRT_df_Overall"] = df_lrt
                out["LRT_Overall_Reason"] = ""
                print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} df={df_lrt} p={out['P_LRT_Overall']:.3e}", flush=True)
            else:
                out["LRT_df_Overall"] = 0
                out["LRT_Overall_Reason"] = "no_df"
                print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=no_df", flush=True)
        else:
            reasons = []
            if fit_red is None: reasons.append(f"reduced_model_failed:{r_reason}")
            if fit_full is None: reasons.append(f"full_model_failed:{f_reason}")
            if (fit_red is not None) and (fit_full is not None) and (fit_full.llf < fit_red.llf):
                reasons.append("full_llf_below_reduced_llf")
            out["LRT_Overall_Reason"] = ";".join(reasons) if reasons else "fit_failed"
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason={out['LRT_Overall_Reason']}", flush=True)

        io.atomic_write_json(result_path, out)
        io.atomic_write_json(meta_path, {
            "kind": "lrt_overall", "s_name": s_name, "category": category,
            "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
            "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
            "target": target_inversion, "core_index_fp": _index_fingerprint(core_index),
            "case_idx_fp": case_fp, "created_at": datetime.now(timezone.utc).isoformat()
        })
    except Exception:
        print(f"[LRT-Stage1-Worker-{os.getpid()}] {task.get('name','?')} FAILED with exception, writing error stub", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        s_name = task.get("name", "unknown")
        io.atomic_write_json(os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name}.json"), {
            "Phenotype": s_name, "P_LRT_Overall": float('nan'),
            "LRT_df_Overall": float('nan'), "LRT_Overall_Reason": "exception"
        })


def lrt_followup_worker(task):
    """
    Worker for Stage-2 ancestry×dosage LRT and per-ancestry splits for selected phenotypes.
    Caches one JSON per phenotype in the follow-up cache directory.
    """
    try:
        s_name = task["name"]
        category = task["category"]
        cdr_codename = task["cdr_codename"]
        target_inversion = task["target"]
        result_path = os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name}.json")
        meta_path = result_path + ".meta.json"

        pheno_cache_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{cdr_codename}.parquet")
        if not os.path.exists(pheno_cache_path):
            io.atomic_write_json(result_path, {
                'Phenotype': s_name, 'P_LRT_AncestryxDosage': float('nan'), 'LRT_df': float('nan'),
                'LRT_Ancestry_Levels': "", 'LRT_Reason': "missing_case_cache"
            })
            io.atomic_write_json(meta_path, {
                "kind": "lrt_followup", "s_name": s_name, "category": category,
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": "", "created_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=missing_case_cache", flush=True)
            return

        ph = pd.read_parquet(pheno_cache_path, columns=['is_case'])
        case_ids = ph.index[ph['is_case'] == 1].astype(str)
        core_index = worker_core_df.index
        case_idx = core_index.get_indexer(case_ids)
        case_idx = case_idx[case_idx >= 0].astype(np.int32)
        case_ids_for_fp = core_index[case_idx] if case_idx.size > 0 else pd.Index([], name=core_index.name)
        case_fp = _index_fingerprint(case_ids_for_fp)

        if os.path.exists(result_path) and _lrt_meta_should_skip(
            meta_path, worker_core_df.columns, _index_fingerprint(core_index),
            case_fp, category, target_inversion
        ):
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} CACHE_HIT", flush=True)
            return

        allowed_mask = allowed_mask_by_cat.get(category, np.ones(len(core_index), dtype=bool))
        case_mask = np.zeros(len(core_index), dtype=bool)
        if case_idx.size > 0: case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask_worker
        if int(valid_mask.sum()) == 0:
            io.atomic_write_json(result_path, {
                'Phenotype': s_name, 'P_LRT_AncestryxDosage': float('nan'), 'LRT_df': float('nan'),
                'LRT_Ancestry_Levels': "", 'LRT_Reason': "no_valid_rows_after_mask"
            })
            io.atomic_write_json(meta_path, {
                "kind": "lrt_followup", "s_name": s_name, "category": category,
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp, "created_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=no_valid_rows_after_mask", flush=True)
            return

        pc_cols_local = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        anc_cols = [c for c in worker_core_df.columns if c.startswith("ANC_")]
        base_cols = ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE_c', 'AGE_c_sq'] + anc_cols
        X_base = worker_core_df.loc[valid_mask, base_cols].astype(np.float64, copy=False)
        y = np.zeros(X_base.shape[0], dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0: y[case_positions] = 1
        y_series = pd.Series(y, index=X_base.index, name='is_case')

        anc_vec = worker_anc_series.loc[X_base.index]
        anc_levels_local = anc_vec.dropna().unique().tolist()
        if 'eur' in anc_levels_local:
            anc_levels_local = ['eur'] + [a for a in anc_levels_local if a != 'eur']

        out = {
            'Phenotype': s_name, 'P_LRT_AncestryxDosage': float('nan'), 'LRT_df': float('nan'),
            'LRT_Ancestry_Levels': ",".join(anc_levels_local), 'LRT_Reason': ""
        }

        if len(anc_levels_local) < 2:
            out['LRT_Reason'] = "only_one_ancestry_level"
            io.atomic_write_json(result_path, out)
            io.atomic_write_json(meta_path, {
                "kind": "lrt_followup", "s_name": s_name, "category": category,
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(core_index),
                "case_idx_fp": case_fp, "created_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=only_one_ancestry_level", flush=True)
            return

        keep = anc_vec.notna()
        X_base = X_base.loc[keep]
        y_series = y_series.loc[keep]
        anc_keep = anc_vec.loc[keep]
        anc_keep = pd.Series(pd.Categorical(anc_keep, categories=anc_levels_local, ordered=False), index=anc_keep.index, name='ANCESTRY')
        A = pd.get_dummies(anc_keep, prefix='ANC', drop_first=True, dtype=np.float64)
        X_red = pd.concat([X_base, A], axis=1, join='inner').astype(np.float64, copy=False)
        X_full = X_red.copy()
        for c in A.columns:
            X_full[f"{target_inversion}:{c}"] = X_full[target_inversion] * X_full[c]

        def _fit_logit(X, y_in, require_target):
            if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
            X = X.astype(np.float64, copy=False)
            y_arr = np.asarray(y_in, dtype=np.float64).reshape(-1)
            if require_target:
                if target_inversion not in X.columns or pd.Series(X[target_inversion]).nunique(dropna=False) <= 1:
                    return None, "target_constant"
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception: pass
            try:
                fit_try = sm.Logit(y_arr, X).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                return fit_try, ""
            except Exception: pass
            try:
                p = X.shape[1] - (1 if 'const' in X.columns else 0)
                n = max(1, X.shape[0])
                alpha = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / float(n)), 1e-6)
                ridge_fit = sm.Logit(y_arr, X).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=800)
                return ridge_fit, ""
            except Exception as e: return None, f"fit_exception:{type(e).__name__}"

        fit_red, rr = _fit_logit(X_red, y_series, require_target=False)
        fit_full, fr = _fit_logit(X_full, y_series, require_target=True)

        if (fit_red is not None) and (fit_full is not None) and hasattr(fit_full, "llf") and hasattr(fit_red, "llf") and (fit_full.llf >= fit_red.llf):
            df_lrt = int(max(0, X_full.shape[1] - X_red.shape[1]))
            if df_lrt > 0:
                llr = 2.0 * (fit_full.llf - fit_red.llf)
                out['P_LRT_AncestryxDosage'] = float(sp_stats.chi2.sf(llr, df_lrt))
                out['LRT_df'] = df_lrt
                out['LRT_Reason'] = ""
                print(f"[Ancestry-Worker-{os.getpid()}] {s_name} df={df_lrt} p={out['P_LRT_AncestryxDosage']:.3e}", flush=True)
            else:
                out['LRT_Reason'] = "no_interaction_df"
        else:
            reasons = []
            if fit_red is None: reasons.append(f"reduced_model_failed:{rr}")
            if fit_full is None: reasons.append(f"full_model_failed:{fr}")
            if (fit_red is not None) and (fit_full is not None) and (fit_full.llf < fit_red.llf):
                reasons.append("full_llf_below_reduced_llf")
            out['LRT_Reason'] = ";".join(reasons) if reasons else "fit_failed"

        for anc in anc_levels_local:
            group_mask = valid_mask & worker_anc_series.eq(anc).to_numpy()
            if not group_mask.any():
                out[f"{anc.upper()}_N"] = 0; out[f"{anc.upper()}_N_Cases"] = 0; out[f"{anc.upper()}_N_Controls"] = 0
                out[f"{anc.upper()}_OR"] = float('nan'); out[f"{anc.upper()}_CI95"] = float('nan'); out[f"{anc.upper()}_P"] = float('nan')
                out[f"{anc.upper()}_REASON"] = "no_rows_in_group"
                continue
            X_g = worker_core_df.loc[group_mask, ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE_c', 'AGE_c_sq']].astype(np.float64, copy=False)
            y_g = np.zeros(X_g.shape[0], dtype=np.int8)
            case_positions_g = np.nonzero(case_mask[group_mask])[0]
            if case_positions_g.size > 0: y_g[case_positions_g] = 1
            n_cases_g = int(y_g.sum()); n_tot_g = int(len(y_g)); n_ctrl_g = n_tot_g - n_cases_g
            out[f"{anc.upper()}_N"] = n_tot_g; out[f"{anc.upper()}_N_Cases"] = n_cases_g; out[f"{anc.upper()}_N_Controls"] = n_ctrl_g
            if (n_cases_g < CTX["PER_ANC_MIN_CASES"]) or (n_ctrl_g < CTX["PER_ANC_MIN_CONTROLS"]):
                out[f"{anc.upper()}_OR"] = float('nan'); out[f"{anc.upper()}_CI95"] = float('nan'); out[f"{anc.upper()}_P"] = float('nan')
                out[f"{anc.upper()}_REASON"] = "insufficient_stratum_counts"
                continue
            try:
                fit_g = sm.Logit(y_g, X_g).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
            except Exception:
                try: fit_g = sm.Logit(y_g, X_g).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                except Exception: fit_g = None
            if (fit_g is None) or (target_inversion not in getattr(fit_g, "params", {})):
                out[f"{anc.upper()}_OR"] = float('nan'); out[f"{anc.upper()}_CI95"] = float('nan'); out[f"{anc.upper()}_P"] = float('nan')
                out[f"{anc.upper()}_REASON"] = "subset_fit_failed"
                continue
            beta = float(fit_g.params[target_inversion]); or_val = float(np.exp(beta))
            if hasattr(fit_g, "bse"):
                try:
                    se = float(fit_g.bse[target_inversion])
                    lo = float(np.exp(beta - 1.96 * se)); hi = float(np.exp(beta + 1.96 * se))
                    out[f"{anc.upper()}_CI95"] = f"{lo:.3f},{hi:.3f}"
                except Exception: out[f"{anc.upper()}_CI95"] = float('nan')
            else: out[f"{anc.upper()}_CI95"] = float('nan')
            out[f"{anc.upper()}_OR"] = or_val
            try: out[f"{anc.upper()}_P"] = float(fit_g.pvalues[target_inversion])
            except Exception: out[f"{anc.upper()}_P"] = float('nan')
            out[f"{anc.upper()}_REASON"] = ""

        io.atomic_write_json(result_path, out)
        io.atomic_write_json(meta_path, {
            "kind": "lrt_followup", "s_name": s_name, "category": category,
            "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
            "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
            "target": target_inversion, "core_index_fp": _index_fingerprint(core_index),
            "case_idx_fp": case_fp, "created_at": datetime.now(timezone.utc).isoformat()
        })
    except Exception:
        print(f"[Ancestry-Worker-{os.getpid()}] {task.get('name','?')} FAILED with exception, writing error stub", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        s_name = task.get("name", "unknown")
        io.atomic_write_json(os.path.join(CTX["LRT_FOLLOWUP_CACHE_DIR"], f"{s_name}.json"), {
            'Phenotype': s_name, 'P_LRT_AncestryxDosage': float('nan'), 'LRT_df': float('nan'),
            'LRT_Ancestry_Levels': "", 'LRT_Reason': "exception"
        })
