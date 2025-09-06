import os
import gc
import hashlib
import warnings
from datetime import datetime
import traceback
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from phewas import io

# --- Module-level globals for worker processes ---
# These are populated by initializer functions.
worker_core_df = None
allowed_mask_by_cat = None
N_core = 0
worker_anc_series = None
CTX = {}  # Worker context with constants from run.py


def init_worker(df_to_share, masks, ctx):
    """Sends the large core_df, precomputed masks, and context to each worker process."""
    warnings.filterwarnings("ignore", message=r"^overflow encountered in exp", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r"^divide by zero encountered in log", category=RuntimeWarning)

    global worker_core_df, allowed_mask_by_cat, N_core, CTX
    worker_core_df = df_to_share
    allowed_mask_by_cat = masks
    N_core = len(df_to_share)
    CTX = ctx

    required_keys = ["NUM_PCS", "MIN_CASES_FILTER", "MIN_CONTROLS_FILTER", "CACHE_DIR", "RESULTS_CACHE_DIR", "RIDGE_L2_BASE"]
    for key in required_keys:
        if key not in CTX:
            raise ValueError(f"Required key '{key}' not found in worker context for init_worker.")

    print(f"[Worker-{os.getpid()}] Initialized and received shared core dataframe, masks, and context.", flush=True)


def init_lrt_worker(df_to_share, masks, anc_series, ctx):
    """Initializer for LRT pools that also provides ancestry labels and context."""
    warnings.filterwarnings("ignore", message=r"^overflow encountered in exp", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=r"^divide by zero encountered in log", category=RuntimeWarning)

    global worker_core_df, allowed_mask_by_cat, N_core, worker_anc_series, CTX
    worker_core_df = df_to_share
    allowed_mask_by_cat = masks
    N_core = len(df_to_share)
    worker_anc_series = anc_series.reindex(df_to_share.index).str.lower()
    CTX = ctx

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
    category = pheno_data["category"]
    case_idx = pheno_data["case_idx"]
    result_path = os.path.join(results_cache_dir, f"{s_name}.json")
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

        # Ensure modeling matrix contains only finite values across all covariates to prevent silent NaN/Inf propagation.
        finite_mask_worker = np.isfinite(worker_core_df.to_numpy()).all(axis=1)
        valid_mask = (allowed_mask_by_cat[category] | case_mask) & finite_mask_worker

        n_total = int(valid_mask.sum())
        if n_total == 0:
            print(f"[Worker-{os.getpid()}] - [SKIP] {s_name:<40s} | Reason=no_valid_rows_after_mask", flush=True)
            return

        # Construct response aligned to valid rows.
        y = np.zeros(n_total, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        # Harden design matrix for numeric stability and emit immediate diagnostics if any non-finite values slip through.
        X_clean = worker_core_df[valid_mask].copy().astype(np.float64, copy=False)
        try:
            _X = X_clean.copy()
            _num_cols = [c for c in _X.columns if pd.api.types.is_numeric_dtype(_X[c])]
            _X = _X[_num_cols].astype(np.float64, copy=False)
            _t = target_inversion
            _tvec = _X[_t].to_numpy()

            # Basic stats for target
            _u = pd.Series(_tvec).nunique(dropna=False)
            _mn = float(np.nanmin(_tvec)) if _u > 0 else float('nan')
            _mx = float(np.nanmax(_tvec)) if _u > 0 else float('nan')
            _sd = float(np.nanstd(_tvec))

            # How many unique AGE values? (quadratic degeneracy check)
            _age_unique = _X['AGE'].nunique(dropna=True) if 'AGE' in _X.columns else np.nan
            _age_flag = (_age_unique <= 2) if pd.notna(_age_unique) else False

            # Columns equal/affine to target (exact within float tol)
            _eq_cols = []
            for c in _X.columns:
                if c == _t:
                    continue
                v = _X[c].to_numpy()
                if np.allclose(v, _tvec, equal_nan=True):
                    _eq_cols.append(c)
                else:
                    A = np.vstack([v, np.ones_like(v)]).T
                    try:
                        coef, *_ = np.linalg.lstsq(A, _tvec, rcond=None)
                        resid = _tvec - (coef[0]*v + coef[1])
                        if np.nanmax(np.abs(resid)) < 1e-10:
                            _eq_cols.append(c + "[affine]")
                    except Exception:
                        pass

            # R^2 of target explained by other covariates
            _others = [c for c in _X.columns if c not in ['const', _t]]
            _r2 = np.nan
            if len(_others) > 0:
                _A = _X[_others].to_numpy()
                try:
                    coef, *_ = np.linalg.lstsq(_A, _tvec, rcond=None)
                    pred = _A.dot(coef)
                    ss_res = float(np.nansum(( _tvec - pred )**2))
                    ss_tot = float(np.nansum(( _tvec - np.nanmean(_tvec))**2))
                    _r2 = 1.0 - (ss_res/ss_tot) if ss_tot > 0 else np.nan
                except Exception:
                    pass

            # Matrix rank & condition number
            try:
                _arr = _X.to_numpy()
                _rank = int(np.linalg.matrix_rank(_arr))
                _s = np.linalg.svd(_arr, compute_uv=False)
                _cond = float((_s[0] / _s[-1]) if (_s[-1] != 0) else np.inf)
            except Exception:
                _rank = -1
                _cond = np.nan

            # Zero-variance columns (after valid_mask)
            _zv = [c for c in _X.columns if c not in ['const', _t] and pd.Series(_X[c]).nunique(dropna=False) <= 1]

            # Print concise one-line summary
            _age_note = " AGE_DEGENERATE(const,AGE,AGE_sq collinear!)" if _age_flag else ""
            print(f"[DEBUG_COLLINEARITY] {s_name} N={len(_X)} rank={_rank} cond={_cond:.2e} "
                  f"target_unique={_u} min={_mn:.4g} max={_mx:.4g} sd={_sd:.4g} "
                  f"R2(target~others)={_r2:.6f} zero_var={_zv} eq_cols={_eq_cols}{_age_note}",
                  flush=True)
        except Exception as _e:
            print(f"[DEBUG_COLLINEARITY] {s_name} failed: {type(_e).__name__}: {_e}", flush=True)

        if not np.isfinite(X_clean.to_numpy()).all():
            bad_cols = [c for c in X_clean.columns if not np.isfinite(X_clean[c].to_numpy()).all()]
            bad_rows_mask = ~np.isfinite(X_clean.to_numpy()).all(axis=1)
            bad_idx_sample = X_clean.index[bad_rows_mask][:10].tolist()
            print(f"[TRACEBACK] Non-finite in worker design for phenotype '{s_name}'", flush=True)
            print(f"[DEBUG] columns={','.join(bad_cols)} sample_rows={bad_idx_sample}", flush=True)
            traceback.print_stack()
            sys.stderr.flush()
        y_clean = pd.Series(y, index=X_clean.index, name='is_case')

        try:
            predictor_counts = X_clean[target_inversion].value_counts(dropna=False)
            if len(predictor_counts) > 10:
                counts_str = (
                    f"{predictor_counts.head(5).to_string()}\n"
                    f"...\n"
                    f"{predictor_counts.tail(5).to_string()}"
                )
            else:
                counts_str = predictor_counts.to_string()

            print(
                f"\n--- [DEBUG] Pre-fit diagnostics for phenotype '{s_name}' in Worker-{os.getpid()} ---"
                f"\n[DEBUG] Total rows in design matrix: {len(X_clean):,}"
                f"\n[DEBUG] Predictor '{target_inversion}' statistics:\n{X_clean[target_inversion].describe().to_string()}"
                f"\n[DEBUG] Predictor '{target_inversion}' value counts (truncated):\n{counts_str}"
                f"\n[DEBUG] Outcome 'is_case' value counts:\n{y_clean.value_counts().to_string()}"
                "\n--- [DEBUG] End of diagnostics ---\n",
                flush=True
            )
        except Exception as diag_e:
            print(f"[DEBUG] Diagnostic printing failed for '{s_name}': {diag_e}", flush=True)

        n_cases = int(y_clean.sum())
        n_ctrls = int(n_total - n_cases)
        if n_cases < CTX["MIN_CASES_FILTER"] or n_ctrls < CTX["MIN_CONTROLS_FILTER"]:
            result_data = {
                "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                "Beta": float('nan'), "OR": float('nan'), "P_Value": float('nan'),
                "Skip_Reason": "insufficient_cases_or_controls"
            }
            io.atomic_write_json(result_path, result_data)
            io.atomic_write_json(meta_path, {
                "kind": "phewas_result", "s_name": s_name, "category": category, "model": "Logit",
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": case_idx_fp, "created_at": datetime.utcnow().isoformat() + "Z",
                "skip_reason": "insufficient_cases_or_controls"
            })
            print(
                f"[Worker-{os.getpid()}] - [SKIP] {s_name:<40s} | N={n_total:,} Cases={n_cases:,} Ctrls={n_ctrls:,} "
                f"| Reason=insufficient_cases_or_controls thresholds(cases>={CTX['MIN_CASES_FILTER']}, ctrls>={CTX['MIN_CONTROLS_FILTER']})",
                flush=True
            )
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
            io.atomic_write_json(meta_path, {
                "kind": "phewas_result", "s_name": s_name, "category": category, "model": "Logit",
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": case_idx_fp, "created_at": datetime.utcnow().isoformat() + "Z",
            })
            uvals = X_clean[target_inversion].dropna().unique().tolist()
            print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta=nan | OR=nan | P=nan | CONSTANT_DOSAGE unique_values={uvals}", flush=True)
            return

        def _converged(fit_obj):
            try:
                if hasattr(fit_obj, "mle_retvals") and isinstance(fit_obj.mle_retvals, dict):
                    return bool(fit_obj.mle_retvals.get("converged", False))
                if hasattr(fit_obj, "converged"):
                    return bool(fit_obj.converged)
                return False
            except Exception:
                return False

        X_work = X_clean
        y_work = y_clean
        model_notes_worker = []
        if 'sex' in X_work.columns:
            try:
                tab = pd.crosstab(X_work['sex'], y_work).reindex(index=[0.0, 1.0], columns=[0, 1], fill_value=0)
                valid_sexes = []
                for s in [0.0, 1.0]:
                    if s in tab.index:
                        has_ctrl = bool(tab.loc[s, 0] > 0)
                        has_case = bool(tab.loc[s, 1] > 0)
                        if has_ctrl and has_case:
                            valid_sexes.append(s)
                if len(valid_sexes) == 1:
                    mask = X_work['sex'].isin(valid_sexes)
                    X_work = X_work.loc[mask]
                    y_work = y_work.loc[X_work.index]
                    model_notes_worker.append("sex_restricted")
                elif len(valid_sexes) == 0:
                    X_work = X_work.drop(columns=['sex'])
                    model_notes_worker.append("sex_dropped_for_separation")
            except Exception:
                pass

        fit = None
        fit_reason = ""
        try:
            fit_try = sm.Logit(y_work, X_work).fit(disp=0, method='newton', maxiter=200, tol=1e-8, warn_convergence=True)
            if _converged(fit_try):
                setattr(fit_try, "_model_note", ";".join(model_notes_worker) if model_notes_worker else "")
                setattr(fit_try, "_used_ridge", False)
                fit = fit_try
        except Exception as e:
            print("[TRACEBACK] run_single_model_worker newton failed:", flush=True)
            traceback.print_exc()
            fit_reason = f"newton_exception:{type(e).__name__}:{e}"

        if fit is None:
            try:
                fit_try = sm.Logit(y_work, X_work).fit(disp=0, maxiter=800, method='bfgs', gtol=1e-8, warn_convergence=True)
                if _converged(fit_try):
                    setattr(fit_try, "_model_note", ";".join(model_notes_worker) if model_notes_worker else "")
                    setattr(fit_try, "_used_ridge", False)
                    fit = fit_try
                else:
                    fit_reason = "bfgs_not_converged"
            except Exception as e:
                print("[TRACEBACK] run_single_model_worker bfgs failed:", flush=True)
                traceback.print_exc()
                fit_reason = f"bfgs_exception:{type(e).__name__}:{e}"

        if fit is None:
            try:
                p = X_work.shape[1] - (1 if 'const' in X_work.columns else 0)
                n = max(1, X_work.shape[0])
                alpha = max(CTX.get("RIDGE_L2_BASE", 1.0) * (float(p) / float(n)), 1e-6)
                ridge_fit = sm.Logit(y_work, X_work).fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=800)
                try:
                    refit = sm.Logit(y_work, X_work).fit(disp=0, method='newton', maxiter=400, tol=1e-8, start_params=ridge_fit.params, warn_convergence=True)
                    if _converged(refit):
                        model_notes_worker.append("ridge_seeded_refit")
                        setattr(refit, "_model_note", ";".join(model_notes_worker))
                        setattr(refit, "_used_ridge", True)
                        fit = refit
                    else:
                        model_notes_worker.append("ridge_only")
                        setattr(ridge_fit, "_model_note", ";".join(model_notes_worker))
                        setattr(ridge_fit, "_used_ridge", True)
                        fit = ridge_fit
                except Exception:
                    model_notes_worker.append("ridge_only")
                    setattr(ridge_fit, "_model_note", ";".join(model_notes_worker))
                    setattr(ridge_fit, "_used_ridge", True)
                    fit = ridge_fit
            except Exception as e:
                fit = None
                fit_reason = f"ridge_exception:{type(e).__name__}:{e}"

        if fit is None or target_inversion not in fit.params:
            result_data = {
                "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
                "Beta": float('nan'), "OR": float('nan'), "P_Value": float('nan')
            }
            io.atomic_write_json(result_path, result_data)
            io.atomic_write_json(meta_path, {
                "kind": "phewas_result", "s_name": s_name, "category": category, "model": "Logit",
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": case_idx_fp, "created_at": datetime.utcnow().isoformat() + "Z",
            })
            what = f"AUTO_FIT_FAILED:{fit_reason}" if fit is None else "COEFFICIENT_MISSING_IN_FIT_PARAMS"
            print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta=nan | OR=nan | P=nan | {what}", flush=True)
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
        or_ci95_str = None
        if se is not None and np.isfinite(se) and se > 0.0:
            lo = float(np.exp(beta - 1.96 * se))
            hi = float(np.exp(beta + 1.96 * se))
            or_ci95_str = f"{lo:.3f},{hi:.3f}"

        suffix = f" | P_CAUSE={pval_reason}" if (not np.isfinite(pval) and pval_reason) else ""
        print(f"[Worker-{os.getpid()}] - [OK] {s_name:<40s} | N={n_total:,} | Cases={n_cases:,} | Beta={beta:+.3f} | OR={np.exp(beta):.3f} | P={pval:.2e}{suffix}", flush=True)

        result_data = {
            "Phenotype": s_name, "N_Total": n_total, "N_Cases": n_cases, "N_Controls": n_ctrls,
            "Beta": beta, "OR": float(np.exp(beta)), "P_Value": pval, "OR_CI95": or_ci95_str
        }
        io.atomic_write_json(result_path, result_data)
        io.atomic_write_json(meta_path, {
            "kind": "phewas_result", "s_name": s_name, "category": category, "model": "Logit",
            "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
            "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
            "target": target_inversion, "core_index_fp": _index_fingerprint(worker_core_df.index),
            "case_idx_fp": case_idx_fp, "created_at": datetime.utcnow().isoformat() + "Z",
        })

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
        category = task["category"]
        cdr_codename = task["cdr_codename"]
        target_inversion = task["target"]
        result_path = os.path.join(CTX["LRT_OVERALL_CACHE_DIR"], f"{s_name}.json")
        meta_path = result_path + ".meta.json"

        pheno_cache_path = os.path.join(CTX["CACHE_DIR"], f"pheno_{s_name}_{cdr_codename}.parquet")
        if not os.path.exists(pheno_cache_path):
            io.atomic_write_json(result_path, {
                "Phenotype": s_name, "P_LRT_Overall": float('nan'), "LRT_df_Overall": float('nan'),
                "LRT_Overall_Reason": "missing_case_cache"
            })
            io.atomic_write_json(meta_path, {
                "kind": "lrt_overall", "s_name": s_name, "category": category,
                "model_columns": list(worker_core_df.columns), "num_pcs": CTX["NUM_PCS"],
                "min_cases": CTX["MIN_CASES_FILTER"], "min_ctrls": CTX["MIN_CONTROLS_FILTER"],
                "target": target_inversion, "core_index_fp": _index_fingerprint(worker_core_df.index),
                "case_idx_fp": "", "created_at": datetime.utcnow().isoformat() + "Z"
            })
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

        finite_mask = np.isfinite(worker_core_df.to_numpy()).all(axis=1)
        allowed_mask = allowed_mask_by_cat.get(category, np.ones(len(core_index), dtype=bool))
        case_mask = np.zeros(len(core_index), dtype=bool)
        if case_idx.size > 0:
            case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask
        n_valid = int(valid_mask.sum())
        y = np.zeros(n_valid, dtype=np.int8)
        case_positions = np.nonzero(case_mask[valid_mask])[0]
        if case_positions.size > 0:
            y[case_positions] = 1

        pc_cols_local = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        base_cols = ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE']
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
                "case_idx_fp": case_fp, "created_at": datetime.utcnow().isoformat() + "Z"
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
                "case_idx_fp": case_fp, "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[LRT-Stage1-Worker-{os.getpid()}] {s_name} SKIP reason=target_constant", flush=True)
            return

        def _design_rank(X):
            try: return int(np.linalg.matrix_rank(X.values))
            except Exception: return int(np.linalg.matrix_rank(np.asarray(X)))

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
            rank_red = _design_rank(X_red)
            rank_full = _design_rank(X_full)
            df_lrt = int(max(0, rank_full - rank_red))
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
            "case_idx_fp": case_fp, "created_at": datetime.utcnow().isoformat() + "Z"
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
                "case_idx_fp": "", "created_at": datetime.utcnow().isoformat() + "Z"
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

        finite_mask = np.isfinite(worker_core_df.to_numpy()).all(axis=1)
        allowed_mask = allowed_mask_by_cat.get(category, np.ones(len(core_index), dtype=bool))
        case_mask = np.zeros(len(core_index), dtype=bool)
        if case_idx.size > 0: case_mask[case_idx] = True
        valid_mask = (allowed_mask | case_mask) & finite_mask
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
                "case_idx_fp": case_fp, "created_at": datetime.utcnow().isoformat() + "Z"
            })
            print(f"[Ancestry-Worker-{os.getpid()}] {s_name} SKIP reason=no_valid_rows_after_mask", flush=True)
            return

        pc_cols_local = [f"PC{i}" for i in range(1, CTX["NUM_PCS"] + 1)]
        base_cols = ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE']
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
                "case_idx_fp": case_fp, "created_at": datetime.utcnow().isoformat() + "Z"
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

        def _rank(X):
            try: return int(np.linalg.matrix_rank(X.values))
            except Exception: return int(np.linalg.matrix_rank(np.asarray(X)))

        if (fit_red is not None) and (fit_full is not None) and hasattr(fit_full, "llf") and hasattr(fit_red, "llf") and (fit_full.llf >= fit_red.llf):
            df_lrt = int(max(0, _rank(X_full) - _rank(X_red)))
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
            X_g = worker_core_df.loc[group_mask, ['const', target_inversion, 'sex'] + pc_cols_local + ['AGE']].astype(np.float64, copy=False)
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
            "case_idx_fp": case_fp, "created_at": datetime.utcnow().isoformat() + "Z"
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
