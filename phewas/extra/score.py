import os as _os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if _os.environ.get(_k) is None:
        _os.environ[_k] = "1"

# -------------------- Imports --------------------
import os, sys, json, hashlib, glob, warnings, time, math, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import statsmodels.api as sm

from . import iox  # atomic writers, cache utils

# ===================== HARD-CODED CONFIG =====================

CACHE_DIR = "./phewas_cache"
LOCK_DIR = os.path.join(CACHE_DIR, "locks")
CACHE_VERSION_TAG = iox.CACHE_VERSION_TAG
DOSAGES_TSV = "imputed_inversion_dosages.tsv"  # resolved upward from CWD

# Covariates from pipeline caches (AGE, AGE_sq, sex, PCs, ancestry labels)
USE_PIPELINE_COVARS = True
REMOVE_RELATED = True
INCLUDE_ANCESTRY = True  # ancestry one-hots (drop_first) as in run.py

PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

NUM_PCS = 10

# Phenotypes to score (binary disease status)
PHENOLIST = [
    "Ectopic_pregnancy",
    "Pityriasis",
    "Inflammation_of_the_heart_Carditis",
    "Chronic_atrial_fibrillation",
    "Disorders_of_bilirubin_excretion",
    "Atrioventricular_block_complete",
    "Psoriatic_arthropathy",
    "Congestive_heart_failure",
    "Chronic_kidney_disease",
    "Chronic_bronchitis",
    "Liver_cell_carcinoma",
    "Epilepsy",
    "Peripheral_vascular_disease",
    "Pulmonary_embolism",
    "Hypothyroidism",
]

# Split & modeling knobs
TEST_SIZE = 0.20
SEED = 42

# Elastic-net for PGS (inversions only, trained on TRAIN)
ALPHA = 0.50                 # elastic-net mixing (0=L2, 1=L1) — keep as-is (no extra sparsity pressure)
N_LAM = 30                   # reduced lambda path length (was 100)
LAMBDA_MIN_RATIO = 1e-2      # tightened grid tail (was 1e-3)
MAX_ITER = 2000
CLASS_WEIGHT = None          # FIX BIC/weights mismatch: use UNWEIGHTED fit for PGS/BIC consistency
NEAR_CONST_SD = 1e-6
BIC_EARLY_STOP = 3           # more aggressive early stop (was 5)
DEBIAS_REFIT = True          # unpenalized refit on selected support (slopes-only used for PGS)

# Test-time paired bootstrap for ΔAUC (Model1 - Model0)
BOOT_B = int(os.environ.get("SCORE_BOOT_B", "1000"))
BOOT_SEED = SEED

# Parallelization (default: cap to avoid oversubscription)
_default_workers = min(8, max(1, (os.cpu_count() or 4)))
N_WORKERS = int(os.environ.get("SCORE_N_JOBS", str(_default_workers)))
PRINT_LOCK = threading.Lock()

OUT_ROOT = os.path.join(CACHE_DIR, "scores_nested_pgs")

# ===================== PROGRESS & UTILS =====================

_ID_CANDIDATES = ("person_id", "SampleID", "sample_id", "research_id", "participant_id", "ID")

def _now(): return time.strftime("%H:%M:%S")

def _p(msg: str):
    with PRINT_LOCK:
        print(f"[{_now()}] {msg}", flush=True)

def _hash_cfg(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

def _find_upwards(pathname: str) -> str:
    if os.path.isabs(pathname): return pathname
    name = os.path.basename(pathname)
    cur = os.getcwd()
    while True:
        candidate = os.path.join(cur, name)
        if os.path.exists(candidate): return candidate
        parent = os.path.dirname(cur)
        if parent == cur: break
        cur = parent
    return pathname

def _summ(v, name):
    v = np.asarray(v, dtype=float)
    return (f"{name}[n={v.size}] mean={np.nanmean(v):.6g} sd={np.nanstd(v):.6g} "
            f"min={np.nanmin(v):.6g} max={np.nanmax(v):.6g} uniq~{len(np.unique(v))}")

def _bar(i: int, total: int, width: int = 24) -> str:
    # textual progress bar (not in-place to avoid thread clashes)
    frac = 0 if total <= 0 else max(0.0, min(1.0, i / total))
    n_full = int(round(frac * width))
    return "[" + ("#" * n_full).ljust(width, ".") + f"] {i}/{total} ({frac*100:5.1f}%)"

# ===================== DATA LOADING =====================

def _read_wide_tsv(path: str) -> pd.DataFrame:
    _p(f"[load/dosages] Reading TSV: {path}")
    df = pd.read_csv(path, sep="\t")
    id_col = next((c for c in df.columns if c in _ID_CANDIDATES), None)
    if id_col is None:
        raise RuntimeError(f"No ID column found in {path}. Expected one of {_ID_CANDIDATES}.")
    df = df.rename(columns={id_col: "person_id"})
    df["person_id"] = df["person_id"].astype(str)
    return df.set_index("person_id")

def _load_dosages() -> pd.DataFrame:
    t0 = time.time()
    path = _find_upwards(DOSAGES_TSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find '{DOSAGES_TSV}' from CWD or any parent.")
    _p(f"[load/dosages] Path: {path}")
    df = _read_wide_tsv(path)
    _p(f"[load/dosages] Raw: {df.shape[0]:,} samples x {df.shape[1]:,} inversions")

    # numeric coercion + drop constant/NA-only columns
    _p("[load/dosages] Coercing to numeric + dropping constant/NA-only columns...")
    t1 = time.time()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nunique = df.nunique(dropna=True)
    keep = nunique > 1
    kept = df.loc[:, keep]
    dropped = int((~keep).sum())
    _p(f"[load/dosages] Dropped {dropped:,} constant/all-NA inversion columns; kept {kept.shape[1]:,}. "
       f"(elapsed {time.time()-t1:.2f}s)")
    if kept.shape[1] == 0:
        raise RuntimeError("No variable inversion columns after filtering.")
    _p(f"[load/dosages] Final shape: {kept.shape[0]:,} x {kept.shape[1]:,} (elapsed {time.time()-t0:.2f}s)")
    return kept

def _resolve_env():
    cdr_dataset_id = os.environ.get("WORKSPACE_CDR")
    gcp_project = os.environ.get("GOOGLE_PROJECT")
    cdr_codename = cdr_dataset_id.split(".")[-1] if cdr_dataset_id else None
    return cdr_dataset_id, cdr_codename, gcp_project


def _pcs_cache_path(gcp_project):
    return os.path.join(
        CACHE_DIR,
        f"pcs_{NUM_PCS}_{iox.stable_hash((gcp_project or '', PCS_URI, NUM_PCS, CACHE_VERSION_TAG))}.parquet",
    )


def _sex_cache_path(gcp_project):
    return os.path.join(
        CACHE_DIR,
        f"genetic_sex_{iox.stable_hash((gcp_project or '', SEX_URI, CACHE_VERSION_TAG))}.parquet",
    )


def _ancestry_cache_path(gcp_project):
    return os.path.join(
        CACHE_DIR,
        f"ancestry_labels_{iox.stable_hash((gcp_project or '', PCS_URI, CACHE_VERSION_TAG))}.parquet",
    )

def _autodetect_cdr_codename() -> str | None:
    pats = sorted(glob.glob(os.path.join(CACHE_DIR, "demographics_*.parquet")))
    if not pats: return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    fn = os.path.basename(pats[0])
    try:
        return Path(fn).stem.split("demographics_")[-1]
    except Exception:
        return None

def _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project):
    from google.cloud import bigquery
    _p("[covars] Materializing missing caches via BigQuery (run.py-compatible)")
    bq_client = bigquery.Client(project=gcp_project)
    os.makedirs(LOCK_DIR, exist_ok=True)
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
        iox.load_demographics_with_stable_age,
        bq_client=bq_client,
        cdr_id=cdr_dataset_id,
        lock_dir=LOCK_DIR,
    )
    _ = iox.get_cached_or_generate(
        _pcs_cache_path(gcp_project),
        iox.load_pcs,
        gcp_project,
        PCS_URI,
        NUM_PCS,
        validate_num_pcs=NUM_PCS,
        lock_dir=LOCK_DIR,
    )
    _ = iox.get_cached_or_generate(
        _sex_cache_path(gcp_project),
        iox.load_genetic_sex,
        gcp_project,
        SEX_URI,
        lock_dir=LOCK_DIR,
    )
    _ = iox.get_cached_or_generate(
        _ancestry_cache_path(gcp_project),
        iox.load_ancestry_labels,
        gcp_project,
        LABELS_URI=PCS_URI,
        lock_dir=LOCK_DIR,
    )

def _load_pipeline_covars() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    cdr_dataset_id, cdr_codename, gcp_project = _resolve_env()
    if not cdr_codename:
        cdr_codename = _autodetect_cdr_codename()
        _p(f"[covars] Autodetected CDR codename: {cdr_codename}")

    demo_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet") if cdr_codename else None
    pcs_path  = _pcs_cache_path(gcp_project)
    sex_path  = _sex_cache_path(gcp_project)
    anc_path  = _ancestry_cache_path(gcp_project)

    needed = [demo_path, pcs_path, sex_path, anc_path]
    missing = [p for p in needed if (p is None or not os.path.exists(p))]
    if missing:
        _p(f"[covars] Missing cache(s): {missing}")
        if not all([cdr_dataset_id, cdr_codename, gcp_project]):
            raise RuntimeError("Covariate caches missing and WORKSPACE_CDR/GOOGLE_PROJECT not set.")
        _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project)

    _p("[covars] Loading cached demographics/PCs/sex/ancestry...")
    demographics_df = pd.read_parquet(os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"))[["AGE","AGE_sq"]]
    pc_df          = pd.read_parquet(pcs_path)[[f"PC{i}" for i in range(1, NUM_PCS+1)]]
    sex_df         = pd.read_parquet(sex_path)[["sex"]]
    ancestry_df    = pd.read_parquet(anc_path)[["ANCESTRY"]]

    for df in (demographics_df, pc_df, sex_df, ancestry_df):
        df.index = df.index.astype(str)

    base_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
    _p(f"[covars] Base covariates shape before related removal: {base_df.shape}")

    if REMOVE_RELATED:
        _, _, gcp_project = _resolve_env()
        if not gcp_project:
            raise RuntimeError("GOOGLE_PROJECT must be set to remove related individuals.")
        _p("    -> Loading list of related individuals to exclude...")
        related_ids = iox.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)
        n_before = len(base_df)
        base_df = base_df[~base_df.index.isin(related_ids)]
        _p(f"[covars] Removed related: {n_before - len(base_df):,} | Remaining: {len(base_df):,}")

    _p(f"[covars] Final base covariates shape: {base_df.shape} (elapsed {time.time()-t0:.2f}s)")
    return base_df, ancestry_df

def _find_pheno_cache(sanitized_name: str) -> str | None:
    pat = os.path.join(CACHE_DIR, f"pheno_{sanitized_name}_*.parquet")
    hits = sorted(glob.glob(pat))
    if not hits: return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def _build_phenotype_matrix(sample_index: pd.Index, phenos: list[str]) -> pd.DataFrame:
    _p(f"[labels] Building Y for {len(phenos)} phenotypes x {len(sample_index):,} samples")
    Y = pd.DataFrame(index=sample_index.copy())
    Y.index.name = "person_id"
    missing = []
    for i, name in enumerate(phenos, 1):
        if i % max(1, len(phenos)//10) == 0 or i == len(phenos):
            _p(f"[labels] Progress: {i}/{len(phenos)} ({i/len(phenos)*100:.1f}%)")
        f = _find_pheno_cache(name)
        if f is None:
            missing.append(name); Y[name] = 0; continue
        try:
            df = pd.read_parquet(f, columns=["is_case"])
        except Exception as e:
            warnings.warn(f"Failed to read cache for {name}: {e}")
            missing.append(name); Y[name] = 0; continue
        if df.index.name != "person_id":
            if "person_id" in df.columns:
                df = df.set_index("person_id")
            else:
                missing.append(name); Y[name] = 0; continue
        case_ids = df.index[df["is_case"].astype("int8") == 1].astype(str)
        y = pd.Series(0, index=sample_index, dtype=np.int8)
        y.loc[y.index.intersection(case_ids)] = 1
        Y[name] = y
    if missing:
        _p(f"[labels/warn] Missing phenotype caches for: {', '.join(missing)}")
    _p(f"[labels] Done. Shape: {Y.shape}")
    return Y

# ===================== PREP & ALIGNMENT =====================

def _align_core(X: pd.DataFrame, base_cov: pd.DataFrame, Y: pd.DataFrame):
    X.index = X.index.astype(str)
    base_cov.index = base_cov.index.astype(str)
    Y.index = Y.index.astype(str)

    _p(f"[align] |X|={len(X):,}, |cov|={len(base_cov):,}, |Y|={len(Y):,}")
    common = X.index.intersection(base_cov.index)
    _p(f"[align] |X ∩ cov| = {len(common):,}")
    common = common.intersection(Y.index)
    _p(f"[align] |(X ∩ cov) ∩ Y| = {len(common):,}")
    if len(common) == 0:
        _p("[align DEBUG] |X ∩ Y|        = {:,}".format(len(X.index.intersection(Y.index))))
        _p("[align DEBUG] |cov ∩ Y|      = {:,}".format(len(base_cov.index.intersection(Y.index))))
        raise RuntimeError("Empty intersection between dosages, covariates, and labels.")
    _p(f"[align] Final N = {len(common):,}")
    return X.reindex(common), base_cov.reindex(common), Y.reindex(common), common

def _train_test_split_stratified(y: pd.Series):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(sss.split(np.zeros(len(y)), y.values))
    return y.index[tr], y.index[te]

def _prep_X_standardize(X: pd.DataFrame, train_ids, test_ids, tag: str):
    _p(f"[{tag}] [prep/X] Split sizes: {len(train_ids):,}/{len(test_ids):,}")
    Xtr, Xte = X.loc[train_ids].copy(), X.loc[test_ids].copy()
    mu = Xtr.mean(axis=0)
    Xtr = Xtr.fillna(mu); Xte = Xte.fillna(mu)
    sd = Xtr.std(axis=0, ddof=0)
    keep = sd > NEAR_CONST_SD
    n_drop = int((~keep).sum())
    if n_drop:
        _p(f"[{tag}] [prep/X] Dropping {n_drop:,} near-constant features")
    Xtr = Xtr.loc[:, keep]; Xte = Xte.loc[:, keep]
    if Xtr.shape[1] == 0:
        raise RuntimeError(f"[{tag}] All inversion features near-constant on train.")
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0)
    eps = 1e-6
    Ztr = (Xtr - mu) / (sd + eps)
    Zte = (Xte - mu) / (sd + eps)
    _p(f"[{tag}] [prep/X] Z-scored P={Ztr.shape[1]:,}")
    return Ztr, Zte

def _build_covariates_splits(C_base: pd.DataFrame, ancestry_df: pd.DataFrame | None, train_ids, test_ids, tag: str):
    cov_tr = C_base.loc[train_ids].copy()
    cov_te = C_base.loc[test_ids].copy()

    age_mean = cov_tr['AGE'].mean()
    cov_tr['AGE_c'] = cov_tr['AGE'] - age_mean
    cov_tr['AGE_c_sq'] = cov_tr['AGE_c'] ** 2
    cov_te['AGE_c'] = cov_te['AGE'] - age_mean
    cov_te['AGE_c_sq'] = cov_te['AGE_c'] ** 2

    pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
    base_cols = ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
    cov_tr = cov_tr[base_cols]
    cov_te = cov_te[base_cols]

    if INCLUDE_ANCESTRY and (ancestry_df is not None) and (not ancestry_df.empty):
        anc_core = ancestry_df.reindex(C_base.index).copy()
        anc_core['ANCESTRY'] = anc_core['ANCESTRY'].astype(str).str.lower()
        anc_cat = pd.Categorical(anc_core['ANCESTRY'])
        A = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True, dtype=np.float32)
        A.index = A.index.astype(str)
        Atr = A.reindex(train_ids).fillna(0.0)
        Ate = A.reindex(test_ids).fillna(0.0)
        cov_tr = pd.concat([cov_tr, Atr], axis=1)
        cov_te = pd.concat([cov_te, Ate], axis=1)
        _p(f"[{tag}] [prep/covars] Added ancestry: +{Atr.shape[1]}")

    keep = cov_tr.nunique(dropna=True) > 1
    dropped = int((~keep).sum())
    if dropped: _p(f"[{tag}] [prep/covars] Dropped {dropped} constant covariate cols")
    cov_tr = cov_tr.loc[:, keep]
    cov_te = cov_te.loc[:, keep]
    _p(f"[{tag}] [prep/covars] Final: train {cov_tr.shape[1]} cols, test {cov_te.shape[1]} cols")
    return cov_tr, cov_te

# ===================== PGS TRAINING (inversions only) =====================

def _lambda_path(X: np.ndarray, y: np.ndarray, alpha: float, n: int, lmin_ratio: float) -> np.ndarray:
    # strong-rule style upper bound for lam_max; robustified
    p0 = y.mean()
    r0 = (y - p0)
    lam_max = np.abs(X.T @ r0).max() / (len(y) * max(alpha, 1e-6))
    lam_max = float(max(lam_max, 1e-3))
    lam_min = lam_max * lmin_ratio
    return np.geomspace(lam_max, lam_min, num=n)

def _fit_pgs_bic(Ztr: pd.DataFrame, ytr: pd.Series, tag: str):
    X = Ztr.values
    y = ytr.values.astype(float)
    lam_grid = _lambda_path(X, y, ALPHA, N_LAM, LAMBDA_MIN_RATIO)

    # Create a single estimator and warm-start along the path (large -> small lambda)
    lr = LogisticRegression(
        penalty="elasticnet", l1_ratio=ALPHA, solver="saga",
        C=1.0 / lam_grid[0], max_iter=MAX_ITER, tol=1e-4, class_weight=CLASS_WEIGHT,
        fit_intercept=True, n_jobs=1, warm_start=True
    )

    best = None
    inc = 0
    t0 = time.time()
    _p(f"[{tag}] [PGS] Lambda sweep (grid={len(lam_grid)}): P={Ztr.shape[1]:,}, N={len(ytr):,}")
    for i, lam in enumerate(lam_grid, 1):
        # warm-start by reusing previous coef_ / intercept_ if available
        try:
            lr.set_params(C=float(1.0 / lam))
        except Exception:
            lr = LogisticRegression(
                penalty="elasticnet", l1_ratio=ALPHA, solver="saga",
                C=float(1.0 / lam), max_iter=MAX_ITER, tol=1e-4, class_weight=CLASS_WEIGHT,
                fit_intercept=True, n_jobs=1, warm_start=True
            )
        lr.fit(X, ytr.values)

        # Progress bar print for this grid step
        _p(f"[{tag}] [PGS] {_bar(i, len(lam_grid))}")

        # Unweighted log-likelihood for BIC (consistent with CLASS_WEIGHT=None)
        p = lr.predict_proba(X)[:, 1]
        eps = 1e-15
        ll = float(np.sum(ytr*np.log(p+eps) + (1-ytr)*np.log(1-p+eps)))
        k = int(np.count_nonzero(lr.coef_[0])) + 1  # + intercept
        bic = -2.0*ll + k*np.log(len(ytr))

        if (best is None) or (bic < best["bic"]):
            best = {
                "lambda": float(lam),
                "C": float(1.0 / lam),
                "bic": float(bic),
                "intercept": float(lr.intercept_[0]),
                "coef": pd.Series(lr.coef_[0], index=Ztr.columns),
                "nonzero": int((lr.coef_[0] != 0).sum()),
                "grid_i": i
            }
            inc = 0
            _p(f"[{tag}] [PGS] New BEST @step {i}: BIC={best['bic']:.3f} nnz={best['nonzero']} C={best['C']:.4g}")
        else:
            inc += 1

        if inc >= BIC_EARLY_STOP:
            _p(f"[{tag}] [PGS] Early stop: BIC rose {BIC_EARLY_STOP}x | elapsed {time.time()-t0:.1f}s")
            break

    if best is None:
        raise RuntimeError(f"[{tag}] [PGS] No solution found on path.")

    _p(f"[{tag}] [PGS] BEST: step={best['grid_i']}/{len(lam_grid)} lambda={best['lambda']:.4g} "
       f"C={best['C']:.4g} nnz={best['nonzero']} BIC={best['bic']:.2f} | total {time.time()-t0:.1f}s")
    return best

def _pgs_scores(Ztr: pd.DataFrame, Zte: pd.DataFrame, ytr: pd.Series, tag: str):
    best = _fit_pgs_bic(Ztr, ytr, tag)
    coef = best["coef"].copy()

    # Build linear predictors from SLOPES ONLY (no intercept added to PGS)
    if DEBIAS_REFIT and best["nonzero"] > 0:
        sel = coef.index[coef != 0]
        _p(f"[{tag}] [PGS] Debias attempt with support={len(sel)} (slopes-only)")
        Xdeb_tr = Ztr[sel]
        Xdeb_te = Zte[sel]
        try:
            Xglm_tr = sm.add_constant(Xdeb_tr, has_constant='add')
            res = sm.GLM(ytr.values, Xglm_tr, family=sm.families.Binomial()).fit()
            beta = pd.Series(res.params, index=Xglm_tr.columns)
            beta_slopes = beta.drop(labels=["const"], errors="ignore")
            lin_tr = (Xdeb_tr.values @ beta_slopes.values)
            lin_te = (Xdeb_te.values @ beta_slopes.values)
            _p(f"[{tag}] [PGS] Debiased refit OK. slopes={len(beta_slopes)} "
               f"L1={float(np.sum(np.abs(beta_slopes.values))):.6g}")
        except Exception as e:
            _p(f"[{tag}] [PGS WARN] Debias failed ({e}); using penalized slopes.")
            lin_tr = (Ztr.values @ coef.values)
            lin_te = (Zte.values @ coef.values)
    else:
        lin_tr = (Ztr.values @ coef.values)
        lin_te = (Zte.values @ coef.values)

    _p(f"[{tag}] [PGS/RAW] " + _summ(lin_tr, "lin_tr") + " | " + _summ(lin_te, "lin_te"))

    # z-score on TRAIN
    mu = float(np.mean(lin_tr))
    sd = float(np.std(lin_tr))
    if not np.isfinite(sd) or sd < 1e-12:
        _p(f"[{tag}] [PGS/CHK] Train linear predictor ~zero variance (sd={sd:.3e}). No usable PGS.")
        return {
            "best": best,
            "pgs_tr": None,
            "pgs_te": None,
            "pgs_mu": mu,
            "pgs_sd": sd,
            "pgs_has_signal": False,
            "support_nonzero": int(best["nonzero"]),
        }

    pgs_tr = (lin_tr - mu) / sd
    pgs_te = (lin_te - mu) / sd
    _p(f"[{tag}] [PGS/Z] " + _summ(pgs_tr, "pgs_tr") + " | " + _summ(pgs_te, "pgs_te"))

    # quick diagnostics
    try:
        r = float(np.corrcoef(pgs_tr, ytr.values)[0,1])
    except Exception:
        r = np.nan
    try:
        auc_tr = float(roc_auc_score(ytr.values, pgs_tr))
    except Exception:
        auc_tr = np.nan
    _p(f"[{tag}] [PGS/DIAG] corr(pgs_tr,y_tr)={r if np.isfinite(r) else float('nan'):.4f} | "
       f"AUC(PGS-only,TRAIN)={auc_tr if np.isfinite(auc_tr) else float('nan'):.4f}")

    return {
        "best": best,
        "pgs_tr": pgs_tr,
        "pgs_te": pgs_te,
        "pgs_mu": mu,
        "pgs_sd": sd,
        "pgs_has_signal": True,
        "support_nonzero": int(best["nonzero"]),
    }

# ===================== NESTED MODEL TESTS =====================

def _fit_model0_model1_train(cov_tr: pd.DataFrame, ytr: pd.Series, pgs_tr: np.ndarray | None, tag: str):
    # Model 0: baseline covariates only (GLM Binomial for numerical stability)
    X0 = sm.add_constant(cov_tr, has_constant='add')
    _p(f"[{tag}] [TRAIN/M0] X0 shape={X0.shape} cols={list(X0.columns)[:6]}..."
       f" (+{len(X0.columns)-6 if len(X0.columns)>6 else 0})")
    m0 = sm.GLM(ytr.values, X0, family=sm.families.Binomial()).fit()
    _p(f"[{tag}] [TRAIN/M0] dev={getattr(m0, 'deviance', float('nan')):.6g} "
       f"llf={getattr(m0, 'llf', float('nan')):.6g}")

    if pgs_tr is None:
        _p(f"[{tag}] [TRAIN/M1] SKIP — PGS has no usable variance.")
        return {
            "m0": m0, "m1": None,
            "p_lrt": np.nan, "lrt_stat": np.nan,
            "beta_pgs": np.nan, "se_pgs": np.nan, "wald_z": np.nan, "p_wald_two": np.nan,
            "m0_cols": list(X0.columns)
        }

    # Model 1: baseline + PGS (aligned)
    X1 = X0.copy()
    X1["PGS"] = pd.Series(pgs_tr, index=X1.index)
    _p(f"[{tag}] [TRAIN/M1] X1 shape={X1.shape} (added PGS). " + _summ(X1['PGS'].values, "PGS_tr"))
    m1 = sm.GLM(ytr.values, X1, family=sm.families.Binomial()).fit()
    _p(f"[{tag}] [TRAIN/M1] dev={getattr(m1, 'deviance', float('nan')):.6g} "
       f"llf={getattr(m1, 'llf', float('nan')):.6g}")

    # LRT via deviance difference (df=1)
    dev0 = getattr(m0, "deviance", None)
    dev1 = getattr(m1, "deviance", None)
    if dev0 is not None and dev1 is not None:
        lrt = float(dev0 - dev1)
        p_lrt = float(1.0 - chi2.cdf(lrt, df=1))
    else:
        ll0 = getattr(m0, "llf", None)
        ll1 = getattr(m1, "llf", None)
        if ll0 is not None and ll1 is not None:
            lrt = float(2.0 * (ll1 - ll0))
            p_lrt = float(1.0 - chi2.cdf(lrt, df=1))
        else:
            lrt, p_lrt = np.nan, np.nan

    # Wald for PGS
    try:
        pgsi = list(X1.columns).index("PGS")
        beta_pgs = float(m1.params[pgsi])
        se_pgs = float(np.sqrt(m1.cov_params().iloc[pgsi, pgsi]))
        z = beta_pgs / se_pgs if se_pgs > 0 else np.nan
        p_wald_two = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    except Exception:
        beta_pgs = np.nan; se_pgs = np.nan; z = np.nan; p_wald_two = np.nan

    _p(f"[{tag}] [TRAIN/TESTS] LRT stat={lrt if np.isfinite(lrt) else float('nan'):.6g} "
       f"p={p_lrt if np.isfinite(p_lrt) else float('nan'):.3e} | "
       f"Wald z={z if np.isfinite(z) else float('nan'):.3f} "
       f"p={p_wald_two if np.isfinite(p_wald_two) else float('nan'):.3e}")

    return {
        "m0": m0, "m1": m1,
        "p_lrt": p_lrt, "lrt_stat": lrt,
        "beta_pgs": beta_pgs, "se_pgs": se_pgs, "wald_z": z, "p_wald_two": p_wald_two,
        "m0_cols": list(X0.columns)
    }

# ---- DeLong for paired AUCs (one-sided) ----
def _midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1
        ranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=float)
    out[sorted_idx] = ranks
    return out

def _fast_delong(y_true: np.ndarray, pred: np.ndarray):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(float)
    assert set(np.unique(y_true)) <= {0,1}
    pos = pred[y_true == 1]
    neg = pred[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, None, None
    all_scores = np.concatenate([pos, neg])
    r_all = _midrank(all_scores)
    r_pos = _midrank(pos)
    r_neg = _midrank(neg)
    auc = (np.sum(r_all[:m]) - m*(m+1)/2.0) / (m*n)
    v10 = (r_all[:m] - r_pos) / n
    v01 = 1.0 - (r_all[m:] - r_neg) / m
    return float(auc), v10, v01

def _delong_test_paired(y_true: np.ndarray, s0: np.ndarray, s1: np.ndarray, alternative="greater") -> float:
    a0, v10_0, v01_0 = _fast_delong(y_true, s0)
    a1, v10_1, v01_1 = _fast_delong(y_true, s1)
    if any(v is None for v in (v10_0, v01_0, v10_1, v01_1)) or np.isnan(a0) or np.isnan(a1):
        return np.nan
    m, n = len(v10_0), len(v01_0)
    cov_v10 = np.cov(np.vstack([v10_0, v10_1]), bias=False)
    cov_v01 = np.cov(np.vstack([v01_0, v01_1]), bias=False)
    s_00 = cov_v10[0,0]/m + cov_v01[0,0]/n
    s_11 = cov_v10[1,1]/m + cov_v01[1,1]/n
    s_01 = cov_v10[0,1]/m + cov_v01[0,1]/n
    var_diff = s_00 + s_11 - 2*s_01
    if var_diff <= 0 or not np.isfinite(var_diff):
        return np.nan
    z = (a1 - a0) / math.sqrt(var_diff)
    if alternative == "greater":
        p = float(1.0 - norm.cdf(z))
    elif alternative == "less":
        p = float(norm.cdf(z))
    else:
        p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return p

def _paired_bootstrap_auc(y, s0, s1, B=1000, seed=SEED, tag=""):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    s0 = np.asarray(s0).astype(float)
    s1 = np.asarray(s1).astype(float)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    m, n = len(idx_pos), len(idx_neg)
    if m == 0 or n == 0:
        return np.nan, (np.nan, np.nan), np.nan
    deltas = np.empty(B, dtype=float)
    count_gt = 0
    t0 = time.time()
    step_print = max(1, B // 20)  # print ~5% steps
    for b in range(B):
        rp = rng.choice(idx_pos, size=m, replace=True)
        rn = rng.choice(idx_neg, size=n, replace=True)
        sel = np.concatenate([rp, rn])
        yb = y[sel]
        d0 = roc_auc_score(yb, s0[sel])
        d1 = roc_auc_score(yb, s1[sel])
        d = d1 - d0
        deltas[b] = d
        if d > 0: count_gt += 1
        if (b+1) % step_print == 0 or (b+1) == B:
            elapsed = time.time() - t0
            pct = (b+1)/B*100
            eta = elapsed/(b+1)*(B-(b+1))
            _p(f"[{tag}] [TEST/bootstrap] {_bar(b+1, B)} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")
    prob = float(count_gt / B)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return prob, (lo, hi), float(np.mean(deltas))

# ===================== EVALUATION (TEST) =====================

def _evaluate_test(m0, m1, cov_te: pd.DataFrame, pgs_te: np.ndarray | None, yte: pd.Series, tag: str):
    X0_te = sm.add_constant(cov_te, has_constant='add')
    _p(f"[{tag}] [TEST] X0_te shape={X0_te.shape}")

    # Predict with Model 0
    p0 = m0.predict(X0_te)
    _p(f"[{tag}] [TEST] p0 summary: " + _summ(p0, "p0"))

    # If Model 1 was skipped
    if (m1 is None) or (pgs_te is None):
        auc0 = float(roc_auc_score(yte.values, p0))
        _p(f"[{tag}] [TEST] Model 1 skipped (no PGS). AUC_M0={auc0:.4f}")
        return {
            "AUC_M0": auc0,
            "AUC_M1": np.nan,
            "DeltaAUC": np.nan,
            "p_DeLong_one_sided": np.nan,
            "Prob_DeltaAUC_gt0_boot": np.nan,
            "DeltaAUC_CI95_lo": np.nan,
            "DeltaAUC_CI95_hi": np.nan,
            "DeltaAUC_boot_mean": np.nan
        }

    # Predict with Model 1 (insert aligned PGS)
    X1_te = X0_te.copy()
    X1_te["PGS"] = pd.Series(pgs_te, index=X1_te.index)
    p1 = m1.predict(X1_te)
    _p(f"[{tag}] [TEST] p1 summary: " + _summ(p1, "p1"))

    same = np.allclose(p0, p1)
    _p(f"[{tag}] [TEST] p0==p1 allclose? {bool(same)}")

    auc0 = float(roc_auc_score(yte.values, p0))
    auc1 = float(roc_auc_score(yte.values, p1))
    d_auc = auc1 - auc0

    # DeLong one-sided: H0 AUC1 <= AUC0 vs H1 AUC1 > AUC0
    p_delong = _delong_test_paired(yte.values, p0, p1, alternative="greater")

    # Paired bootstrap (probability view & CI)
    prob_gt, (ci_lo, ci_hi), mean_delta = _paired_bootstrap_auc(
        yte.values, p0, p1, B=BOOT_B, seed=BOOT_SEED, tag=tag
    )

    _p(f"[{tag}] [TEST] AUC_M0={auc0:.4f}  AUC_M1={auc1:.4f}  ΔAUC={d_auc:+.4f}  "
       f"p_DeLong={(p_delong if np.isfinite(p_delong) else float('nan')):.3e}  "
       f"P(ΔAUC>0)={prob_gt if np.isfinite(prob_gt) else float('nan'):.3f}  "
       f"CI_95%=[{ci_lo if np.isfinite(ci_lo) else float('nan'):+.4f},"
       f"{ci_hi if np.isfinite(ci_hi) else float('nan'):+.4f}]")

    return {
        "AUC_M0": auc0,
        "AUC_M1": auc1,
        "DeltaAUC": d_auc,
        "p_DeLong_one_sided": p_delong,
        "Prob_DeltaAUC_gt0_boot": prob_gt,
        "DeltaAUC_CI95_lo": ci_lo,
        "DeltaAUC_CI95_hi": ci_hi,
        "DeltaAUC_boot_mean": mean_delta,
    }

# ===================== PER-PHENOTYPE PIPE =====================

def _run_one(ph: str, X: pd.DataFrame, Y: pd.DataFrame, C_base: pd.DataFrame, ancestry_df: pd.DataFrame | None):
    t0 = time.time()
    y = Y[ph].astype(int)
    pos = int(y.sum()); neg = int((1 - y).sum())
    _p(f"[{ph}] [start] N={len(y):,} | Cases={pos:,} Controls={neg:,}")
    if y.nunique() < 2:
        _p(f"[{ph}] [skip] Only one class present.")
        return None

    train_ids, test_ids = _train_test_split_stratified(y)
    _p(f"[{ph}] [split] Train={len(train_ids):,}  Test={len(test_ids):,}")

    cfg = {
        "phenotype": ph,
        "alpha": ALPHA, "test_size": TEST_SIZE, "seed": SEED,
        "use_covars": bool(USE_PIPELINE_COVARS),
        "include_ancestry": bool(INCLUDE_ANCESTRY), "remove_related": bool(REMOVE_RELATED),
        "N_LAM": N_LAM, "LAMBDA_MIN_RATIO": LAMBDA_MIN_RATIO, "BIC_EARLY_STOP": BIC_EARLY_STOP,
        "DEBIAS_REFIT": DEBIAS_REFIT, "BOOT_B": BOOT_B
    }
    key = _hash_cfg(cfg)
    outdir = os.path.join(OUT_ROOT, f"score_{ph}_{key}")
    os.makedirs(outdir, exist_ok=True)

    weights_pq = os.path.join(outdir, "pgs_weights.parquet")
    metrics_js = os.path.join(outdir, "metrics.json")
    if os.path.exists(weights_pq) and os.path.exists(metrics_js):
        try:
            mets = iox.read_meta_json(metrics_js)
            _p(f"[{ph}] [cache] Using cached results.")
            return {"phenotype": ph, **(mets or {})}
        except Exception:
            pass

    # Standardize inversions (PGS design)
    Ztr, Zte = _prep_X_standardize(X, train_ids, test_ids, ph)

    # Build covariates (Model 0/1 bases)
    cov_tr, cov_te = _build_covariates_splits(C_base, ancestry_df, train_ids, test_ids, ph)

    # Train PGS on TRAIN (inversions only), produce TRAIN/TEST PGS (z-scored on TRAIN)
    pgs = _pgs_scores(Ztr, Zte, y.loc[train_ids], ph)

    # Fit nested models on TRAIN
    fit = _fit_model0_model1_train(cov_tr, y.loc[train_ids], pgs["pgs_tr"] if pgs["pgs_has_signal"] else None, ph)

    # Evaluate discrimination on TEST (ΔAUC, DeLong, bootstrap)
    mets_test = _evaluate_test(
        fit["m0"], fit["m1"], cov_te, pgs["pgs_te"] if pgs["pgs_has_signal"] else None, y.loc[test_ids], ph
    )

    # Save artifacts
    coef = pgs["best"]["coef"]
    weights_df = pd.DataFrame({"feature": coef.index, "beta": coef.values})
    iox.atomic_write_parquet(weights_pq, weights_df)
    iox.atomic_write_json(os.path.join(outdir, "pgs_model.json"), {
        "alpha": ALPHA, "lambda": pgs["best"]["lambda"], "C": pgs["best"]["C"],
        "intercept_penalized": pgs["best"]["intercept"],  # for record; not used in PGS
        "nonzero": pgs["best"]["nonzero"], "bic": pgs["best"]["bic"],
        "pgs_mu_train": pgs["pgs_mu"], "pgs_sd_train": pgs["pgs_sd"],
        "pgs_has_signal": bool(pgs["pgs_has_signal"])
    })

    if pgs["pgs_has_signal"]:
        test_scores = pd.DataFrame(
            {"person_id": test_ids, "PGS": pgs["pgs_te"], "Y": y.loc[test_ids].values}
        ).set_index("person_id")
        iox.atomic_write_parquet(os.path.join(outdir, "test_pgs.parquet"), test_scores)

    # Consolidate metrics
    out = {
        "phenotype": ph,
        "TRAIN_LRT_p": fit["p_lrt"], "TRAIN_LRT_stat": fit["lrt_stat"],
        "TRAIN_Wald_beta_PGS": fit["beta_pgs"], "TRAIN_Wald_se_PGS": fit["se_pgs"],
        "TRAIN_Wald_z": fit["wald_z"], "TRAIN_Wald_p_two_sided": fit["p_wald_two"],
        **mets_test,
        "PGS_nonzero": pgs["best"]["nonzero"],
        "PGS_support_signal": bool(pgs["pgs_has_signal"])
    }
    iox.atomic_write_json(metrics_js, out)
    _p(f"[{ph}] [done] nnz={pgs['best']['nonzero']} | "
       f"LRT p={fit['p_lrt'] if np.isfinite(fit['p_lrt']) else float('nan'):.3e} | "
       f"ΔAUC={out['DeltaAUC'] if np.isfinite(out['DeltaAUC']) else float('nan'):+.4f} | "
       f"p_DeLong={out['p_DeLong_one_sided'] if np.isfinite(out['p_DeLong_one_sided']) else float('nan'):.3e} | "
       f"wall {time.time()-t0:.1f}s")

    return out

# ===================== MAIN =====================

def main():
    t_all = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    _p(f"[init] N_WORKERS={N_WORKERS}  ALPHA={ALPHA}  N_LAM={N_LAM}  TEST_SIZE={TEST_SIZE}  BOOT_B={BOOT_B}")
    _p("[init] Threading guards: "
       f"OMP={os.environ.get('OMP_NUM_THREADS')} "
       f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} "
       f"MKL={os.environ.get('MKL_NUM_THREADS')} "
       f"VECLIB={os.environ.get('VECLIB_MAXIMUM_THREADS')} "
       f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}")

    # 1) Load inversions
    X = _load_dosages()

    # 2) Load pipeline covariates
    if not USE_PIPELINE_COVARS:
        raise RuntimeError("This script expects pipeline covariates (Model 0). Set USE_PIPELINE_COVARS=True.")
    base_cov, ancestry_df = _load_pipeline_covars()

    # 3) Labels
    Y = _build_phenotype_matrix(X.index, PHENOLIST)

    # 4) Align
    X, base_cov, Y, core_ids = _align_core(X, base_cov, Y)

    # 5) Select runnable phenotypes (two classes)
    run_list = []
    for ph in PHENOLIST:
        if Y[ph].astype(int).nunique() < 2:
            _p(f"[plan] {ph}: SKIP (only one class).")
        else:
            run_list.append(ph)
    _p(f"[plan] Runnable phenotypes: {len(run_list)}/{len(PHENOLIST)}")

    # 6) Parallel execution
    results = []
    total = len(run_list)
    if total == 0:
        _p("[run] Nothing to do.")
    else:
        _p("[run] Starting parallel per-phenotype execution...")
        done = 0
        start = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            fut_to_ph = {ex.submit(_run_one, ph, X, Y, base_cov, ancestry_df): ph for ph in run_list}
            for fut in as_completed(fut_to_ph):
                ph = fut_to_ph[fut]
                try:
                    res = fut.result()
                    if res: results.append(res)
                except Exception as e:
                    _p(f"[{ph}] [FAIL] {e}")
                finally:
                    done += 1
                    pct = done / total * 100.0
                    elapsed = time.time() - start
                    rem = elapsed/done*(total-done) if done > 0 else float('nan')
                    _p(f"[progress] {_bar(done, total)} | elapsed {elapsed:.1f}s | ETA ~{rem:.1f}s")

    # 7) Summary
    if results:
        summary = pd.DataFrame(results)
        out_csv = os.path.join(OUT_ROOT, "summary_metrics.csv")
        if os.path.exists(out_csv):
            try:
                old = pd.read_csv(out_csv)
                keep = old[~old["phenotype"].isin(summary["phenotype"])]
                summary = pd.concat([keep, summary], axis=0, ignore_index=True)
            except Exception:
                pass
        summary.to_csv(out_csv, index=False)
        _p(f"[summary] Wrote {out_csv} ({summary.shape[0]} rows)")
    else:
        _p("[summary] No results to write.")

    _p(f"[done] Total wall time {time.time()-t_all:.2f}s")

if __name__ == "__main__":
    main()
