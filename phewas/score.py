import os, sys, json, hashlib, glob, warnings, time, math, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import norm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import iox  # atomic writers, cache utils

# ===================== HARD-CODED CONFIG =====================

CACHE_DIR = "./phewas_cache"
DOSAGES_TSV = "imputed_inversion_dosages.tsv"  # resolved via _find_upwards()

# Use pipeline covariates / caches (AGE, AGE_sq, sex, PCs, ancestry one-hots). No external covariate TSV.
USE_PIPELINE_COVARS = True
REMOVE_RELATED = True
INCLUDE_ANCESTRY = True  # added AFTER alignment; never used to gate sample intersection

# Data sources (only used if caches are missing and we must materialize them)
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

NUM_PCS = 10

# Fixed phenotypes (sanitized names that match cached parquet files pheno_<name>_*.parquet)
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

# Modeling knobs
TEST_SIZE = 0.20
SEED = 42
ALPHA = 0.50                # elastic-net mixing (0=L2, 1=L1) — fixed, no CV
N_LAM_MAX = 60              # hard cap for path length (upper bound); adaptive early-stop will cut this
LAMBDA_MIN_RATIO = 1e-3     # min(lambda) = lambda_max * ratio
NEAR_CONST_SD = 1e-6
MAX_ITER = 2000
CLASS_WEIGHT = "balanced"

# BIC early-stopping along the path
BIC_WORSE_STOP = 4          # stop after this many successive BIC increases

# Bootstrap for ΔAUC
BOOTSTRAP_B = 1000          # paired stratified bootstrap resamples for ΔAUC prob & CI
BOOT_PRINT_EVERY = 100

# Parallelization
N_WORKERS = int(os.environ.get("SCORE_N_JOBS", str(max(1, (os.cpu_count() or 4) - 0))))  # default: all cores
PRINT_LOCK = threading.Lock()

OUT_ROOT = os.path.join(CACHE_DIR, "scores_elasticnet")

# ===================== UTILS & PROGRESS =====================

_ID_CANDIDATES = ("person_id","SampleID","sample_id","research_id","participant_id","ID")

def _now():
    return time.strftime("%H:%M:%S")

def _p(msg: str):
    with PRINT_LOCK:
        print(f"[{_now()}] {msg}", flush=True)

def _hash_cfg(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

def _find_upwards(pathname: str) -> str:
    if os.path.isabs(pathname):
        return pathname
    name = os.path.basename(pathname)
    cur = os.getcwd()
    while True:
        candidate = os.path.join(cur, name)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return pathname  # fallback (will error later if missing)

# ===================== LOADING DATA =====================

def _read_wide_tsv(path: str) -> pd.DataFrame:
    _p(f"[dosages] Reading TSV: {path}")
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
    _p(f"[dosages] Resolved path: {path}")
    df = _read_wide_tsv(path)
    _p(f"[dosages] Raw shape: {df.shape[0]:,} samples x {df.shape[1]:,} inversions (pre-numeric)")
    # coerce everything except index to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nunique = df.nunique(dropna=True)
    keep = nunique > 1
    kept = df.loc[:, keep]
    dropped = int((~keep).sum())
    _p(f"[dosages] Dropped {dropped:,} constant/all-NA inversion columns. Kept {kept.shape[1]:,}.")
    if kept.shape[1] == 0:
        raise RuntimeError("No variable inversion columns after filtering (all were NA/constant).")
    _p(f"[dosages] Final shape: {kept.shape[0]:,} x {kept.shape[1]:,} (elapsed {time.time()-t0:.2f}s)")
    return kept

def _resolve_env():
    cdr_dataset_id = os.environ.get("WORKSPACE_CDR")
    gcp_project = os.environ.get("GOOGLE_PROJECT")
    cdr_codename = cdr_dataset_id.split(".")[-1] if cdr_dataset_id else None
    return cdr_dataset_id, cdr_codename, gcp_project

def _autodetect_cdr_codename() -> str | None:
    pats = sorted(glob.glob(os.path.join(CACHE_DIR, "demographics_*.parquet")))
    if not pats:
        return None
    pats.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    fn = os.path.basename(pats[0])
    try:
        stem = Path(fn).stem  # demographics_{cdr}
        return stem.split("demographics_")[-1]
    except Exception:
        return None

def _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project):
    from google.cloud import bigquery
    _p("[covars] Materializing missing caches via BigQuery (run.py-compatible)")
    bq_client = bigquery.Client(project=gcp_project)

    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
        iox.load_demographics_with_stable_age, bq_client=bq_client, cdr_id=cdr_dataset_id
    )
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet"),
        iox.load_pcs, gcp_project, PCS_URI, NUM_PCS, validate_num_pcs=NUM_PCS
    )
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, "genetic_sex.parquet"),
        iox.load_genetic_sex, gcp_project, SEX_URI
    )
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, "ancestry_labels.parquet"),
        iox.load_ancestry_labels, gcp_project, LABELS_URI=PCS_URI
    )

def _load_pipeline_covars() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    cdr_dataset_id, cdr_codename, gcp_project = _resolve_env()
    if not cdr_codename:
        cdr_codename = _autodetect_cdr_codename()
        _p(f"[covars] Autodetected CDR codename: {cdr_codename}")

    demo_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet") if cdr_codename else None
    pcs_path  = os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet")
    sex_path  = os.path.join(CACHE_DIR, "genetic_sex.parquet")
    anc_path  = os.path.join(CACHE_DIR, "ancestry_labels.parquet")

    needed = [demo_path, pcs_path, sex_path, anc_path]
    missing = [p for p in needed if (p is None or not os.path.exists(p))]
    if missing:
        _p(f"[covars] Missing cache(s): {missing}")
        if not all([cdr_dataset_id, cdr_codename, gcp_project]):
            raise RuntimeError("Covariate caches missing and WORKSPACE_CDR/GOOGLE_PROJECT not set to materialize.")
        _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project)

    _p("[covars] Loading cached demographics/PCs/sex/ancestry...")
    demographics_df = pd.read_parquet(os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"))[["AGE","AGE_sq"]]
    pc_df          = pd.read_parquet(os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet"))[[f"PC{i}" for i in range(1, NUM_PCS+1)]]
    sex_df         = pd.read_parquet(os.path.join(CACHE_DIR, "genetic_sex.parquet"))[["sex"]]
    ancestry_df    = pd.read_parquet(os.path.join(CACHE_DIR, "ancestry_labels.parquet"))[["ANCESTRY"]]

    for df in (demographics_df, pc_df, sex_df, ancestry_df):
        df.index = df.index.astype(str)

    base_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
    _p(f"[covars] Base covariates before related removal: {base_df.shape}")

    if REMOVE_RELATED:
        _, _, gcp_project = _resolve_env()
        if not gcp_project:
            raise RuntimeError("GOOGLE_PROJECT must be set to remove related individuals.")
        related_ids = iox.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)
        n_before = len(base_df)
        base_df = base_df[~base_df.index.isin(related_ids)]
        _p(f"[covars] Removed related: {n_before - len(base_df):,} | Remaining: {len(base_df):,}")

    _p(f"[covars] Final base covariates shape: {base_df.shape} (elapsed {time.time()-t0:.2f}s)")
    return base_df, ancestry_df

def _find_pheno_cache(sanitized_name: str) -> str | None:
    pat = os.path.join(CACHE_DIR, f"pheno_{sanitized_name}_*.parquet")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def _build_phenotype_matrix(sample_index: pd.Index, phenos: list[str]) -> pd.DataFrame:
    _p(f"[labels] Building phenotype matrix for {len(phenos)} phenotypes vs {len(sample_index):,} samples...")
    Y = pd.DataFrame(index=sample_index.copy())
    Y.index.name = "person_id"
    missing = []
    for i, name in enumerate(phenos, 1):
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
        if (i % max(1, len(phenos)//5) == 0) or (i == len(phenos)):
            _p(f"[labels] Progress {i}/{len(phenos)} ({i/len(phenos)*100:.1f}%)")
    if missing:
        _p(f"[labels/warn] Missing phenotype caches for: {', '.join(missing)}")
    _p(f"[labels] Done. Label matrix shape: {Y.shape}")
    return Y

# ===================== ALIGNMENT & SPLIT =====================

def _align_core(X: pd.DataFrame, base_cov: pd.DataFrame, Y: pd.DataFrame):
    X.index = X.index.astype(str)
    base_cov.index = base_cov.index.astype(str)
    Y.index = Y.index.astype(str)

    _p(f"[align] X samples={len(X):,}, cov samples={len(base_cov):,}, label samples={len(Y):,}")
    common = X.index.intersection(base_cov.index)
    _p(f"[align] |X ∩ cov| = {len(common):,}")
    common = common.intersection(Y.index)
    _p(f"[align] |(X ∩ cov) ∩ Y| = {len(common):,}")
    if len(common) == 0:
        _p("[align DEBUG] Sizes: X={:,} cov={:,} Y={:,}".format(len(X), len(base_cov), len(Y)))
        _p("[align DEBUG] |X ∩ Y|   = {:,}".format(len(X.index.intersection(Y.index))))
        _p("[align DEBUG] |cov ∩ Y| = {:,}".format(len(base_cov.index.intersection(Y.index))))
        raise RuntimeError("Empty intersection between dosages, covariates, and phenotype labels.")

    _p(f"[align] Final aligned N = {len(common):,}")
    return X.reindex(common), base_cov.reindex(common), Y.reindex(common), common

def _load_or_make_global_split(Y: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """
    One global 80/20 split shared across phenotypes.
    Stratify on 'any_case' (>=1 among selected phenos) to stabilize case fraction across split.
    Persist and reuse.
    """
    tr_path = os.path.join(CACHE_DIR, "global_train_ids.parquet")
    te_path = os.path.join(CACHE_DIR, "global_test_ids.parquet")
    if os.path.exists(tr_path) and os.path.exists(te_path):
        train_ids = pd.read_parquet(tr_path, columns=["person_id"]).index.astype(str)
        test_ids  = pd.read_parquet(te_path, columns=["person_id"]).index.astype(str)
        _p(f"[split] Loaded existing global split: Train={len(train_ids):,} Test={len(test_ids):,}")
        return train_ids, test_ids

    _p("[split] Creating new global 80/20 split (stratified on ANY-case among phenotypes)...")
    y_any = (Y.sum(axis=1) > 0).astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_idx, te_idx = next(sss.split(np.zeros(len(y_any)), y_any.values))
    train_ids = Y.index[tr_idx]
    test_ids  = Y.index[te_idx]

    df_tr = pd.DataFrame(index=train_ids); df_tr.index.name = "person_id"
    df_te = pd.DataFrame(index=test_ids);  df_te.index.name  = "person_id"
    os.makedirs(CACHE_DIR, exist_ok=True)
    iox.atomic_write_parquet(tr_path, df_tr)
    iox.atomic_write_parquet(te_path, df_te)
    _p(f"[split] Saved global split: Train={len(train_ids):,} Test={len(test_ids):,}")
    return train_ids, test_ids

# ===================== PREP MATRICES (ONCE) =====================

def _zscore_inversions_once(X: pd.DataFrame, train_ids, test_ids):
    _p(f"[prep/X] Z-scoring inversions once on global TRAIN ({len(train_ids):,})...")
    Xtr, Xte = X.loc[train_ids].copy(), X.loc[test_ids].copy()
    mu = Xtr.mean(axis=0)
    Xtr = Xtr.fillna(mu); Xte = Xte.fillna(mu)
    sd = Xtr.std(axis=0, ddof=0)
    keep = sd > NEAR_CONST_SD
    n_drop = int((~keep).sum())
    if n_drop:
        _p(f"[prep/X] Dropping {n_drop:,} near-constant features from TRAIN")
    Xtr = Xtr.loc[:, keep]; Xte = Xte.loc[:, keep]
    if Xtr.shape[1] == 0:
        raise RuntimeError("All inversion features were near-constant in training set.")
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0)
    eps = 1e-6
    Ztr = (Xtr - mu) / (sd + eps)
    Zte = (Xte - mu) / (sd + eps)
    _p(f"[prep/X] Z-scored features: {Ztr.shape[1]:,}")
    # Persist for reuse
    os.makedirs(OUT_ROOT, exist_ok=True)
    iox.atomic_write_parquet(os.path.join(OUT_ROOT, "Z_train.parquet"), Ztr)
    iox.atomic_write_parquet(os.path.join(OUT_ROOT, "Z_test.parquet"), Zte)
    iox.atomic_write_json(os.path.join(OUT_ROOT, "scalers.json"), {"mu": mu.to_dict(), "sd": sd.to_dict()})
    return Ztr, Zte

def _prep_covars_once(C_base: pd.DataFrame, ancestry_df: pd.DataFrame | None, train_ids, test_ids):
    _p(f"[prep/covars] Preparing covariates once on global split...")
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
        # Build dummies based on TRAIN categories to avoid leakage
        anc_core = ancestry_df.reindex(C_base.index).copy()
        anc_core['ANCESTRY'] = anc_core['ANCESTRY'].astype(str).str.lower()
        anc_cat_all = pd.Categorical(anc_core['ANCESTRY'])
        A_all = pd.get_dummies(anc_cat_all, prefix='ANC', drop_first=True, dtype=np.float32)
        A_all.index = A_all.index.astype(str)
        Atr = A_all.reindex(train_ids).fillna(0.0)
        Ate = A_all.reindex(test_ids).fillna(0.0)
        cov_tr = pd.concat([cov_tr, Atr], axis=1)
        cov_te = pd.concat([cov_te, Ate], axis=1)
        _p(f"[prep/covars] Added ancestry dummies: +{Atr.shape[1]} cols")

    keep = cov_tr.nunique(dropna=True) > 1
    dropped = int((~keep).sum())
    if dropped:
        _p(f"[prep/covars] Dropped {dropped} constant covariate columns post-split")
    cov_tr = cov_tr.loc[:, keep]
    cov_te = cov_te.loc[:, keep]
    _p(f"[prep/covars] Final covars: train {cov_tr.shape[1]} cols, test {cov_te.shape[1]} cols")
    # Persist for reuse
    iox.atomic_write_parquet(os.path.join(OUT_ROOT, "C_train.parquet"), cov_tr)
    iox.atomic_write_parquet(os.path.join(OUT_ROOT, "C_test.parquet"), cov_te)
    return cov_tr, cov_te

# ===================== LAMBDA PATH + BIC SELECTION (NO CV) =====================

def _lambda_path(X: np.ndarray, y: np.ndarray, alpha: float, n: int, lmin_ratio: float) -> np.ndarray:
    # strong-rule style upper bound
    p0 = y.mean()
    r0 = (y - p0)
    lam_max = np.abs(X.T @ r0).max() / (len(y) * max(alpha, 1e-6))
    lam_max = float(max(lam_max, 1e-3))
    lam_min = lam_max * lmin_ratio
    return np.geomspace(lam_max, lam_min, num=n)

def _fit_bic_path(Ztr: pd.DataFrame, ytr: pd.Series, alpha: float, ph_name: str):
    """
    Fit elastic-net logistic on inversions only. Choose lambda via BIC with early-stop on worsening.
    Returns dict(best) with keys: lambda, C, bic, intercept, coef (Series), nonzero
    """
    X = Ztr.values
    y = ytr.values.astype(float)
    feats = Ztr.columns.tolist()
    lam_grid = _lambda_path(X, y, alpha, N_LAM_MAX, LAMBDA_MIN_RATIO)
    _p(f"[{ph_name}] [fit] Lambda path length (max): {len(lam_grid)} | features={len(feats):,} | n={len(y):,}")

    best = None
    worse_count = 0
    t0 = time.time()

    # single estimator reused with warm_start
    lr = LogisticRegression(
        penalty="elasticnet", l1_ratio=alpha, solver="saga",
        C=1.0/lam_grid[0], max_iter=MAX_ITER, tol=1e-4, class_weight=CLASS_WEIGHT,
        fit_intercept=True, n_jobs=1, warm_start=True
    )
    last_C = None

    for i, lam in enumerate(lam_grid, 1):
        C_inv = float(1.0 / lam)
        if last_C is None:
            lr.set_params(C=C_inv)
        else:
            lr.set_params(C=C_inv)
        last_C = C_inv

        lr.fit(X, y)
        # train likelihood
        p = lr.predict_proba(X)[:, 1]
        eps = 1e-15
        ll = float(np.sum(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)))
        k = int(np.count_nonzero(lr.coef_[0])) + 1  # intercept
        bic = -2.0*ll + k*np.log(len(y))

        if (best is None) or (bic < best["bic"]):
            best = {
                "lambda": float(lam),
                "C": C_inv,
                "bic": float(bic),
                "intercept": float(lr.intercept_[0]),
                "coef": pd.Series(lr.coef_[0], index=feats),
                "nonzero": int((lr.coef_[0] != 0).sum()),
            }
            worse_count = 0
        else:
            worse_count += 1

        if (i % max(1, len(lam_grid)//10) == 0) or (i == len(lam_grid)):
            _p(f"[{ph_name}] [fit] λ step {i}/{len(lam_grid)} ({i/len(lam_grid)*100:.1f}%) | "
               f"nnz={best['nonzero']} | best BIC={best['bic']:.2f}")

        if worse_count >= BIC_WORSE_STOP:
            _p(f"[{ph_name}] [fit] Early stop after {BIC_WORSE_STOP} successive worse BICs at step {i}.")
            break

    _p(f"[{ph_name}] [fit] Done path in {time.time()-t0:.2f}s | best nnz={best['nonzero']} | best C={best['C']:.3g}")
    return best

def _refit_debias_inversion(Ztr: pd.DataFrame, ytr: pd.Series, best_coef: pd.Series, best_intercept: float, ph_name: str):
    """Optional de-bias: refit unpenalized logistic on the selected support (if any)."""
    active = best_coef[best_coef != 0.0]
    if active.empty:
        _p(f"[{ph_name}] [debias] No active features — skip debias refit.")
        return best_intercept, best_coef  # nothing to debias
    cols = list(active.index)
    X = Ztr[cols].values
    y = ytr.values
    lr = LogisticRegression(penalty='none', solver='lbfgs', max_iter=2000, n_jobs=1)
    lr.fit(X, y)
    coef = pd.Series(0.0, index=best_coef.index)
    coef.loc[cols] = lr.coef_[0]
    _p(f"[{ph_name}] [debias] Refit on {len(cols)} features complete.")
    return float(lr.intercept_[0]), coef

# ===================== EVALUATION & TESTS =====================

def _eval_auc_from_lin(lin: np.ndarray, y: np.ndarray) -> float:
    # AUC invariant to monotone transform; use linear predictor for tiny speed win
    return float(roc_auc_score(y, lin))

# -- DeLong (paired) implementation (adapted from standard approach) --

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(preds_sorted_transposed, label_1_count):
    """
    preds_sorted_transposed: shape (2, n) for 2 models, each sorted by labels (all positives first)
    label_1_count: #positives
    Returns: (aucs, S) aucs shape(2,), S covariance matrix 2x2
    """
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    pos_preds = preds_sorted_transposed[:, :m]
    neg_preds = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    # compute AUCs
    v01 = np.zeros((k, m))
    v10 = np.zeros((k, n))
    for r in range(k):
        pos_rank = _compute_midrank(np.concatenate([pos_preds[r], neg_preds[r]]))[:m]
        neg_rank = _compute_midrank(np.concatenate([pos_preds[r], neg_preds[r]]))[m:]
        v01[r] = (pos_rank - (m + 1) / 2.0) / n
        v10[r] = (neg_rank - (n + 1) / 2.0) / m
    aucs = v01.mean(axis=1)

    # covariance
    S01 = np.cov(v01)
    S10 = np.cov(v10)
    S = S01 / m + S10 / n
    return aucs, S

def _delong_one_sided_pvalue(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray) -> float:
    """
    Test H0: AUC1 <= AUC2 vs H1: AUC1 > AUC2 (one-sided).
    Returns p-value (small p supports AUC1 > AUC2).
    """
    assert y_true.ndim == 1
    assert scores1.shape == scores2.shape == y_true.shape

    # sort by y (positives first)
    order = np.argsort(-y_true)  # y=1 first
    y_sorted = y_true[order]
    pos = int(y_sorted.sum())
    if pos == 0 or pos == len(y_true):
        return float('nan')

    preds = np.vstack([scores1[order], scores2[order]])
    aucs, S = _fast_delong(preds, pos)
    delta = aucs[0] - aucs[1]
    var = S[0,0] + S[1,1] - 2*S[0,1]
    if var <= 0:
        return float('nan')
    z = delta / math.sqrt(var)
    # one-sided p for AUC1 > AUC2
    p = 1.0 - norm.cdf(z)
    return float(p)

def _paired_stratified_boot_auc(y_true: np.ndarray, s1: np.ndarray, s2: np.ndarray, B: int, ph_name: str):
    """
    Paired, stratified bootstrap over test rows. No refits; resample indices.
    Returns: (prob_win, (ci_lo, ci_hi), deltas_array)
    """
    rng = np.random.default_rng(SEED)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    n_pos = len(idx_pos); n_neg = len(idx_neg)
    deltas = np.empty(B, dtype=float)
    wins = 0
    t0 = time.time()
    for b in range(1, B+1):
        samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
        samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
        samp = np.concatenate([samp_pos, samp_neg])
        yb = y_true[samp]
        s1b = s1[samp]
        s2b = s2[samp]
        auc1 = roc_auc_score(yb, s1b)
        auc2 = roc_auc_score(yb, s2b)
        d = auc1 - auc2
        deltas[b-1] = d
        if d > 0:
            wins += 1
        if (b % BOOT_PRINT_EVERY == 0) or (b == B):
            _p(f"[{ph_name}] [bootstrap] {b}/{B} ({b/B*100:.1f}%) | wins={wins} | mean ΔAUC={deltas[:b].mean():+.4f}")
    prob_win = wins / B
    ci_lo, ci_hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    _p(f"[{ph_name}] [bootstrap] Done in {time.time()-t0:.2f}s | prob(inv>cov)={prob_win:.3f} | ΔAUC 95% CI=({ci_lo:+.4f},{ci_hi:+.4f})")
    return prob_win, (ci_lo, ci_hi), deltas

# ===================== PER-PHENOTYPE RUN =====================

def _run_one(ph: str, Z_train: pd.DataFrame, Z_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
             C_train: pd.DataFrame, C_test: pd.DataFrame):
    pos_tr = int(y_train.sum()); neg_tr = int((1 - y_train).sum())
    pos_te = int(y_test.sum());  neg_te = int((1 - y_test).sum())
    _p(f"[{ph}] [start] TRAIN cases/ctrl={pos_tr:,}/{neg_tr:,} | TEST cases/ctrl={pos_te:,}/{neg_te:,}")
    if (y_train.nunique() < 2) or (y_test.nunique() < 2):
        _p(f"[{ph}] [skip] Not enough class variety in train/test.")
        return None

    cfg = {
        "phenotype": ph,
        "alpha": ALPHA,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "X_cols": Z_train.shape[1],
        "use_covars": bool(USE_PIPELINE_COVARS),
        "include_ancestry": bool(INCLUDE_ANCESTRY),
        "remove_related": bool(REMOVE_RELATED),
    }
    key = _hash_cfg(cfg)
    outdir = os.path.join(OUT_ROOT, f"score_{ph}_{key}")
    os.makedirs(outdir, exist_ok=True)

    weights_pq = os.path.join(outdir, "weights.parquet")
    metrics_js = os.path.join(outdir, "metrics.json")
    if os.path.exists(weights_pq) and os.path.exists(metrics_js):
        try:
            mets = iox.read_meta_json(metrics_js)
            _p(f"[{ph}] [cache] Using cached results.")
            return {"phenotype": ph, **(mets or {})}
        except Exception:
            pass

    # ----- Inversion-only model: fit path, BIC select, optional debias -----
    best = _fit_bic_path(Z_train, y_train, ALPHA, ph)
    # Debias refit on support (often improves AUC)
    db_intercept, db_coef = _refit_debias_inversion(Z_train, y_train, best["coef"], best["intercept"], ph)

    # Test predictions for inversion-only (use linear predictor for AUC)
    cols_active = db_coef.index[db_coef != 0.0].tolist()
    if cols_active:
        lin_inv_test = db_intercept + Z_test[cols_active].values @ db_coef.loc[cols_active].values
    else:
        lin_inv_test = np.full(shape=len(y_test), fill_value=db_intercept, dtype=float)

    # ----- Covariate-only baseline -----
    if (C_train is None) or C_train.empty:
        _p(f"[{ph}] [covars] Covariate matrix is empty — baseline will be intercept-only.")
        Xtr_cov = np.ones((len(y_train), 1))
        Xte_cov = np.ones((len(y_test), 1))
        lr_cov = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000, n_jobs=1)
        lr_cov.fit(Xtr_cov, y_train.values)
        lin_cov_test = lr_cov.intercept_[0] + Xte_cov @ lr_cov.coef_[0]
    else:
        lr_cov = LogisticRegression(penalty='none', solver='lbfgs', max_iter=2000, n_jobs=1)
        lr_cov.fit(C_train.values, y_train.values)
        lin_cov_test = lr_cov.intercept_[0] + C_test.values @ lr_cov.coef_[0]

    # ----- AUCs on test -----
    auc_inv = _eval_auc_from_lin(lin_inv_test, y_test.values)
    auc_cov = _eval_auc_from_lin(lin_cov_test, y_test.values)
    dauc = float(auc_inv - auc_cov)
    _p(f"[{ph}] [eval] AUC_inv={auc_inv:.4f} | AUC_cov={auc_cov:.4f} | ΔAUC={dauc:+.4f}")

    # ----- DeLong one-sided p-value for AUC_inv > AUC_cov -----
    try:
        p_delong = _delong_one_sided_pvalue(y_test.values.astype(int), lin_inv_test, lin_cov_test)
    except Exception as e:
        _p(f"[{ph}] [delong/warn] Failed: {e}")
        p_delong = float('nan')
    _p(f"[{ph}] [delong] one-sided p(AUC_inv > AUC_cov) = {p_delong:.3g}" if not np.isnan(p_delong) else f"[{ph}] [delong] p=NaN")

    # ----- Paired stratified bootstrap probability + CI -----
    try:
        p_boot, (ci_lo, ci_hi), _ = _paired_stratified_boot_auc(
            y_test.values.astype(int), lin_inv_test, lin_cov_test, BOOTSTRAP_B, ph
        )
    except Exception as e:
        _p(f"[{ph}] [bootstrap/warn] Failed: {e}")
        p_boot, ci_lo, ci_hi = float('nan'), float('nan'), float('nan')

    # ----- Save artifacts -----
    weights = pd.DataFrame({"feature": db_coef.index, "beta": db_coef.values})
    iox.atomic_write_parquet(weights_pq, weights)
    iox.atomic_write_json(os.path.join(outdir, "model.json"), {
        "alpha": ALPHA,
        "lambda": best["lambda"],
        "C": best["C"],
        "intercept": db_intercept,
        "nonzero": int((db_coef != 0.0).sum()),
        "bic": best["bic"],
        "selected_features": cols_active,
    })
    test_scores = pd.DataFrame(
        {"person_id": y_test.index, "Inv_Lin": lin_inv_test, "Cov_Lin": lin_cov_test, "Y": y_test.values}
    ).set_index("person_id")
    iox.atomic_write_parquet(os.path.join(outdir, "test_scores.parquet"), test_scores)

    mets = {
        "AUC_inv": auc_inv,
        "AUC_cov": auc_cov,
        "DeltaAUC": dauc,
        "p_DeLong_one_sided": p_delong,
        "p_boot_one_sided": p_boot,
        "DeltaAUC_CI95_lo": ci_lo,
        "DeltaAUC_CI95_hi": ci_hi,
        "nonzero": int((db_coef != 0.0).sum()),
    }
    iox.atomic_write_json(metrics_js, mets)
    _p(f"[{ph}] [done] AUC_inv={auc_inv:.4f} ΔAUC={dauc:+.4f} p_DeLong={p_delong:.3g} p_boot={p_boot:.3f} | nnz={mets['nonzero']}")
    return {"phenotype": ph, **mets}

# ===================== MAIN =====================

def main():
    t_all = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    _p(f"[init] N_WORKERS={N_WORKERS}  ALPHA={ALPHA}  N_LAM_MAX={N_LAM_MAX}  TEST_SIZE={TEST_SIZE}  BOOTSTRAP_B={BOOTSTRAP_B}")

    # STEP 1: load inversions
    X = _load_dosages()

    # STEP 2: load pipeline covariates (or disable)
    base_cov, ancestry_df = (None, None)
    if USE_PIPELINE_COVARS:
        base_cov, ancestry_df = _load_pipeline_covars()

    # STEP 3: build labels (uses index domain as X; phenotypes from caches)
    Y = _build_phenotype_matrix(X.index, PHENOLIST)

    # STEP 4: align core matrices on common person_id (DO NOT intersect on ancestry)
    X, base_cov, Y, core_ids = _align_core(
        X,
        base_cov if USE_PIPELINE_COVARS else pd.DataFrame(index=X.index),
        Y
    )

    # STEP 5: create/load ONE global train/test split
    train_ids, test_ids = _load_or_make_global_split(Y)

    # STEP 6: prepare matrices once (persisted)
    Z_train, Z_test = _zscore_inversions_once(X, train_ids, test_ids)
    C_train = C_test = None
    if USE_PIPELINE_COVARS and (base_cov is not None) and (not base_cov.empty):
        C_train, C_test = _prep_covars_once(base_cov, ancestry_df, train_ids, test_ids)

    # STEP 7: select runnable phenotypes (must have both classes in global train & test)
    run_list = []
    for ph in PHENOLIST:
        y = Y[ph].astype(int)
        y_tr = y.reindex(train_ids)
        y_te = y.reindex(test_ids)
        if (y_tr.nunique() < 2) or (y_te.nunique() < 2):
            _p(f"[plan] {ph}: SKIP (needs both classes in TRAIN and TEST).")
            continue
        run_list.append(ph)
    _p(f"[plan] Runnable phenotypes: {len(run_list)}/{len(PHENOLIST)}")

    # STEP 8: run per phenotype in parallel (threads to share memory)
    results = []
    total = len(run_list)
    if total == 0:
        _p("[run] Nothing to do.")
    else:
        _p("[run] Starting parallel execution...")
        done = 0
        start = time.time()
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            fut_to_ph = {
                ex.submit(
                    _run_one,
                    ph,
                    Z_train, Z_test,
                    Y[ph].reindex(train_ids),
                    Y[ph].reindex(test_ids),
                    C_train, C_test
                ): ph
                for ph in run_list
            }
            for fut in as_completed(fut_to_ph):
                ph = fut_to_ph[fut]
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    _p(f"[{ph}] [FAIL] {e}")
                finally:
                    done += 1
                    pct = done / total * 100.0
                    elapsed = time.time() - start
                    rem = (elapsed/done*(total-done)) if done > 0 else float('nan')
                    _p(f"[progress] Completed {done}/{total} ({pct:.1f}%) | elapsed {elapsed:.1f}s | est. remaining {rem:.1f}s")

    # STEP 9: write summary
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
        # pretty print a quick table
        prn = summary[["phenotype","AUC_inv","AUC_cov","DeltaAUC","p_DeLong_one_sided","p_boot_one_sided","nonzero"]].copy()
        prn["AUC_inv"] = prn["AUC_inv"].map(lambda v: f"{v:.4f}")
        prn["AUC_cov"] = prn["AUC_cov"].map(lambda v: f"{v:.4f}")
        prn["DeltaAUC"] = prn["DeltaAUC"].map(lambda v: f"{v:+.4f}")
        prn["p_DeLong_one_sided"] = prn["p_DeLong_one_sided"].map(lambda v: f"{v:.3g}" if pd.notna(v) else "NaN")
        prn["p_boot_one_sided"] = prn["p_boot_one_sided"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "NaN")
        _p("\n[summary] Topline metrics:\n" + prn.to_string(index=False))
    else:
        _p("[summary] No results to write.")

    _p(f"[done] Total wall time {time.time()-t_all:.2f}s")

if __name__ == "__main__":
    main()
