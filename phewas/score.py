import os, sys, json, hashlib, glob, warnings, time, math, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import chi2, norm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import statsmodels.api as sm

from . import iox  # atomic writers, cache utils

# ===================== HARD-CODED CONFIG =====================

CACHE_DIR = "./phewas_cache"
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
ALPHA = 0.50                 # elastic-net mixing (0=L2, 1=L1)
N_LAM = 100                  # maximum lambda path length
LAMBDA_MIN_RATIO = 1e-3
MAX_ITER = 2000
CLASS_WEIGHT = "balanced"
NEAR_CONST_SD = 1e-6
BIC_EARLY_STOP = 5           # stop path after this many consecutive BIC increases
DEBIAS_REFIT = True          # optional unpenalized refit on selected support

# Test-time paired bootstrap for ΔAUC (Model1 - Model0)
BOOT_B = int(os.environ.get("SCORE_BOOT_B", "1000"))
BOOT_SEED = SEED

# Parallelization
N_WORKERS = int(os.environ.get("SCORE_N_JOBS", str(max(1, (os.cpu_count() or 4) - 0)))))
PRINT_LOCK = threading.Lock()

OUT_ROOT = os.path.join(CACHE_DIR, "scores_nested_pgs")

# ===================== PROGRESS & UTILS =====================

_ID_CANDIDATES = ("person_id","SampleID","sample_id","research_id","participant_id","ID")

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

# ===================== DATA LOADING =====================

def _read_wide_tsv(path: str) -> pd.DataFrame:
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
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    nunique = df.nunique(dropna=True)
    keep = nunique > 1
    kept = df.loc[:, keep]
    dropped = int((~keep).sum())
    _p(f"[load/dosages] Dropped {dropped:,} constant/all-NA inversion columns; kept {kept.shape[1]:,}.")
    if kept.shape[1] == 0:
        raise RuntimeError("No variable inversion columns after filtering.")
    _p(f"[load/dosages] Final shape: {kept.shape[0]:,} x {kept.shape[1]:,} (elapsed {time.time()-t0:.2f}s)")
    return kept

def _resolve_env():
    cdr_dataset_id = os.environ.get("WORKSPACE_CDR")
    gcp_project = os.environ.get("GOOGLE_PROJECT")
    cdr_codename = cdr_dataset_id.split(".")[-1] if cdr_dataset_id else None
    return cdr_dataset_id, cdr_codename, gcp_project

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
            raise RuntimeError("Covariate caches missing and WORKSPACE_CDR/GOOGLE_PROJECT not set.")
        _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project)

    _p("[covars] Loading cached demographics/PCs/sex/ancestry...")
    demographics_df = pd.read_parquet(os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"))[["AGE","AGE_sq"]]
    pc_df          = pd.read_parquet(os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet"))[[f"PC{i}" for i in range(1, NUM_PCS+1)]]
    sex_df         = pd.read_parquet(os.path.join(CACHE_DIR, "genetic_sex.parquet"))[["sex"]]
    ancestry_df    = pd.read_parquet(os.path.join(CACHE_DIR, "ancestry_labels.parquet"))[["ANCESTRY"]]

    for df in (demographics_df, pc_df, sex_df, ancestry_df):
        df.index = df.index.astype(str)

    base_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
    _p(f"[covars] Base covariates shape before related removal: {base_df.shape}")

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

    best = None
    inc = 0
    _p(f"[{tag}] [PGS] Lambda sweep: {len(lam_grid)} values, P={Ztr.shape[1]:,}, N={len(ytr):,}")
    for i, lam in enumerate(lam_grid, 1):
        C_inv = 1.0 / lam
        lr = LogisticRegression(
            penalty="elasticnet", l1_ratio=ALPHA, solver="saga",
            C=C_inv, max_iter=MAX_ITER, tol=1e-4, class_weight=CLASS_WEIGHT,
            fit_intercept=True, n_jobs=1
        )
        lr.fit(X, ytr.values)
        p = lr.predict_proba(X)[:, 1]
        eps = 1e-15
        ll = float(np.sum(ytr*np.log(p+eps) + (1-ytr)*np.log(1-p+eps)))
        k = int(np.count_nonzero(lr.coef_[0])) + 1  # + intercept
        bic = -2.0*ll + k*np.log(len(ytr))

        if (best is None) or (bic < best["bic"]):
            best = {
                "lambda": float(lam),
                "C": float(C_inv),
                "bic": float(bic),
                "intercept": float(lr.intercept_[0]),
                "coef": pd.Series(lr.coef_[0], index=Ztr.columns),
                "nonzero": int((lr.coef_[0] != 0).sum()),
            }
            inc = 0
        else:
            inc += 1

        if (i % max(1, len(lam_grid)//10) == 0) or (i == len(lam_grid)):
            _p(f"[{tag}] [PGS] {i}/{len(lam_grid)} ({i/len(lam_grid)*100:.1f}%) | "
               f"best BIC={best['bic']:.2f} nnz={best['nonzero']}")

        if inc >= BIC_EARLY_STOP:
            _p(f"[{tag}] [PGS] Early stop: BIC rose {BIC_EARLY_STOP}x")
            break

    return best

def _pgs_scores(Ztr: pd.DataFrame, Zte: pd.DataFrame, ytr: pd.Series, tag: str):
    best = _fit_pgs_bic(Ztr, ytr, tag)
    coef = best["coef"].copy()

    if DEBIAS_REFIT and best["nonzero"] > 0:
        sel = coef.index[coef != 0]
        Xdeb = sm.add_constant(Ztr[sel], has_constant='add')
        try:
            res = sm.Logit(ytr.values, Xdeb).fit(disp=0, method="lbfgs")
            beta = pd.Series(res.params, index=Xdeb.columns)
            # linear predictors (no sigmoid) as raw PGS
            lin_tr = (Xdeb.values @ beta.values)
            Xdeb_te = sm.add_constant(Zte[sel], has_constant='add')
            lin_te = (Xdeb_te.values @ beta.values)
            _p(f"[{tag}] [PGS] Debiased refit ok. Support={len(sel)}")
        except Exception as e:
            _p(f"[{tag}] [PGS WARN] Debias failed ({e}); using penalized coef.")
            lin_tr = best["intercept"] + (Ztr.values @ coef.values)
            lin_te = best["intercept"] + (Zte.values @ coef.values)
    else:
        lin_tr = best["intercept"] + (Ztr.values @ coef.values)
        lin_te = best["intercept"] + (Zte.values @ coef.values)

    # z-score PGS on TRAIN (for interpretability & stable Wald/LRT)
    mu = float(np.mean(lin_tr))
    sd = float(np.std(lin_tr)) if float(np.std(lin_tr)) > 0 else 1.0
    pgs_tr = (lin_tr - mu) / sd
    pgs_te = (lin_te - mu) / sd

    return {
        "best": best,
        "pgs_tr": pgs_tr,
        "pgs_te": pgs_te,
        "pgs_mu": mu,
        "pgs_sd": sd,
    }

# ===================== NESTED MODEL TESTS =====================

def _fit_model0_model1_train(cov_tr: pd.DataFrame, ytr: pd.Series, pgs_tr: np.ndarray, tag: str):
    # Model 0: baseline covariates only
    X0 = sm.add_constant(cov_tr, has_constant='add')
    m0 = sm.Logit(ytr.values, X0).fit(disp=0, method="lbfgs")

    # Model 1: baseline + PGS
    X1 = X0.copy()
    X1 = X1.assign(PGS=pgs_tr)
    m1 = sm.Logit(ytr.values, X1).fit(disp=0, method="lbfgs")

    # LRT (df=1; only added PGS)
    lrt = 2.0 * (m1.llf - m0.llf)
    p_lrt = float(1.0 - chi2.cdf(lrt, df=1))

    # Wald z for PGS coefficient (two-sided)
    try:
        pgsi = list(X1.columns).index("PGS")
        beta_pgs = float(m1.params[pgsi])
        se_pgs = float(np.sqrt(m1.cov_params().iloc[pgsi, pgsi]))
        z = beta_pgs / se_pgs if se_pgs > 0 else np.nan
        p_wald_two = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    except Exception:
        beta_pgs = np.nan; se_pgs = np.nan; z = np.nan; p_wald_two = np.nan

    _p(f"[{tag}] [TRAIN] LRT p={p_lrt:.3e} | Wald z={z if np.isfinite(z) else float('nan'):.3f} p={p_wald_two:.3e}")
    return {
        "m0": m0, "m1": m1,
        "p_lrt": p_lrt, "lrt_stat": float(lrt),
        "beta_pgs": beta_pgs, "se_pgs": se_pgs, "wald_z": z, "p_wald_two": p_wald_two
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
        # midrank for ties
        mid = 0.5 * (i + j - 1) + 1  # +1 for 1-based ranks
        ranks[i:j] = mid
        i = j
    out = np.empty(n, dtype=float)
    out[sorted_idx] = ranks
    return out

def _fast_delong(y_true: np.ndarray, pred: np.ndarray):
    # One predictor variant: return AUC, V10, V01 vectors per DeLong
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
    # Paired DeLong for two correlated scores on same individuals; one-sided p for AUC1 > AUC0
    a0, v10_0, v01_0 = _fast_delong(y_true, s0)
    a1, v10_1, v01_1 = _fast_delong(y_true, s1)
    if any(v is None for v in (v10_0, v01_0, v10_1, v01_1)) or np.isnan(a0) or np.isnan(a1):
        return np.nan
    m, n = len(v10_0), len(v01_0)
    # cov(AUCa, AUCb) = cov(V10a, V10b)/m + cov(V01a, V01b)/n
    cov_v10 = np.cov(np.vstack([v10_0, v10_1]), bias=False)
    cov_v01 = np.cov(np.vstack([v01_0, v01_1]), bias=False)
    s_00 = cov_v10[0,0]/m + cov_v01[0,0]/n
    s_11 = cov_v10[1,1]/m + cov_v01[1,1]/n
    s_01 = cov_v10[0,1]/m + cov_v01[0,1]/n
    var_diff = s_00 + s_11 - 2*s_01
    if var_diff <= 0:
        return np.nan
    z = (a1 - a0) / math.sqrt(var_diff)
    if alternative == "greater":
        p = float(1.0 - norm.cdf(z))
    elif alternative == "less":
        p = float(norm.cdf(z))
    else:  # two-sided
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
    for b in range(B):
        # stratified resample
        rp = rng.choice(idx_pos, size=m, replace=True)
        rn = rng.choice(idx_neg, size=n, replace=True)
        sel = np.concatenate([rp, rn])
        yb = y[sel]
        d0 = roc_auc_score(yb, s0[sel])
        d1 = roc_auc_score(yb, s1[sel])
        d = d1 - d0
        deltas[b] = d
        if d > 0: count_gt += 1
        if (b+1) % max(1, B//10) == 0 or (b+1) == B:
            _p(f"[{tag}] [TEST/bootstrap] {b+1}/{B} ({(b+1)/B*100:.1f}%)")
    prob = float(count_gt / B)
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return prob, (lo, hi), float(np.mean(deltas))

# ===================== EVALUATION (TEST) =====================

def _evaluate_test(m0, m1, cov_te: pd.DataFrame, pgs_te: np.ndarray, yte: pd.Series, tag: str):
    X0_te = sm.add_constant(cov_te, has_constant='add')
    X1_te = X0_te.assign(PGS=pgs_te)

    # Predicted probabilities (GLM)
    p0 = m0.predict(X0_te)
    p1 = m1.predict(X1_te)

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
       f"p_DeLong={p_delong if not np.isnan(p_delong) else float('nan'):.3e}  "
       f"P(ΔAUC>0)={prob_gt:.3f}  CI_95%=[{ci_lo:+.4f},{ci_hi:+.4f}]")
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

    # Build covariates (Model 0/1)
    cov_tr, cov_te = _build_covariates_splits(C_base, ancestry_df, train_ids, test_ids, ph)

    # Train PGS on TRAIN (inversions only), produce TRAIN/TEST PGS (z-scored on TRAIN)
    pgs = _pgs_scores(Ztr, Zte, y.loc[train_ids], ph)

    # Fit nested models on TRAIN
    fit = _fit_model0_model1_train(cov_tr, y.loc[train_ids], pgs["pgs_tr"], ph)

    # Evaluate discrimination on TEST (ΔAUC, DeLong, bootstrap)
    mets_test = _evaluate_test(
        fit["m0"], fit["m1"], cov_te, pgs["pgs_te"], y.loc[test_ids], ph
    )

    # Save artifacts
    coef = pgs["best"]["coef"]
    weights_df = pd.DataFrame({"feature": coef.index, "beta": coef.values})
    iox.atomic_write_parquet(weights_pq, weights_df)
    iox.atomic_write_json(os.path.join(outdir, "pgs_model.json"), {
        "alpha": ALPHA, "lambda": pgs["best"]["lambda"], "C": pgs["best"]["C"],
        "intercept": pgs["best"]["intercept"], "nonzero": pgs["best"]["nonzero"],
        "bic": pgs["best"]["bic"], "pgs_mu_train": pgs["pgs_mu"], "pgs_sd_train": pgs["pgs_sd"]
    })
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
        "PGS_nonzero": pgs["best"]["nonzero"]
    }
    iox.atomic_write_json(metrics_js, out)
    _p(f"[{ph}] [done] nnz={pgs['best']['nonzero']} | "
       f"LRT p={fit['p_lrt']:.3e} | ΔAUC={out['DeltaAUC']:+.4f} | "
       f"p_DeLong={out['p_DeLong_one_sided'] if not np.isnan(out['p_DeLong_one_sided']) else float('nan'):.3e}")

    return out

# ===================== MAIN =====================

def main():
    t_all = time.time()
    os.makedirs(OUT_ROOT, exist_ok=True)
    _p(f"[init] N_WORKERS={N_WORKERS}  ALPHA={ALPHA}  N_LAM={N_LAM}  TEST_SIZE={TEST_SIZE}  BOOT_B={BOOT_B}")

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
                    _p(f"[progress] {done}/{total} ({pct:.1f}%) | elapsed {elapsed:.1f}s | ETA ~{rem:.1f}s")

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
