import os, json, hashlib, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from . import iox  # atomic writers, cache utils

# ------------------ HARD-CODED CONFIG (consistent with run.py where applicable) ------------------

CACHE_DIR = "./phewas_cache"
DOSAGES_TSV = "imputed_inversion_dosages.tsv"

# Covariates from pipeline caches (no external TSV required)
USE_PIPELINE_COVARS = True
REMOVE_RELATED = True
INCLUDE_ANCESTRY = True

# Data sources (same as run.py; only used if caches are missing and we must materialize them)
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

NUM_PCS = 10

# Fixed phenotypes
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
ALPHA = 0.50        # elastic-net mixing (0=L2, 1=L1)
N_LAM = 100         # lambda path length
LAMBDA_MIN_RATIO = 1e-3
NEAR_CONST_SD = 1e-6
MAX_ITER = 2000
CLASS_WEIGHT = "balanced"

OUT_ROOT = os.path.join(CACHE_DIR, "scores_elasticnet")

# ------------------ HELPERS ------------------

_ID_CANDIDATES = ("person_id","SampleID","sample_id","research_id","participant_id","ID")

def _hash_cfg(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

def _read_wide_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    id_col = next((c for c in df.columns if c in _ID_CANDIDATES), None)
    if id_col is None:
        raise RuntimeError(f"No ID column found in {path}. Expected one of {_ID_CANDIDATES}.")
    df = df.rename(columns={id_col: "person_id"})
    df["person_id"] = df["person_id"].astype(str)
    return df.set_index("person_id")

def _load_dosages() -> pd.DataFrame:
    df = _read_wide_tsv(DOSAGES_TSV)
    # coerce everything except index to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop columns all-NA or constant
    nunique = df.nunique(dropna=True)
    keep = nunique > 1
    kept = df.loc[:, keep]
    if kept.shape[1] == 0:
        raise RuntimeError("No variable inversion columns after filtering (all were NA/constant).")
    return kept

def _resolve_env():
    cdr_dataset_id = os.environ.get("WORKSPACE_CDR")
    gcp_project = os.environ.get("GOOGLE_PROJECT")
    cdr_codename = cdr_dataset_id.split(".")[-1] if cdr_dataset_id else None
    return cdr_dataset_id, cdr_codename, gcp_project

def _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project):
    """
    Use existing iox loaders + get_cached_or_generate to ensure parquet caches exist,
    exactly like run.py does. Only invoked if files are missing.
    """
    try:
        from google.cloud import bigquery
    except Exception as e:
        raise RuntimeError("google-cloud-bigquery is required to materialize missing caches") from e

    bq_client = bigquery.Client(project=gcp_project)

    # demographics (AGE/AGE_sq) — CDR dependent
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"),
        iox.load_demographics_with_stable_age, bq_client=bq_client, cdr_id=cdr_dataset_id
    )

    # PCs — NUM_PCS dependent
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet"),
        iox.load_pcs, gcp_project, PCS_URI, NUM_PCS, validate_num_pcs=NUM_PCS
    )

    # genetic sex
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, "genetic_sex.parquet"),
        iox.load_genetic_sex, gcp_project, SEX_URI
    )

    # ancestry labels (labels and PCs come from the same URI in run.py)
    _ = iox.get_cached_or_generate(
        os.path.join(CACHE_DIR, "ancestry_labels.parquet"),
        iox.load_ancestry_labels, gcp_project, LABELS_URI=PCS_URI
    )

def _load_pipeline_covars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      base_df: index=person_id, columns: AGE, AGE_sq, sex, PC1..PC{NUM_PCS}
      A_global: ancestry one-hot (drop_first=True), aligned to base_df index
    """
    cdr_dataset_id, cdr_codename, gcp_project = _resolve_env()

    demo_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet") if cdr_codename else None
    pcs_path  = os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet")
    sex_path  = os.path.join(CACHE_DIR, "genetic_sex.parquet")
    anc_path  = os.path.join(CACHE_DIR, "ancestry_labels.parquet")

    needed = [demo_path, pcs_path, sex_path, anc_path]
    missing = [p for p in needed if (p is None or not os.path.exists(p))]

    if missing:
        if not all([cdr_dataset_id, cdr_codename, gcp_project]):
            raise RuntimeError(
                "Covariate caches are missing and WORKSPACE_CDR/GOOGLE_PROJECT are not set. "
                "Either set env vars so we can materialize caches, or precreate the parquet files."
            )
        _maybe_materialize_covars(cdr_dataset_id, cdr_codename, gcp_project)

    demographics_df = pd.read_parquet(os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"))[["AGE","AGE_sq"]]
    pc_df          = pd.read_parquet(os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet"))[[f"PC{i}" for i in range(1, NUM_PCS+1)]]
    sex_df         = pd.read_parquet(os.path.join(CACHE_DIR, "genetic_sex.parquet"))[["sex"]]
    ancestry_df    = pd.read_parquet(os.path.join(CACHE_DIR, "ancestry_labels.parquet"))[["ANCESTRY"]]

    base_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")

    if REMOVE_RELATED:
        cdr_dataset_id, cdr_codename, gcp_project = _resolve_env()
        if not gcp_project:
            raise RuntimeError("GOOGLE_PROJECT must be set to remove related individuals.")
        related_ids = iox.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)
        base_df = base_df[~base_df.index.isin(related_ids)]

    if INCLUDE_ANCESTRY:
        anc_series = ancestry_df.reindex(base_df.index)["ANCESTRY"].astype(str).str.lower()
        anc_cat = pd.Categorical(anc_series.reindex(base_df.index))
        A_global = pd.get_dummies(anc_cat, prefix='ANC', drop_first=True, dtype=np.float32)
        A_global.index = A_global.index.astype(str)
    else:
        A_global = pd.DataFrame(index=base_df.index)

    return base_df, A_global

def _find_pheno_cache(sanitized_name: str) -> str | None:
    # pattern: CACHE_DIR/pheno_{name}_{cdr}.parquet
    pat = os.path.join(CACHE_DIR, f"pheno_{sanitized_name}_*.parquet")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # newest
    return hits[0]

def _build_phenotype_matrix(sample_index: pd.Index, phenos: list[str]) -> pd.DataFrame:
    Y = pd.DataFrame(index=sample_index.copy())
    Y.index.name = "person_id"
    missing = []
    for name in phenos:
        f = _find_pheno_cache(name)
        if f is None:
            missing.append(name)
            Y[name] = 0
            continue
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
        print(f"[warn] missing phenotype caches for: {', '.join(missing)}")
    return Y

def _lambda_path(X: np.ndarray, y: np.ndarray, alpha: float, n: int, lmin_ratio: float) -> np.ndarray:
    # simple strong rule upper bound; robustified
    p0 = y.mean()
    r0 = (y - p0)
    lam_max = np.abs(X.T @ r0).max() / (len(y) * max(alpha, 1e-6))
    lam_max = float(max(lam_max, 1e-3))
    lam_min = lam_max * lmin_ratio
    return np.geomspace(lam_max, lam_min, num=n)

def _train_test_split_stratified(y: pd.Series):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y.values))
    return y.index[train_idx], y.index[test_idx]

def _prep_X(X: pd.DataFrame, train_ids, test_ids):
    Xtr, Xte = X.loc[train_ids].copy(), X.loc[test_ids].copy()
    mu = Xtr.mean(axis=0)
    Xtr = Xtr.fillna(mu); Xte = Xte.fillna(mu)
    sd = Xtr.std(axis=0, ddof=0)
    keep = sd > NEAR_CONST_SD
    Xtr = Xtr.loc[:, keep]; Xte = Xte.loc[:, keep]
    if Xtr.shape[1] == 0:
        raise RuntimeError("All inversion features were near-constant in training set.")
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0)
    eps = 1e-6
    Ztr = (Xtr - mu) / (sd + eps)
    Zte = (Xte - mu) / (sd + eps)
    return Ztr, Zte, list(Ztr.columns), {"mu": mu.to_dict(), "sd": sd.to_dict()}

def _prep_covars(base_df: pd.DataFrame, A_global: pd.DataFrame, train_ids, test_ids):
    # center age on TRAIN, derive AGE_c_sq, append ancestry one-hots
    cov_tr = base_df.loc[train_ids].copy()
    cov_te = base_df.loc[test_ids].copy()
    age_mean = cov_tr['AGE'].mean()
    cov_tr['AGE_c'] = cov_tr['AGE'] - age_mean
    cov_tr['AGE_c_sq'] = cov_tr['AGE_c'] ** 2
    cov_te['AGE_c'] = cov_te['AGE'] - age_mean
    cov_te['AGE_c_sq'] = cov_te['AGE_c'] ** 2

    pc_cols = [f"PC{i}" for i in range(1, NUM_PCS+1)]
    base_cols = ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]

    if INCLUDE_ANCESTRY and (A_global is not None) and (not A_global.empty):
        Atr = A_global.reindex(train_ids).fillna(0.0)
        Ate = A_global.reindex(test_ids).fillna(0.0)
        cov_tr = pd.concat([cov_tr[base_cols], Atr], axis=1)
        cov_te = pd.concat([cov_te[base_cols], Ate], axis=1)
    else:
        cov_tr = cov_tr[base_cols]
        cov_te = cov_te[base_cols]

    # drop any columns that became constant (rare but possible)
    keep = cov_tr.nunique(dropna=True) > 1
    cov_tr = cov_tr.loc[:, keep]
    cov_te = cov_te.loc[:, keep]

    return cov_tr, cov_te

def _fit_min_bic(Ztr: pd.DataFrame, ytr: pd.Series, Ctr: pd.DataFrame | None):
    X = Ztr if (Ctr is None or Ctr.empty) else pd.concat([Ztr, Ctr], axis=1)
    lam_grid = _lambda_path(X.values, ytr.values.astype(float), ALPHA, N_LAM, LAMBDA_MIN_RATIO)
    feats = X.columns.tolist()
    best = None
    for lam in lam_grid:
        C_inv = 1.0 / lam
        lr = LogisticRegression(
            penalty="elasticnet", l1_ratio=ALPHA, solver="saga",
            C=C_inv, max_iter=MAX_ITER, tol=1e-4, class_weight=CLASS_WEIGHT,
            fit_intercept=True, n_jobs=1
        )
        lr.fit(X.values, ytr.values)
        p = lr.predict_proba(X.values)[:, 1]
        eps = 1e-15
        ll = np.sum(ytr*np.log(p+eps) + (1-ytr)*np.log(1-p+eps))
        k = int(np.count_nonzero(lr.coef_[0])) + 1  # + intercept
        bic = -2*ll + k*np.log(len(ytr))
        if (best is None) or (bic < best["bic"]):
            best = {
                "lambda": float(lam),
                "C": float(C_inv),
                "bic": float(bic),
                "intercept": float(lr.intercept_[0]),
                "coef": pd.Series(lr.coef_[0], index=feats),
                "nonzero": int((lr.coef_[0] != 0).sum()),
            }
    return best

def _eval_test(intercept, coef: pd.Series, Zte: pd.DataFrame, yte: pd.Series, Cte: pd.DataFrame | None):
    X = Zte if (Cte is None or Cte.empty) else pd.concat([Zte, Cte], axis=1)
    beta = coef[X.columns]
    lin = intercept + X.values @ beta.values
    p = sigmoid(lin)

    # metrics
    auc = float(roc_auc_score(yte.values, p))
    pr_auc = float(average_precision_score(yte.values, p))
    brier = float(brier_score_loss(yte.values, p))

    # calibration (fit logistic y ~ lin on test)
    try:
        cal = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000).fit(lin.reshape(-1,1), yte.values)
        cal_intercept = float(cal.intercept_[0])
        cal_slope = float(cal.coef_[0,0])
    except Exception:
        cal_intercept, cal_slope = np.nan, np.nan

    return {
        "AUC": auc, "PR_AUC": pr_auc, "Brier": brier,
        "Cal_intercept": cal_intercept, "Cal_slope": cal_slope
    }, lin, p

# ------------------ MAIN WORKFLOW ------------------

def _align_all(X: pd.DataFrame, base_cov: pd.DataFrame, A_global: pd.DataFrame, Y: pd.DataFrame):
    """
    Align dosages (X), base covariates, ancestry dummies, and labels on the common person_id set.
    """
    common = X.index.astype(str)
    common = common.intersection(base_cov.index.astype(str))
    common = common.intersection(Y.index.astype(str))
    if A_global is not None and not A_global.empty:
        common = common.intersection(A_global.index.astype(str))
    if len(common) == 0:
        raise RuntimeError("Empty intersection between dosages, covariates, ancestry, and phenotype labels.")
    X = X.reindex(common)
    base_cov = base_cov.reindex(common)
    if A_global is not None and not A_global.empty:
        A_global = A_global.reindex(common)
    Y = Y.reindex(common)
    return X, base_cov, A_global, Y

def _run_one(ph: str, X: pd.DataFrame, Y: pd.DataFrame, base_cov: pd.DataFrame | None, A_global: pd.DataFrame | None):
    y = Y[ph].astype(int)
    if y.nunique() < 2:
        print(f"[skip] {ph}: only one class label present.")
        return None

    train_ids, test_ids = _train_test_split_stratified(y)

    cfg = {
        "phenotype": ph,
        "alpha": ALPHA,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "X_cols": list(X.columns),
        "use_covars": bool(USE_PIPELINE_COVARS),
        "include_ancestry": bool(INCLUDE_ANCESTRY),
        "remove_related": bool(REMOVE_RELATED),
    }
    key = _hash_cfg(cfg)
    outdir = os.path.join(OUT_ROOT, f"score_{ph}_{key}")
    os.makedirs(outdir, exist_ok=True)

    # cache short-circuit
    weights_pq = os.path.join(outdir, "weights.parquet")
    metrics_js = os.path.join(outdir, "metrics.json")
    if os.path.exists(weights_pq) and os.path.exists(metrics_js):
        try:
            mets = iox.read_meta_json(metrics_js)
            print(f"[cache] {ph}: {mets}")
            return {"phenotype": ph, **(mets or {})}
        except Exception:
            pass

    # prepare X
    Ztr, Zte, kept_feats, scalers = _prep_X(X, train_ids, test_ids)

    # prepare covariates (if enabled)
    Ctr = Cte = None
    if USE_PIPELINE_COVARS and (base_cov is not None) and (not base_cov.empty):
        Ctr, Cte = _prep_covars(base_cov, A_global, train_ids, test_ids)

    # fit + evaluate
    best = _fit_min_bic(Ztr, y.loc[train_ids], Ctr)
    mets, lin, prob = _eval_test(best["intercept"], best["coef"], Zte, y.loc[test_ids], Cte)

    # save artifacts
    weights = pd.DataFrame({"feature": best["coef"].index, "beta": best["coef"].values})
    iox.atomic_write_parquet(weights_pq, weights)
    iox.atomic_write_json(os.path.join(outdir, "scalers.json"), {"genotype": scalers})
    iox.atomic_write_json(os.path.join(outdir, "model.json"), {
        "alpha": ALPHA, "lambda": best["lambda"], "C": best["C"],
        "intercept": best["intercept"], "nonzero": best["nonzero"], "bic": best["bic"]
    })
    test_scores = pd.DataFrame(
        {"person_id": test_ids, "PGS": lin, "Prob": prob, "Y": y.loc[test_ids].values}
    ).set_index("person_id")
    iox.atomic_write_parquet(os.path.join(outdir, "test_scores.parquet"), test_scores)
    iox.atomic_write_json(metrics_js, mets)

    print(f"[done] {ph}: {mets} | nonzero={best['nonzero']}")
    return {"phenotype": ph, **mets, "nonzero": best["nonzero"]}

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 1) load inversions
    X = _load_dosages()

    # 2) load pipeline covariates (or disable)
    base_cov, A_global = (None, None)
    if USE_PIPELINE_COVARS:
        base_cov, A_global = _load_pipeline_covars()

    # 3) build labels
    Y = _build_phenotype_matrix(X.index, PHENOLIST)

    # 4) align all matrices on common person_id
    X, base_cov, A_global, Y = _align_all(X, base_cov if USE_PIPELINE_COVARS else pd.DataFrame(index=X.index), A_global if USE_PIPELINE_COVARS else pd.DataFrame(index=X.index), Y)

    # 5) run per phenotype
    results = []
    for ph in PHENOLIST:
        res = _run_one(ph, X, Y, base_cov if USE_PIPELINE_COVARS else None, A_global if USE_PIPELINE_COVARS else None)
        if res:
            results.append(res)

    # 6) write summary
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
        print(f"[summary] wrote {out_csv}")
    else:
        print("[summary] no results to write.")

if __name__ == "__main__":
    main()
