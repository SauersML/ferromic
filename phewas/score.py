import os, json, hashlib, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from . import iox  # atomic writers etc.

# ------------------ HARD-CODED CONFIG ------------------

# keep consistent with run.py
CACHE_DIR = "./phewas_cache"
DOSAGES_TSV = "imputed_inversion_dosages.tsv"

# optional covariates (wide TSV, first col person id). Off by default.
USE_COVARS = False
COVARS_TSV = None  # set path and flip USE_COVARS to True if you want them

# fixed phenolist you asked for
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

# modeling knobs
TEST_SIZE = 0.20
SEED = 42
ALPHA = 0.50        # elastic-net mixing (0=l2, 1=l1)
N_LAM = 100         # length of lambda path
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
    path = DOSAGES_TSV
    df = _read_wide_tsv(path)
    # coerce everything except index to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop columns that are all-NA or constant
    nunique = df.nunique(dropna=True)
    keep = nunique > 1
    kept = df.loc[:, keep]
    if kept.shape[1] == 0:
        raise RuntimeError("No variable inversion columns after filtering (all were NA/constant).")
    return kept

def _find_pheno_cache(sanitized_name: str) -> str | None:
    # pattern: CACHE_DIR/pheno_{name}_{cdr}.parquet
    pat = os.path.join(CACHE_DIR, f"pheno_{sanitized_name}_*.parquet")
    hits = sorted(glob.glob(pat))
    if not hits:
        return None
    # prefer newest by mtime
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def _build_phenotype_matrix(sample_index: pd.Index, phenos: list[str]) -> pd.DataFrame:
    Y = pd.DataFrame(index=sample_index.copy())
    Y.index.name = "person_id"
    missing = []
    for name in phenos:
        f = _find_pheno_cache(name)
        if f is None:
            missing.append(name)
            # create all-zero so the run still completes deterministically
            Y[name] = 0
            continue
        try:
            df = pd.read_parquet(f, columns=["is_case"])
        except Exception as e:
            warnings.warn(f"Failed to read cache for {name}: {e}")
            missing.append(name); Y[name] = 0; continue
        # df index should be person_id for cases with is_case==1
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

def _load_covars(idx: pd.Index) -> pd.DataFrame | None:
    if not USE_COVARS or not COVARS_TSV:
        return None
    C = _read_wide_tsv(COVARS_TSV)
    C = C.reindex(idx)
    # z-score numeric; leave binary/one-hot
    num_cols = [c for c in C.columns if pd.api.types.is_numeric_dtype(C[c])]
    if not num_cols:
        return C
    scaler = StandardScaler()
    C[num_cols] = scaler.fit_transform(C[num_cols].fillna(C[num_cols].mean()))
    return C

def _lambda_path(X: np.ndarray, y: np.ndarray, alpha: float, n: int, lmin_ratio: float) -> np.ndarray:
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
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0).replace(0, np.nan)
    eps = 1e-6
    Ztr = (Xtr - mu) / (sd + eps)
    Zte = (Xte - mu) / (sd + eps)
    return Ztr, Zte, list(Ztr.columns), {"mu": mu.to_dict(), "sd": sd.to_dict()}

def _fit_min_bic(Ztr: pd.DataFrame, ytr: pd.Series, Zc: pd.DataFrame | None):
    X = Ztr if Zc is None else pd.concat([Ztr, Zc], axis=1)
    lam_grid = _lambda_path(X.values, ytr.values.astype(float), ALPHA, N_LAM, LAMBDA_MIN_RATIO)
    feats = X.columns.tolist()
    best = None
    for lam in lam_grid:
        C = 1.0 / lam
        lr = LogisticRegression(
            penalty="elasticnet", l1_ratio=ALPHA, solver="saga",
            C=C, max_iter=MAX_ITER, tol=1e-4, class_weight=CLASS_WEIGHT, fit_intercept=True, n_jobs=1
        )
        lr.fit(X.values, ytr.values)
        p = lr.predict_proba(X.values)[:,1]
        eps = 1e-15
        ll = np.sum(ytr*np.log(p+eps) + (1-ytr)*np.log(1-p+eps))
        k = int(np.count_nonzero(lr.coef_[0])) + 1  # + intercept
        bic = -2*ll + k*np.log(len(ytr))
        if (best is None) or (bic < best["bic"]):
            best = {
                "lambda": float(lam),
                "C": float(C),
                "bic": float(bic),
                "intercept": float(lr.intercept_[0]),
                "coef": pd.Series(lr.coef_[0], index=feats),
                "nonzero": int((lr.coef_[0] != 0).sum()),
            }
    return best

def _eval_test(intercept, coef: pd.Series, Zte: pd.DataFrame, yte: pd.Series, Zc: pd.DataFrame | None):
    X = Zte if Zc is None else pd.concat([Zte, Zc], axis=1)
    beta = coef[X.columns]
    lin = intercept + X.values @ beta.values
    p = sigmoid(lin)
    return {
        "AUC": float(roc_auc_score(yte.values, p)),
        "PR_AUC": float(average_precision_score(yte.values, p)),
        "Brier": float(brier_score_loss(yte.values, p)),
        # quick calibration proxy (OLS of logit on linear score)
        "Cal_intercept": float(np.polyfit(lin, np.log((p+1e-15)/(1-p+1e-15)), 1)[1]),
        "Cal_slope": float(np.polyfit(lin, np.log((p+1e-15)/(1-p+1e-15)), 1)[0]),
    }, lin, p

# ------------------ MAIN WORKFLOW ------------------

def _run_one(ph: str, X: pd.DataFrame, Y: pd.DataFrame, C: pd.DataFrame | None):
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
        "C_cols": ([] if C is None else list(C.columns)),
    }
    key = _hash_cfg(cfg)
    outdir = os.path.join(OUT_ROOT, f"score_{ph}_{key}")
    os.makedirs(outdir, exist_ok=True)

    # if cached, skip compute
    weights_pq = os.path.join(outdir, "weights.parquet")
    metrics_js = os.path.join(outdir, "metrics.json")
    if os.path.exists(weights_pq) and os.path.exists(metrics_js):
        try:
            mets = iox.read_meta_json(metrics_js)
            print(f"[cache] {ph}: {mets}")
            return {"phenotype": ph, **(mets or {})}
        except Exception:
            pass

    Ztr, Zte, kept_feats, scalers = _prep_X(X, train_ids, test_ids)
    Ctr = C.loc[train_ids] if (C is not None) else None
    Cte = C.loc[test_ids]  if (C is not None) else None

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
    X = _load_dosages()
    Y = _build_phenotype_matrix(X.index, PHENOLIST)
    C = _load_covars(X.index) if USE_COVARS else None

    results = []
    for ph in PHENOLIST:
        res = _run_one(ph, X, Y, C)
        if res:
            results.append(res)

    if results:
        summary = pd.DataFrame(results)
        out_csv = os.path.join(OUT_ROOT, "summary_metrics.csv")
        if os.path.exists(out_csv):
            # replace rows for these phenos; keep others
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
