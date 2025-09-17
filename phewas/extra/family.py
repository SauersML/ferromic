# ===================== GLOBALS ===================== #
import os, re, ast, json, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable

# ---- configuration (no CLI args) ----
PROJECT_ID      = os.getenv("GOOGLE_PROJECT")
CDR_DATASET_ID  = os.getenv("WORKSPACE_CDR")

INVERSION_FILE  = "../imputed_inversion_dosages.tsv"
OUTPUT_DIR      = "./assoc_outputs"
CACHE_DIR       = ".bq_cache"

# GCS (requester pays)
GCS_ANCESTRY_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
GCS_SEX_URI      = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"

# target inversion (we’ll auto-detect case; name will be normalized to lowercase)
INV_TARGET      = "chr17-45585160-inv-706887"

NUM_PCS         = 16
np.set_printoptions(suppress=True, linewidth=160)

# pretty printing
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{x:.6g}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# -------------------- logging helpers --------------------
def info(msg: str) -> None:
    print(f"INFO | {msg}")

def debug(msg: str) -> None:
    print(f"DEBUG| {msg}")

def warn(msg: str) -> None:
    print(f"WARN | {msg}")

# -------------------- small utils --------------------
def _normalize_cols_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def _normalize_person_id(df: pd.DataFrame, src_col: str) -> pd.DataFrame:
    df = df.copy()
    if src_col not in df.columns:
        raise RuntimeError(f"Could not find column '{src_col}' to normalize person_id")
    before = len(df)
    df[src_col] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[src_col]).astype({src_col: "int64"})
    debug(f"{src_col} person_id normalization: dropped {before - len(df):,} rows; kept {len(df):,}")
    df = df.rename(columns={src_col: "person_id"})
    return df

def _cache_key(sql: str) -> str:
    payload = json.dumps({"sql": sql, "cdr": CDR_DATASET_ID}, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()

def _cache_path(key: str) -> Path:
    return Path(CACHE_DIR) / f"{key}.parquet"

def _execute_gbq(sql: str, desc: str) -> pd.DataFrame:
    from pandas import read_gbq
    key = _cache_key(sql)
    path = _cache_path(key)
    if path.exists():
        info(f"{desc:40s} | cache -> {path.name}")
        df = pd.read_parquet(path)
    else:
        info(f"{desc:40s} | BigQuery")
        df = read_gbq(sql, project_id=PROJECT_ID, progress_bar_type=None)
        df.to_parquet(path, index=False)
    df = _normalize_cols_lower(df)
    debug(f"{desc:40s} | rows={len(df):,} unique_person_id={df['person_id'].nunique() if 'person_id' in df.columns else 'NA'}")
    return df

def _read_gcs_tsv(uri: str, desc: str) -> pd.DataFrame:
    info(f"{desc:40s} | GCS -> downloading")
    storage_options = {
        "requester_pays": True,
        "project": PROJECT_ID,
        "token": "cloud",
    }
    df = pd.read_csv(uri, sep="\t", dtype=str, storage_options=storage_options)
    df = _normalize_cols_lower(df)
    debug(f"GCS read OK: shape={df.shape}")
    debug(f"{desc:40s} columns (normalized): {list(df.columns)}")
    return df

# =========================================================
#                        LOADERS
# =========================================================

def load_dosages() -> tuple[pd.DataFrame, str]:
    info(f"Loading dosages: {INVERSION_FILE}")
    d = pd.read_csv(INVERSION_FILE, sep="\t", dtype=str)
    d = _normalize_cols_lower(d)
    # SampleID -> person_id
    id_col = None
    for candidate in ["sampleid", "person_id", "research_id", "participant_id"]:
        if candidate in d.columns:
            id_col = candidate
            break
    if id_col is None:
        raise RuntimeError("Could not find SampleID/person_id column in dosage file.")
    d = _normalize_person_id(d, id_col)

    # normalize inversion column names to lowercase (e.g., 'INV' -> 'inv')
    inv_cols = [c for c in d.columns if c.startswith("chr") and "inv" in c]
    if not inv_cols:
        # try also if input used 'INV' uppercase; rename to lowercase
        for c in [c for c in d.columns if c.startswith("chr") and "inv" not in c]:
            if "inv" in c.lower():
                d = d.rename(columns={c: c.lower()})
        inv_cols = [c for c in d.columns if c.startswith("chr") and "inv" in c]
    sample = inv_cols[:4]
    info(f"Dosage shape={d.shape} inversions={len(inv_cols)} sample={sample}")
    # Ensure numeric dosage
    for c in inv_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # pick target inversion if present, otherwise raise
    inv = None
    if INV_TARGET in d.columns:
        inv = INV_TARGET
    else:
        # try case-insensitive match
        lower_map = {c.lower(): c for c in d.columns}
        if INV_TARGET.lower() in lower_map:
            # rename to exact normalized lower-case
            orig = lower_map[INV_TARGET.lower()]
            if orig != INV_TARGET.lower():
                d = d.rename(columns={orig: INV_TARGET.lower()})
            inv = INV_TARGET.lower()
        else:
            raise RuntimeError(f"Target inversion '{INV_TARGET}' not found in dosage file.")
    info(f"Using inversion column: {inv}")
    return d[["person_id", inv]].rename(columns={inv: "G"}), inv

def load_pcs_and_ancestry() -> pd.DataFrame:
    df = _read_gcs_tsv(GCS_ANCESTRY_URI, "ancestry_preds.tsv")
    # person_id from research_id
    if "research_id" not in df.columns:
        raise RuntimeError("ancestry_preds.tsv missing 'research_id'.")
    df = _normalize_person_id(df, "research_id")

    # ancestry label
    if "ancestry_pred" not in df.columns:
        raise RuntimeError("ancestry_preds.tsv missing 'ancestry_pred'.")
    df["ancestry_pred"] = df["ancestry_pred"].str.strip().str.lower()
    # restrict to known (keep 'mid' if present)
    df["ancestry_pred"] = df["ancestry_pred"].replace({
        "european": "eur", "african": "afr", "east_asian": "eas", "south_asian": "sas",
        "admixed_american": "amr", "middle_eastern": "mid"
    })
    # PCs: prefer 'pca_features' (JSON-like list). If absent, fail.
    pc_source = None
    for c in ["pca_features", "pc_features", "pcs", "pca"]:
        if c in df.columns:
            pc_source = c
            break
    if pc_source is None:
        raise RuntimeError("PC columns not found in ancestry_preds.tsv (no 'pca_features').")
    debug(f"Parsing PCs from column: '{pc_source}'")
    # small peek
    ex_vals = df[pc_source].dropna().astype(str).head(3).tolist()
    debug(f"Example {pc_source} values: {ex_vals}")

    # parse list-of-floats safely
    def _parse_list(s):
        if pd.isna(s):
            return [np.nan] * NUM_PCS
        s = str(s).strip()
        try:
            # most rows look like Python list string, use literal_eval
            x = ast.literal_eval(s)
            if not isinstance(x, (list, tuple)):
                return [np.nan] * NUM_PCS
            x = list(x)
        except Exception:
            # fallback: strip brackets and split
            s2 = s.strip("[]")
            x = [float(z) if z.strip() not in ("", "nan", "None") else np.nan for z in s2.split(",")]
        if len(x) < NUM_PCS:
            x = list(x) + [np.nan] * (NUM_PCS - len(x))
        return x[:NUM_PCS]

    pcs = np.vstack(df[pc_source].apply(_parse_list).to_numpy())
    pcs = pd.DataFrame(pcs, columns=[f"pc{i}" for i in range(1, NUM_PCS + 1)])
    out = pd.concat([df[["person_id", "ancestry_pred"]].reset_index(drop=True), pcs], axis=1)

    # standardize PCs (z-score)
    for i in range(1, NUM_PCS + 1):
        col = f"pc{i}"
        mu = out[col].mean(skipna=True)
        sd = out[col].std(skipna=True)
        debug(f"PC standardization PC{i}: μ={mu:.6g} σ={sd:.6g}")
        out[col] = (out[col] - mu) / (sd if sd and np.isfinite(sd) and sd > 0 else 1.0)

    info(f"PCs+Ancestry loaded: rows={len(out):,}; NUM_PCS={NUM_PCS}")
    vc = out["ancestry_pred"].value_counts().to_dict()
    debug(f"Ancestry value counts: {vc}")

    # One-hot ancestry dummies with EUR reference (drop_first manually later)
    for lab in ["afr", "amr", "eas", "mid", "sas"]:
        out[f"ANC_{lab}"] = (out["ancestry_pred"] == lab).astype("float64")

    return out.drop(columns=["ancestry_pred"])

def load_sex_from_genomic_metrics() -> pd.DataFrame:
    df = _read_gcs_tsv(GCS_SEX_URI, "genomic_metrics.tsv")
    if "research_id" not in df.columns:
        raise RuntimeError("genomic_metrics.tsv missing 'research_id'")
    df = _normalize_person_id(df, "research_id")
    # derive sex from dragen_sex_ploidy
    cols_needed = ["dragen_sex_ploidy"]
    if not all(c in df.columns for c in cols_needed):
        raise RuntimeError("genomic_metrics.tsv missing 'dragen_sex_ploidy'")
    debug(f"Attempting sex derivation from columns: {['person_id'] + [c for c in df.columns if c!='person_id']}")
    ploidy = df["dragen_sex_ploidy"].astype(str).str.upper()
    # Map: XX -> 0 (female), XY -> 1 (male). Anything else -> NaN.
    def _ploidy_to_sex(x: str):
        x = (x or "").strip().upper()
        if x == "XX":
            return 0.0
        if x == "XY":
            return 1.0
        return np.nan
    df["sex"] = ploidy.apply(_ploidy_to_sex).astype("float64")
    info(f"Sex loaded: rows={len(df):,}")
    return df[["person_id", "sex"]]

def load_stable_age() -> pd.DataFrame:
    # year_of_birth
    yob_sql = f"""
        SELECT person_id, year_of_birth
        FROM `{CDR_DATASET_ID}.person`
    """
    yob = _execute_gbq(yob_sql, "person.year_of_birth")
    yob = _normalize_person_id(yob, "person_id")
    # observation_period.max_end_year
    o_sql = f"""
        SELECT person_id, CAST(EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS INT64) AS max_end_year
        FROM `{CDR_DATASET_ID}.observation_period`
        GROUP BY person_id
    """
    op = _execute_gbq(o_sql, "observation_period.max_end_year")
    op = _normalize_person_id(op, "person_id")

    df = yob.merge(op, on="person_id", how="inner")
    before = len(df)
    df["age"] = pd.to_numeric(df["max_end_year"], errors="coerce") - pd.to_numeric(df["year_of_birth"], errors="coerce")
    df = df.dropna(subset=["age", "year_of_birth"]).copy()
    df["age"] = df["age"].astype("float64")
    debug(f"Stable AGE rows: {before} -> {len(df)}")
    # z-scores
    mu_age, sd_age = df["age"].mean(), df["age"].std()
    mu_yob, sd_yob = df["year_of_birth"].mean(), df["year_of_birth"].std()
    df["AGE_z"] = (df["age"] - mu_age) / (sd_age if sd_age else 1.0)
    df["AGE_z_sq"] = df["AGE_z"] ** 2
    df["YOB_z"] = (df["year_of_birth"] - mu_yob) / (sd_yob if sd_yob else 1.0)
    debug(f"AGE z-stats: μ={mu_age:.3g} σ={sd_age:.3g}")
    info("Stable age computed for {:,} participants.".format(len(df)))
    return df[["person_id", "age", "AGE_z", "AGE_z_sq", "year_of_birth", "YOB_z", "max_end_year"]]

# =========================================================
#                 FAMILY RELATION PULL/PARSE
# =========================================================

# PFHH_ANY universe: anyone who reached PFHH family-history content
_PFHH_ANY_UNIVERSE = """
    SELECT DISTINCT person_id FROM `{CDR}.ds_survey`
    WHERE question LIKE 'Have you or anyone in your family ever been diagnosed with%%'
       OR question LIKE 'Including yourself, who in your family has had%%'
"""

PHENO_DEF = {
    "Breast Cancer": {
        "universe": _PFHH_ANY_UNIVERSE,
        "family_relation_questions": [
            "Including yourself, who in your family has had breast cancer%",
        ],
    },
    "Obesity": {
        "universe": _PFHH_ANY_UNIVERSE,
        "family_relation_questions": [
            "Including yourself, who in your family has had obesity%",
        ],
    },
    "Heart Failure": {
        "universe": _PFHH_ANY_UNIVERSE,
        "family_relation_questions": [
            "Including yourself, who in your family has had congestive heart failure%",
        ],
    },
    "Cognitive Impairment": {
        "universe": _PFHH_ANY_UNIVERSE,
        "family_relation_questions": [
            "Including yourself, who in your family has had dementia%",
            "Including yourself, who in your family has had memory loss or impairment%",
        ],
    },
}

_REL_KEYS = ["grandparent", "mother", "father", "sibling", "daughter", "son"]

def _extract_relations(answer: str) -> list[str]:
    """Split multi-select answers into normalized relation tokens; keep only known keys; exclude 'Self' upstream."""
    if not isinstance(answer, str) or not answer.strip():
        return []
    tokens = []
    for chunk in re.split(r"[;,|]", answer):
        if " - " not in chunk:
            continue
        rel = chunk.rsplit(" - ", 1)[-1].strip().lower()
        rel = re.sub(r"\s+", " ", rel)
        if rel in _REL_KEYS:
            tokens.append(rel)
    return tokens

def _execute_family_query(pattern: str, desc: str) -> pd.DataFrame:
    sql = f"""
        SELECT person_id, answer
        FROM `{CDR_DATASET_ID}.ds_survey`
        WHERE question LIKE '{pattern}'
          AND answer   LIKE '%% - %%'
          AND answer  NOT LIKE '%% - Self'
    """
    df = _execute_gbq(sql, desc)
    df = _normalize_person_id(df, "person_id")
    return df[["person_id", "answer"]]

def build_family_flags(phenotype: str, defn: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    u_sql = defn["universe"].replace("{CDR}", CDR_DATASET_ID)
    uni = _execute_gbq(u_sql, f"{phenotype}: universe")
    uni = _normalize_person_id(uni, "person_id")[["person_id"]].drop_duplicates()

    frames = []
    for patt in defn["family_relation_questions"]:
        df = _execute_family_query(patt, f"{phenotype}: family relations ({patt})")
        frames.append(df)
    rel_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["person_id","answer"])
    # drop dups & parse
    rel_raw = rel_raw.dropna(subset=["person_id"]).drop_duplicates()
    info(f"{phenotype:24s} | pulled relation rows: {len(rel_raw):,}")

    rows = []
    for pid, ans in rel_raw.itertuples(index=False):
        for r in set(_extract_relations(ans)):
            rows.append((pid, r))
    rel = pd.DataFrame(rows, columns=["person_id","relation"]).drop_duplicates()

    # initialize flags over universe (ensures 0s for people w/o any relation answers)
    flags = uni.copy()
    for r in _REL_KEYS:
        vec = (rel["relation"] == r).groupby(rel["person_id"]).any().rename(r).astype(int).reset_index()
        flags = flags.merge(vec, on="person_id", how="left")
    for r in _REL_KEYS:
        flags[r] = flags[r].fillna(0).astype(int)

    # collapse to 4 groups
    flags["grandparents"] = flags["grandparent"]
    flags["parents"]      = ((flags["mother"] | flags["father"]).astype(int))
    flags["siblings"]     = flags["sibling"]
    flags["children"]     = ((flags["daughter"] | flags["son"]).astype(int))

    flags = flags[["person_id","grandparents","parents","siblings","children"]]

    info(f"{phenotype:24s} | subgroup nonzero counts:  "
         f"grandparents={int(flags['grandparents'].sum()):,}  "
         f"parents={int(flags['parents'].sum()):,}  "
         f"siblings={int(flags['siblings'].sum()):,}  "
         f"children={int(flags['children'].sum()):,}")
    none_ct = int((flags[["grandparents","parents","siblings","children"]].sum(axis=1) == 0).sum())
    debug(f"{phenotype:24s} | 'None of four' participants: {none_ct:,}")

    return flags, uni

# =========================================================
#                      MODEL & REPORT
# =========================================================

def make_covariate_panel(dosage: pd.DataFrame, pcs_anc: pd.DataFrame, sex: pd.DataFrame, age: pd.DataFrame) -> pd.DataFrame:
    # dosage: person_id, G
    cov = dosage.merge(pcs_anc, on="person_id", how="inner")\
                .merge(sex,     on="person_id", how="left")\
                .merge(age,     on="person_id", how="left")
    debug(f"Covariate shapes: pcs_anc={pcs_anc.shape} sex={sex.shape} age={age.shape}")
    # coerce numerics
    num_cols = ["G", "sex", "AGE_z", "AGE_z_sq", "YOB_z"] \
               + [f"pc{i}" for i in range(1, NUM_PCS+1)] \
               + [f"ANC_{g}" for g in ["afr","amr","eas","mid","sas"]]
    for c in num_cols:
        if c in cov.columns:
            cov[c] = pd.to_numeric(cov[c], errors="coerce").astype("float64")
        else:
            cov[c] = 0.0  # safe default if a dummy missing completely
    return cov

def fit_gee_one_beta(df_long: pd.DataFrame) -> dict:
    """
    df_long columns: person_id, group, y (0/1), G, sex, AGE_z, AGE_z_sq, YOB_z, pc1..pc16, ANC_*.
    Returns dict with OR, CI, p, beta, se, N rows, clusters (n_person), etc.
    """
    # kinship weights
    r_map = {"grandparents": 0.25, "parents": 0.5, "siblings": 0.5, "children": 0.5}
    df_long = df_long.copy()
    df_long["group"] = df_long["group"].astype("category")
    df_long["r_g"] = df_long["group"].map(r_map).astype("float64")

    # center dosage at 2p using unique participants to avoid 4x weighting
    p_df = df_long[["person_id","G"]].dropna().drop_duplicates(subset=["person_id"])
    p = float(p_df["G"].mean() / 2.0)
    debug(f"Allele frequency p (from merged unique persons): p={p:.6g}")
    df_long["r_times_Gc"] = df_long["r_g"] * (df_long["G"] - 2.0*p)

    # build formula
    pc_terms = " + ".join([f"pc{i}" for i in range(1, NUM_PCS+1)])
    anc_terms = " + ".join([f"ANC_{g}" for g in ["afr","amr","eas","mid","sas"]])
    form = (
        "y ~ 1 + r_times_Gc + C(group) "
        "+ AGE_z + AGE_z_sq + YOB_z "
        "+ C(group):AGE_z + C(group):AGE_z_sq + C(group):YOB_z "
        "+ sex "
        f"+ {pc_terms} "
        f"+ {anc_terms} "
    )

    # required columns for complete-case
    req = ["y","r_times_Gc","AGE_z","AGE_z_sq","YOB_z","sex"] \
          + [f"pc{i}" for i in range(1, NUM_PCS+1)] \
          + [f"ANC_{g}" for g in ["afr","amr","eas","mid","sas"]]
    before = len(df_long)
    df_cc = df_long.dropna(subset=req).copy()
    dropped = before - len(df_cc)
    info(f"Rows before/after NA drop: {before} -> {len(df_cc)} (removed {dropped})")

    # enforce float64 on all exog columns to avoid 'object' casting
    for c in set(req + ["r_g","G"]):
        df_cc[c] = pd.to_numeric(df_cc[c], errors="coerce").astype("float64")

    # GEE
    gee = smf.gee(formula=form, groups="person_id", data=df_cc,
                  family=sm.families.Binomial(), cov_struct=Exchangeable())
    res = gee.fit()
    beta = float(res.params["r_times_Gc"])
    se   = float(res.bse["r_times_Gc"])
    OR   = float(np.exp(beta))
    ci_l = float(np.exp(beta - 1.96*se))
    ci_h = float(np.exp(beta + 1.96*se))
    pval = float(res.pvalues["r_times_Gc"])

    out = {
        "N_rows": int(len(df_cc)),
        "N_persons": int(df_cc["person_id"].nunique()),
        "beta": beta, "se": se, "p": pval,
        "OR": OR, "CI_low": ci_l, "CI_high": ci_h,
        "p_allele": p,
        "llf": float(getattr(res, "llf", np.nan)),
        "model_df": int(getattr(res, "df_model", np.nan)),
        "converged": bool(getattr(res, "converged", True)),
    }
    return out

def run_for_phenotype(name: str, defn: dict, covars: pd.DataFrame, inv_name: str) -> dict:
    info("#" * 130)
    info(f"Phenotype: {name} (FAMILY ONLY; pooled GEE with kinship-scaled centered dosage)")
    info("#" * 130)

    flags, universe = build_family_flags(name, defn)
    # merge with covariates
    df = flags.merge(covars, on="person_id", how="inner")
    info(f"{name:24s} | merged rows (with dosage & covars): {len(df):,}")

    # quick subgroup presence inside merged
    sg_counts = {g: int(df[g].sum()) for g in ["grandparents","parents","siblings","children"]}
    info(f"{name:24s} | subgroup nonzero counts (in merged): "
         f"grandparents={sg_counts['grandparents']:,},  parents={sg_counts['parents']:,},  "
         f"siblings={sg_counts['siblings']:,},  children={sg_counts['children']:,}")

    # wide -> long
    long = (
        df.melt(id_vars=["person_id","G","sex","AGE_z","AGE_z_sq","YOB_z"]
                      + [f"pc{i}" for i in range(1, NUM_PCS+1)]
                      + [f"ANC_{g}" for g in ["afr","amr","eas","mid","sas"]],
                value_vars=["grandparents","parents","siblings","children"],
                var_name="group", value_name="y")
          .assign(y=lambda x: x["y"].astype("int64"))
    )

    # debug outcome counts by group
    debug("Outcome counts by group in merged-long:")
    for g, sub in long.groupby("group"):
        yvc = sub["y"].value_counts().to_dict()
        info(f"  {g:12s} -> N={len(sub):7d} cases={int(yvc.get(1,0)):6d} controls={int(yvc.get(0,0)):7d}")

    # fit GEE
    debug("Building design & fitting GEE (this may take a moment)...")
    res = fit_gee_one_beta(long)

    # print summary line
    info(f"[RESULT] {name:20s}  inv={inv_name:28s}  persons={res['N_persons']:7d}  rows={res['N_rows']:7d}  "
         f"OR={res['OR']:.4g}  CI=[ {res['CI_low']:.4g},  {res['CI_high']:.4g} ]  p={res['p']:.4g}  "
         f"β={res['beta']:.6g}  SE={res['se']:.6g}  p_allele={res['p_allele']:.6g}")
    return {
        "phenotype": name,
        "inversion": inv_name,
        **res
    }

# =========================================================
#                         MAIN
# =========================================================

def main():
    if not PROJECT_ID or not CDR_DATASET_ID:
        raise RuntimeError("GOOGLE_PROJECT and WORKSPACE_CDR environment variables are required.")

    # 1) Dosages (person_id, G)
    dosage, inv_name = load_dosages()

    # 2) PCs + Ancestry (person_id, pc1..pc16, ANC_*)
    pcs_anc = load_pcs_and_ancestry()

    # 3) Sex (person_id, sex)
    sex = load_sex_from_genomic_metrics()

    # 4) Stable age & cohort (person_id, AGE_z, AGE_z_sq, YOB_z)
    age = load_stable_age()

    # 5) Merge covariate panel
    covars = make_covariate_panel(dosage, pcs_anc, sex, age)
    # quick NA report for core covars
    core = ["sex","AGE_z","AGE_z_sq","YOB_z"] + [f"pc{i}" for i in range(1, NUM_PCS+1)] + [f"ANC_{g}" for g in ["afr","amr","eas","mid","sas"]]
    na_counts = covars[["G"] + core].isna().sum().sort_values(ascending=False).head(10)
    debug("NA counts (top 10):\n" + na_counts.to_string())

    # 6) Phenotype loops
    phenos = {
        name: {"universe": defn["universe"].replace("{CDR}", CDR_DATASET_ID),
               "family_relation_questions": defn["family_relation_questions"]}
        for name, defn in PHENO_DEF.items()
    }

    rows = []
    for name, defn in phenos.items():
        rows.append(run_for_phenotype(name, defn, covars, inv_name))

    # 7) Save results table
    rep = pd.DataFrame(rows)
    out_csv = Path(OUTPUT_DIR) / "assoc_family_groups_gee_single_beta.csv"
    rep.to_csv(out_csv, index=False)

    print("\n" + "="*170)
    print("FINAL — ONE-COEFFICIENT GEE PER PHENOTYPE (per-allele OR for kinship-scaled, centered dosage)")
    print("="*170)
    if rep.empty:
        print("No results produced.")
    else:
        for _, r in rep.iterrows():
            print(f"{r['phenotype']:24s}  {str(r['inversion'])[:28]:28s}  persons={int(r['N_persons']):7d}  rows={int(r['N_rows']):7d}  "
                  f"OR={float(r['OR']):.4g}  CI=[ {float(r['CI_low']):.4g},  {float(r['CI_high']):.4g} ]  p={float(r['p']):.4g}  "
                  f"β={float(r['beta']):.6g}  SE={float(r['se']):.6g}  p_allele={float(r['p_allele']):.6g}")
    print(f"\nSaved results CSV -> {out_csv}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
