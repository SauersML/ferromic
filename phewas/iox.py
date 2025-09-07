import os
import time
import json
import ast
import tempfile
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def get_cached_or_generate(cache_path, generation_func, *args, validate_target=None, validate_num_pcs=None, **kwargs):
    """
    Generic caching wrapper with validation. Compatible with pre-existing caches.
    If the existing file fails shape/schema/NA checks, regenerate it by calling generation_func.
    """
    def _valid_demographics(df):
        ok = all(c in df.columns for c in ["AGE", "AGE_sq"])
        ok = ok and is_numeric_dtype(df["AGE"]) and is_numeric_dtype(df["AGE_sq"])
        if not ok: return False
        # AGE_sq consistency (allow minor float noise)
        return np.nanmax(np.abs(df["AGE_sq"].to_numpy() - (df["AGE"].to_numpy() ** 2))) < 1e-4

    def _valid_inversion(df):
        # exactly one column: the current TARGET_INVERSION; numeric; no NA-only rows
        if validate_target is None:
            return True
        cols = list(df.columns)
        if cols != [validate_target]:
            return False
        return is_numeric_dtype(df[validate_target]) and df[validate_target].notna().any()

    def _valid_pcs(df):
        if validate_num_pcs is None:
            return True
        expected = [f"PC{i}" for i in range(1, validate_num_pcs + 1)]
        if list(df.columns) != expected:
            return False
        # all numeric, no all-NA columns
        return all(is_numeric_dtype(df[c]) and df[c].notna().any() for c in expected)

    def _valid_sex(df):
        if list(df.columns) != ["sex"]:
            return False
        if not is_numeric_dtype(df["sex"]):
            return False
        # allow only 0/1 (with possible missing filtered at join time)
        uniq = set(pd.unique(df["sex"].dropna()))
        return uniq.issubset({0, 1})

    def _needs_validation(path):
        bn = os.path.basename(path)
        return (
            bn.startswith("demographics_")
            or bn.startswith("inversion_")
            or bn.startswith("pcs_")
            or bn == "genetic_sex.parquet"
        )

    def _validate(path, df):
        bn = os.path.basename(path)
        if bn.startswith("demographics_"):
            return _valid_demographics(df)
        if bn.startswith("inversion_"):
            return _valid_inversion(df)
        if bn.startswith("pcs_"):
            return _valid_pcs(df)
        if bn == "genetic_sex.parquet":
            return _valid_sex(df)
        # everything else: accept as-is
        return True

    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"  -> Cache unreadable ({e}); regenerating...")
            df = generation_func(*args, **kwargs)
            df.to_parquet(cache_path)
            return df

        # Basic index hygiene for joins
        try:
            df.index = df.index.astype(str)
            df.index.name = "person_id"
        except Exception:
            pass

        # Only validate known core covariates; regenerate if invalid
        if _needs_validation(cache_path) and not _validate(cache_path, df):
            print(f"  -> Cache at '{cache_path}' failed validation; regenerating...")
            df = generation_func(*args, **kwargs)
            df.to_parquet(cache_path)
        return df

    print(f"  -> No cache found at '{cache_path}'. Generating data...")
    data = generation_func(*args, **kwargs)
    data.to_parquet(cache_path)
    return data


def read_meta_json(path) -> dict | None:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not read corrupted meta file: {path}, Error: {e}")
        return None


def write_meta_json(path, meta: dict):
    pd.Series(meta).to_json(path)


def atomic_write_json(path, data_obj):
    """
    Writes JSON atomically by first writing to a unique temp path and then moving it into place.
    Accepts either a dict-like object or a pandas Series.
    """
    tmpdir = os.path.dirname(path) or "."
    os.makedirs(tmpdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=os.path.basename(path) + '.tmp.')
    os.close(fd)
    try:
        if isinstance(data_obj, pd.Series):
            data_obj = data_obj.to_dict()

        # Custom JSON encoder to handle numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open(tmp_path, 'w') as f:
            json.dump(data_obj, f, cls=NpEncoder)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def rss_gb():
    """Returns the resident set size of the current process in gigabytes for lightweight memory instrumentation."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def load_inversions(TARGET_INVERSION, INVERSION_DOSAGES_FILE):
    """Loads the target inversion dosage."""
    try:
        df = pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", usecols=["SampleID", TARGET_INVERSION])
        df['SampleID'] = df['SampleID'].astype(str)
        return df.set_index('SampleID').rename_axis('person_id')
    except Exception as e:
        raise RuntimeError(f"Failed to load inversion data: {e}")


def load_pcs(gcp_project, PCS_URI, NUM_PCS):
    """Loads genetic PCs."""
    try:
        raw_pcs = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})
        def _parse_and_pad(s):
            vals = ast.literal_eval(s) if pd.notna(s) else []
            return (vals + [np.nan] * NUM_PCS)[:NUM_PCS]

        pc_mat = pd.DataFrame(
            raw_pcs["pca_features"].apply(_parse_and_pad).tolist(),
            columns=[f"PC{i}" for i in range(1, NUM_PCS + 1)]
        )
        pc_df = pc_mat.assign(person_id=raw_pcs["research_id"].astype(str)).set_index("person_id")
        return pc_df
    except Exception as e:
        raise RuntimeError(f"Failed to load PCs: {e}")


def load_genetic_sex(gcp_project, SEX_URI):
    """Loads genetically-inferred sex and encodes it as a numeric variable."""
    print("    -> Loading genetically-inferred sex (ploidy)...")
    sex_df = pd.read_csv(SEX_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True},
                         usecols=['research_id', 'dragen_sex_ploidy'])

    sex_df['sex'] = np.nan
    sex_df.loc[sex_df['dragen_sex_ploidy'] == 'XX', 'sex'] = 0
    sex_df.loc[sex_df['dragen_sex_ploidy'] == 'XY', 'sex'] = 1

    sex_df = sex_df.rename(columns={'research_id': 'person_id'})
    sex_df['person_id'] = sex_df['person_id'].astype(str)

    return sex_df[['person_id', 'sex']].dropna().set_index('person_id')


def load_ancestry_labels(gcp_project, PCS_URI):
    """Loads predicted ancestry labels for each person."""
    print("    -> Loading genetic ancestry labels...")
    raw = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True},
                      usecols=['research_id', 'ancestry_pred'])
    df = raw.rename(columns={'research_id': 'person_id', 'ancestry_pred': 'ANCESTRY'})
    df['person_id'] = df['person_id'].astype(str)
    df = df.dropna(subset=['ANCESTRY'])
    return df.set_index('person_id')[['ANCESTRY']]


def load_related_to_remove(gcp_project, RELATEDNESS_URI):
    """Loads the pre-computed list of related individuals to prune."""
    print("    -> Loading list of related individuals to exclude...")
    related_df = pd.read_csv(RELATEDNESS_URI, sep="\t", header=None, names=['person_id'],
                             storage_options={"project": gcp_project, "requester_pays": True})

    # Return a set for extremely fast filtering
    return set(related_df['person_id'].astype(str))


def load_demographics_with_stable_age(bq_client, cdr_id):
    """
    Loads demographics, calculating a stable and reproducible age for each participant
    based on their last observation date in the dataset.
    """
    print("    -> Generating stable, reproducible age covariate...")

    # Query 1: Get year of birth
    yob_q = f"SELECT person_id, year_of_birth FROM `{cdr_id}.person`"
    yob_df = bq_client.query(yob_q).to_dataframe()
    yob_df['person_id'] = yob_df['person_id'].astype(str)

    # Query 2: Get the year of the last observation for each person
    obs_q = f"""
        SELECT person_id, EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS obs_end_year
        FROM `{cdr_id}.observation_period`
        GROUP BY person_id
    """
    obs_df = bq_client.query(obs_q).to_dataframe()
    obs_df['person_id'] = obs_df['person_id'].astype(str)

    # Merge the two data sources
    demographics = pd.merge(yob_df, obs_df, on='person_id', how='inner')

    # Calculate age and age-squared, handling potential data errors gracefully
    demographics['year_of_birth'] = pd.to_numeric(demographics['year_of_birth'], errors='coerce')
    demographics['AGE'] = demographics['obs_end_year'] - demographics['year_of_birth']
    demographics['AGE_sq'] = demographics['AGE'] ** 2

    # Set index and select final columns, dropping anyone with missing age info
    final_df = demographics[['person_id', 'AGE', 'AGE_sq']].dropna().set_index('person_id')

    print(f"    -> Successfully calculated stable age for {len(final_df):,} participants.")
    return final_df
