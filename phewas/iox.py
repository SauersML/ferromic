import os
import time
import json
import tempfile
import uuid
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _best_effort_fsync(fobj):
    try:
        fobj.flush()
    except Exception:
        pass
    try:
        os.fsync(fobj.fileno())
    except Exception:
        pass


def _fsync_dir(path):
    d = os.path.dirname(os.path.abspath(path)) or "."
    try:
        dir_fd = os.open(d, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def get_cached_or_generate(cache_path, generation_func, *args, validate_target=None, validate_num_pcs=None, **kwargs):
    """
    Generic caching wrapper with validation. Compatible with pre-existing caches.
    If the existing file fails shape/schema/NA checks, regenerate it by calling generation_func.
    """
    def _valid_demographics(df):
        if not all(c in df.columns for c in ("AGE","AGE_sq")): return False
        if not (is_numeric_dtype(df["AGE"]) and is_numeric_dtype(df["AGE_sq"])): return False
        vals = df[["AGE","AGE_sq"]].astype(float)
        if vals.isna().any().any():  # fail caches with NA; you regenerate anyway
            return False
        return np.allclose(vals["AGE_sq"].to_numpy(),
                           (vals["AGE"]**2).to_numpy(),
                           rtol=0, atol=1e-6)

    def _valid_inversion(df):
        """Validate that the inversion dosage file contains the target column, it's numeric, and is not constant."""
        if validate_target is None:
            return True
        if validate_target not in df.columns:
            return False
        s = pd.to_numeric(df[validate_target], errors="coerce")
        return is_numeric_dtype(s) and s.notna().sum() > 0 and s.nunique(dropna=True) > 1

    def _valid_pcs(df):
        if validate_num_pcs is None: return True
        expected = [f"PC{i}" for i in range(1, validate_num_pcs + 1)]
        if not set(expected).issubset(df.columns):
            return False
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

    def _coerce_index(df):
        out = df.copy()
        out.index = out.index.astype(str)
        out.index.name = "person_id"
        return out

    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        try:
            df = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"  -> Cache unreadable ({e}); regenerating...")
            df = _coerce_index(generation_func(*args, **kwargs))
            atomic_write_parquet(cache_path, df)
            return df


        # Basic index hygiene for joins
        df = _coerce_index(df)

        # Only validate known core covariates; regenerate if invalid
        if _needs_validation(cache_path) and not _validate(cache_path, df):
            print(f"  -> Cache at '{cache_path}' failed validation; regenerating...")
            df = _coerce_index(generation_func(*args, **kwargs))
            atomic_write_parquet(cache_path, df)
        return df

    print(f"  -> No cache found at '{cache_path}'. Generating data...")
    df = _coerce_index(generation_func(*args, **kwargs))
    atomic_write_parquet(cache_path, df)
    return df


def get_cached_or_generate_pickle(cache_path, generation_func, *args, **kwargs):
    """Simple cache wrapper for pickled objects."""
    if os.path.exists(cache_path):
        print(f"  -> Found cache, loading from '{cache_path}'...")
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"  -> Cache unreadable ({e}); regenerating...")
    obj = generation_func(*args, **kwargs)
    atomic_write_pickle(cache_path, obj)
    return obj


def read_meta_json(path) -> dict | None:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not read corrupted meta file: {path}, Error: {e}")
        return None


def write_meta_json(path, meta: dict):
    atomic_write_json(path, meta)


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
            _best_effort_fsync(f)
        os.replace(tmp_path, path)
        _fsync_dir(path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def atomic_write_parquet(path, df, **to_parquet_kwargs):
    """
    Atomically writes a parquet file by first writing to a unique temp path and then moving it into place.
    This prevents partial files from being observed if the process is killed mid-write.
    """
    tmpdir = os.path.dirname(path) or "."
    os.makedirs(tmpdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=os.path.basename(path) + '.tmp.')
    os.close(fd)
    try:
        df.to_parquet(tmp_path, **to_parquet_kwargs)
        with open(tmp_path, 'rb') as f:
            _best_effort_fsync(f)
        os.replace(tmp_path, path)
        _fsync_dir(path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def atomic_write_pickle(path, obj):
    """
    Atomically writes a pickle by first writing to a unique temp path and then moving it into place.
    """
    tmpdir = os.path.dirname(path) or "."
    os.makedirs(tmpdir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=os.path.basename(path) + '.tmp.')
    os.close(fd)
    try:
        pd.to_pickle(obj, tmp_path)
        with open(tmp_path, 'rb') as f:
            _best_effort_fsync(f)
        os.replace(tmp_path, path)
        _fsync_dir(path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# --- Shared memory helpers -------------------------------------------------
from multiprocessing import shared_memory


def create_shared_from_ndarray(arr: np.ndarray, readonly: bool = True):
    """Create a shared memory block from ``arr`` and return (meta, handle).

    ``meta`` is a lightweight dictionary describing the shared memory block
    that can be sent to worker processes. ``handle`` is the ``SharedMemory``
    instance which must be kept alive by the parent and later closed/unlinked
    when no longer needed.
    """
    try:
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        view[:] = arr
        if readonly:
            try:
                view.setflags(write=False)
            except Exception:
                pass
        meta = {"kind": "shm", "name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
        return meta, shm
    except Exception:
        # Fallback: disk-backed memmap when /dev/shm is insufficient
        tmpdir = tempfile.gettempdir()
        fname = os.path.join(tmpdir, f"mm_{uuid.uuid4().hex}.dat")
        mm = np.memmap(fname, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mm[:] = arr
        del mm
        meta = {"kind": "memmap", "path": fname, "shape": arr.shape, "dtype": str(arr.dtype)}
        class _Handle:
            def close(self):
                pass
            def unlink(self):
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    pass
        return meta, _Handle()


def attach_shared_ndarray(meta: dict):
    """Attach to a shared memory block created by ``create_shared_from_ndarray``.

    Returns a tuple ``(array, handle)`` where ``array`` is a NumPy view of the
    shared memory and ``handle`` is an object which the caller is responsible for
    closing (but not unlinking).
    """
    if meta.get("kind") == "memmap":
        arr = np.memmap(meta["path"], mode="r", dtype=np.dtype(meta["dtype"]), shape=tuple(meta["shape"]))
        try:
            arr.setflags(write=False)
        except Exception:
            pass
        class _Handle:
            def close(self):
                pass
            def unlink(self):
                pass  # parent removes the file
        return arr, _Handle()

    shm = shared_memory.SharedMemory(name=meta["name"])
    arr = np.ndarray(tuple(meta["shape"]), dtype=np.dtype(meta["dtype"]), buffer=shm.buf)
    try:
        arr.setflags(write=False)
    except Exception:
        pass
    return arr, shm

def rss_gb():
    """Returns the resident set size of the current process in gigabytes for lightweight memory instrumentation."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def load_inversions(TARGET_INVERSION, INVERSION_DOSAGES_FILE):
    """Loads the target inversion dosage."""
    try:
        df = pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", usecols=["SampleID", TARGET_INVERSION])
        df[TARGET_INVERSION] = pd.to_numeric(df[TARGET_INVERSION], errors="coerce")
        df['SampleID'] = df['SampleID'].astype(str)
        return df.set_index('SampleID').rename_axis('person_id')
    except Exception as e:
        raise RuntimeError(f"Failed to load inversion data: {e}")


def load_pcs(gcp_project, PCS_URI, NUM_PCS):
    """Loads genetic PCs."""
    try:
        raw_pcs = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})
        def _parse_and_pad_fast(s):
            """Fast parser for string-encoded lists of floats."""
            if pd.isna(s): return [np.nan] * NUM_PCS
            s = s.strip()
            if not s.startswith('[') or not s.endswith(']'): return [np.nan] * NUM_PCS
            s = s[1:-1]
            if not s: return [np.nan] * NUM_PCS
            try:
                vals = [float(x) for x in s.split(',')]
                # Pad with NaNs if the list is shorter than NUM_PCS
                if len(vals) < NUM_PCS:
                    vals.extend([np.nan] * (NUM_PCS - len(vals)))
                return vals[:NUM_PCS]
            except ValueError:
                return [np.nan] * NUM_PCS

        pc_mat = pd.DataFrame(
            raw_pcs["pca_features"].apply(_parse_and_pad_fast).tolist(),
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


def load_ancestry_labels(gcp_project, LABELS_URI):
    """Loads predicted ancestry labels for each person."""
    print("    -> Loading genetic ancestry labels...")
    raw = pd.read_csv(LABELS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True},
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
    # Sanity-bound ages to handle data glitches
    demographics['AGE'] = demographics['AGE'].clip(lower=0, upper=120)
    demographics['AGE_sq'] = demographics['AGE'] ** 2

    # Set index and select final columns, dropping anyone with missing age info
    final_df = demographics[['person_id', 'AGE', 'AGE_sq']].dropna().set_index('person_id')

    print(f"    -> Successfully calculated stable age for {len(final_df):,} participants.")
    return final_df


def create_lock(path: str, payload: dict) -> bool:
    """Creates a lock file atomically, returning True if the lock was acquired."""
    try:
        # O_CREAT | O_EXCL is the atomic "create if not exists" operation
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f)
        return True
    except FileExistsError:
        return False
    except Exception:
        # In case of other errors, ensure we don't leave a broken lock file
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        return False

def release_lock(path: str) -> None:
    """Deletes a lock file if it exists."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # It's already gone, which is fine.
    except Exception as e:
        print(f"Warning: Failed to release lock '{path}': {e}")

def is_lock_stale(path: str, max_age_sec: float) -> bool:
    """
    Checks if a lock file is stale.
    Returns True if the lock is missing, invalid, or older than the threshold.
    """
    if not os.path.exists(path):
        return True
    try:
        with open(path, 'r') as f:
            payload = json.load(f)
        lock_time = payload.get("ts", 0)
        return (time.time() - lock_time) > max_age_sec
    except (json.JSONDecodeError, TypeError):
        # Invalid JSON or structure, treat as stale
        return True
    except Exception:
        # Any other read error, better to assume it's stale
        return True

def ensure_lock(path: str, max_age_sec: float) -> bool:
    """
    Ensures a lock is acquired. If a stale lock exists, it is removed.
    Returns True if the lock is now held by the current process.
    """
    lock_payload = {"pid": os.getpid(), "ts": time.time()}
    if os.path.exists(path) and is_lock_stale(path, max_age_sec):
        print(f"  -> Stale lock found: '{os.path.basename(path)}'. Reclaiming...")
        release_lock(path)

    return create_lock(path, lock_payload)

def parquet_n_rows(path: str) -> int | None:
    """Fast row-count via pyarrow metadata if present; return None on failure."""
    if not PYARROW_AVAILABLE or not os.path.exists(path):
        return None
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return None


def load_pheno_cases_from_cache(name, cache_dir, cdr_codename):
    path = os.path.join(cache_dir, f"pheno_{name}_{cdr_codename}.parquet")
    if not os.path.exists(path):
        return pd.Index([], dtype=str)
    df = pd.read_parquet(path, columns=['is_case'])
    if df.index.name != 'person_id':
        if 'person_id' in df.columns:
            df = df.set_index('person_id')
        else:
            return pd.Index([], dtype=str)
    if 'is_case' not in df.columns:
        return pd.Index([], dtype=str)
    case_ids = df.index[df['is_case'].astype('int8') == 1].astype(str)
    return case_ids
