from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
import hashlib
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


CACHE_VERSION_TAG = "phewas_v1"


def stable_hash(obj: Any, digest_size: int = 8) -> str:
    """Return a short, stable hash for arbitrary JSON-serializable payloads."""
    try:
        payload = json.dumps(obj, sort_keys=True, default=str)
    except TypeError:
        payload = str(obj)
    return hashlib.blake2s(payload.encode("utf-8"), digest_size=digest_size).hexdigest()

try:
    import psutil  # optional, for rss_gb()
    _PSUTIL = True
except Exception:
    _PSUTIL = False

try:
    import pyarrow.parquet as pq  # optional, for parquet_n_rows()
    _PYARROW = True
except Exception:
    _PYARROW = False

# ---------------------------------------------------------------------------
# Defaults (no env vars)
# ---------------------------------------------------------------------------
# Use SHM only for arrays up to this size (bytes)
_SHM_MAX_BYTES = 64 * 1024 * 1024  # 64 MB per array
# Keep at least this much free space in /dev/shm (bytes) before using it
_SHM_CUSHION = 256 * 1024 * 1024  # 256 MB headroom
# Disk memmap directory and cushion
_MEMMAP_DIR = tempfile.gettempdir()
_MEMMAP_CUSHION = 512 * 1024 * 1024  # 512 MB free required
_MEMMAP_PREFIX = "mm_ferromic_"
# SHM path (Linux); if not present, we skip SHM usage entirely
_SHM_PATH = "/dev/shm"

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def _best_effort_fsync(fobj) -> None:
    try:
        fobj.flush()
    except Exception:
        pass
    try:
        os.fsync(fobj.fileno())
    except Exception:
        pass


def _fsync_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    flags = getattr(os, "O_DIRECTORY", 0)
    if not flags:
        # Platforms without O_DIRECTORY (e.g., Windows/macOS) don't need an explicit dir fsync.
        return
    try:
        dir_fd = os.open(d, flags)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


def _disk_free_bytes(path: str) -> Optional[int]:
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return None


def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Atomic writers & simple caches
# ---------------------------------------------------------------------------
def atomic_write_json(path: str, data_obj: Any) -> None:
    """
    Write JSON atomically via a temp file then rename into place.
    Accepts dict-like or pandas Series.
    """
    tmpdir = os.path.dirname(path) or "."
    _ensure_dir(tmpdir)
    fd, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=os.path.basename(path) + ".tmp.")
    os.close(fd)
    try:
        if isinstance(data_obj, pd.Series):
            data_obj = data_obj.to_dict()

        class NpEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(tmp_path, "w") as f:
            json.dump(data_obj, f, cls=NpEncoder, ensure_ascii=False)
            _best_effort_fsync(f)
        os.replace(tmp_path, path)
        _fsync_dir(path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def atomic_write_parquet(path: str, df: pd.DataFrame, **to_parquet_kwargs) -> None:
    """Atomic parquet write via temp file then rename."""
    tmpdir = os.path.dirname(path) or "."
    _ensure_dir(tmpdir)
    fd, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=os.path.basename(path) + ".tmp.")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, **to_parquet_kwargs)
        with open(tmp_path, "rb") as f:
            _best_effort_fsync(f)
        os.replace(tmp_path, path)
        _fsync_dir(path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def atomic_write_pickle(path: str, obj: Any) -> None:
    """Atomic pickle write via temp file then rename."""
    tmpdir = os.path.dirname(path) or "."
    _ensure_dir(tmpdir)
    fd, tmp_path = tempfile.mkstemp(dir=tmpdir, prefix=os.path.basename(path) + ".tmp.")
    os.close(fd)
    try:
        pd.to_pickle(obj, tmp_path)
        with open(tmp_path, "rb") as f:
            _best_effort_fsync(f)
        os.replace(tmp_path, path)
        _fsync_dir(path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def read_meta_json(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not read corrupted meta file: {path}, Error: {e}")
        return None


def write_meta_json(path: str, meta: dict) -> None:
    atomic_write_json(path, meta)


def get_cached_or_generate(
    cache_path: str,
    generation_func,
    *args,
    validate_target: Optional[str] = None,
    validate_num_pcs: Optional[int] = None,
    lock_dir: Optional[str] = None,
    lock_timeout: float = 600.0,
    **kwargs,
) -> pd.DataFrame:
    """
    Generic caching wrapper with validation. Compatible with pre-existing caches.
    If an existing file fails checks, regenerate via generation_func.
    """

    def _valid_demographics(df: pd.DataFrame) -> bool:
        if not all(c in df.columns for c in ("AGE", "AGE_sq")):
            return False
        if not (is_numeric_dtype(df["AGE"]) and is_numeric_dtype(df["AGE_sq"])):
            return False
        vals = df[["AGE", "AGE_sq"]].astype(float)
        if vals.isna().any().any():
            return False
        return np.allclose(
            vals["AGE_sq"].to_numpy(),
            (vals["AGE"] ** 2).to_numpy(),
            rtol=0,
            atol=1e-6,
        )

    def _valid_inversion(df: pd.DataFrame) -> bool:
        if validate_target is None:
            return True
        if validate_target not in df.columns:
            return False
        s = pd.to_numeric(df[validate_target], errors="coerce")
        return is_numeric_dtype(s) and s.notna().sum() > 0 and s.nunique(dropna=True) > 1

    def _valid_pcs(df: pd.DataFrame) -> bool:
        if validate_num_pcs is None:
            return True
        expected = [f"PC{i}" for i in range(1, validate_num_pcs + 1)]
        if not set(expected).issubset(df.columns):
            return False
        return all(is_numeric_dtype(df[c]) and df[c].notna().any() for c in expected)

    def _valid_sex(df: pd.DataFrame) -> bool:
        if list(df.columns) != ["sex"]:
            return False
        if not is_numeric_dtype(df["sex"]):
            return False
        uniq = set(pd.unique(df["sex"].dropna()))
        return uniq.issubset({0, 1})

    def _needs_validation(path: str) -> bool:
        bn = os.path.basename(path)
        return (
            bn.startswith("demographics_")
            or bn.startswith("inversion_")
            or bn.startswith("pcs_")
            or bn.startswith("genetic_sex_")
        )

    def _validate(path: str, df: pd.DataFrame) -> bool:
        bn = os.path.basename(path)
        if bn.startswith("demographics_"):
            return _valid_demographics(df)
        if bn.startswith("inversion_"):
            return _valid_inversion(df)
        if bn.startswith("pcs_"):
            return _valid_pcs(df)
        if bn.startswith("genetic_sex_"):
            return _valid_sex(df)
        return True

    def _coerce_index(df: pd.DataFrame | Any) -> pd.DataFrame:
        out = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
        out = out.copy()
        out.index = out.index.astype(str)
        out.index.name = "person_id"
        return out

    def _load_existing() -> Optional[pd.DataFrame]:
        if not os.path.exists(cache_path):
            return None
        print(f"  -> Found cache, loading from '{cache_path}'...")
        try:
            df_loaded = pd.read_parquet(cache_path)
        except Exception as e:
            print(f"  -> Cache unreadable ({e}); regenerating...")
            return None

        df_loaded = _coerce_index(df_loaded)
        if _needs_validation(cache_path) and not _validate(cache_path, df_loaded):
            print(f"  -> Cache at '{cache_path}' failed validation; regenerating...")
            return None
        return df_loaded

    existing = _load_existing()
    if existing is not None:
        return existing

    lock_path = None
    lock_acquired = False
    if lock_dir:
        _ensure_dir(lock_dir)
        lock_name = f"{stable_hash(os.path.abspath(cache_path))}.lock"
        lock_path = os.path.join(lock_dir, lock_name)
        start = time.time()
        while True:
            got = ensure_lock(lock_path, max_age_sec=lock_timeout)
            if got:
                lock_acquired = True
                break
            time.sleep(0.25)
            existing = _load_existing()
            if existing is not None:
                return existing
            if (time.time() - start) > lock_timeout:
                raise TimeoutError(f"Timed out waiting for cache lock on '{cache_path}'")

    try:
        existing = _load_existing()
        if existing is not None:
            return existing

        print(f"  -> No cache found at '{cache_path}'. Generating data...")
        df = _coerce_index(generation_func(*args, **kwargs))
        atomic_write_parquet(cache_path, df)
        return df
    finally:
        if lock_acquired and lock_path:
            release_lock(lock_path)


def get_cached_or_generate_pickle(
    cache_path,
    generation_func,
    *args,
    lock_dir: Optional[str] = None,
    lock_timeout: float = 600.0,
    **kwargs,
):
    """Simple cache wrapper for pickled objects with optional locking."""

    def _load_existing_pickle():
        if not os.path.exists(cache_path):
            return None
        print(f"  -> Found cache, loading from '{cache_path}'...")
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"  -> Cache unreadable ({e}); regenerating...")
            return None

    existing = _load_existing_pickle()
    if existing is not None:
        return existing

    lock_path = None
    lock_acquired = False
    if lock_dir:
        _ensure_dir(lock_dir)
        lock_name = f"{stable_hash(os.path.abspath(cache_path))}.lock"
        lock_path = os.path.join(lock_dir, lock_name)
        start = time.time()
        while True:
            got = ensure_lock(lock_path, max_age_sec=lock_timeout)
            if got:
                lock_acquired = True
                break
            time.sleep(0.25)
            existing = _load_existing_pickle()
            if existing is not None:
                return existing
            if (time.time() - start) > lock_timeout:
                raise TimeoutError(f"Timed out waiting for cache lock on '{cache_path}'")

    try:
        existing = _load_existing_pickle()
        if existing is not None:
            return existing

        obj = generation_func(*args, **kwargs)
        atomic_write_pickle(cache_path, obj)
        return obj
    finally:
        if lock_acquired and lock_path:
            release_lock(lock_path)


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------
def create_lock(path: str, payload: dict) -> bool:
    """Create a lock file atomically, returning True if acquired."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        return True
    except FileExistsError:
        return False
    except Exception:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
        return False


def release_lock(path: str) -> None:
    """Delete a lock file if it exists."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: Failed to release lock '{path}': {e}")


def is_lock_stale(path: str, max_age_sec: float) -> bool:
    """Return True if lock missing/invalid/older than threshold."""
    if not os.path.exists(path):
        return True
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        lock_time = payload.get("ts", 0)
        return (time.time() - lock_time) > max_age_sec
    except (json.JSONDecodeError, TypeError):
        return True
    except Exception:
        return True


def ensure_lock(path: str, max_age_sec: float) -> bool:
    """Ensure a lock is acquired; remove stale lock if present."""
    lock_payload = {"pid": os.getpid(), "ts": time.time()}
    if os.path.exists(path) and is_lock_stale(path, max_age_sec):
        print(f"  -> Stale lock found: '{os.path.basename(path)}'. Reclaiming...")
        release_lock(path)
    return create_lock(path, lock_payload)


# ---------------------------------------------------------------------------
# Shared memory helpers (robust, SHM-aware, SIGBUS-resistant)
# ---------------------------------------------------------------------------
from multiprocessing import shared_memory  # noqa: E402  (import after __future__)

_SHM_PROBE_RESULT: Optional[bool] = None


def _shm_supported() -> bool:
    global _SHM_PROBE_RESULT
    if _SHM_PROBE_RESULT is not None:
        return _SHM_PROBE_RESULT
    try:
        probe = shared_memory.SharedMemory(create=True, size=1)
    except Exception:
        _SHM_PROBE_RESULT = False
        return False
    else:
        try:
            probe.close()
            probe.unlink()
        except Exception:
            pass
        _SHM_PROBE_RESULT = True
        return True


def _can_use_shm(arr_bytes: int) -> bool:
    if not _shm_supported():
        return False
    if arr_bytes <= 0 or arr_bytes > _SHM_MAX_BYTES:
        return False
    if os.path.exists(_SHM_PATH):
        free = _disk_free_bytes(_SHM_PATH)
        if free is not None:
            return (arr_bytes + _SHM_CUSHION) <= free
    # If /dev/shm is absent or free space can't be measured, optimistically try SHM;
    # allocation failures will fall back to memmap in create_shared_from_ndarray.
    return True


def _memmap_backing(arr: np.ndarray, readonly: bool) -> Tuple[dict, Any]:
    """Write arr to a disk-backed memmap and return (meta, handle)."""
    _ensure_dir(_MEMMAP_DIR)
    free = _disk_free_bytes(_MEMMAP_DIR)
    if free is not None and (arr.nbytes + _MEMMAP_CUSHION) > free:
        raise OSError(
            f"Insufficient free space in memmap dir '{_MEMMAP_DIR}'. "
            f"Need ~{arr.nbytes} bytes (+ cushion), free {free}."
        )

    fname = os.path.join(_MEMMAP_DIR, f"{_MEMMAP_PREFIX}{uuid.uuid4().hex}.dat")

    # Zero-sized arrays cannot be memmapped: handle separately (caller guards this).
    if arr.size == 0:
        # Build "empty" meta and a no-op handle
        meta = {
            "kind": "empty",
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "readonly": bool(readonly),
        }

        class _Handle:
            def close(self):  # no-op
                pass

            def unlink(self):  # no-op
                pass

        return meta, _Handle()

    # Write once with read/write mode
    mm = np.memmap(fname, mode="w+", dtype=arr.dtype, shape=arr.shape)
    try:
        np.copyto(mm, arr, casting="no")
        mm.flush()
    finally:
        # Release the memmap object; mapping persists for other processes
        del mm

    meta = {
        "kind": "memmap",
        "path": fname,
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "readonly": bool(readonly),
    }

    class _Handle:
        def close(self):  # nothing to close here
            pass

        def unlink(self):
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass
            except Exception:
                pass

    return meta, _Handle()


def create_shared_from_ndarray(arr: np.ndarray, readonly: bool = True) -> Tuple[dict, Any]:
    """
    Create a shareable backing for arr and return (meta, handle).

    Strategy:
      * Special-case zero-sized arrays (no SHM/memmap).
      * Prefer POSIX SHM for small arrays with sufficient headroom.
      * After SHM creation, re-check headroom to mitigate races; if tight, spill to memmap.
      * On any SHM copy failure, cleanly unlink and spill to memmap.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    # Zero-size: represent without SHM/memmap
    if arr.size == 0:
        meta = {
            "kind": "empty",
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "readonly": bool(readonly),
        }

        class _Handle:
            def close(self):  # no-op
                pass

            def unlink(self):  # no-op
                pass

        return meta, _Handle()

    # Fast path: SHM when safe
    if _can_use_shm(arr.nbytes):
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        try:
            # Post-create free-space check to shrink race window
            post_free = _disk_free_bytes(_SHM_PATH)
            if post_free is not None and post_free < _SHM_CUSHION:
                # Too tight – avoid copy and spill to disk
                try:
                    shm.close()
                    shm.unlink()
                finally:
                    pass
                return _memmap_backing(arr, readonly)

            view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            try:
                np.copyto(view, arr, casting="no")
            except BaseException:
                # Copy failed (e.g., SIGBUS window). Clean up & spill.
                try:
                    shm.close()
                    shm.unlink()
                finally:
                    pass
                return _memmap_backing(arr, readonly)

            if readonly:
                try:
                    view.setflags(write=False)
                except Exception:
                    pass

            meta = {
                "kind": "shm",
                "name": shm.name,
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
                "readonly": bool(readonly),
            }
            return meta, shm

        except BaseException:
            # Allocation failed after create=True? Ensure cleanup if shm exists.
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
            return _memmap_backing(arr, readonly)

    # Spill to disk memmap
    return _memmap_backing(arr, readonly)


def attach_shared_ndarray(meta: dict) -> Tuple[np.ndarray, Any]:
    """
    Attach to a shareable array created by create_shared_from_ndarray(meta).
    Returns (array_view, handle). Callers should 'close' the handle; the parent
    that created the resource should later 'unlink' it via dispose_shared(...).
    """
    kind = meta.get("kind")
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    readonly = bool(meta.get("readonly", True))

    if kind == "empty":
        arr = np.empty(shape, dtype=dtype)
        if readonly:
            try:
                arr.setflags(write=False)
            except Exception:
                pass

        class _Handle:
            def close(self):  # no-op
                pass

            def unlink(self):  # no-op
                pass

        return arr, _Handle()

    if kind == "memmap":
        path = meta["path"]
        mode = "r" if readonly else "r+"
        arr = np.memmap(path, mode=mode, dtype=dtype, shape=shape)
        if readonly:
            try:
                arr.setflags(write=False)
            except Exception:
                pass

        class _Handle:
            def close(self):  # deleting the memmap view is enough
                pass

            def unlink(self):  # parent cleans the file; children no-op
                pass

        return arr, _Handle()

    # POSIX SHM
    shm = shared_memory.SharedMemory(name=meta["name"])
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if readonly:
        try:
            arr.setflags(write=False)
        except Exception:
            pass
    return arr, shm


def dispose_shared(meta: Optional[dict], handle: Optional[Any], unlink: bool = True) -> None:
    """
    Best-effort cleanup for resources returned by create_shared_from_ndarray/attach_shared_ndarray.

    - For SHM (parent): call close(); if unlink=True, also unlink().
    - For memmap: parent unlink() removes the file; handles from attach() are no-ops.
    - For "empty": no-ops.
    """
    try:
        if handle is None:
            return
        try:
            handle.close()
        except Exception:
            pass
        if unlink:
            try:
                handle.unlink()
            except Exception:
                pass
    except Exception:
        pass


def rss_gb() -> float:
    """Resident set size (GB) of current process (0.0 if psutil unavailable)."""
    if not _PSUTIL:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Domain-specific loaders
# ---------------------------------------------------------------------------
def load_inversions(TARGET_INVERSION: str, INVERSION_DOSAGES_FILE: str) -> pd.DataFrame:
    """Load target inversion dosage; autodetect the identifier column."""
    try:
        header = pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", nrows=0).columns.tolist()
        id_candidates = [
            "SampleID",
            "sample_id",
            "person_id",
            "research_id",
            "participant_id",
            "ID",
        ]
        id_col = next((col for col in id_candidates if col in header), None)
        if id_col is None:
            raise RuntimeError(
                f"No identifier column found in '{INVERSION_DOSAGES_FILE}'. "
                f"Looked for {id_candidates}."
            )
        usecols = [id_col, TARGET_INVERSION]
        df = pd.read_csv(INVERSION_DOSAGES_FILE, sep="\t", usecols=usecols)
        df[TARGET_INVERSION] = pd.to_numeric(df[TARGET_INVERSION], errors="coerce")
        df[id_col] = df[id_col].astype(str)
        return df.set_index(id_col).rename_axis("person_id")
    except Exception as e:
        raise RuntimeError(f"Failed to load inversion data: {e}") from e


def load_pcs(gcp_project: str, PCS_URI: str, NUM_PCS: int) -> pd.DataFrame:
    """Load genetic PCs from a tsv with columns research_id, pca_features (list-like)."""
    try:
        raw_pcs = pd.read_csv(PCS_URI, sep="\t", storage_options={"project": gcp_project, "requester_pays": True})

        def _parse_and_pad_fast(s) -> list[float]:
            if pd.isna(s):
                return [np.nan] * NUM_PCS
            s = str(s).strip()
            if not (s.startswith("[") and s.endswith("]")):
                return [np.nan] * NUM_PCS
            body = s[1:-1]
            if not body:
                return [np.nan] * NUM_PCS
            try:
                vals = [float(x) for x in body.split(",")]
                if len(vals) < NUM_PCS:
                    vals.extend([np.nan] * (NUM_PCS - len(vals)))
                return vals[:NUM_PCS]
            except Exception:
                return [np.nan] * NUM_PCS

        pc_mat = pd.DataFrame(
            raw_pcs["pca_features"].apply(_parse_and_pad_fast).tolist(),
            columns=[f"PC{i}" for i in range(1, NUM_PCS + 1)],
        )
        pc_df = pc_mat.assign(person_id=raw_pcs["research_id"].astype(str)).set_index("person_id")
        return pc_df
    except Exception as e:
        raise RuntimeError(f"Failed to load PCs: {e}") from e


def load_genetic_sex(gcp_project: str, SEX_URI: str) -> pd.DataFrame:
    """Load genetically inferred sex (0=XX, 1=XY) as numeric column 'sex' indexed by person_id."""
    print("    -> Loading genetically-inferred sex (ploidy)...")
    sex_df = pd.read_csv(
        SEX_URI,
        sep="\t",
        storage_options={"project": gcp_project, "requester_pays": True},
        usecols=["research_id", "dragen_sex_ploidy"],
    )

    sex_df["sex"] = np.nan
    sex_df.loc[sex_df["dragen_sex_ploidy"] == "XX", "sex"] = 0
    sex_df.loc[sex_df["dragen_sex_ploidy"] == "XY", "sex"] = 1

    sex_df = sex_df.rename(columns={"research_id": "person_id"})
    sex_df["person_id"] = sex_df["person_id"].astype(str)

    return sex_df[["person_id", "sex"]].dropna().set_index("person_id")


def load_ancestry_labels(gcp_project: str, LABELS_URI: str) -> pd.DataFrame:
    """Load predicted ancestry labels into column 'ANCESTRY' indexed by person_id."""
    print("    -> Loading genetic ancestry labels...")
    raw = pd.read_csv(
        LABELS_URI,
        sep="\t",
        storage_options={"project": gcp_project, "requester_pays": True},
        usecols=["research_id", "ancestry_pred"],
    )
    df = raw.rename(columns={"research_id": "person_id", "ancestry_pred": "ANCESTRY"})
    df["person_id"] = df["person_id"].astype(str)
    df = df.dropna(subset=["ANCESTRY"])
    return df.set_index("person_id")[["ANCESTRY"]]


def load_related_to_remove(gcp_project: str, RELATEDNESS_URI: str) -> set[str]:
    """Load pre-computed list of related individuals to prune (as a set of strings)."""
    print("    -> Loading list of related individuals to exclude...")
    related_df = pd.read_csv(
        RELATEDNESS_URI,
        sep="\t",
        header=None,
        names=["person_id"],
        storage_options={"project": gcp_project, "requester_pays": True},
    )
    return set(related_df["person_id"].astype(str))


def load_demographics_with_stable_age(bq_client, cdr_id: str) -> pd.DataFrame:
    """
    Compute stable age per participant as (last observation year - YOB),
    returning columns AGE and AGE_sq indexed by person_id.
    """
    print("    -> Generating stable, reproducible age covariate...")

    yob_q = f"SELECT person_id, year_of_birth FROM `{cdr_id}.person`"
    yob_df = bq_client.query(yob_q).to_dataframe()
    yob_df["person_id"] = yob_df["person_id"].astype(str)

    obs_q = f"""
        SELECT person_id, EXTRACT(YEAR FROM MAX(observation_period_end_date)) AS obs_end_year
        FROM `{cdr_id}.observation_period`
        GROUP BY person_id
    """
    obs_df = bq_client.query(obs_q).to_dataframe()
    obs_df["person_id"] = obs_df["person_id"].astype(str)

    demographics = pd.merge(yob_df, obs_df, on="person_id", how="inner")
    demographics["year_of_birth"] = pd.to_numeric(demographics["year_of_birth"], errors="coerce")
    demographics["AGE"] = demographics["obs_end_year"] - demographics["year_of_birth"]
    demographics["AGE"] = demographics["AGE"].clip(lower=0, upper=120)
    demographics["AGE_sq"] = demographics["AGE"] ** 2

    final_df = demographics[["person_id", "AGE", "AGE_sq"]].dropna().set_index("person_id")
    print(f"    -> Successfully calculated stable age for {len(final_df):,} participants.")
    return final_df

# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def parquet_n_rows(path: str) -> Optional[int]:
    """Fast row-count via pyarrow metadata if present; return None on failure."""
    if not _PYARROW or not os.path.exists(path):
        return None
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return None


def load_pheno_cases_from_cache(name: str, cache_dir: str, cdr_codename: str) -> pd.Index:
    path = os.path.join(cache_dir, f"pheno_{name}_{cdr_codename}.parquet")
    if not os.path.exists(path):
        return pd.Index([], dtype=str)
    df = pd.read_parquet(path, columns=["is_case"])
    if df.index.name != "person_id":
        if "person_id" in df.columns:
            df = df.set_index("person_id")
        else:
            return pd.Index([], dtype=str)
    if "is_case" not in df.columns:
        return pd.Index([], dtype=str)
    case_ids = df.index[df["is_case"].astype("int8") == 1].astype(str)
    return case_ids
