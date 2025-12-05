import os
import math
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from datetime import datetime
import time
import warnings
import gc
import threading
import faulthandler
import sys
import traceback
import json
import importlib
import importlib.util
import hashlib
from typing import Callable, Optional, Sequence, Tuple
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False


import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    _bigquery_spec = importlib.util.find_spec("google.cloud.bigquery")
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal environments
    _bigquery_spec = None

if _bigquery_spec is not None:
    bigquery = importlib.import_module("google.cloud.bigquery")
else:  # pragma: no cover - exercised in minimal environments
    from types import SimpleNamespace

    def _missing_bigquery_client(*args, **kwargs):
        raise ModuleNotFoundError(
            "google-cloud-bigquery is required to access remote cohort data. "
            "Install google-cloud-bigquery or patch 'phewas.run.bigquery.Client' with a stub."
        )

    bigquery = SimpleNamespace(Client=_missing_bigquery_client)  # type: ignore
from statsmodels.stats.multitest import multipletests
from scipy import stats

from . import categories
from . import inversion_frequency
from . import iox as io
from . import pheno
from . import pipes
from . import models
from . import testing

from statsmodels.tools.sm_exceptions import ConvergenceWarning

# 1. RuntimeWarning: overflow encountered in exp
warnings.filterwarnings('ignore', message='overflow encountered in exp', category=RuntimeWarning)

# 2. RuntimeWarning: divide by zero encountered in log
warnings.filterwarnings('ignore', message='divide by zero encountered in log', category=RuntimeWarning)

# 3. ConvergenceWarning: QC check did not pass
warnings.filterwarnings('ignore', message=r'QC check did not pass', category=ConvergenceWarning)

# 4. ConvergenceWarning: Could not trim params automatically
warnings.filterwarnings('ignore', message=r'Could not trim params automatically', category=ConvergenceWarning)

try:
    faulthandler.enable()
except Exception:
    pass

def _global_excepthook(exc_type, exc, tb):
    """
    Uncaught exception hook that prints a full stack trace immediately across threads and subprocesses.
    """
    print("[TRACEBACK] Uncaught exception:", flush=True)
    traceback.print_exception(exc_type, exc, tb)
    sys.stderr.flush()

sys.excepthook = _global_excepthook

def _thread_excepthook(args):
    _global_excepthook(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = _thread_excepthook


def _merge_followup_results(
    stage1_df: pd.DataFrame, follow_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge stage-2 follow-up metrics into the consolidated stage-1 dataframe.

    The follow-up cache reuses generic column names (for example, "P_Source")
    that also exist in the stage-1 results.  Pandas will raise a ``MergeError``
    when it encounters overlapping column names and the default suffixes would
    still collide.  To keep the merge well-defined and preserve the stage-1
    metrics, rename the follow-up columns under a "Stage2_"-prefixed namespace.
    If a unique prefixed column cannot be created (for instance, because the
    follow-up file already includes that name), fall back to appending a
    numerical suffix.  As a final guard, drop any columns that still collide
    after renaming.

    Parameters
    ----------
    stage1_df:
        The dataframe that aggregates all stage-1 results.
    follow_df:
        The dataframe containing the stage-2 follow-up metrics to merge.

    Returns
    -------
    pandas.DataFrame
        A new dataframe containing all stage-1 results with follow-up metrics
        merged in on ``["Phenotype", "Inversion"]``.
    """

    if follow_df.empty:
        return stage1_df

    sanitized_follow = follow_df.copy()
    stage1_cols = set(stage1_df.columns)

    overlap_cols = (
        set(sanitized_follow.columns).intersection(stage1_cols)
        - {"Phenotype", "Inversion"}
    )

    rename_map: dict[str, str] = {}
    for col in sorted(overlap_cols):
        base_name = f"Stage2_{col}"
        candidate = base_name
        suffix = 2
        while candidate in sanitized_follow.columns or candidate in stage1_cols:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        rename_map[col] = candidate

    if rename_map:
        sanitized_follow = sanitized_follow.rename(columns=rename_map)

    residual_overlap = (
        set(sanitized_follow.columns).intersection(stage1_cols)
        - {"Phenotype", "Inversion"}
    )
    if residual_overlap:
        sanitized_follow = sanitized_follow.drop(
            columns=list(residual_overlap), errors="ignore"
        )

    return stage1_df.merge(sanitized_follow, on=["Phenotype", "Inversion"], how="left")


def _stable_seed(*parts: object) -> int:
    """Return a deterministic 32-bit seed derived from the provided parts."""
    hasher = hashlib.blake2s(digest_size=4)
    for part in parts:
        if part is None:
            data = b"<none>"
        else:
            data = str(part).encode("utf-8", "surrogatepass")
        hasher.update(len(data).to_bytes(4, "big"))
        hasher.update(data)
    return int.from_bytes(hasher.digest(), "big")


def _infer_bootstrap_ceiling(num_phenotypes: int, num_inversions: int, alpha: float) -> int:
    """Return a bootstrap draw ceiling that resolves BH-adjusted discoveries."""
    try:
        phenos = max(1, int(num_phenotypes))
        inversions = max(1, int(num_inversions))
    except Exception:
        return models.BOOTSTRAP_MAX_B

    total_tests = phenos * inversions
    if total_tests <= 0:
        return models.BOOTSTRAP_MAX_B

    alpha_eff = float(alpha) if alpha and float(alpha) > 0 else 0.05
    alpha_eff = max(alpha_eff, 1e-9)
    required = math.ceil(total_tests / alpha_eff)
    required = max(required, models.BOOTSTRAP_MAX_B)

    # The worker doubles its target draw count, so round up to the next power of two
    if required & (required - 1):
        required = 1 << (required - 1).bit_length()

    hard_cap = 1 << 31  # ~2.1 billion draws; protects against runaway estimates
    return int(min(required, hard_cap))


def _create_progress_emitter(
    total: int,
    log_prefix: str,
    stage_label: str,
    *,
    max_updates: int = 20,
) -> Callable[[int], None]:
    """Return a callable that prints coarse progress updates for long-running stages."""

    try:
        total_int = int(total)
    except Exception:
        total_int = 0

    total_int = max(0, total_int)
    if total_int == 0:
        prefix = f"{log_prefix} " if log_prefix else ""
        printed = threading.Event()
        lock = threading.Lock()

        def _noop(_: int) -> None:
            if printed.is_set():
                return
            with lock:
                if printed.is_set():
                    return
                print(
                    f"{prefix}[Stage] {stage_label}: 0/0 (0.0%)",
                    flush=True,
                )
                printed.set()

        return _noop

    step = max(1, total_int // max(1, int(max_updates)))
    prefix = f"{log_prefix} " if log_prefix else ""
    lock = threading.Lock()

    def _emit(current: int) -> None:
        nonlocal step
        try:
            cur = int(current)
        except Exception:
            cur = total_int

        cur = max(0, min(cur, total_int))
        should_print = cur == 0 or cur == total_int or (cur % step == 0)
        if not should_print:
            return

        pct = (cur / total_int) * 100 if total_int else 100.0
        with lock:
            print(
                f"{prefix}[Stage] {stage_label}: {cur}/{total_int} ({pct:5.1f}%)",
                flush=True,
            )

    return _emit


class SystemMonitor(threading.Thread):
    """
    A thread that monitors and reports system resource usage periodically.
    It is also a thread-safe data provider for the ResourceGovernor.
    """
    def __init__(self, interval=2):
        super().__init__(daemon=True)
        self.interval = interval
        self._lock = threading.Lock()
        self._main_process = None
        if PSUTIL_AVAILABLE:
            try:
                self._main_process = psutil.Process()
            except psutil.NoSuchProcess:
                self._main_process = None

        # Thread-safe public fields
        self.sys_cpu_percent = 0.0
        self.sys_available_gb = 0.0
        self.app_rss_gb = 0.0
        self.app_cpu_percent = 0.0
        self._sample_ts = 0.0

    def snapshot(self) -> 'ResourceSnapshot':
        """Returns a thread-safe snapshot of the current stats."""
        with self._lock:
            return ResourceSnapshot(
                ts=self._sample_ts,
                sys_cpu_percent=self.sys_cpu_percent,
                sys_available_gb=self.sys_available_gb,
                app_rss_gb=self.app_rss_gb,
                app_cpu_percent=self.app_cpu_percent,
            )

    def run(self):
        """Monitors and reports system stats until the main program exits."""
        if not self._main_process:
            print("[SysMonitor] Could not find main process to monitor.", flush=True)
            return

        # Prime per-process cpu_percent baselines
        try:
            self._main_process.cpu_percent(interval=None)
            for p in self._main_process.children(recursive=True):
                try:
                    p.cpu_percent(interval=None)
                except Exception:
                    pass
        except Exception:
            pass

        while True:
            try:
                cpu = psutil.cpu_percent(interval=self.interval)
                mem = psutil.virtual_memory()
                ram_percent = mem.percent
                host_available_gb = mem.available / (1024**3)
                cgroup_available = pipes.cgroup_available_gb()
                if cgroup_available is not None:
                    available_gb = min(host_available_gb, cgroup_available)
                else:
                    available_gb = host_available_gb
                child_processes = self._main_process.children(recursive=True)
                main_mem = self._main_process.memory_info()
                child_mem = sum(p.memory_info().rss for p in child_processes)
                total_rss_gb = (main_mem.rss + child_mem) / (1024**3)
                n_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 1

                app_cpu_raw = 0.0
                try:
                    app_cpu_raw += self._main_process.cpu_percent(interval=None)
                except Exception:
                    pass
                for c in child_processes:
                    try:
                        app_cpu_raw += c.cpu_percent(interval=None)
                    except Exception:
                        pass
                app_cpu = min(100.0, app_cpu_raw / n_cpus)

                with self._lock:
                    self.sys_cpu_percent = cpu
                    self.sys_available_gb = available_gb
                    self.app_rss_gb = total_rss_gb
                    self.app_cpu_percent = app_cpu
                    self._sample_ts = time.time()

                if cgroup_available is not None:
                    avail_str = f"{available_gb:.2f}GB (cg:{cgroup_available:.2f}GB)"
                else:
                    avail_str = f"{available_gb:.2f}GB"

                print(
                    f"[SysMonitor] CPU: {cpu:5.1f}% | AppCPU: {app_cpu:5.1f}% | RAM: {ram_percent:5.1f}% "
                    f"(avail: {avail_str}) | App RSS: {total_rss_gb:.2f}GB | "
                    f"Budget: {pipes.BUDGET.remaining_gb():.2f}/{pipes.BUDGET._total_gb:.2f}GB",
                    flush=True,
                )

                try:
                    prog = pipes.PROGRESS.snapshot()
                    by_inv = {}
                    for (inv, stage), (d, q, ts) in prog.items():
                        prev = by_inv.get(inv)
                        if (prev is None) or (ts > prev[-1]):
                            pct = int((100*d/q)) if q else 0
                            by_inv[inv] = (stage, d, q, pct, ts)
                    if by_inv:
                        parts = [
                            f"{inv}:{stage} {pct}%" for inv, (stage, _, _, pct, _) in sorted(by_inv.items())
                        ]
                        print("[Progress] " + " | ".join(parts), flush=True)
                except Exception:
                    pass
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                print(f"[SysMonitor] Error: {e}", flush=True)

from collections import deque
from dataclasses import dataclass

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

@dataclass
class ResourceSnapshot:
    ts: float
    sys_cpu_percent: float
    sys_available_gb: float
    app_rss_gb: float
    app_cpu_percent: float

class ResourceGovernor:
    def __init__(self, monitor: SystemMonitor | None, history_sec: int = 30):
        self._monitor = monitor
        self._lock = threading.Lock()
        interval = 1
        if monitor is not None:
            try:
                interval = int(getattr(monitor, "interval", 1) or 1)
            except Exception:
                interval = 1
        maxlen = max(1, history_sec // interval) if history_sec else None
        self._history = deque(maxlen=maxlen)
        self.observed_core_df_gb = []
        self.observed_steady_state_gb = []
        self.mem_guard_gb = 4.0

    def _update_history(self):
        if not self._monitor:
            return
        snap = self._monitor.snapshot()
        if snap.ts <= 0:
            return
        if not self._history or self._history[-1].ts != snap.ts:
            self._history.append(snap)


    def can_admit_next(self, predicted_extra_gb: float) -> bool:
        self._update_history()
        if not self._history:
            return True

        with self._lock:
            latest_mem_gb = self._history[-1].sys_available_gb
            budget_avail = pipes.BUDGET.remaining_gb()
            effective_avail = min(latest_mem_gb, budget_avail)
            mem_ok = (effective_avail - predicted_extra_gb) >= self.mem_guard_gb
            if not mem_ok:
                print(
                    f"[Governor] Hold: mem_avail~{effective_avail:.2f}GB (budget {budget_avail:.2f}GB), "
                    f"pred_cost={predicted_extra_gb:.2f}GB",
                    flush=True,
                )
            return mem_ok

    def predict_extra_gb_before_pool(self, N: int, C: int) -> float:
        base_estimate = (N * C * 8 / 1024**3) * 1.6
        with self._lock:
            if not self.observed_core_df_gb: return base_estimate
            return max(base_estimate, np.percentile(self.observed_core_df_gb, 75))

    def predict_extra_gb_after_pool(self) -> float:
        with self._lock:
            if not self.observed_steady_state_gb:
                core_df_pred = np.percentile(self.observed_core_df_gb, 75) if self.observed_core_df_gb else 1.0
                return core_df_pred * 1.3
            return np.percentile(self.observed_steady_state_gb, 75)

    def update_after_core_df(self, delta_gb: float):
        with self._lock:
            self.observed_core_df_gb.append(delta_gb)
        print(f"[Governor] Observed core_df memory delta: +{delta_gb:.2f}GB")

    def update_steady_state(self, delta_gb: float):
        with self._lock:
            self.observed_steady_state_gb.append(delta_gb)
        print(f"[Governor] Observed steady-state memory delta: +{delta_gb:.2f}GB")

    def dynamic_floor_callable(self) -> float:
        """The memory floor for the submission throttle, made empirical."""
        base_floor = self.mem_guard_gb
        predicted_ss_footprint = self.predict_extra_gb_after_pool()
        # Raise the floor based on the predicted steady-state footprint of one inversion
        return max(base_floor, predicted_ss_footprint + 0.5)

class MultiTenantGovernor(ResourceGovernor):
    def __init__(self, monitor, history_sec=30):
        super().__init__(monitor, history_sec)
        self.inv_pools = {}
        self.inv_rss_gb = {}
        self.observed_steady_state_gb_per_inv = []

    def register_pool(self, inv_id, pids):
        self.inv_pools[inv_id] = list(pids)

    def deregister_pool(self, inv_id):
        self.inv_pools.pop(inv_id, None)
        self.inv_rss_gb.pop(inv_id, None)

    def measure_inv(self, inv_id):
        """
        Returns the proportional working set of the pool keyed by inv_id in GB.
        Uses PSS from /proc/<pid>/smaps_rollup to avoid double-counting shared pages.
        Falls back to USS via psutil if PSS is unavailable. As a last resort uses RSS.
        """
        if not PSUTIL_AVAILABLE:
            self.inv_rss_gb[inv_id] = 0.0
            return 0.0
    
        def _pss_kb(pid: int) -> int:
            path = f"/proc/{pid}/smaps_rollup"
            try:
                with open(path, "r") as fh:
                    for line in fh:
                        if line.startswith("Pss:"):
                            parts = line.split()
                            return int(parts[1])
            except FileNotFoundError:
                return -1
            except PermissionError:
                return -1
            except Exception:
                return -1
            return -1
    
        total_bytes = 0
        pids = self.inv_pools.get(inv_id, [])
        if not pids:
            self.inv_rss_gb[inv_id] = 0.0
            return 0.0
    
        valid_pids = []
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                pss_kb = _pss_kb(pid)
                if pss_kb >= 0:
                    total_bytes += pss_kb * 1024
                else:
                    try:
                        finfo = proc.memory_full_info()
                        # USS is private bytes; avoids counting shared pages multiple times.
                        total_bytes += getattr(finfo, "uss", 0)
                    except Exception:
                        total_bytes += proc.memory_info().rss
                valid_pids.append(pid)
            except psutil.NoSuchProcess:
                pass
            except Exception:
                pass
    
        if len(valid_pids) < len(pids):
            self.inv_pools[inv_id] = valid_pids
    
        measured_gb = total_bytes / (1024**3)
        self.inv_rss_gb[inv_id] = measured_gb
        return measured_gb


    def total_active_footprint(self):
        return sum(self.inv_rss_gb.values())

    def update_steady_state(self, inv_id, measured_gb):
        with self._lock:
            self.observed_steady_state_gb_per_inv.append(measured_gb)
        print(f"[Governor] Observed steady-state for {inv_id}: {measured_gb:.2f}GB")

    def predict_extra_gb_after_pool(self) -> float:
        with self._lock:
            values = list(self.observed_steady_state_gb_per_inv)
        if not values:
            return super().predict_extra_gb_after_pool()
        return np.percentile(values, 75)

    def can_admit_next_inv(self, predicted_gb):
        self._update_history()
        if not self._history:
            return True
        latest_avail = self._history[-1].sys_available_gb
        budget_avail = pipes.BUDGET.remaining_gb()
        effective_avail = min(latest_avail, budget_avail)
        active = self.total_active_footprint()
        guard = self.mem_guard_gb if active > 0 else 0.0
        mem_ok = (effective_avail - predicted_gb) >= guard
        if active == 0 and mem_ok:
            return True
        if not mem_ok:
            print(
                f"[Governor] Hold: mem_avail~{effective_avail:.2f}GB (budget {budget_avail:.2f}GB), "
                f"active={active:.2f}GB, next_pred={predicted_gb:.2f}GB",
                flush=True,
            )
        return mem_ok

# --- Configuration ---
TARGET_INVERSIONS = {
    #'chr7-73113990-INV-1685041',
    #'chr16-15028481-INV-133352',
    'chr10-79542902-INV-674513',
    #'chr16-28471894-INV-165758',
    #'chr7-65219158-INV-312667',
    'chr17-45585160-INV-706887',
    #'chr15-30618104-INV-1535102',
    'chr8-7301025-INV-5297356',
    'chr12-46897663-INV-16289',
    #'chr7-54220528-INV-101153'
}

"""
Low imputation:
1. **chr7-73113990-INV-1685041**
2. **chr16-15028481-INV-133352**
3. **chr16-28471894-INV-165758**
4. **chr7-65219158-INV-312667**
5. **chr15-30618104-INV-1535102**
6. **chr7-54220528-INV-101153**
"""

# chr17-45974480-INV-29218 is arbitrary number for tagging SNP 17q21

PHENOTYPE_DEFINITIONS_URL = "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/significant_heritability_diseases.tsv"
MASTER_RESULTS_CSV = f"phewas_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.tsv"

# --- Performance & Memory Tuning ---
MIN_AVAILABLE_MEMORY_GB = 4.0
QUEUE_MAX_SIZE = os.cpu_count() * 4
LOADER_THREADS = 32
LOADER_CHUNK_SIZE = 128

# --- Data sources and caching ---
CACHE_DIR = "./phewas_cache"
LOCK_DIR = os.path.join(CACHE_DIR, "locks")
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

CACHE_VERSION_TAG = io.CACHE_VERSION_TAG

# --- Model parameters ---
NUM_PCS = 16
DEFAULT_MIN_CASES_FILTER = int(pheno.MIN_CASES_FILTER)
DEFAULT_MIN_CONTROLS_FILTER = int(pheno.MIN_CONTROLS_FILTER)
MIN_CASES_FILTER = DEFAULT_MIN_CASES_FILTER
MIN_CONTROLS_FILTER = DEFAULT_MIN_CONTROLS_FILTER
CLI_MIN_CASES_CONTROLS_OVERRIDE: Optional[int] = None
MIN_NEFF_FILTER = 0 # Default off
FDR_ALPHA = 0.05

# --- Population filter ---
POPULATION_FILTER = "all"

_pop_filter_env = os.getenv("FERROMIC_POPULATION_FILTER")
if _pop_filter_env is not None:
    POPULATION_FILTER = _pop_filter_env.strip() or "all"

# --- Phenotype filter ---
PHENOTYPE_FILTER: Optional[str] = None

_pheno_filter_env = os.getenv("FERROMIC_PHENOTYPE_FILTER")
if _pheno_filter_env is not None:
    PHENOTYPE_FILTER = _pheno_filter_env.strip() or None


def _apply_pipeline_config(pipeline_config: Optional[dict[str, object]] = None) -> None:
    """Synchronize module globals and environment variables with the pipeline config."""

    global CLI_MIN_CASES_CONTROLS_OVERRIDE, MIN_CASES_FILTER, MIN_CONTROLS_FILTER
    global POPULATION_FILTER, PHENOTYPE_FILTER

    config = pipeline_config or {}

    min_override = config.get("min_cases_controls")
    if min_override is not None:
        CLI_MIN_CASES_CONTROLS_OVERRIDE = int(min_override)
        MIN_CASES_FILTER = int(min_override)
        MIN_CONTROLS_FILTER = int(min_override)
    else:
        CLI_MIN_CASES_CONTROLS_OVERRIDE = None
        MIN_CASES_FILTER = DEFAULT_MIN_CASES_FILTER
        MIN_CONTROLS_FILTER = DEFAULT_MIN_CONTROLS_FILTER

    pheno.MIN_CASES_FILTER = MIN_CASES_FILTER
    pheno.MIN_CONTROLS_FILTER = MIN_CONTROLS_FILTER

    population_label = config.get("population_filter", POPULATION_FILTER)
    normalized_population = str(population_label).strip() or "all"
    POPULATION_FILTER = normalized_population
    if normalized_population == "all":
        os.environ.pop("FERROMIC_POPULATION_FILTER", None)
    else:
        os.environ["FERROMIC_POPULATION_FILTER"] = normalized_population

    phenotype_filter = config.get("phenotype_filter")
    PHENOTYPE_FILTER = (str(phenotype_filter).strip() or None) if phenotype_filter is not None else None
    if PHENOTYPE_FILTER is None:
        os.environ.pop("FERROMIC_PHENOTYPE_FILTER", None)
    else:
        os.environ["FERROMIC_PHENOTYPE_FILTER"] = PHENOTYPE_FILTER


def _normalize_population_label(label: Optional[str]) -> str:
    """Return a canonical, lower-case population label suitable for comparisons."""
    if label is None:
        return ""
    normalized = str(label).strip().lower()
    return normalized


def _apply_population_filter(
    covariates_df: pd.DataFrame,
    ancestry_series: pd.Series,
    population_filter: str,
) -> tuple[pd.DataFrame, pd.Series, str, bool]:
    """Filter shared covariates and ancestry labels by the requested population."""

    normalized = _normalize_population_label(population_filter)
    ancestry_str = ancestry_series.astype("string").str.strip().str.lower()
    ancestry_str = ancestry_str.reindex(covariates_df.index)

    def _restore_object_dtype(series: pd.Series) -> pd.Series:
        series_obj = series.astype(object)
        # Ensure pandas uses ``np.nan`` sentinels for missing values to match historical behaviour.
        return series_obj.where(~series.isna(), np.nan)

    if not normalized or normalized == "all":
        return covariates_df, _restore_object_dtype(ancestry_str), "all", True
    available_labels = sorted(
        {label for label in ancestry_str.dropna().unique().tolist() if label is not None}
    )
    if normalized not in available_labels:
        raise RuntimeError(
            "Requested population filter '"
            f"{population_filter}"
            "' does not match any available ancestry labels."
        )

    mask = ancestry_str.eq(normalized)
    keep_ids = ancestry_str.index[mask.fillna(False)]
    filtered_covariates = covariates_df.loc[keep_ids]
    filtered_ancestry = ancestry_str.loc[keep_ids]
    if filtered_covariates.empty:
        raise RuntimeError(
            "Population filter '"
            f"{population_filter}"
            "' removed all participants; check your labels and filter."
        )

    print(
        f"[Config] Restricting analysis to population '{normalized}' "
        f"({len(filtered_covariates)} participants).",
        flush=True,
    )
    return filtered_covariates, _restore_object_dtype(filtered_ancestry), normalized, False

# --- Testing configuration (centralized in testing.py) ---
_cat_env_overrides = {}

def _maybe_parse_env(name: str, cast):
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return cast(raw)
    except Exception:
        print(f"[Config] Ignoring invalid value for {name!s}: {raw!r}", flush=True)
        return None

env_method = os.getenv("CAT_METHOD")
if env_method:
    _cat_env_overrides["CAT_METHOD"] = env_method.strip()

env_shrinkage = os.getenv("CAT_SHRINKAGE")
if env_shrinkage:
    _cat_env_overrides["CAT_SHRINKAGE"] = env_shrinkage.strip()

gbj_draws = _maybe_parse_env("CAT_GBJ_B", int)
if gbj_draws is not None:
    _cat_env_overrides["CAT_GBJ_B"] = max(int(gbj_draws), 1)

gbj_max_draws = _maybe_parse_env("CAT_GBJ_MAX", int)
if gbj_max_draws is not None:
    _cat_env_overrides["CAT_GBJ_MAX"] = max(int(gbj_max_draws), 1)

z_cap = _maybe_parse_env("CAT_Z_CAP", float)
if z_cap is not None:
    _cat_env_overrides["CAT_Z_CAP"] = float(z_cap)

lambda_override = _maybe_parse_env("CAT_LAMBDA", float)
if lambda_override is not None:
    _cat_env_overrides["CAT_LAMBDA"] = float(lambda_override)

min_k_override = _maybe_parse_env("CAT_MIN_K", int)
if min_k_override is not None:
    _cat_env_overrides["CAT_MIN_K"] = max(int(min_k_override), 1)

seed_override = _maybe_parse_env("CAT_SEED_BASE", int)
if seed_override is not None:
    _cat_env_overrides["CAT_SEED_BASE"] = int(seed_override)

tctx = testing.get_testing_ctx(_cat_env_overrides or None)
PHENO_PROTECT = set()
MAX_CONCURRENT_INVERSIONS_DEFAULT = tctx["MAX_CONCURRENT_INVERSIONS_DEFAULT"]
MAX_CONCURRENT_INVERSIONS_BOOT = tctx["MAX_CONCURRENT_INVERSIONS_BOOT"]

# --- Per-ancestry thresholds and multiple-testing for ancestry splits ---
PER_ANC_MIN_CASES = 100
PER_ANC_MIN_CONTROLS = 100
ANCESTRY_ALPHA = 0.05
ANCESTRY_P_ADJ_METHOD = "fdr_bh"
LRT_SELECT_ALPHA = 0.05

# --- Regularization strength for ridge fallback in unstable fits ---
RIDGE_L2_BASE = 1.0

# --- Suppress pandas warnings ---
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)

class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start_time = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

def _find_upwards(pathname: str) -> str:
    """
    Resolves a filesystem path for a filename by searching the current working directory
    and then walking up parent directories until the file is found. Returns the absolute
    path when found; returns the original pathname if not found.
    """
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
    return pathname


def _source_key(*parts) -> str:
    return io.stable_hash({"parts": parts, "version": CACHE_VERSION_TAG})


def _effective_case_control_thresholds() -> Tuple[int, int]:
    """Return the minimum case/control thresholds enforced for this run."""
    if CLI_MIN_CASES_CONTROLS_OVERRIDE is not None:
        threshold = int(CLI_MIN_CASES_CONTROLS_OVERRIDE)
        return threshold, threshold
    return int(MIN_CASES_FILTER), int(MIN_CONTROLS_FILTER)


def _prefilter_thresholds() -> Tuple[int, int]:
    """Return the case/control thresholds used to prefilter phenotypes."""
    return _effective_case_control_thresholds()


def _pipeline_once(pipeline_config: Optional[dict[str, object]] = None):
    """
    Entry point for the PheWAS pipeline. Uses module-level configuration directly.
    """
    _apply_pipeline_config(pipeline_config)

    script_start_time = time.time()

    # Keep the phenotype module thresholds in sync with any overrides applied here.
    global MIN_CASES_FILTER, MIN_CONTROLS_FILTER
    effective_min_cases, effective_min_ctrls = _effective_case_control_thresholds()
    MIN_CASES_FILTER = effective_min_cases
    MIN_CONTROLS_FILTER = effective_min_ctrls
    pheno.MIN_CASES_FILTER = effective_min_cases
    pheno.MIN_CONTROLS_FILTER = effective_min_ctrls
    prefilter_min_cases, prefilter_min_ctrls = _prefilter_thresholds()

    if PSUTIL_AVAILABLE:
        monitor_thread = SystemMonitor(interval=3)
        monitor_thread.start()
    else:
        monitor_thread = None
    pipes.BUDGET.init_total(fraction=0.92)

    def mem_floor_callable():
        """
        Returns the memory floor used by the submission throttle.
        Uses the empirical governor prediction of steady-state footprint for this workload,
        with a small guard to prevent thrashing. This keeps task submission moving even when
        the static fraction-based floor would be overly conservative.
        """
        # Lower-bound at 4 GB guard; raise to predicted steady state + small cushion.
        return max(4.0, governor.dynamic_floor_callable())


    allow_ancestry_followups = True
    population_filter_label = "all"


    print("=" * 70)
    print(" Starting Robust, Parallel PheWAS Pipeline")
    print("=" * 70)

    global TARGET_INVERSIONS
    if isinstance(TARGET_INVERSIONS, str):
        TARGET_INVERSIONS = {TARGET_INVERSIONS}

    from types import SimpleNamespace
    run = SimpleNamespace(TARGET_INVERSIONS=TARGET_INVERSIONS)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LOCK_DIR, exist_ok=True)

    bootstrap_draw_cap = models.BOOTSTRAP_MAX_B

    try:
        with Timer() as t_setup:
            print("\n--- Loading shared data... ---")
            pheno_defs_df = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)
            
            # Apply phenotype filter if specified
            if PHENOTYPE_FILTER is not None:
                original_count = len(pheno_defs_df)
                pheno_defs_df = pheno_defs_df[pheno_defs_df['sanitized_name'] == PHENOTYPE_FILTER].copy()
                filtered_count = len(pheno_defs_df)
                if filtered_count == 0:
                    raise ValueError(f"No phenotype found matching filter: '{PHENOTYPE_FILTER}'")
                print(f"[Config] Phenotype filter applied: '{PHENOTYPE_FILTER}' ({filtered_count}/{original_count} phenotypes)", flush=True)
            
            bootstrap_draw_cap = _infer_bootstrap_ceiling(
                num_phenotypes=pheno_defs_df.shape[0],
                num_inversions=len(run.TARGET_INVERSIONS),
                alpha=FDR_ALPHA,
            )
            print(
                f"[Config] Bootstrap draw ceiling set to {bootstrap_draw_cap} "
                f"for {pheno_defs_df.shape[0]} phenotypes across {len(run.TARGET_INVERSIONS)} inversions (alpha={FDR_ALPHA}).",
                flush=True,
            )
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split(".")[-1]
            demographics_cache_path = os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet")
            demographics_df = io.get_cached_or_generate(
                demographics_cache_path,
                io.load_demographics_with_stable_age,
                bq_client=bq_client,
                cdr_id=cdr_dataset_id,
                lock_dir=LOCK_DIR,
            )
            pcs_cache = os.path.join(
                CACHE_DIR,
                f"pcs_{NUM_PCS}_{_source_key(gcp_project, PCS_URI, NUM_PCS)}.parquet",
            )
            pc_df = io.get_cached_or_generate(
                pcs_cache,
                io.load_pcs,
                gcp_project,
                PCS_URI,
                NUM_PCS,
                validate_num_pcs=NUM_PCS,
                lock_dir=LOCK_DIR,
            )
            sex_cache = os.path.join(
                CACHE_DIR,
                f"genetic_sex_{_source_key(gcp_project, SEX_URI)}.parquet",
            )
            sex_df = io.get_cached_or_generate(
                sex_cache,
                io.load_genetic_sex,
                gcp_project,
                SEX_URI,
                lock_dir=LOCK_DIR,
            )
            related_ids_to_remove = io.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)
            demographics_df.index, pc_df.index, sex_df.index = [df.index.astype(str) for df in (demographics_df, pc_df, sex_df)]
            shared_covariates_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
            shared_covariates_df = shared_covariates_df[~shared_covariates_df.index.isin(related_ids_to_remove)]
            if not shared_covariates_df.index.is_unique:
                dup_idx = shared_covariates_df.index[shared_covariates_df.index.duplicated()].unique()
                dup_list = sorted(map(str, dup_idx))
                sample = ", ".join(dup_list[:5])
                if len(dup_list) > 5:
                    sample += ", ..."
                raise ValueError(
                    "Duplicate person_id entries detected in shared covariates after merging: "
                    f"{sample}"
                )

            LABELS_URI = PCS_URI # Clarify that PCs and Ancestry labels are from the same source
            ancestry_cache = os.path.join(
                CACHE_DIR,
                f"ancestry_labels_{_source_key(gcp_project, LABELS_URI)}.parquet",
            )
            ancestry = io.get_cached_or_generate(
                ancestry_cache,
                io.load_ancestry_labels,
                gcp_project,
                LABELS_URI=LABELS_URI,
                lock_dir=LOCK_DIR,
            )
            anc_series_raw = ancestry.reindex(shared_covariates_df.index)["ANCESTRY"]
            shared_covariates_df, anc_series, population_filter_label, allow_ancestry_followups = (
                _apply_population_filter(
                    shared_covariates_df,
                    anc_series_raw,
                    POPULATION_FILTER,
                )
            )

            anc_cat_global = pd.Categorical(
                anc_series.reindex(shared_covariates_df.index)
            )
            A_global = pd.get_dummies(anc_cat_global, prefix='ANC', drop_first=True, dtype=np.float32)
            A_global.index = A_global.index.astype(str)
            A_cols = list(A_global.columns)
        print(f"\n--- Shared Setup Time: {t_setup.duration:.2f}s ---")

        # --- Filter TARGET_INVERSIONS to only those present in the dosages TSV ---
        dosages_path = _find_upwards(INVERSION_DOSAGES_FILE)
        dosages_resolved = os.path.abspath(dosages_path)
        try:
            hdr = pd.read_csv(dosages_path, sep="\t", nrows=0).columns.tolist()
            id_candidates = {"SampleID", "sample_id", "person_id", "research_id", "participant_id", "ID"}
            id_col = next((c for c in hdr if c in id_candidates), None)
            available_inversions = set(hdr) - ({id_col} if id_col else set())
        
            missing = sorted(run.TARGET_INVERSIONS - available_inversions)
            if missing:
                print(f"[Config] Skipping {len(missing)} inversions not present in dosages file. "
                      f"Examples: {', '.join(missing[:5])}")

            run.TARGET_INVERSIONS = run.TARGET_INVERSIONS & available_inversions
            if not run.TARGET_INVERSIONS:
                raise RuntimeError("No target inversions remain after filtering; check your dosages file and configuration.")
        except Exception as e:
            print(f"[Config WARN] Could not inspect dosages header at '{dosages_path}': {e}")

        try:
            print("\n--- Computing inversion population allele frequencies... ---", flush=True)
            frequency_output = os.path.join(os.getcwd(), "inversion_population_frequencies.tsv")
            dosages_df_all = inversion_frequency.load_all_inversion_dosages(dosages_resolved)
            aligned_ids = shared_covariates_df.index.intersection(dosages_df_all.index)
            dosages_for_freq = dosages_df_all.loc[aligned_ids]
            ancestry_for_freq = anc_series.reindex(aligned_ids)
            freq_df = inversion_frequency.summarize_population_frequencies(
                dosages_for_freq,
                ancestry_for_freq,
                include_all_inversions=True,
            )
            freq_df.to_csv(frequency_output, sep="\t", index=False)
            print(
                f"[Summary] Population allele frequencies saved to {frequency_output} ({len(freq_df)} rows).",
                flush=True,
            )
        except Exception as e:
            print(f"[WARN] Population allele frequency computation failed: {e}", flush=True)

        try:
            pheno.populate_caches_prepass(pheno_defs_df, bq_client, cdr_dataset_id, shared_covariates_df.index, CACHE_DIR, cdr_codename)
        except Exception as e:
            print(f"[Prepass WARN] Cache prepass failed: {e}", flush=True)

        # --- One-time, inversion-independent phenotype dedup manifest ---
        try:
            pheno.deduplicate_phenotypes(
                pheno_defs_df=pheno_defs_df,
                core_index=shared_covariates_df.index,
                cdr_codename=cdr_codename,
                cache_dir=CACHE_DIR,
                min_cases=prefilter_min_cases,
                phi_threshold=pheno.PHI_THRESHOLD,
                share_threshold=pheno.SHARE_THRESHOLD,
                protect=PHENO_PROTECT
            )
            print("[Setup]    - Global phenotype dedup manifest ready.", flush=True)
        except Exception as e:
            print(f"[Dedup WARN] Global dedup pass failed: {e}", flush=True)

        dosages_key = _source_key(dosages_resolved)
        covar_key = _source_key(
            demographics_cache_path,
            pcs_cache,
            sex_cache,
            ancestry_cache,
            gcp_project,
            PCS_URI,
            SEX_URI,
            RELATEDNESS_URI,
        )
        data_keys = {
            "dosages": dosages_key,
            "covars": covar_key,
            "population_filter": population_filter_label,
            "phenotype_filter": PHENOTYPE_FILTER,
        }

        def _inversion_cache_path(inv: str) -> str:
            inv_safe = models.safe_basename(inv)
            key = _source_key(dosages_resolved, inv)
            return os.path.join(CACHE_DIR, f"inversion_{inv_safe}_{key}.parquet")

        def _ctx_tag_for(inv: str) -> str:
            payload = {
                "version": CACHE_VERSION_TAG,
                "cdr_codename": cdr_codename,
                "target_inversion": inv,
                "NUM_PCS": NUM_PCS,
                "MIN_CASES_FILTER": pheno.MIN_CASES_FILTER,
                "MIN_CONTROLS_FILTER": pheno.MIN_CONTROLS_FILTER,
                "MIN_NEFF_FILTER": MIN_NEFF_FILTER,
                "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES,
                "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
                "FDR_ALPHA": FDR_ALPHA,
                "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA,
                "RIDGE_L2_BASE": RIDGE_L2_BASE,
                "ALLOW_POST_FIRTH_MLE_REFIT": bool(
                    tctx.get(
                        "ALLOW_POST_FIRTH_MLE_REFIT",
                        models.DEFAULT_ALLOW_POST_FIRTH_MLE_REFIT,
                    )
                ),
                "MODE": tctx.get("MODE"),
                "SELECTION": tctx.get("SELECTION"),
                "BOOTSTRAP_B": tctx.get("BOOTSTRAP_B"),
                "BOOT_SEED_BASE": tctx.get("BOOT_SEED_BASE"),
                "DATA_KEYS": data_keys,
                "POPULATION_FILTER": population_filter_label,
                "PHENOTYPE_FILTER": PHENOTYPE_FILTER,
            }
            return io.stable_hash(payload)

        ctx_tag_by_inversion = {inv: _ctx_tag_for(inv) for inv in run.TARGET_INVERSIONS}

        governor = MultiTenantGovernor(monitor_thread)

        category_summary_frames: list[pd.DataFrame] = []
        category_summary_lock = threading.Lock()
        skipped_low_variance_inversions: set[str] = set()
        skipped_low_variance_lock = threading.Lock()
        
        def run_single_inversion(target_inversion: str, baseline_rss_gb: float, shared_data: dict):
            inv_safe_name = models.safe_basename(target_inversion)
            log_prefix = f"[INV {inv_safe_name}]"
            try:
                print(f"{log_prefix} Started.", flush=True)
                stage_total = 5
                print(
                    f"{log_prefix} [Stage 1/{stage_total}] Loading inversion dosages...",
                    flush=True,
                )
                inversion_cache_dir = os.path.join(CACHE_DIR, inv_safe_name)
                results_cache_dir = os.path.join(inversion_cache_dir, "results_atomic")
                lrt_overall_cache_dir = os.path.join(inversion_cache_dir, "lrt_overall")
                lrt_followup_cache_dir = os.path.join(inversion_cache_dir, "lrt_followup")
                boot_overall_cache_dir = os.path.join(inversion_cache_dir, "boot_overall")
                os.makedirs(results_cache_dir, exist_ok=True)
                os.makedirs(lrt_followup_cache_dir, exist_ok=True)
                if tctx["MODE"] == "bootstrap":
                    os.makedirs(boot_overall_cache_dir, exist_ok=True)
                else:
                    os.makedirs(lrt_overall_cache_dir, exist_ok=True)

                ctx = {
                    "NUM_PCS": NUM_PCS,
                    "MIN_CASES_FILTER": pheno.MIN_CASES_FILTER,
                    "MIN_CONTROLS_FILTER": pheno.MIN_CONTROLS_FILTER,
                    "MIN_NEFF_FILTER": MIN_NEFF_FILTER,
                    "FDR_ALPHA": FDR_ALPHA,
                    "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES,
                    "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
                    "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA,
                    "CACHE_DIR": CACHE_DIR,
                    "RIDGE_L2_BASE": RIDGE_L2_BASE,
                    "ALLOW_POST_FIRTH_MLE_REFIT": bool(
                        tctx.get(
                            "ALLOW_POST_FIRTH_MLE_REFIT",
                            models.DEFAULT_ALLOW_POST_FIRTH_MLE_REFIT,
                        )
                    ),
                    "RESULTS_CACHE_DIR": results_cache_dir,
                    "LRT_OVERALL_CACHE_DIR": lrt_overall_cache_dir,
                    "LRT_FOLLOWUP_CACHE_DIR": lrt_followup_cache_dir,
                    "BOOT_OVERALL_CACHE_DIR": boot_overall_cache_dir,
                    "BOOTSTRAP_B": tctx["BOOTSTRAP_B"],
                    "BOOTSTRAP_B_MAX": bootstrap_draw_cap,
                    "BOOT_SEED_BASE": tctx["BOOT_SEED_BASE"],
                    "cdr_codename": shared_data['cdr_codename'],
                    "REPAIR_META_IF_MISSING": True,
                    "CACHE_VERSION_TAG": CACHE_VERSION_TAG,
                    "MODE": tctx.get("MODE"),
                    "SELECTION": tctx.get("SELECTION"),
                    "TARGET_INVERSION": target_inversion,
                    "CTX_TAG": ctx_tag_by_inversion.get(target_inversion),
                    "DATA_KEYS": data_keys,
                    "STAGE1_REPORTS_FINAL": True,
                    "STAGE1_MATCH_PHEWAS_DESIGN": True,
                    "STAGE1_EMIT_PHEWAS_EXTRAS": True,
                    "POPULATION_FILTER": population_filter_label,
                    "PHENOTYPE_FILTER": PHENOTYPE_FILTER,
                    "ALLOW_ANCESTRY_FOLLOWUP": allow_ancestry_followups,
                }
                pheno.configure_from_ctx(ctx)
                try:
                    inversion_df = io.get_cached_or_generate(
                        _inversion_cache_path(target_inversion),
                        io.load_inversions,
                        target_inversion,
                        dosages_resolved,
                        validate_target=target_inversion,
                        lock_dir=LOCK_DIR,
                    )
                except io.LowVarianceInversionError as exc:
                    if np.isfinite(exc.std):
                        print(
                            f"{log_prefix} [WARN] Skipping inversion: LOW VARIANCE "
                            f"(std={exc.std:.4f}).",
                            flush=True,
                        )
                    else:
                        print(
                            f"{log_prefix} [WARN] Skipping inversion: NO DATA / ALL NAN.",
                            flush=True,
                        )
                    with skipped_low_variance_lock:
                        skipped_low_variance_inversions.add(target_inversion)
                    return
                inversion_df.index = inversion_df.index.astype(str)
                print(
                    f"{log_prefix} [Stage 1/{stage_total}] Loaded inversion dosages for "
                    f"{len(inversion_df):,} participants.",
                    flush=True,
                )
                print(
                    f"{log_prefix} [Stage 2/{stage_total}] Assembling covariate matrix...",
                    flush=True,
                )
                core_df = shared_data['covariates'].join(inversion_df, how="inner")
                if not core_df.index.is_unique:
                    dupes = core_df.index[core_df.index.duplicated()].unique()
                    sample = ", ".join(map(str, dupes[:5]))
                    raise RuntimeError(
                        "Join between covariates and inversion dosages produced non-unique "
                        f"person_id entries (e.g., {sample})."
                    )

                age_mean = core_df['AGE'].mean()
                core_df['AGE_c'] = core_df['AGE'] - age_mean
                core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2
                pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                covariate_cols = [target_inversion] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
                core_df_subset = core_df[covariate_cols].astype(np.float32, copy=False)
                core_df_subset["const"] = np.float32(1.0)
                A_slice = shared_data['A_global'].reindex(core_df_subset.index).fillna(0.0).astype(np.float32)
                core_df_with_const = pd.concat([core_df_subset, A_slice], axis=1, copy=False).astype(np.float32, copy=False)
                print(
                    f"{log_prefix} [Stage 2/{stage_total}] Covariate matrix ready with "
                    f"shape {core_df_with_const.shape}.",
                    flush=True,
                )

                delta_core_df_gb = (monitor_thread.snapshot().app_rss_gb - baseline_rss_gb) if monitor_thread else 0.0
                governor.update_after_core_df(delta_core_df_gb)

                core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
                global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)
                print(
                    f"{log_prefix} [Stage 3/{stage_total}] Resolving allowed-control masks...",
                    flush=True,
                )
                pan_path = os.path.join(CACHE_DIR, f"pan_category_cases_{shared_data['cdr_codename']}.pkl")
                category_to_pan_cases = io.get_cached_or_generate_pickle(
                    pan_path,
                    pheno.build_pan_category_cases,
                    shared_data['pheno_defs'], shared_data['bq_client'], shared_data['cdr_id'], CACHE_DIR, shared_data['cdr_codename'],
                    lock_dir=LOCK_DIR,
                )
                allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(
                    core_index,
                    category_to_pan_cases,
                    global_notnull_mask,
                    log_prefix=log_prefix,
                    progress_label=f"Stage 3/{stage_total}: Building allowed-control masks",
                )

                sex_vec = core_df_with_const['sex'].to_numpy(dtype=np.float32, copy=False)

                # --- Build Stage-1 testing worklist without running main PheWAS ---
                print(
                    f"{log_prefix} [Stage 4/{stage_total}] Prefiltering phenotypes for Stage-1 queue...",
                    flush=True,
                )
                phenos_list = []
                pheno_records = shared_data['pheno_defs'][['sanitized_name', 'disease_category']].to_dict('records')
                prefilter_progress = _create_progress_emitter(
                    len(pheno_records),
                    log_prefix,
                    f"Stage 4/{stage_total}: Prefiltering phenotypes",
                )
                prefilter_progress(0)
                for idx, row in enumerate(pheno_records, start=1):
                    info = {
                        'sanitized_name': row['sanitized_name'],
                        'disease_category': row['disease_category'],
                        'cdr_codename': shared_data['cdr_codename'],
                        'cache_dir': CACHE_DIR
                    }
                    ok, case_ix = pheno._prequeue_should_run(
                        info,
                        core_index,
                        allowed_mask_by_cat,
                        sex_vec,
                        prefilter_min_cases,
                        prefilter_min_ctrls,
                        sex_mode="majority",
                        sex_prop=models.DEFAULT_SEX_RESTRICT_PROP,
                        max_other=ctx.get("SEX_RESTRICT_MAX_OTHER_CASES", 0),
                        min_neff=(MIN_NEFF_FILTER if MIN_NEFF_FILTER > 0 else None),
                        return_case_idx=True,
                    )
                    if ok:
                        case_ix = np.asarray(case_ix, dtype=np.int32)
                        if case_ix.size:
                            case_index = core_index.take(case_ix)
                        else:
                            case_index = pd.Index([], dtype=core_index.dtype, name=core_index.name)
                        case_fp = models._index_fingerprint(case_index)
                        phenos_list.append({
                            "name": row['sanitized_name'],
                            "case_idx": case_ix.tolist(),
                            "case_fp": case_fp,
                        })
                    prefilter_progress(idx)

                print(
                    f"{log_prefix} [Stage 4/{stage_total}] Prefilter complete: {len(phenos_list)} phenotypes queued.",
                    flush=True,
                )

                print(f"{log_prefix} Queued {len(phenos_list)} phenotypes for Stage-1 testing (pre-filtered).")

                def on_pool_started_callback(num_procs, worker_pids):
                    governor.register_pool(inv_safe_name, worker_pids)
                    time.sleep(10)
                    measured_gb = governor.measure_inv(inv_safe_name)
                    core_shm_gb = float(pipes.BUDGET._reserved_by_inv.get(inv_safe_name, {}).get("core_shm", 0.0))
                    boot_shm_gb = float(pipes.BUDGET._reserved_by_inv.get(inv_safe_name, {}).get("boot_shm", 0.0))
                    # Do not double-charge shared matrices: reserve only the pool's incremental footprint.
                    pool_steady_gb = max(0.0, measured_gb - core_shm_gb - boot_shm_gb)
                    governor.update_steady_state(inv_safe_name, pool_steady_gb)
                    pipes.BUDGET.revise(inv_safe_name, "pool_steady", pool_steady_gb)
                    per_worker = max(0.25, pool_steady_gb / max(1, num_procs))
                    pipes._WORKER_GB_EST = 0.5 * pipes._WORKER_GB_EST + 0.5 * per_worker
                    print(f"[Budget] {inv_safe_name}.pool_steady: set {pool_steady_gb:.2f}GB | remaining {pipes.BUDGET.remaining_gb():.2f}GB", flush=True)

                name_to_cat = shared_data['pheno_defs'].set_index('sanitized_name')['disease_category'].to_dict()
                if phenos_list:
                    print(
                        f"{log_prefix} [Stage 5/{stage_total}] Dispatching Stage-1 models for {len(phenos_list)} phenotypes...",
                        flush=True,
                    )
                    testing.run_overall(
                        core_df_with_const,
                        allowed_mask_by_cat,
                        shared_data['anc_series'],
                        phenos_list,
                        name_to_cat,
                        shared_data['cdr_codename'],
                        target_inversion,
                        ctx,
                        mem_floor_callable,
                        on_pool_started=on_pool_started_callback,
                        mode=tctx["MODE"],
                    )
                    print(
                        f"{log_prefix} [Stage 5/{stage_total}] Stage-1 testing complete.",
                        flush=True,
                    )
                else:
                    print(
                        f"{log_prefix} [Stage 5/{stage_total}] No phenotypes qualified for Stage-1 testing after prefiltering.",
                        flush=True,
                    )

                inv_df = pd.DataFrame()
                p_col = None

                try:
                    rows = []
                    if tctx["MODE"] == "lrt_bh":
                        lrt_files = [f for f in os.listdir(lrt_overall_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                        for fn in lrt_files:
                            meta_path = os.path.join(lrt_overall_cache_dir, fn.replace(".json", ".meta.json"))
                            meta = io.read_meta_json(meta_path)
                            if not meta:
                                continue
                            if meta.get("ctx_tag") != ctx.get("CTX_TAG") or meta.get("cdr_codename") != shared_data['cdr_codename'] or meta.get("target") != target_inversion:
                                continue
                            s = pd.read_json(os.path.join(lrt_overall_cache_dir, fn), typ="series")
                            rows.append({"Phenotype": os.path.splitext(fn)[0], "P_LRT_Overall": pd.to_numeric(s.get("P_LRT_Overall"), errors="coerce")})
                        p_col = "P_LRT_Overall"
                    else:
                        boot_files = [f for f in os.listdir(boot_overall_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                        for fn in boot_files:
                            meta_path = os.path.join(boot_overall_cache_dir, fn.replace(".json", ".meta.json"))
                            meta = io.read_meta_json(meta_path)
                            if not meta:
                                continue
                            if meta.get("ctx_tag") != ctx.get("CTX_TAG") or meta.get("cdr_codename") != shared_data['cdr_codename'] or meta.get("target") != target_inversion:
                                continue
                            s = pd.read_json(os.path.join(boot_overall_cache_dir, fn), typ="series")
                            rows.append({"Phenotype": os.path.splitext(fn)[0], "P_EMP": pd.to_numeric(s.get("P_EMP"), errors="coerce")})
                        p_col = "P_EMP"
                    lrt_df = pd.DataFrame(rows)
                    res_files = [f for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                    rrows = []
                    for fn in res_files:
                        meta_path = os.path.join(results_cache_dir, fn.replace(".json", ".meta.json"))
                        meta = io.read_meta_json(meta_path)
                        if not meta:
                            continue
                        if meta.get("ctx_tag") != ctx.get("CTX_TAG") or meta.get("cdr_codename") != shared_data['cdr_codename'] or meta.get("target") != target_inversion:
                            continue
                        s = pd.read_json(os.path.join(results_cache_dir, fn), typ="series")
                        rrows.append({
                            "Phenotype": os.path.splitext(fn)[0],
                            "N_Total": pd.to_numeric(s.get("N_Total"), errors="coerce"),
                            "OR": pd.to_numeric(s.get("OR"), errors="coerce"),
                            "Beta": pd.to_numeric(s.get("Beta"), errors="coerce"),
                            "P_Value": pd.to_numeric(s.get("P_Value"), errors="coerce"),
                            "OR_CI95": s.get("OR_CI95"),
                            "N_Cases": pd.to_numeric(s.get("N_Cases"), errors="coerce"),
                            "N_Controls": pd.to_numeric(s.get("N_Controls"), errors="coerce"),
                        })
                    res_df = pd.DataFrame(rrows)
                    inv_df = lrt_df.merge(res_df, on="Phenotype", how="left") if not lrt_df.empty else pd.DataFrame(columns=["Phenotype", p_col, "N_Total", "OR", "Beta", "P_Value", "N_Cases", "N_Controls"])
                    m = int(inv_df[p_col].notna().sum()) if not inv_df.empty else 0
                    if m > 0:
                        mask = inv_df[p_col].notna()
                        _, q_within, _, _ = multipletests(inv_df.loc[mask, p_col], alpha=FDR_ALPHA, method="fdr_bh")
                        inv_df.loc[mask, "Q_within"] = q_within

                    def _fmt(v, fmt_str):
                        return f"{float(v):{fmt_str}}" if pd.notna(v) else ""
                    top = inv_df.sort_values(p_col).head(10).copy() if m > 0 else inv_df.head(0)
                    top["P"] = top[p_col].apply(lambda v: _fmt(v, ".3e"))
                    top["Q"] = top["Q_within"].apply(lambda v: _fmt(v, ".3f"))
                    top["OR"] = top["OR"].apply(lambda v: _fmt(v, "0.3f"))
                    top["Beta"] = top["Beta"].apply(lambda v: _fmt(v, "+0.4f"))
                    top["N"] = (
                        pd.to_numeric(top["N_Total"], errors="coerce").fillna(0).astype(int).astype(str)
                        + ";"
                        + (pd.to_numeric(top["N_Cases"], errors="coerce").fillna(0).astype(int)).astype(str)
                        + "/"
                        + (pd.to_numeric(top["N_Controls"], errors="coerce").fillna(0).astype(int)).astype(str)
                    )
                    print(f"\n{log_prefix} --- Top Hits Summary (provisional) ---\n" + top[["Phenotype","P","Q","OR","Beta","N"]].to_string(index=False) + "\n")

                    if not inv_df.empty and p_col:
                        try:
                            dedup_manifest = categories.load_dedup_manifest(CACHE_DIR, shared_data['cdr_codename'], core_index)
                            valid_mask = pd.to_numeric(inv_df[p_col], errors="coerce").notna()
                            phenos_for_plan = inv_df.loc[valid_mask, "Phenotype"].tolist()
                            min_k = int(tctx.get("CAT_MIN_K", 3))
                            category_sets, _dropped_small = categories.plan_category_sets(
                                phenos_for_plan,
                                name_to_cat,
                                dedup_manifest,
                                min_k=min_k,
                            )
                            if category_sets:
                                null_structs = categories.build_category_null_structure(
                                    core_df_with_const,
                                    allowed_mask_by_cat,
                                    category_sets,
                                    cache_dir=CACHE_DIR,
                                    cdr_codename=shared_data['cdr_codename'],
                                    method=str(tctx.get("CAT_METHOD", "fast_phi")).lower(),
                                    shrinkage=str(tctx.get("CAT_SHRINKAGE", "ridge")).lower(),
                                    lambda_value=float(tctx.get("CAT_LAMBDA", 0.05)),
                                    min_k=min_k,
                                    global_mask=global_notnull_mask,
                                )
                                if null_structs:
                                    base_seed = int(tctx.get("CAT_SEED_BASE", 1729))
                                    stable_component = _stable_seed(target_inversion, ctx.get("CTX_TAG"))
                                    seed = (base_seed + stable_component) % (2 ** 32)
                                    z_cap_value = categories._sanitize_z_cap(
                                        tctx.get("CAT_Z_CAP")
                                    )

                                    base_draws = int(tctx.get("CAT_GBJ_B", 5000))
                                    max_draws_cfg = int(
                                        tctx.get(
                                            "CAT_GBJ_MAX",
                                            max(base_draws * 10, base_draws),
                                        )
                                    )

                                    cat_df = categories.compute_category_metrics(
                                        inv_df,
                                        p_col=p_col,
                                        beta_col="Beta",
                                        null_structures=null_structs,
                                        gbj_draws=base_draws,
                                        adaptive_max_draws=max_draws_cfg,
                                        z_cap=z_cap_value,
                                        rng_seed=seed,
                                        min_k=min_k,
                                    )
                                    if not cat_df.empty:
                                        if "P_GBJ" in cat_df.columns and cat_df["P_GBJ"].notna().any():
                                            mask = cat_df["P_GBJ"].notna()
                                            _, q_cat_gbj, _, _ = multipletests(
                                                cat_df.loc[mask, "P_GBJ"], alpha=0.05, method="fdr_bh"
                                            )
                                            cat_df.loc[mask, "Q_GBJ"] = q_cat_gbj
                                        if "P_GLS" in cat_df.columns and cat_df["P_GLS"].notna().any():
                                            mask = cat_df["P_GLS"].notna()
                                            _, q_cat_gls, _, _ = multipletests(
                                                cat_df.loc[mask, "P_GLS"], alpha=0.05, method="fdr_bh"
                                            )
                                            cat_df.loc[mask, "Q_GLS"] = q_cat_gls
                                        cat_df.insert(0, "Inversion", target_inversion)
                                        with category_summary_lock:
                                            category_summary_frames.append(cat_df)
                                        print(
                                            f"{log_prefix} Category summary recorded with {len(cat_df)} rows.",
                                            flush=True,
                                        )
                                    else:
                                        print(f"{log_prefix} Category metrics produced no eligible categories.", flush=True)
                                else:
                                    print(f"{log_prefix} No category null structures built (insufficient data).", flush=True)
                            else:
                                print(f"{log_prefix} No categories met minimum phenotype requirement.", flush=True)
                        except NotImplementedError as exc:
                            print(f"{log_prefix} [WARN] Category metrics unavailable: {exc}", flush=True)
                        except Exception as exc:
                            print(f"{log_prefix} [WARN] Category metrics failed: {exc}", flush=True)
                            traceback.print_exc()
                except Exception:
                    print(f"{log_prefix} [WARN] Could not produce per-inversion summary.", flush=True)

                print(f"{log_prefix} Finished.", flush=True)
            except Exception as e:
                print(f"{log_prefix} [FAIL] Failed with error: {e}", flush=True)
                traceback.print_exc()

        shared_data_for_threads = {
            "covariates": shared_covariates_df,
            "anc_series": anc_series,
            "A_global": A_global,
            "A_cols": A_cols,
            "pheno_defs": pheno_defs_df,
            "bq_client": bq_client,
            "cdr_id": cdr_dataset_id,
            "cdr_codename": cdr_codename,
        }
        num_ancestry_dummies = len(A_cols)
        C = 1 + 1 + 1 + NUM_PCS + 2 + num_ancestry_dummies
        MAX_CONCURRENT_INVERSIONS = MAX_CONCURRENT_INVERSIONS_DEFAULT
        if tctx["MODE"] == "bootstrap":
            MAX_CONCURRENT_INVERSIONS = MAX_CONCURRENT_INVERSIONS_BOOT
        pending_inversions = deque(sorted(list(run.TARGET_INVERSIONS)))
        running_inversions = {}

        print("\n--- Starting Parallel Inversion Orchestrator ---")
        while pending_inversions or running_inversions:
            finished_threads = [t for t in running_inversions if not t.is_alive()]
            for t in finished_threads:
                inv = running_inversions.pop(t)
                governor.deregister_pool(inv)
                pipes.BUDGET.release(inv, "pool_steady")
                pipes.BUDGET.release(inv, "core_shm")
                print(f"[Orchestrator] Inversion '{inv}' thread finished.")

            if len(running_inversions) >= MAX_CONCURRENT_INVERSIONS:
                time.sleep(0.5)
                continue

            for inv_name in running_inversions.values():
                governor.measure_inv(inv_name)

            if pending_inversions:
                predicted_gb = governor.predict_extra_gb_after_pool()
                if not governor.can_admit_next_inv(predicted_gb):
                    time.sleep(0.5)
                    continue

                target_inv = pending_inversions[0]
                try:
                    inversion_path = _inversion_cache_path(target_inv)
                    if os.path.exists(inversion_path):
                        inversion_index = pd.read_parquet(inversion_path, columns=[target_inv]).index.astype(str)
                        N = shared_covariates_df.index.intersection(inversion_index).size
                    else:
                        N = len(shared_covariates_df)
                    core_bytes = N * C * 4
                    core_gb = core_bytes / (1024**3)
                except Exception as e:
                    print(f"[Orchestrator] Could not predict memory for {target_inv}, using fallback. Error: {e}")
                    core_gb = 2.0

                if pipes.BUDGET.reserve(target_inv, "core_shm", core_gb, block=False):
                    target_inv = pending_inversions.popleft()
                    print(f"[Orchestrator] Admitted {target_inv} | reserved core_shm={core_gb:.2f}GB | budget {pipes.BUDGET.remaining_gb():.2f}/{pipes.BUDGET._total_gb:.2f}GB")
                    baseline_rss = (monitor_thread.snapshot().app_rss_gb if monitor_thread else 0.0)
                    thread = threading.Thread(target=run_single_inversion, args=(target_inv, baseline_rss, shared_data_for_threads))
                    running_inversions[thread] = target_inv
                    thread.start()
                else:
                    time.sleep(0.5)

            time.sleep(1.0)

        print("\n--- All inversions processed. ---")

        if category_summary_frames:
            combined_category_summary = pd.concat(category_summary_frames, ignore_index=True)
            output_dir = os.getcwd()
            summary_path = os.path.join(output_dir, "category_summary.tsv")
            combined_category_summary.to_csv(summary_path, sep="\t", index=False)
            print(
                f"[Summary] Combined category summary saved to {summary_path} ({len(combined_category_summary)} rows).",
                flush=True,
            )
        else:
            print("[Summary] No category summaries were generated.", flush=True)

        # --- PART 3: CONSOLIDATE & ANALYZE RESULTS (ACROSS ALL INVERSIONS) ---
        print("\n" + "=" * 70)
        print(" Part 3: Consolidating final results across all inversions")
        print("=" * 70)

        all_results_from_disk = []
        for target_inversion in run.TARGET_INVERSIONS:
            if target_inversion in skipped_low_variance_inversions:
                inv_safe_name = models.safe_basename(target_inversion)
                print(
                    f"[INV {inv_safe_name}] Skipping consolidation due to low-variance dosage column.",
                    flush=True,
                )
                continue
            inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
            results_cache_dir = os.path.join(inversion_cache_dir, "results_atomic")
            result_files = [f for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
            for filename in result_files:
                try:
                    meta_path = os.path.join(results_cache_dir, filename.replace(".json", ".meta.json"))
                    meta = io.read_meta_json(meta_path)
                    expected_tag = ctx_tag_by_inversion.get(target_inversion)
                    if not meta:
                        continue
                    if meta.get("ctx_tag") != expected_tag or meta.get("cdr_codename") != cdr_codename or meta.get("target") != target_inversion:
                        continue
                    result = pd.read_json(os.path.join(results_cache_dir, filename), typ="series").to_dict()
                    result['Inversion'] = target_inversion
                    all_results_from_disk.append(result)
                except Exception as e:
                    print(f"Warning: Could not read corrupted result file: {filename}, Error: {e}")

        if not all_results_from_disk:
            print("No results found to process.")
        else:
            df = pd.DataFrame(all_results_from_disk)
            processed_inversion_count = len(set(run.TARGET_INVERSIONS) - skipped_low_variance_inversions)
            print(
                f"Successfully consolidated {len(df)} results across {processed_inversion_count} inversions."
            )

            if "OR_CI95" not in df.columns: df["OR_CI95"] = np.nan
            def _compute_overall_or_ci(beta_val, p_val):
                if pd.isna(beta_val) or pd.isna(p_val): return np.nan
                b = float(beta_val); p = float(p_val)
                if not (np.isfinite(b) and np.isfinite(p) and 0.0 < p < 1.0): return np.nan
                z = stats.norm.ppf(1.0 - p / 2.0)
                if not (np.isfinite(z) and z > 0): return np.nan
                se = abs(b) / z
                lo, hi = np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)
                return f"{lo:.3f},{hi:.3f}"
            missing_ci_mask = (df["OR_CI95"].isna() | (df["OR_CI95"].astype(str) == "") | (df["OR_CI95"].astype(str).str.lower() == "nan"))
            if "Used_Ridge" in df.columns:
                missing_ci_mask &= (df["Used_Ridge"] == False)
            df.loc[missing_ci_mask, "OR_CI95"] = df.loc[missing_ci_mask, ["Beta", "P_Value"]].apply(lambda r: _compute_overall_or_ci(r["Beta"], r["P_Value"]), axis=1)

            df, _ = testing.consolidate_and_select(
                df,
                sorted(set(run.TARGET_INVERSIONS) - skipped_low_variance_inversions),
                CACHE_DIR,
                alpha=FDR_ALPHA,
                mode=tctx["MODE"],
                selection=tctx["SELECTION"],
                ctx_tags=ctx_tag_by_inversion,
                cdr_codename=cdr_codename,
            )

            # --- PART 4: SCHEDULE AND RUN STAGE-2 FOLLOW-UPS ---
            print("\n" + "=" * 70)
            print(" Part 4: Running Stage-2 Follow-up Analyses for Global Hits")
            print("=" * 70)
            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()

            for target_inversion in run.TARGET_INVERSIONS:
                if target_inversion in skipped_low_variance_inversions:
                    inv_safe_name = models.safe_basename(target_inversion)
                    print(
                        f"[INV {inv_safe_name}] Skipping follow-up due to low-variance dosage column.",
                        flush=True,
                    )
                    continue
                # Re-create the inversion-specific context and data to ensure correct follow-up
                dosages_path = _find_upwards(INVERSION_DOSAGES_FILE)
                try:
                    inversion_df = io.get_cached_or_generate(
                        _inversion_cache_path(target_inversion),
                        io.load_inversions,
                        target_inversion,
                        dosages_path,
                        validate_target=target_inversion,
                        lock_dir=LOCK_DIR,
                    )
                except io.LowVarianceInversionError as exc:
                    inv_safe_name = models.safe_basename(target_inversion)
                    if np.isfinite(exc.std):
                        print(
                            f"[INV {inv_safe_name}] Skipping follow-up: LOW VARIANCE "
                            f"(std={exc.std:.4f}).",
                            flush=True,
                        )
                    else:
                        print(
                            f"[INV {inv_safe_name}] Skipping follow-up: NO DATA / ALL NAN.",
                            flush=True,
                        )
                    with skipped_low_variance_lock:
                        skipped_low_variance_inversions.add(target_inversion)
                    continue
                inversion_df.index = inversion_df.index.astype(str)

                core_df = shared_covariates_df.join(inversion_df, how="inner")
                if not core_df.index.is_unique:
                    dupes = core_df.index[core_df.index.duplicated()].unique()
                    sample = ", ".join(map(str, dupes[:5]))
                    raise RuntimeError(
                        "Join between covariates and inversion dosages produced non-unique "
                        f"person_id entries (e.g., {sample})."
                    )
                age_mean = core_df['AGE'].mean()
                core_df['AGE_c'] = core_df['AGE'] - age_mean
                core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2
                pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                covariate_cols = [target_inversion] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
                core_df_subset = core_df[covariate_cols].astype(np.float32, copy=False)
                core_df_subset["const"] = np.float32(1.0)
                A_slice = A_global.reindex(core_df_subset.index).fillna(0.0).astype(np.float32)
                core_df_with_const = pd.concat([core_df_subset, A_slice], axis=1, copy=False).astype(np.float32, copy=False)
                core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
                global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)
                pan_path = os.path.join(CACHE_DIR, f"pan_category_cases_{cdr_codename}.pkl")
                category_to_pan_cases = io.get_cached_or_generate_pickle(
                    pan_path,
                    pheno.build_pan_category_cases,
                    pheno_defs_df, bq_client, cdr_dataset_id, CACHE_DIR, cdr_codename,
                    lock_dir=LOCK_DIR,
                )
                allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask)

                inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
                ctx = {
                    "NUM_PCS": NUM_PCS, "MIN_CASES_FILTER": pheno.MIN_CASES_FILTER, "MIN_CONTROLS_FILTER": pheno.MIN_CONTROLS_FILTER,
                    "MIN_NEFF_FILTER": MIN_NEFF_FILTER,
                    "FDR_ALPHA": FDR_ALPHA, "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES, "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
                    "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA, "CACHE_DIR": CACHE_DIR, "RIDGE_L2_BASE": RIDGE_L2_BASE,
                    "ALLOW_POST_FIRTH_MLE_REFIT": bool(
                        tctx.get(
                            "ALLOW_POST_FIRTH_MLE_REFIT",
                            models.DEFAULT_ALLOW_POST_FIRTH_MLE_REFIT,
                        )
                    ),
                    "RESULTS_CACHE_DIR": os.path.join(inversion_cache_dir, "results_atomic"),
                    "LRT_OVERALL_CACHE_DIR": os.path.join(inversion_cache_dir, "lrt_overall"),
                    "LRT_FOLLOWUP_CACHE_DIR": os.path.join(inversion_cache_dir, "lrt_followup"),
                    "BOOT_OVERALL_CACHE_DIR": os.path.join(inversion_cache_dir, "boot_overall"),
                    "cdr_codename": cdr_codename,
                    "CACHE_VERSION_TAG": CACHE_VERSION_TAG,
                    "MODE": tctx.get("MODE"),
                    "SELECTION": tctx.get("SELECTION"),
                    "TARGET_INVERSION": target_inversion,
                    "CTX_TAG": ctx_tag_by_inversion.get(target_inversion),
                    "DATA_KEYS": data_keys,
                    "STAGE1_REPORTS_FINAL": True,
                    "STAGE1_MATCH_PHEWAS_DESIGN": True,
                    "STAGE1_EMIT_PHEWAS_EXTRAS": True,
                    "PHENOTYPE_FILTER": PHENOTYPE_FILTER,
                }

                pheno.configure_from_ctx(ctx)

                # Select hits for the current inversion and run follow-up
                hit_phenos = df.loc[(df["Sig_Global"] == True) & (df["Inversion"] == target_inversion), "Phenotype"].astype(str).tolist()
                if hit_phenos:
                    if allow_ancestry_followups:
                        print(f"--- Running follow-up for {len(hit_phenos)} hits in {target_inversion} ---")
                        pipes.run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_phenos, name_to_cat, cdr_codename, target_inversion, ctx, mem_floor_callable)
                    else:
                        print(
                            f"--- Skipping ancestry follow-up for {len(hit_phenos)} hits in {target_inversion} "
                            f"due to active population filter '{population_filter_label}'. ---"
                        )

            if allow_ancestry_followups:
                # Consolidate all follow-up results
                print("\n--- Consolidating all Stage-2 follow-up results ---")
                follow_records = []
                for target_inversion in run.TARGET_INVERSIONS:
                    if target_inversion in skipped_low_variance_inversions:
                        inv_safe_name = models.safe_basename(target_inversion)
                        print(
                            f"[INV {inv_safe_name}] Skipping follow-up consolidation due to low-variance dosage column.",
                            flush=True,
                        )
                        continue
                    lrt_followup_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion), "lrt_followup")
                    if not os.path.isdir(lrt_followup_cache_dir): continue
                    files_follow = [f for f in os.listdir(lrt_followup_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                    for filename in files_follow:
                        try:
                            meta_path = os.path.join(lrt_followup_cache_dir, filename.replace(".json", ".meta.json"))
                            meta = io.read_meta_json(meta_path)
                            expected_tag = ctx_tag_by_inversion.get(target_inversion)
                            if not meta:
                                continue
                            if meta.get("ctx_tag") != expected_tag or meta.get("cdr_codename") != cdr_codename or meta.get("target") != target_inversion:
                                continue
                            rec = pd.read_json(os.path.join(lrt_followup_cache_dir, filename), typ="series").to_dict()
                            rec['Inversion'] = target_inversion
                            follow_records.append(rec)
                        except Exception as e:
                            print(f"Warning: Could not read LRT follow-up file: {filename}, Error: {e}")

                if follow_records:
                    follow_df = pd.DataFrame(follow_records)
                    print(f"Collected {len(follow_df)} follow-up records.")
                    df = _merge_followup_results(df, follow_df)
            else:
                print("\n--- Stage-2 ancestry follow-up consolidation skipped due to population filter. ---")

            df = testing.apply_followup_fdr(df, FDR_ALPHA, LRT_SELECT_ALPHA)

            print(f"\n--- Saving final results to '{MASTER_RESULTS_CSV}' ---")
            # Atomic write of the master results TSV to guard against partial files.
            _tmp_dir = os.path.dirname(MASTER_RESULTS_CSV) or "."
            os.makedirs(_tmp_dir, exist_ok=True)
            import tempfile
            _fd, _tmp_path = tempfile.mkstemp(dir=_tmp_dir, prefix=os.path.basename(MASTER_RESULTS_CSV) + ".tmp.")
            os.close(_fd)
            try:
                df.to_csv(_tmp_path, index=False, sep='\t')
                os.replace(_tmp_path, MASTER_RESULTS_CSV)
            finally:
                try:
                    if _tmp_path and os.path.exists(_tmp_path):
                        os.remove(_tmp_path)
                except Exception:
                    pass

            out_df = df[df['Sig_Global'] == True].copy()
            if not out_df.empty:
                print("\n--- Top Hits Summary ---")
                for col in ["N_Total", "N_Cases", "N_Controls"]:
                    if col in out_df.columns:
                        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").apply(lambda v: f"{int(v):,}" if pd.notna(v) else "")
                for col, fmt in {"Beta": "+0.4f", "OR": "0.3f", "P_Value": ".3e", "Q_GLOBAL": ".3f"}.items():
                    if col in out_df.columns:
                        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").apply(lambda v: f"{v:{fmt}}" if pd.notna(v) else "")
                out_df["Sig_Global"] = out_df["Sig_Global"].fillna(False).map(lambda x: "✓" if bool(x) else "")
                print(out_df.to_string(index=False))

    except Exception as e:
        print("\nSCRIPT HALTED DUE TO A CRITICAL ERROR:", flush=True)
        traceback.print_exc()

    finally:
        script_duration = time.time() - script_start_time
        print("\n" + "=" * 70)
        print(f" Script finished in {script_duration:.2f} seconds.")
        print("=" * 70)


def supervisor_main(max_restarts=100, backoff_sec=10, *, pipeline_config=None):
    import multiprocessing as mp, time, signal, os

    ctx = mp.get_context("spawn")
    should_stop = {"flag": False}
    current_child = {"proc": None}

    def _stop(signum, _frame):
        should_stop["flag"] = True
        proc = current_child.get("proc")
        if proc is not None and proc.is_alive():
            try:
                proc.send_signal(signum)
            except AttributeError:
                try:
                    os.kill(proc.pid, signum)
                except Exception:
                    pass
            except Exception:
                pass

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    restarts = 0
    while not should_stop["flag"] and restarts <= max_restarts:
        p = ctx.Process(
            target=_pipeline_once,
            name="ferromic-pipeline",
            args=(pipeline_config,),
        )
        current_child["proc"] = p
        p.start()
        while p.is_alive():
            if should_stop["flag"]:
                try:
                    p.terminate()
                except Exception:
                    pass
                p.join(timeout=5)
                current_child["proc"] = None
                return
            time.sleep(0.2)
        current_child["proc"] = None
        code = p.exitcode
        if code == 0:
            break
        if code in (-2, -15):
            break
        restarts += 1
        print(f"[Supervisor] Child exited with code {code}. Restart {restarts}/{max_restarts} in {backoff_sec}s...", flush=True)
        for _ in range(backoff_sec * 5):
            if should_stop["flag"]:
                return
            time.sleep(0.2)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for running the pipeline directly via ``python -m phewas.run``."""

    if argv is None:
        argv = []

    pipeline_config = None
    if argv:
        # Import locally to avoid circular imports at module load time.
        from . import cli as _cli

        args = _cli.parse_args(argv)
        pipeline_config = _cli.apply_cli_configuration(args)

    supervisor_main(pipeline_config=pipeline_config)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
