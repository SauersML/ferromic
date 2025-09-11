import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from datetime import datetime
import time
import warnings
import gc
import threading
import queue
import faulthandler
import sys
import traceback
import json
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False


import numpy as np
import pandas as pd
import statsmodels.api as sm
from google.cloud import bigquery
from statsmodels.stats.multitest import multipletests
from scipy import stats

from . import iox as io
from . import pheno
from . import pipes
from . import models

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

    def snapshot(self) -> 'ResourceSnapshot':
        """Returns a thread-safe snapshot of the current stats."""
        with self._lock:
            return ResourceSnapshot(
                ts=time.time(),
                sys_cpu_percent=self.sys_cpu_percent,
                sys_available_gb=self.sys_available_gb,
                app_rss_gb=self.app_rss_gb,
            )

    def run(self):
        """Monitors and reports system stats until the main program exits."""
        if not self._main_process:
            print("[SysMonitor] Could not find main process to monitor.", flush=True)
            return

        while True:
            try:
                cpu = psutil.cpu_percent(interval=1.0)
                mem = psutil.virtual_memory()
                ram_percent = mem.percent
                available_gb = mem.available / (1024**3)
                child_processes = self._main_process.children(recursive=True)
                main_mem = self._main_process.memory_info()
                child_mem = sum(p.memory_info().rss for p in child_processes)
                total_rss_gb = (main_mem.rss + child_mem) / (1024**3)
                n_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 1
                app_cpu_raw = sum(c.cpu_percent(interval=None) for c in child_processes)
                app_cpu = min(100.0, app_cpu_raw / n_cpus)

                with self._lock:
                    self.sys_cpu_percent = cpu
                    self.sys_available_gb = available_gb
                    self.app_rss_gb = total_rss_gb

                print(f"[SysMonitor] CPU: {cpu:5.1f}% | AppCPU: {app_cpu:5.1f}% | RAM: {ram_percent:5.1f}% (avail: {available_gb:.2f}GB) | App RSS: {total_rss_gb:.2f}GB | Budget: {pipes.BUDGET.remaining_gb():.2f}/{pipes.BUDGET._total_gb:.2f}GB", flush=True)
                try:
                    prog = pipes.PROGRESS.snapshot()
                    by_inv = {}
                    for (inv, stage), (d, q, ts) in prog.items():
                        prev = by_inv.get(inv)
                        if (prev is None) or (ts > prev[-1]):
                            pct = int((100*d/q)) if q else 0
                            by_inv[inv] = (stage, d, q, pct, ts)
                    if by_inv:
                        parts = [f"{inv}:{stage} {pct}%" for inv,(stage,_,_,pct,_) in sorted(by_inv.items())]
                        print("[Progress] " + " | ".join(parts), flush=True)
                except Exception:
                    pass
            except psutil.NoSuchProcess:
                break
            except Exception as e:
                print(f"[SysMonitor] Error: {e}", flush=True)
            time.sleep(self.interval)

from collections import deque
from dataclasses import dataclass
import statistics

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

class ResourceGovernor:
    def __init__(self, monitor: SystemMonitor, history_sec: int = 30):
        self._monitor = monitor
        self._lock = threading.Lock()
        self._history = deque(maxlen=history_sec // (monitor.interval or 1))
        self.observed_core_df_gb = []
        self.observed_steady_state_gb = []
        self.mem_guard_gb = 4.0
        self.min_cpu_idle_frac = 0.20

    def _update_history(self):
        self._history.append(self._monitor.snapshot())

    def can_admit_next(self, predicted_extra_gb: float) -> bool:
        self._update_history()
        if not self._history: return False

        with self._lock:
            cpu_hist = [s.sys_cpu_percent for s in self._history]
            avg_cpu_usage = statistics.mean(cpu_hist) if cpu_hist else 100.0
            avg_idle = 1.0 - (avg_cpu_usage / 100.0)
            cpu_ok = avg_idle >= self.min_cpu_idle_frac
            latest_mem_gb = self._history[-1].sys_available_gb
            mem_ok = (latest_mem_gb - predicted_extra_gb) >= self.mem_guard_gb
            if not (cpu_ok and mem_ok):
                print(f"[Governor] Hold: cpu_idle={avg_idle:.2f}, mem_avail={latest_mem_gb:.2f}GB, pred_cost={predicted_extra_gb:.2f}GB")
            return cpu_ok and mem_ok

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
        total = 0
        pids = self.inv_pools.get(inv_id, [])
        if not pids:
            self.inv_rss_gb[inv_id] = 0
            return 0

        valid_pids = []
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                total += proc.memory_info().rss
                valid_pids.append(pid)
            except psutil.NoSuchProcess:
                pass
            except Exception:
                pass

        if len(valid_pids) < len(pids):
            self.inv_pools[inv_id] = valid_pids

        measured_gb = total / (1024**3)
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
            if not self.observed_steady_state_gb_per_inv:
                return super().predict_extra_gb_after_pool()
            return np.percentile(self.observed_steady_state_gb_per_inv, 75)

    def can_admit_next_inv(self, predicted_gb):
        self._update_history()
        if not self._history: return False
        latest_avail = self._history[-1].sys_available_gb
        active = self.total_active_footprint()
        mem_ok = (latest_avail - predicted_gb) >= self.mem_guard_gb
        cpu_hist = [s.sys_cpu_percent for s in self._history]
        avg_idle = 1.0 - (sum(cpu_hist) / len(cpu_hist)) / 100.0 if cpu_hist else 0.0
        cpu_ok = avg_idle >= self.min_cpu_idle_frac
        if not (cpu_ok and mem_ok):
            print(f"[Governor] Hold: cpu_idle={avg_idle:.2f}, mem_avail={latest_avail:.2f}GB,"
                  f" active={active:.2f}GB, next_pred={predicted_gb:.2f}GB")
        return cpu_ok and mem_ok

# --- Configuration ---
TARGET_INVERSIONS = {
    "chr3-195680867-INV-272256",
    "chr3-195749464-INV-230745",
    "chr6-76111919-INV-44661",
    "chr12-46897663-INV-16289",
    "chr6-141867315-INV-29159",
    "chr3-131969892-INV-7927",
    "chr6-167181003-INV-209976",
    "chr11-71571191-INV-6980",
    "chr9-102565835-INV-4446",
    "chr4-33098029-INV-7075",
    "chr7-57835189-INV-284465",
    "chr10-46135869-INV-77646",
    "chr11-24263185-INV-392",
    "chr13-79822252-INV-17591",
    "chr1-60775308-INV-5023",
    "chr6-130527042-INV-4267",
    "chr13-48199211-INV-7451",
    "chr21-13992018-INV-65632",
    "chr8-7301025-INV-5297356",
    "chr9-30951702-INV-5595",
    "chr17-45585160-INV-706887",
    "chr12-131333944-INV-289865",
    "chr7-70955928-INV-18020",
    "chr16-28471894-INV-165758",
    "chr7-65219158-INV-312667",
    "chr10-79542902-INV-674513",
    "chr1-13084312-INV-62181",
    "chr10-37102555-INV-11157",
    "chr4-40233409-INV-2010",
    "chr2-138246733-INV-5010",
}

PHENOTYPE_DEFINITIONS_URL = "https://github.com/SauersML/ferromic/raw/refs/heads/main/data/significant_heritability_diseases.tsv"
MASTER_RESULTS_CSV = f"phewas_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.tsv"

# --- Performance & Memory Tuning ---
MIN_AVAILABLE_MEMORY_GB = 4.0
QUEUE_MAX_SIZE = os.cpu_count() * 4
LOADER_THREADS = 32
LOADER_CHUNK_SIZE = 128

# --- Data sources and caching ---
CACHE_DIR = "./phewas_cache"
INVERSION_DOSAGES_FILE = "imputed_inversion_dosages.tsv"
PCS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
SEX_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/qc/genomic_metrics.tsv"
RELATEDNESS_URI = "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"

# --- Model parameters ---
NUM_PCS = 10
MIN_CASES_FILTER = 1000
MIN_CONTROLS_FILTER = 1000
MIN_NEFF_FILTER = 0 # Default off
FDR_ALPHA = 0.05

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

def _pipeline_once():
    """
    Entry point for the PheWAS pipeline. Uses module-level configuration directly.
    """
    script_start_time = time.time()

    if PSUTIL_AVAILABLE:
        monitor_thread = SystemMonitor(interval=3)
        monitor_thread.start()
    else:
        monitor_thread = None
    pipes.BUDGET.init_total(fraction=0.92)

    def mem_floor_callable():
        return pipes.BUDGET.floor_gb()

    print("=" * 70)
    print(" Starting Robust, Parallel PheWAS Pipeline")
    print("=" * 70)

    global TARGET_INVERSIONS
    if isinstance(TARGET_INVERSIONS, str):
        TARGET_INVERSIONS = {TARGET_INVERSIONS}

    from types import SimpleNamespace
    run = SimpleNamespace(TARGET_INVERSIONS=TARGET_INVERSIONS)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "locks"), exist_ok=True)

    try:
        with Timer() as t_setup:
            print("\n--- Loading shared data... ---")
            pheno_defs_df = pheno.load_definitions(PHENOTYPE_DEFINITIONS_URL)
            cdr_dataset_id = os.environ["WORKSPACE_CDR"]
            gcp_project = os.environ["GOOGLE_PROJECT"]
            bq_client = bigquery.Client(project=gcp_project)
            cdr_codename = cdr_dataset_id.split(".")[-1]
            demographics_df = io.get_cached_or_generate(os.path.join(CACHE_DIR, f"demographics_{cdr_codename}.parquet"), io.load_demographics_with_stable_age, bq_client=bq_client, cdr_id=cdr_dataset_id)
            pc_df = io.get_cached_or_generate(os.path.join(CACHE_DIR, f"pcs_{NUM_PCS}.parquet"), io.load_pcs, gcp_project, PCS_URI, NUM_PCS, validate_num_pcs=NUM_PCS)
            sex_df = io.get_cached_or_generate(os.path.join(CACHE_DIR, "genetic_sex.parquet"), io.load_genetic_sex, gcp_project, SEX_URI)
            related_ids_to_remove = io.load_related_to_remove(gcp_project=gcp_project, RELATEDNESS_URI=RELATEDNESS_URI)
            demographics_df.index, pc_df.index, sex_df.index = [df.index.astype(str) for df in (demographics_df, pc_df, sex_df)]
            shared_covariates_df = demographics_df.join(pc_df, how="inner").join(sex_df, how="inner")
            shared_covariates_df = shared_covariates_df[~shared_covariates_df.index.isin(related_ids_to_remove)]

            LABELS_URI = PCS_URI # Clarify that PCs and Ancestry labels are from the same source
            ancestry = io.get_cached_or_generate(os.path.join(CACHE_DIR, "ancestry_labels.parquet"), io.load_ancestry_labels, gcp_project, LABELS_URI=LABELS_URI)
            anc_series = ancestry.reindex(shared_covariates_df.index)["ANCESTRY"].str.lower()
            anc_cat_global = pd.Categorical(anc_series.reindex(shared_covariates_df.index))
            A_global = pd.get_dummies(anc_cat_global, prefix='ANC', drop_first=True, dtype=np.float32)
            A_global.index = A_global.index.astype(str)
            A_cols = list(A_global.columns)
        print(f"\n--- Shared Setup Time: {t_setup.duration:.2f}s ---")
        
        # --- Filter TARGET_INVERSIONS to only those present in the dosages TSV ---
        dosages_path = _find_upwards(INVERSION_DOSAGES_FILE)
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
            pheno.populate_caches_prepass(pheno_defs_df, bq_client, cdr_dataset_id, shared_covariates_df.index, CACHE_DIR, cdr_codename)
        except Exception as e:
            print(f"[Prepass WARN] Cache prepass failed: {e}", flush=True)


        governor = MultiTenantGovernor(monitor_thread)
        
        def run_single_inversion(target_inversion: str, baseline_rss_gb: float, shared_data: dict):
            inv_safe_name = models.safe_basename(target_inversion)
            log_prefix = f"[INV {inv_safe_name}]"
            try:
                print(f"{log_prefix} Started.", flush=True)
                inversion_cache_dir = os.path.join(CACHE_DIR, inv_safe_name)
                results_cache_dir = os.path.join(inversion_cache_dir, "results_atomic")
                lrt_overall_cache_dir = os.path.join(inversion_cache_dir, "lrt_overall")
                lrt_followup_cache_dir = os.path.join(inversion_cache_dir, "lrt_followup")
                os.makedirs(results_cache_dir, exist_ok=True)
                os.makedirs(lrt_overall_cache_dir, exist_ok=True)
                os.makedirs(lrt_followup_cache_dir, exist_ok=True)

                ctx = {"NUM_PCS": NUM_PCS, "MIN_CASES_FILTER": MIN_CASES_FILTER, "MIN_CONTROLS_FILTER": MIN_CONTROLS_FILTER, "MIN_NEFF_FILTER": MIN_NEFF_FILTER, "FDR_ALPHA": FDR_ALPHA, "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES, "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS, "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA, "CACHE_DIR": CACHE_DIR, "RIDGE_L2_BASE": RIDGE_L2_BASE, "RESULTS_CACHE_DIR": results_cache_dir, "LRT_OVERALL_CACHE_DIR": lrt_overall_cache_dir, "LRT_FOLLOWUP_CACHE_DIR": lrt_followup_cache_dir, "cdr_codename": shared_data['cdr_codename']}
                dosages_path = _find_upwards(INVERSION_DOSAGES_FILE)
                inversion_df = io.get_cached_or_generate(os.path.join(CACHE_DIR, f"inversion_{target_inversion}.parquet"), io.load_inversions, target_inversion, dosages_path, validate_target=target_inversion)
                inversion_df.index = inversion_df.index.astype(str)
                core_df = shared_data['covariates'].join(inversion_df, how="inner")

                age_mean = core_df['AGE'].mean()
                core_df['AGE_c'] = core_df['AGE'] - age_mean
                core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2
                pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                covariate_cols = [target_inversion] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
                core_df_subset = core_df[covariate_cols]
                core_df_with_const = sm.add_constant(core_df_subset, prepend=True)
                A_slice = shared_data['A_global'].reindex(core_df_with_const.index).fillna(0.0)
                core_df_with_const = pd.concat([core_df_with_const, A_slice], axis=1, copy=False)

                delta_core_df_gb = monitor_thread.snapshot().app_rss_gb - baseline_rss_gb
                governor.update_after_core_df(delta_core_df_gb)

                core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
                global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)
                pan_path = os.path.join(CACHE_DIR, f"pan_category_cases_{shared_data['cdr_codename']}.pkl")
                category_to_pan_cases = io.get_cached_or_generate_pickle(
                    pan_path,
                    pheno.build_pan_category_cases,
                    shared_data['pheno_defs'], shared_data['bq_client'], shared_data['cdr_id'], CACHE_DIR, shared_data['cdr_codename']
                )
                allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask)

                sex_vec = core_df_with_const['sex'].to_numpy(dtype=np.float32, copy=False)

                pheno_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
                fetcher_thread = threading.Thread(
                    target=pheno.phenotype_fetcher_worker,
                    args=(pheno_queue, shared_data['pheno_defs'], shared_data['bq_client'], shared_data['cdr_id'], shared_data['cdr_codename'], core_index, CACHE_DIR, LOADER_CHUNK_SIZE, LOADER_THREADS, True),
                    kwargs=dict(
                        allowed_mask_by_cat=allowed_mask_by_cat,
                        sex_vec=sex_vec,
                        min_cases=MIN_CASES_FILTER,
                        min_ctrls=MIN_CONTROLS_FILTER,
                        sex_mode="majority",
                        sex_prop=models.DEFAULT_SEX_RESTRICT_PROP,
                        max_other=ctx.get("SEX_RESTRICT_MAX_OTHER_CASES", 0),
                        min_neff=MIN_NEFF_FILTER,
                    ),
                )
                fetcher_thread.start()

                def on_pool_started_callback(num_procs, worker_pids):
                    governor.register_pool(inv_safe_name, worker_pids)
                    time.sleep(10)
                    measured_gb = governor.measure_inv(inv_safe_name)
                    governor.update_steady_state(inv_safe_name, measured_gb)
                    pipes.BUDGET.revise(inv_safe_name, "pool_steady", measured_gb)
                    per_worker = max(0.25, measured_gb / max(1, num_procs))
                    pipes._WORKER_GB_EST = 0.5 * pipes._WORKER_GB_EST + 0.5 * per_worker
                    print(f"[Budget] {inv_safe_name}.pool_steady: set {measured_gb:.2f}GB | remaining {pipes.BUDGET.remaining_gb():.2f}GB", flush=True)

                pipes.run_fits(pheno_queue, core_df_with_const, allowed_mask_by_cat, target_inversion, results_cache_dir, ctx, mem_floor_callable, on_pool_started=on_pool_started_callback)
                fetcher_thread.join()

                name_to_cat = shared_data['pheno_defs'].set_index('sanitized_name')['disease_category'].to_dict()
                result_paths = [os.path.join(results_cache_dir, f) for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]

                def is_valid_result(path, min_cases, min_controls):
                    try:
                        with open(path, 'r') as f:
                            res = json.load(f)
                        if pd.notna(res.get("Skip_Reason", np.nan)): return False
                        if (res.get("N_Cases", 0) < min_cases) or (res.get("N_Controls", 0) < min_controls): return False
                        return True
                    except Exception:
                        return False
                phenos_list = [os.path.splitext(os.path.basename(p))[0] for p in result_paths if is_valid_result(p, MIN_CASES_FILTER, MIN_CONTROLS_FILTER)]

                print(f"{log_prefix} Found {len(phenos_list)} valid models for Stage-1 LRT.")
                pipes.run_lrt_overall(core_df_with_const, allowed_mask_by_cat, shared_data['anc_series'], phenos_list, name_to_cat, shared_data['cdr_codename'], target_inversion, ctx, mem_floor_callable, on_pool_started=on_pool_started_callback)

                try:
                    lrt_files = [f for f in os.listdir(lrt_overall_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                    rows = []
                    for fn in lrt_files:
                        s = pd.read_json(os.path.join(lrt_overall_cache_dir, fn), typ="series")
                        rows.append({"Phenotype": os.path.splitext(fn)[0], "P_LRT_Overall": pd.to_numeric(s.get("P_LRT_Overall"), errors="coerce")})
                    lrt_df = pd.DataFrame(rows)
                    res_files = [f for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                    rrows = []
                    for fn in res_files:
                        s = pd.read_json(os.path.join(results_cache_dir, fn), typ="series")
                        rrows.append({"Phenotype": os.path.splitext(fn)[0], "OR": pd.to_numeric(s.get("OR"), errors="coerce"), "Beta": pd.to_numeric(s.get("Beta"), errors="coerce"), "N_Cases": pd.to_numeric(s.get("N_Cases"), errors="coerce"), "N_Controls": pd.to_numeric(s.get("N_Controls"), errors="coerce")})
                    res_df = pd.DataFrame(rrows)
                    inv_df = lrt_df.merge(res_df, on="Phenotype", how="left") if not lrt_df.empty else pd.DataFrame(columns=["Phenotype","P_LRT_Overall","OR","Beta","N_Cases","N_Controls"])
                    m = int(inv_df["P_LRT_Overall"].notna().sum()) if not inv_df.empty else 0
                    if m > 0:
                        mask = inv_df["P_LRT_Overall"].notna()
                        _, q_within, _, _ = multipletests(inv_df.loc[mask, "P_LRT_Overall"], alpha=FDR_ALPHA, method="fdr_bh")
                        inv_df.loc[mask, "Q_within"] = q_within

                    def _fmt(v, fmt_str): return f"{float(v):{fmt_str}}" if pd.notna(v) else ""
                    top = inv_df.sort_values("P_LRT_Overall").head(10).copy() if m > 0 else inv_df.head(0)
                    top["P"] = top["P_LRT_Overall"].apply(lambda v: _fmt(v, ".3e"))
                    top["Q"] = top["Q_within"].apply(lambda v: _fmt(v, ".3f"))
                    top["OR"] = top["OR"].apply(lambda v: _fmt(v, "0.3f"))
                    top["Beta"] = top["Beta"].apply(lambda v: _fmt(v, "+0.4f"))
                    top["N"] = (pd.to_numeric(top["N_Cases"], errors="coerce").fillna(0).astype(int)).astype(str) + "/" + (pd.to_numeric(top["N_Controls"], errors="coerce").fillna(0).astype(int)).astype(str)
                    print(f"\n{log_prefix} --- Top Hits Summary (provisional) ---\n" + top[["Phenotype","P","Q","OR","Beta","N"]].to_string(index=False) + "\n")
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

            for inv_name in running_inversions.values():
                governor.measure_inv(inv_name)

            if pending_inversions:
                target_inv = pending_inversions[0]
                try:
                    inversion_path = os.path.join(CACHE_DIR, f"inversion_{target_inv}.parquet")
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

        # --- PART 3: CONSOLIDATE & ANALYZE RESULTS (ACROSS ALL INVERSIONS) ---
        print("\n" + "=" * 70)
        print(" Part 3: Consolidating final results across all inversions")
        print("=" * 70)

        all_results_from_disk = []
        for target_inversion in run.TARGET_INVERSIONS:
            inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
            results_cache_dir = os.path.join(inversion_cache_dir, "results_atomic")
            result_files = [f for f in os.listdir(results_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
            for filename in result_files:
                try:
                    result = pd.read_json(os.path.join(results_cache_dir, filename), typ="series").to_dict()
                    result['Inversion'] = target_inversion
                    all_results_from_disk.append(result)
                except Exception as e:
                    print(f"Warning: Could not read corrupted result file: {filename}, Error: {e}")

        if not all_results_from_disk:
            print("No results found to process.")
        else:
            df = pd.DataFrame(all_results_from_disk)
            print(f"Successfully consolidated {len(df)} results across {len(run.TARGET_INVERSIONS)} inversions.")

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

            # Collect all Stage-1 LRT results from their per-inversion cache directories
            print("\n--- Collecting Stage-1 LRT results ---")
            overall_records = []
            for target_inversion in run.TARGET_INVERSIONS:
                lrt_overall_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion), "lrt_overall")
                if not os.path.isdir(lrt_overall_cache_dir): continue
                files_overall = [f for f in os.listdir(lrt_overall_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                for filename in files_overall:
                    try:
                        rec = pd.read_json(os.path.join(lrt_overall_cache_dir, filename), typ="series").to_dict()
                        rec['Inversion'] = target_inversion
                        overall_records.append(rec)
                    except Exception as e:
                        print(f"Warning: Could not read LRT overall file: {filename}, Error: {e}")

            if overall_records:
                overall_df = pd.DataFrame(overall_records)
                print(f"Collected {len(overall_df)} Stage-1 LRT records across {len(run.TARGET_INVERSIONS)} inversions.")
                # Merge LRT results into the main dataframe
                df = df.merge(overall_df, on=["Phenotype", "Inversion"], how="left")
            else:
                print("No Stage-1 LRT records found.")

            # Perform global FDR correction on the merged dataframe
            mask_overall = pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna()
            m_total = int(mask_overall.sum())
            df["Q_GLOBAL"] = np.nan
            if m_total > 0:
                print(f"\n--- Applying global BH-FDR correction to {m_total} valid P-values ---")
                _, q_adj_global, _, _ = multipletests(df.loc[mask_overall, "P_LRT_Overall"], alpha=FDR_ALPHA, method="fdr_bh")
                df.loc[mask_overall, "Q_GLOBAL"] = q_adj_global

            df["Sig_Global"] = df["Q_GLOBAL"] < FDR_ALPHA

            # --- PART 4: SCHEDULE AND RUN STAGE-2 FOLLOW-UPS ---
            print("\n" + "=" * 70)
            print(" Part 4: Running Stage-2 Follow-up Analyses for Global Hits")
            print("=" * 70)
            name_to_cat = pheno_defs_df.set_index('sanitized_name')['disease_category'].to_dict()

            for target_inversion in run.TARGET_INVERSIONS:
                # Re-create the inversion-specific context and data to ensure correct follow-up
                dosages_path = _find_upwards(INVERSION_DOSAGES_FILE)
                inversion_df = io.get_cached_or_generate(
                    os.path.join(CACHE_DIR, f"inversion_{target_inversion}.parquet"),
                    io.load_inversions, target_inversion, dosages_path, validate_target=target_inversion,
                )
                inversion_df.index = inversion_df.index.astype(str)

                core_df = shared_covariates_df.join(inversion_df, how="inner")
                age_mean = core_df['AGE'].mean()
                core_df['AGE_c'] = core_df['AGE'] - age_mean
                core_df['AGE_c_sq'] = core_df['AGE_c'] ** 2
                pc_cols = [f"PC{i}" for i in range(1, NUM_PCS + 1)]
                covariate_cols = [target_inversion] + ["sex"] + pc_cols + ["AGE_c", "AGE_c_sq"]
                core_df_subset = core_df[covariate_cols]
                core_df_with_const = sm.add_constant(core_df_subset, prepend=True)
                A_slice = A_global.reindex(core_df_with_const.index).fillna(0.0)
                core_df_with_const = pd.concat([core_df_with_const, A_slice], axis=1, copy=False)
                core_index = pd.Index(core_df_with_const.index.astype(str), name="person_id")
                global_notnull_mask = np.isfinite(core_df_with_const.to_numpy()).all(axis=1)
                pan_path = os.path.join(CACHE_DIR, f"pan_category_cases_{cdr_codename}.pkl")
                category_to_pan_cases = io.get_cached_or_generate_pickle(
                    pan_path,
                    pheno.build_pan_category_cases,
                    pheno_defs_df, bq_client, cdr_dataset_id, CACHE_DIR, cdr_codename
                )
                allowed_mask_by_cat = pheno.build_allowed_mask_by_cat(core_index, category_to_pan_cases, global_notnull_mask)

                inversion_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion))
                ctx = {
                    "NUM_PCS": NUM_PCS, "MIN_CASES_FILTER": MIN_CASES_FILTER, "MIN_CONTROLS_FILTER": MIN_CONTROLS_FILTER,
                    "MIN_NEFF_FILTER": MIN_NEFF_FILTER,
                    "FDR_ALPHA": FDR_ALPHA, "PER_ANC_MIN_CASES": PER_ANC_MIN_CASES, "PER_ANC_MIN_CONTROLS": PER_ANC_MIN_CONTROLS,
                    "LRT_SELECT_ALPHA": LRT_SELECT_ALPHA, "CACHE_DIR": CACHE_DIR, "RIDGE_L2_BASE": RIDGE_L2_BASE,
                    "RESULTS_CACHE_DIR": os.path.join(inversion_cache_dir, "results_atomic"),
                    "LRT_OVERALL_CACHE_DIR": os.path.join(inversion_cache_dir, "lrt_overall"),
                    "LRT_FOLLOWUP_CACHE_DIR": os.path.join(inversion_cache_dir, "lrt_followup"),
                    "cdr_codename": cdr_codename,
                }

                # Select hits for the current inversion and run follow-up
                hit_phenos = df.loc[(df["Sig_Global"] == True) & (df["Inversion"] == target_inversion), "Phenotype"].astype(str).tolist()
                if hit_phenos:
                    print(f"--- Running follow-up for {len(hit_phenos)} hits in {target_inversion} ---")
                    pipes.run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_phenos, name_to_cat, cdr_codename, target_inversion, ctx, mem_floor_callable)

            # Consolidate all follow-up results
            print("\n--- Consolidating all Stage-2 follow-up results ---")
            follow_records = []
            for target_inversion in run.TARGET_INVERSIONS:
                lrt_followup_cache_dir = os.path.join(CACHE_DIR, models.safe_basename(target_inversion), "lrt_followup")
                if not os.path.isdir(lrt_followup_cache_dir): continue
                files_follow = [f for f in os.listdir(lrt_followup_cache_dir) if f.endswith(".json") and not f.endswith(".meta.json")]
                for filename in files_follow:
                    try:
                        rec = pd.read_json(os.path.join(lrt_followup_cache_dir, filename), typ="series").to_dict()
                        rec['Inversion'] = target_inversion
                        follow_records.append(rec)
                    except Exception as e:
                        print(f"Warning: Could not read LRT follow-up file: {filename}, Error: {e}")

            if follow_records:
                follow_df = pd.DataFrame(follow_records)
                print(f"Collected {len(follow_df)} follow-up records.")
                df = df.merge(follow_df, on=["Phenotype", "Inversion"], how="left")

            m_total = int(pd.to_numeric(df["P_LRT_Overall"], errors="coerce").notna().sum())
            R_selected = int(pd.to_numeric(df["Sig_Global"], errors="coerce").fillna(False).astype(bool).sum())
            alpha_within = (FDR_ALPHA * (R_selected / m_total)) if m_total > 0 else 0.0

            if R_selected > 0 and alpha_within > 0.0:
                selected_idx = df.index[df["Sig_Global"] == True].tolist()
                for idx in selected_idx:
                    p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
                    if (not pd.notna(p_lrt)) or (p_lrt >= LRT_SELECT_ALPHA): continue
                    levels_str = df.at[idx, "LRT_Ancestry_Levels"] if "LRT_Ancestry_Levels" in df.columns else ""
                    anc_levels = [s for s in str(levels_str).split(",") if s]
                    anc_upper = [s.upper() for s in anc_levels]
                    pvals, keys = [], []
                    for anc in anc_upper:
                        pcol, rcol = f"{anc}_P", f"{anc}_REASON"
                        if pcol in df.columns and rcol in df.columns:
                            pval, reason = df.at[idx, pcol], df.at[idx, rcol]
                            if pd.notna(pval) and reason != "insufficient_stratum_counts" and reason != "not_selected_by_LRT":
                                pvals.append(float(pval)); keys.append(anc)
                    if len(pvals) > 0:
                        _, p_adj_vals, _, _ = multipletests(pvals, alpha=alpha_within, method="fdr_bh")
                        for anc_key, adj_val in zip(keys, p_adj_vals):
                            df.at[idx, f"{anc_key}_P_FDR"] = float(adj_val)

            if "EUR_P_Source" in df.columns: df = df.drop(columns=["EUR_P_Source"], errors="ignore")

            if "Sig_Global" in df.columns:
                df["FINAL_INTERPRETATION"] = ""
                for idx in df.index[df['Sig_Global'] == True].tolist():
                    p_lrt = df.at[idx, "P_LRT_AncestryxDosage"] if "P_LRT_AncestryxDosage" in df.columns else np.nan
                    if pd.isna(p_lrt) or p_lrt >= LRT_SELECT_ALPHA:
                        df.at[idx, "FINAL_INTERPRETATION"] = "overall"
                        continue

                    levels_str = df.at[idx, "LRT_Ancestry_Levels"] if "LRT_Ancestry_Levels" in df.columns else ""
                    anc_levels = [s.upper() for s in str(levels_str).split(",") if s]
                    sig_groups = []
                    for anc in anc_levels:
                        adj_col, rcol = f"{anc}_P_FDR", f"{anc}_REASON"
                        if adj_col in df.columns:
                            p_adj, reason = df.at[idx, adj_col], df.at[idx, rcol] if rcol in df.columns else ""
                            if pd.notna(p_adj) and p_adj < alpha_within and reason != "insufficient_stratum_counts" and reason != "not_selected_by_LRT":
                                sig_groups.append(anc)
                    df.at[idx, "FINAL_INTERPRETATION"] = ",".join(sig_groups) if sig_groups else "unable to determine"

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


def supervisor_main(max_restarts=100, backoff_sec=10):
    import multiprocessing as mp, time, signal
    ctx = mp.get_context("spawn")
    should_stop = {"flag": False}

    def _stop(*_):
        should_stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    restarts = 0
    while not should_stop["flag"] and restarts <= max_restarts:
        p = ctx.Process(target=_pipeline_once, name="ferromic-pipeline")
        p.start()
        while p.is_alive():
            if should_stop["flag"]:
                try:
                    p.terminate()
                except Exception:
                    pass
                p.join(timeout=5)
                return
            time.sleep(0.2)
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


def main():
    supervisor_main()


if __name__ == "__main__":
    supervisor_main()
