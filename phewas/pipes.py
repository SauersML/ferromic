import threading
import sys
from multiprocessing import get_context, cpu_count
import os
import json
import math
import gc
import traceback
import numpy as np

from . import models
from . import iox as io
import time
import random

# PHEWAS main removed: Stage-1 (LRT/Bootstrap) is the only engine.


def _log_worker_exception(stage: str, exc: BaseException) -> None:
    """Emit a detailed message for a worker exception before re-raising."""

    message = str(exc).strip()
    if message:
        header = f"\n[pool ERR] {stage} worker raised {type(exc).__name__}: {message}"
    else:
        header = f"\n[pool ERR] {stage} worker raised {type(exc).__name__}"
    print(header, flush=True)

    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    if formatted:
        print(formatted, flush=True)


def cgroup_available_gb():
    """Return remaining memory permitted by the active cgroup, if known.

    Cgroup usage counters include the page cache, which is typically reclaimable.
    Deducting that cache from the limit can significantly under-report available
    memory in data-heavy workloads (for example, after reading large input
    files).  To avoid artificially stalling the governor, estimate the
    reclaimable cache from ``memory.stat`` and subtract it from the usage before
    computing the available bytes.
    """

    def _read_int(path: str) -> int:
        with open(path, "r") as fh:
            return int(fh.read().strip())

    def _reclaimable_bytes(stat_path: str, keys: tuple[str, ...]) -> int:
        try:
            with open(stat_path, "r") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    key, value = parts
                    if key in keys:
                        try:
                            return int(value)
                        except ValueError:
                            return 0
        except Exception:
            pass
        return 0

    def _available(limit_path: str, usage_path: str, stat_path: str, stat_keys: tuple[str, ...]):
        if not os.path.exists(limit_path) or not os.path.exists(usage_path):
            return None

        limit_raw = _read_int(limit_path)
        if limit_raw <= 0 or limit_raw >= (1 << 60):
            return None

        usage_raw = _read_int(usage_path)
        reclaimable = _reclaimable_bytes(stat_path, stat_keys)
        adjusted_usage = max(0, usage_raw - reclaimable)
        return max(0.0, (limit_raw - adjusted_usage) / (1024**3))

    try:
        avail_v2 = _available(
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory.current",
            "/sys/fs/cgroup/memory.stat",
            ("inactive_file", "file"),
        )
        if avail_v2 is not None:
            return avail_v2

        avail_v1 = _available(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.stat",
            ("total_inactive_file", "total_cache"),
        )
        if avail_v1 is not None:
            return avail_v1
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _prob_is_valid(value) -> bool:
    """Return True if *value* is a finite probability in [0, 1]."""
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(prob) and (0.0 <= prob <= 1.0)


def _cached_lrt_result_is_usable(res_obj: dict) -> bool:
    """Determine whether a cached Stage-1 LRT result should be reused."""
    if not isinstance(res_obj, dict):
        return False

    if _prob_is_valid(res_obj.get("P_LRT_Overall")):
        return True

    # Results with explicit failure/skip reasons should not be evicted—the rerun
    # would simply reproduce the same outcome.
    if res_obj.get("LRT_Overall_Reason"):
        return True

    p_overall_valid = res_obj.get("P_Overall_Valid")
    if p_overall_valid is False:
        return True

    if p_overall_valid:
        if _prob_is_valid(res_obj.get("P_Value")):
            source = res_obj.get("P_Source") or res_obj.get("P_Method")
            if isinstance(source, str):
                source = source.lower()
            allowed_sources = getattr(models, "ALLOWED_P_SOURCES", set())
            if source in allowed_sources:
                return True

    return False


def _evict_if_ctx_mismatch(meta_path, res_path, ctx, expected_target):
    """Remove stale metadata/results when the recorded context no longer matches."""
    if not os.path.exists(meta_path):
        return False
    meta = io.read_meta_json(meta_path)
    if not meta:
        return False

    stale = False
    ctx_tag = ctx.get("CTX_TAG")
    if ctx_tag and meta.get("ctx_tag") != ctx_tag:
        stale = True
    cdr = ctx.get("cdr_codename")
    if cdr and meta.get("cdr_codename") != cdr:
        stale = True
    version_tag = ctx.get("CACHE_VERSION_TAG")
    if version_tag and meta.get("cache_version_tag") != version_tag:
        stale = True
    if expected_target and meta.get("target") != expected_target:
        stale = True
    data_keys = ctx.get("DATA_KEYS")
    if data_keys and meta.get("data_keys") != data_keys:
        stale = True

    if not stale:
        return False

    try:
        os.remove(meta_path)
    except Exception:
        pass

    if res_path and os.path.exists(res_path):
        stale_path = res_path + ".stale"
        try:
            if os.path.exists(stale_path):
                os.remove(stale_path)
        except Exception:
            pass
        try:
            os.replace(res_path, stale_path)
        except Exception:
            pass
    return True

# ---- Global Budget Manager (no new files) ----

class BudgetManager:
    """
    Global memory token manager (GB). Respects container cgroup limit.
    Thread-safe; suitable for orchestrator + pool workers.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._total_gb = self._detect_container_limit_gb()
        self._guard_gb = 2.0  # minimum headroom to avoid OOM killer (kept small; we rely on real reservations)
        self._reserved_by_inv = {}   # inv_id -> {component -> gb}
        self._total_reserved = 0.0

    def _detect_container_limit_gb(self):
        # cgroups v2 then v1; fallback to psutil
        try:
            # v2
            path = "/sys/fs/cgroup/memory.max"
            if os.path.exists(path):
                val = open(path).read().strip()
                if val.isdigit():
                    return max(1.0, int(val) / (1024**3))
            # v1
            path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            if os.path.exists(path):
                lim = int(open(path).read().strip())
                if lim > 0 and lim < (1<<60):
                    return max(1.0, lim / (1024**3))
        except Exception:
            pass
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 16.0  # conservative fallback

    def init_total(self, fraction=0.92):
        # Reserve a fraction for ourselves (avoid kernel/OOM headroom)
        with self._lock:
            if not getattr(self, "_init_done", False):
                self._total_gb *= float(fraction)
                self._init_done = True

    def remaining_gb(self):
        with self._lock:
            return max(0.0, self._total_gb - self._total_reserved)

    def floor_gb(self):
        with self._lock:
            return max(self._guard_gb, 0.05 * self._total_gb)

    def reserve(self, inv_id: str, component: str, gb: float, block=True):
        gb = max(0.0, float(gb))
        with self._cond:
            while block and (self._total_reserved + gb + self._guard_gb) > self._total_gb:
                self._cond.wait(timeout=0.5)
            if (self._total_reserved + gb + self._guard_gb) > self._total_gb:
                return False  # non-blocking and can't reserve
            self._reserved_by_inv.setdefault(inv_id, {})
            self._reserved_by_inv[inv_id][component] = self._reserved_by_inv[inv_id].get(component, 0.0) + gb
            self._total_reserved += gb
            return True

    def revise(self, inv_id: str, component: str, new_gb: float):
        new_gb = max(0.0, float(new_gb))
        with self._cond:
            self._reserved_by_inv.setdefault(inv_id, {})
            cur = self._reserved_by_inv.get(inv_id, {}).get(component, 0.0)
            delta = new_gb - cur
            if delta <= 0:
                self._reserved_by_inv[inv_id][component] = new_gb
                self._total_reserved += delta
                self._cond.notify_all()
                return True
            while (self._total_reserved + delta + self._guard_gb) > self._total_gb:
                self._cond.wait(timeout=0.5)
            self._reserved_by_inv[inv_id][component] = new_gb
            self._total_reserved += delta
            return True

    def release(self, inv_id: str, component: str):
        with self._cond:
            cur = self._reserved_by_inv.get(inv_id, {}).pop(component, 0.0)
            if not self._reserved_by_inv.get(inv_id):
                self._reserved_by_inv.pop(inv_id, None)
            self._total_reserved -= cur
            self._total_reserved = max(0.0, self._total_reserved)
            self._cond.notify_all()

# module-global singleton
BUDGET = BudgetManager()


class ProgressRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {}

    def update(self, inv, stage, done, total):
        with self._lock:
            self._data[(inv, stage)] = (int(done), int(total), time.time())

    def snapshot(self):
        with self._lock:
            return dict(self._data)


PROGRESS = ProgressRegistry()

_WORKER_GB_EST = 0.5
POOL_PROCS_PER_INV = 42

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

MP_CONTEXT = 'forkserver' if sys.platform.startswith('linux') else 'spawn'

def mem_ok_for_submission(min_free_gb: float = None):
    from math import isfinite
    rem = BUDGET.remaining_gb()
    floor = BUDGET.floor_gb() if min_free_gb is None else float(min_free_gb)
    return isfinite(rem) and rem >= floor

def _resolve_floor(v):
    """
    Resolves a numeric floor value from either a float or a zero-argument callable returning a float.
    """
    if callable(v):
        try:
            return float(v())
        except Exception:
            # If callable fails, it's safer to fallback to a default value
            # rather than trying to cast the callable itself.
            # Here, we might just return a very high floor to be safe, or a default.
            # For now, let's assume the callable is trusted and this is for edge cases.
            return 4.0 # Default fallback
    return float(v)

class MemoryMonitor(threading.Thread):
    def __init__(self, interval=1):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.available_memory_gb = 0
        self.rss_gb = 0
        self.sys_cpu_percent = 0.0
        self.app_cpu_percent = 0.0

    def run(self):
        try:
            import psutil, os, time
        except Exception:
            while not self.stop_event.is_set():
                time.sleep(self.interval)
            return

        p = psutil.Process()
        n_cpus = psutil.cpu_count(logical=True) or os.cpu_count() or 1

        def _cgroup_bytes():
            for path in ("/sys/fs/cgroup/memory.current",
                         "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                try:
                    return int(open(path, "r").read().strip())
                except Exception:
                    pass
            return None

        while not self.stop_event.is_set():
            try:
                vm = psutil.virtual_memory()
                host_avail = vm.available / (1024**3)
                cg_avail = cgroup_available_gb()
                self.available_memory_gb = min(host_avail, cg_avail) if cg_avail is not None else host_avail

                cg = _cgroup_bytes()
                if cg is not None:
                    self.rss_gb = cg / (1024**3)
                else:
                    total = 0
                    for proc in [p] + (p.children(recursive=True) or []):
                        try:
                            finfo = proc.memory_full_info()
                            total += getattr(finfo, "uss", finfo.rss)
                        except Exception:
                            try:
                                total += proc.memory_info().rss
                            except Exception:
                                pass
                    self.rss_gb = total / (1024**3)

                cpu0 = 0.0
                for proc in (p.children(recursive=True) or []):
                    try:
                        cpu0 += proc.cpu_percent(interval=None)
                    except Exception:
                        pass
                time.sleep(0.25)
                cpu1 = 0.0
                for proc in (p.children(recursive=True) or []):
                    try:
                        cpu1 += proc.cpu_percent(interval=None)
                    except Exception:
                        pass
                self.app_cpu_percent = min(100.0, (cpu0 + cpu1) / 2.0 / n_cpus)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


def run_lrt_overall(core_df_with_const, allowed_mask_by_cat, anc_series, phenos_list, name_to_cat, cdr_codename, target_inversion, ctx, min_available_memory_gb, on_pool_started=None):
    """
    Same pool pattern; submits models.lrt_overall_worker.
    """
    tasks = []
    for item in phenos_list:
        if isinstance(item, dict):
            name = item.get("name") or item.get("sanitized_name")
            if name is None:
                raise KeyError("phenotype payload missing 'name'")
            category = item.get("category", name_to_cat.get(name, None))
            task = {
                "name": name,
                "category": category,
                "cdr_codename": item.get("cdr_codename", cdr_codename),
                "target": target_inversion,
            }
            for key, value in item.items():
                if key not in task:
                    task[key] = value
        else:
            name = item
            task = {
                "name": name,
                "category": name_to_cat.get(name, None),
                "cdr_codename": cdr_codename,
                "target": target_inversion,
            }
        tasks.append(task)
    random.shuffle(tasks)

    monitor = MemoryMonitor()
    monitor.start()
    try:
        BUDGET.reserve(target_inversion, "core_shm", 0.0, block=True)
        # core_df_with_const is cast to float32 immediately after; 4 bytes per value
        bytes_needed = int(core_df_with_const.index.size) * int(len(core_df_with_const.columns)) * 4
        shm_gb = bytes_needed / (1024**3)
        BUDGET.revise(target_inversion, "core_shm", shm_gb)
        print(f"[Budget] {target_inversion}.core_shm: set {shm_gb:.2f}GB | remaining {BUDGET.remaining_gb():.2f}GB", flush=True)

        C = cpu_count()
        W_gb = max(0.25, _WORKER_GB_EST)
        max_by_budget = max(1, int(BUDGET.remaining_gb() // W_gb))
        n_procs = max(1, min(C, max_by_budget, POOL_PROCS_PER_INV))
        print(f"[LRT-Stage1] Scheduling {len(tasks)} phenotypes for overall LRT with atomic caching ({n_procs} workers).", flush=True)
        bar_len = 40
        queued = 0
        done = 0

        def _print_bar(q, d, label):
            q = int(q)
            d = int(d)
            pct = int((d * 100) / q) if q else 0
            filled = int(bar_len * (d / q)) if q else 0
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            mem_info = (f"| App≈{monitor.rss_gb:.2f}GB  "
                        f"SysAvail≈{monitor.available_memory_gb:.2f}GB  "
                        f"Budget≈{BUDGET.remaining_gb():.2f}GB")
            PROGRESS.update(target_inversion, label, d, q)
            print(f"\r[{label}] {bar} {d}/{q} ({pct}%) {mem_info}", end="", flush=True)

        X_base = core_df_with_const.to_numpy(dtype=np.float32, copy=True)
        base_meta, base_shm = io.create_shared_from_ndarray(X_base, readonly=True)
        core_cols = list(core_df_with_const.columns)
        core_index = core_df_with_const.index.astype(str)
        del X_base, core_df_with_const
        gc.collect()

        BUDGET.reserve(target_inversion, "pool_steady", 0.0, block=True)

        def _task_iter():
            nonlocal queued
            for task in tasks:
                floor = _resolve_floor(min_available_memory_gb)
                while BUDGET.remaining_gb() < floor:
                    print(
                        f"\n[gov WARN] Budget low (remain: {BUDGET.remaining_gb():.2f}GB, floor: {floor:.2f}GB), pausing task submission...",
                        flush=True,
                    )
                    time.sleep(2)

                # Cache policy: if a previous Stage-1 LRT result exists but has an invalid or NA P_LRT_Overall,
                # evict the meta to force a fresh run. LRT tasks are only scheduled for non-skipped models.
                try:
                    _res_path = os.path.join(ctx["LRT_OVERALL_CACHE_DIR"], f"{task['name']}.json")
                    _meta_path = os.path.join(ctx["LRT_OVERALL_CACHE_DIR"], f"{task['name']}.meta.json")
                    _evict_if_ctx_mismatch(_meta_path, _res_path, ctx, target_inversion)
                    if os.path.exists(_res_path) and os.path.exists(_meta_path):
                        with open(_res_path, "r") as _rf:
                            _res_obj = json.load(_rf)
                        if not _cached_lrt_result_is_usable(_res_obj):
                            try:
                                os.remove(_meta_path)
                                print(
                                    f"\n[cache POLICY] Invalid or missing P_LRT_Overall for '{task['name']}'. Forcing re-run by removing meta.",
                                    flush=True,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

                queued += 1
                _print_bar(queued, done, "LRT-Stage1")
                yield task

        try:
            with get_context(MP_CONTEXT).Pool(
                processes=n_procs,
                initializer=models.init_lrt_worker,
                initargs=(base_meta, core_cols, core_index, allowed_mask_by_cat, anc_series, ctx),
                maxtasksperchild=500,
            ) as pool:
                if on_pool_started:
                    try:
                        worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p and p.pid]
                    except Exception:
                        worker_pids = []
                    try:
                        on_pool_started(n_procs, worker_pids)
                    except Exception as e:
                        print(f"\n[WARN] on_pool_started callback failed: {e}", flush=True)

                try:
                    for _ in pool.imap_unordered(models.lrt_overall_worker, _task_iter(), chunksize=1):
                        done += 1
                        _print_bar(queued, done, "LRT-Stage1")
                except Exception as exc:
                    _log_worker_exception("LRT-Stage1", exc)
                    raise
                finally:
                    _print_bar(queued, done, "LRT-Stage1")
                    print("")
        finally:
            base_shm.close()
            base_shm.unlink()
            BUDGET.release(target_inversion, "pool_steady")
            BUDGET.release(target_inversion, "core_shm")
    finally:
        monitor.stop()


def run_bootstrap_overall(core_df_with_const, allowed_mask_by_cat, anc_series,
                          phenos_list, name_to_cat, cdr_codename, target_inversion,
                          ctx, min_available_memory_gb, on_pool_started=None):
    """Stage-1 parametric bootstrap with shared U matrix."""
    import gc, os, numpy as np, random, threading, time, hashlib
    tasks = []
    for item in phenos_list:
        if isinstance(item, dict):
            name = item.get("name") or item.get("sanitized_name")
            if name is None:
                raise KeyError("phenotype payload missing 'name'")
            category = item.get("category", name_to_cat.get(name, None))
            task = {
                "name": name,
                "category": category,
                "cdr_codename": item.get("cdr_codename", cdr_codename),
                "target": target_inversion,
            }
            for key, value in item.items():
                if key not in task:
                    task[key] = value
        else:
            name = item
            task = {
                "name": name,
                "category": name_to_cat.get(name, None),
                "cdr_codename": cdr_codename,
                "target": target_inversion,
            }
        tasks.append(task)
    random.shuffle(tasks)

    monitor = MemoryMonitor()
    monitor.start()
    try:
        BUDGET.reserve(target_inversion, "core_shm", 0.0, block=True)
        X_base = core_df_with_const.to_numpy(dtype=np.float32, copy=True)
        base_meta, base_shm = io.create_shared_from_ndarray(X_base, readonly=True)
        core_cols = list(core_df_with_const.columns)
        core_index = core_df_with_const.index.astype(str)
        shm_gb = (X_base.nbytes) / (1024**3)
        BUDGET.revise(target_inversion, "core_shm", shm_gb)
        del X_base, core_df_with_const
        gc.collect()

        B = int(ctx.get("BOOTSTRAP_B", 1000))
        seed_base = int(ctx.get("BOOT_SEED_BASE", 2025))
        inv_tag = str(target_inversion).encode()
        inv_hash = int(hashlib.blake2s(inv_tag, digest_size=8).hexdigest(), 16)
        rng = np.random.default_rng(seed_base ^ inv_hash)
        N = len(core_index)
        U_gb = (N * B * 4) / (1024**3)
        BUDGET.reserve(target_inversion, "boot_shm", U_gb, block=True)
        U = np.empty((N, B), dtype=np.float32)
        step = max(1, min(B, 64))
        for j0 in range(0, B, step):
            j1 = min(B, j0 + step)
            U[:, j0:j1] = rng.random((N, j1 - j0), dtype=np.float32)
        boot_meta, boot_shm = io.create_shared_from_ndarray(U, readonly=True)
        BUDGET.revise(target_inversion, "boot_shm", U_gb)
        del U
        gc.collect()

        C = cpu_count()
        W_gb = max(0.25, _WORKER_GB_EST)
        max_by_budget = max(1, int(BUDGET.remaining_gb() // W_gb))
        n_procs = max(1, min(C, max_by_budget, POOL_PROCS_PER_INV))
        print(f"[Bootstrap-Stage1] Scheduling {len(tasks)} phenotypes (B={B}) with {n_procs} workers.", flush=True)

        bar_len, queued, done = 40, 0, 0

        def _print(q, d):
            pct = int((d * 100) / q) if q else 0
            filled = int(bar_len * (d / max(1, q)))
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            print(
                f"\r[Bootstrap-Stage1] {bar} {d}/{q} ({pct}%) | App≈{monitor.rss_gb:.2f}GB  Sys≈{monitor.available_memory_gb:.2f}GB  Budget≈{BUDGET.remaining_gb():.2f}GB",
                end="",
                flush=True,
            )

        BUDGET.reserve(target_inversion, "pool_steady", 0.0, block=True)

        def _task_iter():
            nonlocal queued
            for task in tasks:
                floor = _resolve_floor(min_available_memory_gb)
                while BUDGET.remaining_gb() < floor:
                    print(
                        f"\n[gov WARN] Budget low (remain: {BUDGET.remaining_gb():.2f}GB, floor: {floor:.2f}GB), pause...",
                        flush=True,
                    )
                    time.sleep(2)
                boot_dir = ctx.get("BOOT_OVERALL_CACHE_DIR")
                if boot_dir:
                    res_path = os.path.join(boot_dir, f"{task['name']}.json")
                    meta_path = os.path.join(boot_dir, f"{task['name']}.meta.json")
                    _evict_if_ctx_mismatch(meta_path, res_path, ctx, target_inversion)
                queued += 1
                _print(queued, done)
                yield task

        try:
            with get_context(MP_CONTEXT).Pool(
                processes=n_procs,
                initializer=models.init_boot_worker,
                initargs=(base_meta, boot_meta, core_cols, core_index, allowed_mask_by_cat, anc_series, ctx),
                maxtasksperchild=500,
            ) as pool:
                if on_pool_started:
                    try:
                        worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p and p.pid]
                    except Exception:
                        worker_pids = []
                    try:
                        on_pool_started(n_procs, worker_pids)
                    except Exception as e:
                        print(f"\n[WARN] on_pool_started callback failed: {e}", flush=True)

                try:
                    for _ in pool.imap_unordered(models.bootstrap_overall_worker, _task_iter(), chunksize=1):
                        done += 1
                        _print(queued, done)
                except Exception as exc:
                    _log_worker_exception("Bootstrap-Stage1", exc)
                    raise
                finally:
                    _print(queued, done)
                    print("")
        finally:
            try:
                BUDGET.release(target_inversion, "pool_steady")
            except Exception:
                pass
            try:
                boot_shm.close()
                boot_shm.unlink()
            except Exception:
                pass
            try:
                base_shm.close()
                base_shm.unlink()
            except Exception:
                pass
            try:
                BUDGET.release(target_inversion, "boot_shm")
            except Exception:
                pass
            try:
                BUDGET.release(target_inversion, "core_shm")
            except Exception:
                pass
    finally:
        monitor.stop()


def run_lrt_followup(core_df_with_const, allowed_mask_by_cat, anc_series, hit_names, name_to_cat, cdr_codename, target_inversion, ctx, min_available_memory_gb, on_pool_started=None):
    if len(hit_names) > 0:
        tasks_follow = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion} for s in hit_names]
        random.shuffle(tasks_follow)

        monitor = MemoryMonitor()
        monitor.start()
        BUDGET.reserve(target_inversion, "core_shm", 0.0, block=True)
        # core_df_with_const is cast to float32 immediately after; 4 bytes per value
        bytes_needed = int(core_df_with_const.index.size) * int(len(core_df_with_const.columns)) * 4
        shm_gb = bytes_needed / (1024**3)
        BUDGET.revise(target_inversion, "core_shm", shm_gb)
        print(f"[Budget] {target_inversion}.core_shm: set {shm_gb:.2f}GB | remaining {BUDGET.remaining_gb():.2f}GB", flush=True)

        C = cpu_count()
        W_gb = max(0.25, _WORKER_GB_EST)
        max_by_budget = max(1, int(BUDGET.remaining_gb() // W_gb))
        n_procs = max(1, min(C, max_by_budget, POOL_PROCS_PER_INV))
        print(f"[Ancestry] Scheduling follow-up for {len(tasks_follow)} FDR-significant phenotypes ({n_procs} workers).", flush=True)
        bar_len = 40
        queued = 0
        done = 0

        def _print_bar(q, d, label):
            q = int(q); d = int(d)
            pct = int((d * 100) / q) if q else 0
            filled = int(bar_len * (d / q)) if q else 0
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            mem_info = (f" | App≈{monitor.rss_gb:.2f}GB  "
                        f"SysAvail≈{monitor.available_memory_gb:.2f}GB  "
                        f"Budget≈{BUDGET.remaining_gb():.2f}GB")
            PROGRESS.update(target_inversion, label, d, q)
            print(f"\r[{label}] {bar} {d}/{q} ({pct}%)" + mem_info, end="", flush=True)

        X_base = core_df_with_const.to_numpy(dtype=np.float32, copy=True)
        base_meta, base_shm = io.create_shared_from_ndarray(X_base, readonly=True)
        core_cols = list(core_df_with_const.columns)
        core_index = core_df_with_const.index.astype(str)
        del X_base, core_df_with_const
        gc.collect()
        BUDGET.reserve(target_inversion, "pool_steady", 0.0, block=True)

        def _task_iter():
            nonlocal queued
            for task in tasks_follow:
                floor = _resolve_floor(min_available_memory_gb)
                while BUDGET.remaining_gb() < floor:
                    print(
                        f"\n[gov WARN] Budget low (remain: {BUDGET.remaining_gb():.2f}GB, floor: {floor:.2f}GB), pausing task submission...",
                        flush=True,
                    )
                    time.sleep(2)

                follow_dir = ctx.get("LRT_FOLLOWUP_CACHE_DIR")
                if follow_dir:
                    res_path = os.path.join(follow_dir, f"{task['name']}.json")
                    meta_path = os.path.join(follow_dir, f"{task['name']}.meta.json")
                    _evict_if_ctx_mismatch(meta_path, res_path, ctx, target_inversion)

                queued += 1
                _print_bar(queued, done, "Ancestry")
                yield task

        try:
            with get_context(MP_CONTEXT).Pool(
                processes=n_procs,
                initializer=models.init_lrt_worker,
                initargs=(base_meta, core_cols, core_index, allowed_mask_by_cat, anc_series, ctx),
                maxtasksperchild=500,
            ) as pool:
                if on_pool_started:
                    try:
                        worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p and p.pid]
                    except Exception:
                        worker_pids = []
                    try:
                        on_pool_started(n_procs, worker_pids)
                    except Exception as e:
                        print(f"\n[WARN] on_pool_started callback failed: {e}", flush=True)

                try:
                    for _ in pool.imap_unordered(models.lrt_followup_worker, _task_iter(), chunksize=1):
                        done += 1
                        _print_bar(queued, done, "Ancestry")
                except Exception as exc:
                    _log_worker_exception("Followup", exc)
                    raise
                finally:
                    _print_bar(queued, done, "Ancestry")
                    print("")
        finally:
            base_shm.close()
            base_shm.unlink()
            BUDGET.release(target_inversion, "pool_steady")
            BUDGET.release(target_inversion, "core_shm")
            monitor.stop()
