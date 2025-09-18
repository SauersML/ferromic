import threading
import sys
from multiprocessing import get_context, cpu_count
import os
import json
import math
import gc
import numpy as np

from . import models
from . import iox as io
import time
import random

# PHEWAS main removed: Stage-1 (LRT/Bootstrap) is the only engine.


def cgroup_available_gb():
    """Return remaining memory permitted by the active cgroup, if known."""
    try:
        max_v2 = "/sys/fs/cgroup/memory.max"
        cur_v2 = "/sys/fs/cgroup/memory.current"
        if os.path.exists(max_v2) and os.path.exists(cur_v2):
            with open(max_v2, "r") as fh:
                raw = fh.read().strip()
            if raw != "max":
                limit = int(raw)
                with open(cur_v2, "r") as fh:
                    usage = int(fh.read().strip())
                return max(0.0, (limit - usage) / (1024**3))

        max_v1 = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        cur_v1 = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        if os.path.exists(max_v1) and os.path.exists(cur_v1):
            with open(max_v1, "r") as fh:
                limit = int(fh.read().strip())
            if limit < (1 << 60):  # guard "unlimited" sentinel
                with open(cur_v1, "r") as fh:
                    usage = int(fh.read().strip())
                return max(0.0, (limit - usage) / (1024**3))
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

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
POOL_PROCS_PER_INV = 8

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
    tasks = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion} for s in phenos_list]
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
        lock = threading.Lock()

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
        pool = get_context(MP_CONTEXT).Pool(
            processes=n_procs,
            initializer=models.init_lrt_worker,
            initargs=(base_meta, core_cols, core_index, allowed_mask_by_cat, anc_series, ctx),
            maxtasksperchild=500,
        )
        try:
            if on_pool_started:
                try:
                    worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p and p.pid]
                except Exception:
                    worker_pids = []
                try:
                    on_pool_started(n_procs, worker_pids)
                except Exception as e:
                    print(f"\n[WARN] on_pool_started callback failed: {e}", flush=True)

            inflight = []

            def _cb(_):
                nonlocal done, queued
                with lock:
                    done += 1
                    _print_bar(queued, done, "LRT-Stage1")

            failed_tasks = []
            def _err_cb(e):
                nonlocal failed_tasks
                print(f"[pool ERR] Worker failed: {e}", flush=True)
                failed_tasks.append(e)

            for task in tasks:
                floor = _resolve_floor(min_available_memory_gb)
                while BUDGET.remaining_gb() < floor:
                    print(f"\n[gov WARN] Budget low (remain: {BUDGET.remaining_gb():.2f}GB, floor: {floor:.2f}GB), pausing task submission...", flush=True)
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
                        _p = _res_obj.get("P_LRT_Overall", None)
                        _valid = False
                        try:
                            _pf = float(_p)
                            _valid = math.isfinite(_pf) and (0.0 < _pf < 1.0)
                        except Exception:
                            _valid = False
                        if not _valid:
                            try:
                                os.remove(_meta_path)
                                print(f"\n[cache POLICY] Invalid or missing P_LRT_Overall for '{task['name']}'. Forcing re-run by removing meta.", flush=True)
                            except Exception:
                                pass
                except Exception:
                    pass

                queued += 1
                ar = pool.apply_async(models.lrt_overall_worker, (task,), callback=_cb, error_callback=_err_cb)
                inflight.append(ar)
                _print_bar(queued, done, "LRT-Stage1")

            pool.close()
            for ar in inflight:
                ar.wait()
            pool.join()
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
    tasks = [{"name": s, "category": name_to_cat.get(s, None), "cdr_codename": cdr_codename, "target": target_inversion}
             for s in phenos_list]
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
        lock = threading.Lock()

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
        pool = None
        try:
            pool = get_context(MP_CONTEXT).Pool(
                processes=n_procs,
                initializer=models.init_boot_worker,
                initargs=(base_meta, boot_meta, core_cols, core_index, allowed_mask_by_cat, anc_series, ctx),
                maxtasksperchild=500,
            )
            if on_pool_started:
                try:
                    worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p and p.pid]
                except Exception:
                    worker_pids = []
                try:
                    on_pool_started(n_procs, worker_pids)
                except Exception as e:
                    print(f"\n[WARN] on_pool_started callback failed: {e}", flush=True)

            inflight = []

            def _cb(_):
                nonlocal done
                with lock:
                    done += 1
                    _print(queued, done)

            def _err_cb(e):
                print(f"[pool ERR] Worker failed: {e}", flush=True)

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
                ar = pool.apply_async(models.bootstrap_overall_worker, (task,), callback=_cb, error_callback=_err_cb)
                inflight.append(ar)
                _print(queued, done)

            pool.close()
            for ar in inflight:
                ar.wait()
            pool.join()
            _print(queued, done)
            print("")
        finally:
            try:
                if pool is not None:
                    pool.close()
                    pool.join()
            except Exception:
                try:
                    if pool is not None:
                        pool.terminate()
                except Exception:
                    pass
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
        lock = threading.Lock()

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
        pool = get_context(MP_CONTEXT).Pool(
            processes=n_procs,
            initializer=models.init_lrt_worker,
            initargs=(base_meta, core_cols, core_index, allowed_mask_by_cat, anc_series, ctx),
            maxtasksperchild=500,
        )
        try:
            if on_pool_started:
                try:
                    worker_pids = [p.pid for p in getattr(pool, "_pool", []) if p and p.pid]
                except Exception:
                    worker_pids = []
                try:
                    on_pool_started(n_procs, worker_pids)
                except Exception as e:
                    print(f"\n[WARN] on_pool_started callback failed: {e}", flush=True)

            inflight = []

            def _cb2(_):
                nonlocal done, queued
                with lock:
                    done += 1
                    _print_bar(queued, done, "Ancestry")

            failed_tasks = []

            def _err_cb(e):
                nonlocal failed_tasks
                print(f"[pool ERR] Worker failed: {e}", flush=True)
                failed_tasks.append(e)

            for task in tasks_follow:
                floor = _resolve_floor(min_available_memory_gb)
                while BUDGET.remaining_gb() < floor:
                    print(f"\n[gov WARN] Budget low (remain: {BUDGET.remaining_gb():.2f}GB, floor: {floor:.2f}GB), pausing task submission...", flush=True)
                    time.sleep(2)

                follow_dir = ctx.get("LRT_FOLLOWUP_CACHE_DIR")
                if follow_dir:
                    res_path = os.path.join(follow_dir, f"{task['name']}.json")
                    meta_path = os.path.join(follow_dir, f"{task['name']}.meta.json")
                    _evict_if_ctx_mismatch(meta_path, res_path, ctx, target_inversion)

                queued += 1
                ar = pool.apply_async(models.lrt_followup_worker, (task,), callback=_cb2, error_callback=_err_cb)
                inflight.append(ar)
                _print_bar(queued, done, "Ancestry")

            pool.close()
            for ar in inflight:
                ar.wait()
            pool.join()
            _print_bar(queued, done, "Ancestry")
            print("")
        finally:
            base_shm.close()
            base_shm.unlink()
            BUDGET.release(target_inversion, "pool_steady")
            BUDGET.release(target_inversion, "core_shm")
            monitor.stop()
